import sounddevice as sd
import numpy as np
import pyautogui
import pyperclip
import requests
from funasr import AutoModel
from queue import Queue, Empty
import time
import threading
import re
import json
from difflib import SequenceMatcher

# =========================
# 参数区（你后面调参就调这里）
# =========================
SILENCE_TIMEOUT = 0.6          # 静音超过多少秒 -> commit
ENERGY_THRESHOLD = 0.008       # 静音能量阈值（不同麦克风要调，偏小更敏感）
MIN_COMMIT_GAP = 0.8           # 两次 commit 最小间隔（防抖）

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3:1.7b"

LLM_MODE = "smart_markdown"    # "clean" / "markdown" / "smart_markdown"

# 输出安全闸门阈值（更适配“结构重排”）
SAFE_SIM_HIGH = 0.70
SAFE_SIM_LOW  = 0.52
SAFE_NGRAM_COV = 0.48
SAFE_LEN_RATIO_MIN = 0.55
SAFE_LEN_RATIO_MAX = 1.60

# 粘贴节奏
pyautogui.PAUSE = 0.005


# =========================
# 工具：diff 新增文本
# =========================
def diff_new_part(prev: str, curr: str) -> str:
    i = 0
    while i < len(prev) and i < len(curr) and prev[i] == curr[i]:
        i += 1
    return curr[i:]


# =========================
# 工具：paste（核心）
# =========================
def paste_text(text: str):
    if not text:
        return
    pyperclip.copy(text)
    pyautogui.hotkey("command", "v")


# =========================
# 工具：回删 N 个字符（用于替换 preview）
# =========================
def delete_chars(n: int):
    if n <= 0:
        return
    pyautogui.press("backspace", presses=n, interval=0)


# =========================
# Step0：工程级去口语/语气词（已移除，改为LLM处理）
# =========================
# 语气词处理已改为通过LLM提示词完成，不再使用工程方式


# =========================
# Step1：工程预清洗（结构化前掰开粘连）
# =========================
_ORD_PATTERN = r'(第[一二三四五六七八九十]+个\s*[:：])'

def split_ordered_items(text: str) -> str:
    """把 '第一个：xx第二个：yy' 拆成多行（仅拆结构，不新增内容）"""
    if not text:
        return text
    if text.count("第") < 2:
        return text
    if not (("第二个" in text) or ("第三个" in text) or ("第四个" in text)):
        return text
    if not re.search(_ORD_PATTERN, text):
        return text

    parts = re.split(_ORD_PATTERN, text)
    if len(parts) <= 1:
        return text

    lines = []
    current = ""
    for part in parts:
        if re.match(_ORD_PATTERN, part):
            if current.strip():
                lines.append(current.strip())
            current = part
        else:
            current += part

    if current.strip():
        lines.append(current.strip())

    return "\n".join(lines)

def add_soft_breaks(text: str) -> str:
    """只做"更利于结构化"的轻度换行，不新增信息"""
    if not text:
        return ""
    t = text
    
    # 优先处理"第X点"模式（包括"第一点"、"第二点"、"第三点"等）
    # 匹配"第[一二三四五六七八九十]+点"或"第[0-9]+点"
    t = re.sub(r"([，。！？,\.\!\?\s]*)(第[一二三四五六七八九十0-9]+点)", r"\1\n\2", t)
    
    # 常见结构连接词/序号词前换行，帮助 LLM 感知"分点"
    keywords = [
        "首先", "其次", "然后", "另外", "最后",
        "第一", "第二", "第三", "第四", "第五",
    ]
    for w in keywords:
        # 情况1：句号/分号之后
        t = re.sub(rf"(。|；|;)\s*({w})", r"\1\n\2", t)
        # 情况2：行首或空格之后（但避免重复换行）
        t = re.sub(rf"(^|\s+)({w})", r"\1\n\2", t)
    
    # 清理多余的连续换行
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t

def preprocess_before_llm(raw_text: str) -> str:
    t = (raw_text or "").strip()
    if not t:
        return ""

    # 只做基本的空白规范化，保持原始格式（换行、列表等）
    t = re.sub(r"[ \t]+", " ", t)
    # 语气词处理已改为通过LLM提示词完成
    t = split_ordered_items(t)
    t = add_soft_breaks(t)

    # 你之前的处理保留
    t = t.replace("。-", "。\n-")

    # 统一换行
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


# =========================
# 输出安全闸门：更适配结构重排
# =========================
def normalize_for_guard(t: str) -> str:
    t = (t or "").replace("\r", "\n")
    # 轻度去 markdown
    t = re.sub(r"[#*`>\-]", "", t)
    # 去空白
    t = re.sub(r"\s+", "", t)
    return t

def build_ngrams(text: str, n: int = 3) -> set:
    text = normalize_for_guard(text)
    if len(text) < n:
        return set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}

def ngram_coverage(source: str, target: str, n: int = 3) -> float:
    src = build_ngrams(source, n)
    if not src:
        return 0.0
    tgt = normalize_for_guard(target)
    if len(tgt) < n:
        return 0.0

    hit = 0
    total = 0
    for i in range(len(tgt) - n + 1):
        total += 1
        if tgt[i:i+n] in src:
            hit += 1
    return hit / total if total else 0.0

def is_llm_output_safe(raw_text, out_text, mode="format"):
    """
    LLM 输出安全检查

    mode:
      - format  : 原有的“格式化 / 清洗”模式（偏严格）
      - reorder : 结构重排模式（允许重排、分组、结构词）
    """

    # 基础兜底
    if not out_text or not out_text.strip():
        return False

    raw_text = raw_text or ""

    # =========================
    # 结构重排模式（新）
    # =========================
    if mode == "reorder":
        # 1️⃣ 输出不能极端膨胀（防胡编）
        if len(out_text) > len(raw_text) * 3:
            return False

        # 2️⃣ 不能完全脱离原文（词汇完全不重合）
        raw_tokens = set(raw_text.replace("\n", " ").split())
        out_tokens = set(out_text.replace("\n", " ").split())

        # 允许的结构词（白名单，可慢慢加）
        structural_tokens = {
            "-", "*", "：", ":",
            "要点", "子要点", "功能", "特点", "描述", "说明"
        }

        out_tokens = {t for t in out_tokens if t not in structural_tokens}

        if not raw_tokens:
            return True  # 原文太短，直接放行

        overlap_ratio = len(raw_tokens & out_tokens) / max(1, len(out_tokens))

        # 经验阈值：30% 已经很宽松
        return overlap_ratio >= 0.3

    # =========================
    # 原有格式化模式（默认）
    # =========================
    # 👉 保留你原来的逻辑即可
    # 下面是一个“保守示例”，你可以替换成你原来的实现
    else:
        raw_simple = strip_formatting(raw_text)
        out_simple = strip_formatting(out_text)

        if not out_simple:
            return False

        # 简单相似度兜底（示意）
        if len(out_simple) < len(raw_simple) * 0.3:
            return False

        return True

# 这是一个我本地部署的i语音系统- 对一个语音输入法，可以去除口音，同时进行其他操作格式化的展示然后以及包括像一些要点比如说第一点是那个


# =========================
# LLM：通用调用（加 stop，减少 # /think 污染）
# =========================
def call_ollama(prompt: str, timeout: int = 40) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "top_p": 0.75,
            "repeat_penalty": 1.15
        },
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return (data.get("response") or "").strip()


# =========================
# 传统后处理（clean/markdown）
# =========================
def build_prompt_edit(raw_text: str, mode: str) -> str:
    base_rules = (
        "你是一个【文本后处理器】，只做编辑，不做解释。\n"
        "只允许修改原文表达，不允许补充、推测、解释。\n\n"
        "编辑规则：\n"
        "1. 删除口语填充词、语气词、重复词（如：嗯、呃、啊、那个、这个、然后、其实、就是、你知道、就是说）。\n"
        "2. 修正明显错别字和病句，使表达更通顺。\n"
        "3. 不新增任何信息，不推测、不补充、不评论。\n"
        "4. 必须保持原文的格式结构：保留所有换行、列表符号（-、1.等）、段落分隔。\n"
        "5. 如果原文有列表结构（有序或无序），必须保持列表格式，不要合并成一段。\n"
        "6. 只输出最终结果，不要输出编辑说明。\n"
        "7. 禁止输出：#、/think、<think>、解释性段落。\n"
    )

    if mode == "clean":
        # 关键修复：必须保持格式化，不要合并成一段，强制识别列表结构
        fmt = (
            "输出要求：\n"
            "- 输出为通顺中文。\n"
            "- 如果原文包含'第一点'、'第二点'、'第三点'、'首先'、'其次'、'最后'等列表标识词，必须格式化为列表，每项单独一行。\n"
            "- 列表格式：使用 '- ' 或 '1. ' 开头，每一点独立一行。\n"
            "- 必须保留原文的所有换行和段落分隔。\n"
            "- 如果原文有列表结构（有序或无序），必须保持列表格式，不要合并成一段。\n"
            "- 绝对不要将所有内容合并成一段连续文本。\n"
            "- 不要写标题，不要写解释。\n"
        )
    else:
        fmt = (
            "输出要求：\n"
            "- 如果原文包含'第一点'、'第二点'、'第三点'、'首先'、'其次'、'最后'等列表标识词，必须格式化为列表，每项单独一行。\n"
            "- 列表格式：使用 '- ' 或 '1. ' 开头，每一点独立一行。\n"
            "- 即使原文只有'第X点'（如'第三点是可以做这个'），也要格式化为列表项（如 '- 第三点是可以做这个'）。\n"
            "- 如果原文本本身是列点结构，必须使用列表符号（- 或 1.），每项单独一行。\n"
            "- 必须保留或恢复所有合理的换行和段落分隔。\n"
            "- 绝对不要将所有内容合并成一段连续文本。\n"
            "- 不要输出说明性词语，不要解释。\n"
            "- 不要使用 # 作为标题。\n"
        )

    return base_rules + fmt + "\n原始文本如下：\n" + (raw_text or "").strip()

def call_ollama_postprocess(raw_text: str, mode: str) -> str:
    raw_text = (raw_text or "").strip()
    if not raw_text:
        return ""
    prompt = build_prompt_edit(raw_text, mode)
    try:
        text = call_ollama(prompt, timeout=40)
        return text if text else raw_text
    except Exception as e:
        print("⚠️ Ollama call failed:", repr(e))
        return raw_text


# =========================
# Step2：结构化理解（LLM 输出 JSON）
# =========================

def build_prompt_reorder(raw_text: str) -> str:
    return (
        "你是一个【文本结构重排器】，不是总结器、不是解释器。\n"
        "目标：把口语化、零散的表达，重排为逻辑清晰的结构化文本。\n\n"

        "只允许做的事情：\n"
        "- 删除口语填充词（如：嗯、呃、啊、那个、然后、其实、就是）\n"
        "- 合并重复意思\n"
        "- 拆分长句\n"
        "- 调整顺序，让表达更清晰\n\n"

        "禁止：\n"
        "- 新增事实\n"
        "- 推测原文未提及的内容\n"
        "- 总结、升华、评价\n\n"

        "输出格式要求（必须遵守）：\n"
        "- 使用 Markdown\n"
        "- 一级结构使用无序列表 `-`\n"
        "- 子结构使用缩进两格的 `-`\n"
        "- 不要使用标题符号 `#`\n"
        "- 不要输出任何解释性文字\n\n"

        "示例格式：\n"
        "- 要点一\n"
        "  - 子要点\n"
        "- 要点二\n\n"

        "原始文本：\n"
        + (raw_text or "").strip()
    )
def build_prompt_struct(raw_text: str) -> str:
    return (
        "你是一个【结构重排器】，不是解释器、不是总结器。\n"
        "只做：删除口语填充词、拆分、换行、分组。禁止：推测、解释、补全。\n\n"
        "硬性约束：\n"
        "1) 只输出一个 JSON 对象，除此之外不要输出任何字符。\n"
        "2) 不新增事实，不推测，不补充未提及信息。\n"
        "3) 删除口语填充词/语气词/口头禅（如：嗯、呃、啊、那个、这个、然后、其实、就是、你知道、就是说）。\n"
        "4) 每条要点尽量短，一句话一个要点。\n"
        "5) 必须保持结构化输出：如果原文有列表结构，必须在JSON中正确分组为bullets和sub。\n"
        "6) 最多两层：bullets + sub。\n"
        "7) 禁止输出：#、/think、<think>、解释性句子。\n\n"
        "JSON 结构必须严格为：\n"
        "{\"title\":\"\",\"bullets\":[{\"text\":\"\",\"sub\":[{\"text\":\"\"}]}]}\n\n"
        "原始文本：\n"
        + (raw_text or "").strip()
    )

def strip_formatting(text: str) -> str:
    """
    仅用于 guard 比对：去掉排版符号、把换行当空格
    """
    if not text:
        return ""
    t = text
    t = re.sub(r"^\s*-\s*", "", t, flags=re.MULTILINE)
    t = re.sub(r"[#*`>]", "", t)
    t = re.sub(r"\n+", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def clean_json_string(json_str: str) -> str:
    """清理和修复常见的 JSON 格式问题"""
    if not json_str:
        return ""
    
    # 移除 markdown 代码块标记
    json_str = re.sub(r"^```(?:json)?\s*", "", json_str, flags=re.MULTILINE)
    json_str = re.sub(r"```\s*$", "", json_str, flags=re.MULTILINE)
    
    # 移除前后的非 JSON 字符（保留可能的空白）
    json_str = json_str.strip()
    
    # 尝试修复常见的 JSON 错误
    # 1. 修复键名中的单引号为双引号（使用更精确的正则）
    # 匹配 'key': 或 'key' : 这种模式
    json_str = re.sub(r"'([^']+)'\s*:", r'"\1":', json_str)
    
    # 2. 移除尾随逗号（在 } 或 ] 前，但要小心字符串中的逗号）
    # 使用负向前瞻确保不在字符串内
    json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
    
    # 3. 移除可能的注释（虽然 JSON 标准不支持）
    json_str = re.sub(r'//.*?$', '', json_str, flags=re.MULTILINE)
    json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
    
    return json_str.strip()

def extract_first_json(text: str) -> str:
    if not text:
        return ""
    start = text.find("{")
    if start < 0:
        return ""
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                extracted = text[start:i+1].strip()
                # 清理提取的 JSON
                return clean_json_string(extracted)
    return ""

def parse_outline(json_text: str):
    if not json_text:
        return None
    
    # 先尝试直接解析
    try:
        obj = json.loads(json_text)
    except json.JSONDecodeError as e:
        # 如果失败，尝试清理后再解析
        cleaned = clean_json_string(json_text)
        try:
            obj = json.loads(cleaned)
        except json.JSONDecodeError:
            # 如果还是失败，打印调试信息
            print(f"[debug] JSON parse failed at position {e.pos}: {e.msg}")
            print(f"[debug] JSON preview: {repr(json_text[:200])}")
            print(f"[debug] Error context: {repr(json_text[max(0, e.pos-20):e.pos+20])}")
            return None
    
    if not isinstance(obj, dict):
        return None

    title = obj.get("title", "")
    bullets = obj.get("bullets", [])

    if not isinstance(title, str):
        title = ""
    if not isinstance(bullets, list):
        return None

    cleaned = []
    for b in bullets:
        if not isinstance(b, dict):
            continue
        text = b.get("text", "")
        if not isinstance(text, str):
            continue

        # 保持LLM输出的原始格式，不再进行工程级清洗
        text = text.strip()
        if not text:
            continue

        sub_list = b.get("sub", [])
        sub_clean = []
        if isinstance(sub_list, list):
            for s in sub_list:
                st = s.get("text", "") if isinstance(s, dict) else s
                if isinstance(st, str):
                    st = st.strip()
                    if st:
                        sub_clean.append({"text": st})

        cleaned.append({"text": text, "sub": sub_clean})

    if not cleaned:
        return None

    cleaned = cleaned[:8]
    for b in cleaned:
        b["sub"] = b["sub"][:8]

    # 保持LLM输出的原始格式，不再进行工程级清洗
    return {"title": title.strip(), "bullets": cleaned}

def outline_to_markdown(outline: dict) -> str:
    lines = []
    title = (outline.get("title") or "").strip()
    if title:
        lines.append(f"**{title}**")

    for b in outline.get("bullets", []):
        lines.append(f"- {b['text']}")
        for s in b.get("sub", []):
            lines.append(f"  - {s['text']}")

    return "\n".join(lines).strip()

def normalize_markdown(text: str) -> str:
    lines = []
    for line in text.splitlines():
        line = line.rstrip()

        # 丢掉明显的废话
        if not line:
            continue
        if line.startswith(("解释", "说明", "注意")):
            continue

        # 只保留 markdown 列表行
        if line.lstrip().startswith("-"):
            lines.append(line)

    return "\n".join(lines)

def smart_struct_then_render(raw_text: str) -> str:
    raw_text = (raw_text or "").strip()
    if not raw_text:
        return ""

    pre = preprocess_before_llm(raw_text)

    try:
        prompt = build_prompt_reorder(pre)  # 注意：不再是 build_prompt_struct
        resp = call_ollama(prompt, timeout=50)

        md = normalize_markdown(resp)

        if not md.strip():
            print("[debug] struct_reorder: empty markdown output")
            return ""

        result = md.strip()
        if result:
            print("[debug] smart_struct: success, output preview:", repr(result[:200]))
        else:
            print("[debug] smart_struct: markdown conversion returned empty")
        return result
    except json.JSONDecodeError as e:
        # JSON 解析错误已经在 parse_outline 中处理了，这里只是兜底
        print(f"⚠️ smart_struct: JSON decode error at position {e.pos}: {e.msg}")
        return ""
    except Exception as e:
        print("⚠️ smart_struct_then_render failed:", repr(e))
        import traceback
        traceback.print_exc()
        return ""


# =========================
# 1. 初始化 ASR 模型
# =========================
model = AutoModel(
    model="paraformer-zh-streaming",
    device="mps"
)

# =========================
# 2. 流式参数
# =========================
sample_rate = 16000
chunk_size = [0, 10, 5]
encoder_chunk_look_back = 4
decoder_chunk_look_back = 1
chunk_stride = chunk_size[1] * 960

cache = {}
audio_buffer = np.zeros((0,), dtype=np.float32)
last_text = ""

text_queue = Queue()

# =========================
# 3. 运行时状态（preview + commit）
# =========================
state_lock = threading.Lock()

preview_raw_text = ""
preview_len = 0
last_voice_time = time.time()
last_commit_time = 0.0
committing = False


# =========================
# 4. 音频回调（只负责 ASR + 能量检测 + 推送增量）
# =========================
def record_callback(indata, frames, time_info, status):
    global audio_buffer, last_text, last_voice_time

    audio_mono = indata[:, 0].astype(np.float32)
    rms = float(np.sqrt(np.mean(audio_mono * audio_mono)) + 1e-12)
    if rms > ENERGY_THRESHOLD:
        last_voice_time = time.time()

    audio_buffer = np.concatenate([audio_buffer, audio_mono])

    while len(audio_buffer) >= chunk_stride:
        chunk = audio_buffer[:chunk_stride]
        audio_buffer = audio_buffer[chunk_stride:]

        res = model.generate(
            input=chunk,
            cache=cache,
            is_final=False,
            chunk_size=chunk_size,
            encoder_chunk_look_back=encoder_chunk_look_back,
            decoder_chunk_look_back=decoder_chunk_look_back,
        )

        # 修复点：这里不要 return（会中断本轮后续 chunk 处理），改 continue
        if not res or not res[0].get("text"):
            continue

        text = res[0]["text"]
        new_part = diff_new_part(last_text, text)

        if new_part and new_part.strip():
            text_queue.put(new_part)

        last_text = text


# =========================
# 5. Commit：静音后触发 LLM → 安全闸门 → 替换 preview
# =========================
def try_commit_if_needed():
    global preview_raw_text, preview_len, last_commit_time, committing

    now = time.time()

    with state_lock:
        if committing:
            return
        if preview_len <= 0:
            return
        if now - last_voice_time < SILENCE_TIMEOUT:
            return
        if now - last_commit_time < MIN_COMMIT_GAP:
            return

        committing = True
        raw_to_process = preview_raw_text
        chars_to_delete = preview_len

    print("\n🧠 commit trigger -> post-process...")

    # 1️⃣ 工程预清洗（只做安全、确定性的事）
    raw_clean = preprocess_before_llm(raw_to_process)

    processed = ""

    # ===============================
    # 2️⃣ 结构重排主路径（smart_markdown）
    # ===============================
    if LLM_MODE == "smart_markdown":
        processed = smart_struct_then_render(raw_clean)

        if processed:
            # ⚠️ 注意：结构重排模式下，只做“底线 guard”
            if not is_llm_output_safe(
                raw_clean,
                processed,
                mode="reorder"   # 👈 关键：告诉 guard 这是重排
            ):
                print("🧯 guard rejected reordered output -> fallback")
                processed = ""

        # ===============================
        # 3️⃣ fallback：markdown → clean
        # ===============================
        if not processed:
            print("[debug] smart_struct empty/rejected, trying markdown mode...")
            processed = call_ollama_postprocess(raw_clean, mode="markdown").strip()

            # markdown 也失败（没结构）
            if not processed or not any(c in processed for c in ['\n', '-', '*', '1.', '2.', '3.']):
                print("[debug] markdown mode weak, trying clean mode...")
                processed = call_ollama_postprocess(raw_clean, mode="clean").strip()

    # ===============================
    # 4️⃣ 非 smart_markdown 模式（旧模式）
    # ===============================
    else:
        processed = call_ollama_postprocess(raw_clean, LLM_MODE).strip()

        if processed:
            if not is_llm_output_safe(raw_clean, processed, mode="format"):
                print("🧯 guard rejected output -> keep raw_clean")
                processed = raw_clean

    # ===============================
    # 5️⃣ 最终兜底
    # ===============================
    if not processed:
        processed = raw_clean if raw_clean else raw_to_process

    # ===============================
    # 6️⃣ Debug 观察
    # ===============================
    try:
        print("[debug] raw_clean preview:", repr(raw_clean[:200]))
        print("[debug] processed preview:", repr(processed[:200]))
    except Exception:
        pass

    # ===============================
    # 7️⃣ 提交到“文档”
    # ===============================
    delete_chars(chars_to_delete)
    paste_text(processed)

    with state_lock:
        preview_raw_text = ""
        preview_len = 0
        last_commit_time = time.time()
        committing = False

    print("✅ commit done\n")

# 这是一个本地部署的ai语音

# =========================
# 6. 启动麦克风 & 主线程输出
# =========================
print("🎙 请把光标放在任意输入框（微信 / 记事本 / 浏览器都行）")
print("👉 preview 实时出字，停顿后 commit 会用结构化+美化替换（带安全闸门）")
print(f"👉 模式：{LLM_MODE} | 静音阈值：{SILENCE_TIMEOUT}s | 模型：{OLLAMA_MODEL}")

with sd.InputStream(
    samplerate=sample_rate,
    channels=1,
    dtype="float32",
    blocksize=1024,
    callback=record_callback,
):
    try:
        while True:
            # 更稳的 queue 读取方式
            while True:
                try:
                    new_text = text_queue.get_nowait()
                except Empty:
                    break

                paste_text(new_text)

                with state_lock:
                    preview_raw_text += new_text
                    preview_len += len(new_text)

            try_commit_if_needed()
            sd.sleep(20)

    except KeyboardInterrupt:
        print("\n🛑 stopped")

# 