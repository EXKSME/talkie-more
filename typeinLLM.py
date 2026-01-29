import sounddevice as sd
import numpy as np
import pyautogui
import pyperclip
import requests
from funasr import AutoModel
from queue import Queue
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

# 输出安全闸门阈值（建议先用这组，后面再微调）
SAFE_SIM_HIGH = 0.70
SAFE_SIM_LOW  = 0.55
SAFE_NGRAM_COV = 0.55

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


def preprocess_before_llm(raw_text: str) -> str:
    t = (raw_text or "").strip()
    if not t:
        return ""
    t = re.sub(r"[ \t]+", " ", t)
    t = split_ordered_items(t)
    t = t.replace("。-", "。\n-")
    return t.strip()


# =========================
# 输出安全闸门：字符相似度 + ngram 覆盖率（实时友好）
# =========================
def normalize_for_guard(t: str) -> str:
    # 去掉 markdown 符号、空白、常见噪声
    t = (t or "")
    t = t.replace("\r", "\n")
    t = re.sub(r"[#*`>\-]", "", t)     # 轻度去 markdown
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

def is_llm_output_safe(raw_text: str, processed_text: str) -> bool:
    raw_n = normalize_for_guard(raw_text)
    out_n = normalize_for_guard(processed_text)
    if not raw_n or not out_n:
        return False

    sim = SequenceMatcher(None, raw_n, out_n).ratio()
    cov = ngram_coverage(raw_n, out_n, n=3)

    # 你可以把这两行 print 打开，调参用
    # print(f"[guard] sim={sim:.3f} cov={cov:.3f}")

    if sim >= SAFE_SIM_HIGH:
        return True
    if sim >= SAFE_SIM_LOW and cov >= SAFE_NGRAM_COV:
        return True
    return False


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
            "repeat_penalty": 1.15,
            # 关键：一旦开始吐这些，就截断
            "stop": ["\n#", "\n/think", "/think", "<think>", "</think>"],
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
        "1. 删除口语填充词、重复词（如：呃、啊、然后、其实、就是）。\n"
        "2. 修正明显错别字和病句，使表达更通顺。\n"
        "3. 不新增任何信息，不推测、不补充、不评论。\n"
        "4. 只输出最终结果，不要输出编辑说明。\n"
        "5. 禁止输出：#、/think、<think>、解释性段落。\n"
    )

    if mode == "clean":
        fmt = (
            "输出要求：\n"
            "- 只输出一段连续中文文本。\n"
            "- 不要标题，不要列表，不要空行。\n"
        )
    else:
        fmt = (
            "输出要求：\n"
            "- 仅在原文本本身明显是列点时，才使用列表符号（- 或 1.）。\n"
            "- 不要输出说明性词语，不要解释。\n"
            "- 不要使用 # 作为标题。\n"
        )

    return base_rules + fmt + "\n原始文本如下：\n" + raw_text.strip()


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
def build_prompt_struct(raw_text: str) -> str:
    return (
        "你是一个【结构重排器】，不是解释器、不是总结器。\n"
        "只做：拆分、换行、分组。禁止：推测、解释、补全。\n\n"
        "硬性约束：\n"
        "1) 只输出一个 JSON 对象，除此之外不要输出任何字符。\n"
        "2) 不新增事实，不推测，不补充未提及信息。\n"
        "3) 每条要点尽量短，一句话一个要点。\n"
        "4) 最多两层：bullets + sub。\n"
        "5) 禁止输出：#、/think、<think>、解释性句子（如“询问/是否/可能/用于/表示”）。\n\n"
        "JSON 结构必须严格为：\n"
        "{\"title\":\"\",\"bullets\":[{\"text\":\"\",\"sub\":[{\"text\":\"\"}]}]}\n\n"
        "原始文本：\n"
        + raw_text.strip()
    )

def strip_formatting(text: str) -> str:
    """
    去掉所有排版信息，只保留“纯内容”
    """
    if not text:
        return ""

    t = text
    # 去 markdown 符号
    t = re.sub(r"^\s*-\s*", "", t, flags=re.MULTILINE)
    t = re.sub(r"^\s*\*\*|\*\*\s*$", "", t, flags=re.MULTILINE)
    t = re.sub(r"[#*`>]", "", t)

    # 把换行当成空格
    t = re.sub(r"\n+", " ", t)

    # 压缩空白
    t = re.sub(r"\s+", " ", t)

    return t.strip()

def extract_first_json(text: str) -> str:
    """括号匹配抽取第一个完整 JSON 对象"""
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
                return text[start:i+1].strip()
    return ""


def parse_outline(json_text: str):
    """解析并校验 outline 结构"""
    obj = json.loads(json_text)
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

    return {"title": title.strip(), "bullets": cleaned}


def outline_to_markdown(outline: dict) -> str:
    """工程渲染 Markdown（稳定）"""
    lines = []
    title = (outline.get("title") or "").strip()
    if title:
        lines.append(f"**{title}**")

    for b in outline.get("bullets", []):
        lines.append(f"- {b['text']}")
        for s in b.get("sub", []):
            lines.append(f"  - {s['text']}")

    return "\n".join(lines).strip()


def smart_struct_then_render(raw_text: str) -> str:
    """两阶段：结构化(JSON) -> 工程渲染 Markdown；失败返回空串"""
    raw_text = (raw_text or "").strip()
    if not raw_text:
        return ""

    pre = preprocess_before_llm(raw_text)

    try:
        prompt = build_prompt_struct(pre)
        resp = call_ollama(prompt, timeout=50)
        js = extract_first_json(resp)
        if not js:
            return ""

        outline = parse_outline(js)
        if not outline:
            return ""

        md = outline_to_markdown(outline)
        return md.strip() if md.strip() else ""
    except Exception as e:
        print("⚠️ smart_struct_then_render failed:", repr(e))
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
# 3. 运行时状态（核心：preview + commit）
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

        if not res or not res[0].get("text"):
            return

        text = res[0]["text"]
        new_part = diff_new_part(last_text, text)

        if new_part.strip():
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

    raw_clean = preprocess_before_llm(raw_to_process)

    processed = ""
    if LLM_MODE == "smart_markdown":
        processed = smart_struct_then_render(raw_clean)

        # 安全闸门：挡掉推测性输出
        if processed:
            raw_for_guard = strip_formatting(raw_to_process)
            out_for_guard = strip_formatting(processed)

            if not is_llm_output_safe(raw_for_guard, out_for_guard):
                print("🧯 guard rejected output -> fallback clean")
                processed = ""

        if not processed:
            processed = call_ollama_postprocess(raw_clean, mode="clean").strip()

    else:
        processed = call_ollama_postprocess(raw_clean, LLM_MODE).strip()

        if processed and not is_llm_output_safe(raw_to_process, processed):
            print("🧯 guard rejected output -> keep raw")
            processed = raw_to_process

    if not processed:
        processed = raw_to_process

    delete_chars(chars_to_delete)
    paste_text(processed)

    with state_lock:
        preview_raw_text = ""
        preview_len = 0
        last_commit_time = time.time()
        committing = False

    print("✅ commit done\n")


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
            while not text_queue.empty():
                new_text = text_queue.get()

                paste_text(new_text)

                with state_lock:
                    preview_raw_text += new_text
                    preview_len += len(new_text)

            try_commit_if_needed()
            sd.sleep(20)

    except KeyboardInterrupt:
        print("\n🛑 stopped")