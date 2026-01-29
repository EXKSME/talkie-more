import sounddevice as sd
import numpy as np
import time
import requests
from funasr import AutoModel

# =========================
# 基本配置
# =========================
SAMPLE_RATE = 16000
CHUNK_SIZE = [0, 10, 5]          # FunASR 官方推荐
ENCODER_LOOK_BACK = 4
DECODER_LOOK_BACK = 1
CHUNK_STRIDE = CHUNK_SIZE[1] * 960  # 600ms
SILENCE_TIMEOUT = 0.5             # 句子结束阈值（秒）

OLLAMA_MODEL = "qwen2.5:1.5b"
OLLAMA_URL = "http://localhost:11434/api/generate"

# =========================
# 初始化 ASR 模型
# =========================
model = AutoModel(
    model="paraformer-zh-streaming",
    device="mps"  # Intel Mac 改成 "cpu"
)

cache = {}
audio_buffer = np.zeros((0,), dtype=np.float32)

last_partial_text = ""
last_text_change_time = time.time()

# =========================
# Ollama 调用
# =========================
def call_ollama(prompt: str) -> str:
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": "qwen3:0.6b",
            "prompt": prompt,
            "stream": False
        },
        timeout=60
    )

    data = resp.json()

    # 情况 1：经典 generate API
    if "response" in data:
        return data["response"]

    # 情况 2：chat 风格
    if "message" in data and "content" in data["message"]:
        return data["message"]["content"]

    # 情况 3：错误
    if "error" in data:
        raise RuntimeError(f"Ollama error: {data['error']}")

    # 兜底
    raise RuntimeError(f"Unknown Ollama response: {data}")

# =========================
# Router（规则优先，Demo 稳定）
# =========================
def route(text: str) -> str:
    if any(k in text for k in ["公式", "平方", "分之", "根号", "求和", "积分", "上标", "下标", "latex"]):
        return "latex"
    if any(k in text for k in ["流程图", "画个流程", "流程是", "如果", "否则", "mermaid"]):
        return "mermaid"
    if any(k in text for k in ["总结", "列一下", "要点", "几点", "步骤", "清单"]):
        return "markdown"
    return "plain"

# =========================
# Prompt 模板
# =========================
def build_prompt(text: str, mode: str) -> str:
    if mode == "markdown":
        return f"""
你是一个文本编辑器，而不是聊天助手。

请将下面的口述内容：
- 删除口语废话（如“我觉得”“然后”“其实”）
- 修正语法
- 如果是清单、步骤或要点，请结构化成 Markdown

【只输出 JSON，不要解释】
格式：
{{
  "type": "markdown",
  "title": "...",
  "blocks": [
    {{ "type": "paragraph", "text": "..." }},
    {{ "type": "bullets", "items": ["...", "..."] }},
    {{ "type": "steps", "items": ["...", "..."] }}
  ]
}}

【口述内容】
{text}
"""
    if mode == "latex":
        return f"""
你是一个公式转写器。

请将下面的口述数学表达转写为 LaTeX 公式。

【要求】
- 不做数学推导
- 只做表达映射
- 只输出 JSON

格式：
{{
  "type": "latex",
  "latex": "..."
}}

【口述内容】
{text}
"""
    if mode == "mermaid":
        return f"""
你是一个流程图生成器。

请根据下面的口述内容生成 Mermaid flowchart TD。

【要求】
- 只输出 JSON
- diagram 中必须是合法 Mermaid

格式：
{{
  "type": "mermaid",
  "diagram": "flowchart TD\\nA[开始] --> B[处理]"
}}

【口述内容】
{text}
"""
    return f"""
请将下面口述内容整理成简洁、通顺的书面语。

【只输出 JSON】
格式：
{{
  "type": "plain",
  "text": "..."
}}

【口述内容】
{text}
"""

# =========================
# 渲染器
# =========================
def render(result: dict) -> str:
    t = result.get("type")
    if t == "plain":
        return result["text"]

    if t == "markdown":
        lines = []
        if result.get("title"):
            lines.append(f"## {result['title']}\n")
        for b in result["blocks"]:
            if b["type"] == "paragraph":
                lines.append(b["text"] + "\n")
            elif b["type"] == "bullets":
                for i in b["items"]:
                    lines.append(f"- {i}")
                lines.append("")
            elif b["type"] == "steps":
                for idx, i in enumerate(b["items"], 1):
                    lines.append(f"{idx}. {i}")
                lines.append("")
        return "\n".join(lines)

    if t == "latex":
        return f"```latex\n{result['latex']}\n```"

    if t == "mermaid":
        return f"```mermaid\n{result['diagram']}\n```"

    return str(result)

# =========================
# 句子结束 → LLM 处理
# =========================
def process_final_sentence(text: str):
    mode = route(text)
    print(f"\n\n🧠 Router → {mode}")

    prompt = build_prompt(text, mode)
    response = call_ollama(prompt)

    try:
        data = eval(response)  # Demo 阶段可接受，后续换 json.loads
        output = render(data)
        print("\n📄 结构化输出：\n")
        print(output)
        print("\n" + "=" * 50)
    except Exception as e:
        print("⚠️ 解析失败，原始输出：")
        print(response)

# =========================
# 音频回调
# =========================
def record_callback(indata, frames, time_info, status):
    global audio_buffer, last_partial_text, last_text_change_time

    audio = indata[:, 0].astype(np.float32)
    audio_buffer = np.concatenate([audio_buffer, audio])

    while len(audio_buffer) >= CHUNK_STRIDE:
        chunk = audio_buffer[:CHUNK_STRIDE]
        audio_buffer = audio_buffer[CHUNK_STRIDE:]

        res = model.generate(
            input=chunk,
            cache=cache,
            is_final=False,
            chunk_size=CHUNK_SIZE,
            encoder_chunk_look_back=ENCODER_LOOK_BACK,
            decoder_chunk_look_back=DECODER_LOOK_BACK,
        )

        if res and res[0].get("text"):
            text = res[0]["text"]
            if text != last_partial_text:
                print(text, end="", flush=True)
                last_partial_text = text
                last_text_change_time = time.time()

    # 判断一句话结束
    if last_partial_text and (time.time() - last_text_change_time) > SILENCE_TIMEOUT:
        final_text = last_partial_text.strip()
        last_partial_text = ""
        process_final_sentence(final_text)

# =========================
# 主程序
# =========================
print("🎙 Typeless-like 本地语音输入 Demo")
print("👉 开始说话，停顿 0.5s 自动结构化")
print("👉 Ctrl+C 退出\n")

with sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32",
    callback=record_callback,
):
    try:
        while True:
            sd.sleep(1000)
    except KeyboardInterrupt:
        print("\n🛑 结束")