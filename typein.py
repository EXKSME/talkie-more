import sounddevice as sd
import numpy as np
import pyautogui
import pyperclip
from funasr import AutoModel
from queue import Queue
import time

# =========================
# 工具：diff 新增文本
# =========================
def diff_new_part(prev: str, curr: str):
    i = 0
    while i < len(prev) and i < len(curr) and prev[i] == curr[i]:
        i += 1
    return curr[i:]


# =========================
# 0. paste 工具（核心）
# =========================
def paste_text(text: str):
    pyperclip.copy(text)
    pyautogui.hotkey("command", "v")


# =========================
# 1. 初始化模型
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
# 3. 音频回调（只负责 ASR）
# =========================
def record_callback(indata, frames, time_info, status):
    global audio_buffer, last_text

    audio = indata[:, 0].astype(np.float32)
    audio_buffer = np.concatenate([audio_buffer, audio])

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
            print("🆕 new_part:", repr(new_part))
            text_queue.put(new_part)

        last_text = text


# =========================
# 4. 启动麦克风 & 主线程粘贴
# =========================
print("🎙 请把光标放在任意输入框（微信 / 记事本 / 浏览器都行）")
print("👉 连续说话 3~5 秒")

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
                text = text_queue.get()
                paste_text(text)   # ⭐⭐⭐ 核心在这里

            sd.sleep(20)

    except KeyboardInterrupt:
        print("\n🛑 stopped")

# 这是一个我本地部署的ai语音输入法然后呢第一点是可以做换行第二点是可以做处理第三点是可以做这个这个