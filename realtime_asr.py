import sounddevice as sd
import numpy as np
from funasr import AutoModel

# =========================
# 1. 初始化模型
# =========================
model = AutoModel(
    model="paraformer-zh-streaming",
    device="mps"  # Intel Mac 改成 "cpu"
)

# =========================
# 2. 流式参数（官方推荐）
# =========================
sample_rate = 16000
chunk_size = [0, 10, 5]  # 600ms
encoder_chunk_look_back = 4
decoder_chunk_look_back = 1

chunk_stride = chunk_size[1] * 960  # 10 * 60ms * 16000 = 9600 samples

# 全局 cache（重点）
cache = {}

# 累计音频 buffer
audio_buffer = np.zeros((0,), dtype=np.float32)

# 上一次打印的文本（防止重复刷屏）
last_text = ""


# =========================
# 3. 回调函数
# =========================
def record_callback(indata, frames, time, status):
    global audio_buffer, last_text

    if status:
        print(status)

    # sounddevice: (frames, channels) → 1D float32
    audio = indata[:, 0].astype(np.float32)
    audio_buffer = np.concatenate([audio_buffer, audio])

    # 每满一个 chunk 才送模型
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

        if res and res[0].get("text"):
            text = res[0]["text"]
            if text != last_text:
                print(text, end="", flush=True)
                last_text = text


# =========================
# 4. 启动麦克风
# =========================
print("🎙 正在监听麦克风，说话即可（Ctrl+C 结束）")
with sd.InputStream(
    samplerate=sample_rate,
    channels=1,
    dtype="float32",
    callback=record_callback,
):
    try:
        while True:
            sd.sleep(1000)
    except KeyboardInterrupt:
        print("\n🛑 停止录音")

        # 通知模型最后一段
        model.generate(
            input=np.zeros((0,), dtype=np.float32),
            cache=cache,
            is_final=True,
            chunk_size=chunk_size,
            encoder_chunk_look_back=encoder_chunk_look_back,
            decoder_chunk_look_back=decoder_chunk_look_back,
        )
