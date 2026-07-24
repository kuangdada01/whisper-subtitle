import time
import os

# ===== 国内镜像设置 =====
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import whisper
import numpy as np
from scipy.io import wavfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

print("Loading audio...")
audio_file = os.path.join(SCRIPT_DIR, "temp_audio.wav")
sample_rate, audio_data = wavfile.read(audio_file)
if audio_data.ndim > 1:
    audio_data = audio_data.mean(axis=1)
audio = audio_data.astype(np.float32) / 32768.0
print("Audio: {:.1f}s".format(len(audio) / sample_rate))

print("Loading whisper model (medium)...")
model_path = os.path.join(SCRIPT_DIR, "medium.pt")
model = whisper.load_model(model_path)
print("Device: {}".format(model.device))

print("Transcribing...")
start = time.time()
result = model.transcribe(audio, language="zh", verbose=False)
elapsed = time.time() - start
print("Done in {:.1f}s".format(elapsed))
print("Segments: {}".format(len(result["segments"])))
print("")
print("=== Subtitles ===")
for seg in result["segments"]:
    s, e = seg["start"], seg["end"]
    m1, s1 = divmod(s, 60)
    m2, s2 = divmod(e, 60)
    print("[{:02d}:{:05.2f} -> {:02d}:{:05.2f}] {}".format(int(m1), s1, int(m2), s2, seg["text"].strip()))
