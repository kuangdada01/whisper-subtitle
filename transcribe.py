import os
import sys
import json
import numpy as np

# ===== 国内镜像设置 =====
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import whisper
from scipy.io import wavfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

print("Loading audio...")
audio_file = os.path.join(SCRIPT_DIR, "temp_audio.wav")
sample_rate, audio_data = wavfile.read(audio_file)
# Convert to float32 mono
if audio_data.ndim > 1:
    audio_data = audio_data.mean(axis=1)
audio = audio_data.astype(np.float32) / 32768.0
print(f"Audio loaded: {len(audio)/sample_rate:.1f}s, {sample_rate}Hz")

print("Loading whisper model (medium for better Chinese accuracy)...")
model_path = os.path.join(SCRIPT_DIR, "medium.pt")
model = whisper.load_model(model_path)
print("Model loaded. Transcribing...")

result = model.transcribe(
    audio,
    language="zh",
    verbose=False
)

print("Transcription complete.")
print(f"Total segments: {len(result['segments'])}")

# Save SRT
srt_lines = []
for i, seg in enumerate(result["segments"], 1):
    start = seg["start"]
    end = seg["end"]
    text = seg["text"].strip()

    sh = int(start // 3600)
    sm = int((start % 3600) // 60)
    ss = start % 60
    eh = int(end // 3600)
    em = int((end % 3600) // 60)
    es = end % 60

    srt_lines.append(str(i))
    srt_lines.append(
        "{:02d}:{:02d}:{:06.3f}".format(sh, sm, ss).replace(".", ",")
        + " --> "
        + "{:02d}:{:02d}:{:06.3f}".format(eh, em, es).replace(".", ",")
    )
    srt_lines.append(text)
    srt_lines.append("")

srt_path = os.path.join(SCRIPT_DIR, "subtitles.srt")
with open(srt_path, "w", encoding="utf-8") as f:
    f.write("\n".join(srt_lines))

# Save JSON
json_path = os.path.join(SCRIPT_DIR, "subtitles.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(result["segments"], f, ensure_ascii=False, indent=2)

print("SRT saved to: " + srt_path)
print("JSON saved to: " + json_path)
print("")
print("=== 字幕内容 ===")
for seg in result["segments"]:
    start = seg["start"]
    end = seg["end"]
    text = seg["text"].strip()
    m1, s1 = divmod(start, 60)
    m2, s2 = divmod(end, 60)
    print("[{:02d}:{:05.2f} -> {:02d}:{:05.2f}] {}".format(int(m1), s1, int(m2), s2, text))
