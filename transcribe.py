"""FunASR 命令行转录：读取同目录 temp_audio.wav（16k 单声道），生成 SRT 字幕"""

import argparse
import os
import sys

# ===== pythonw.exe 无控制台修复 =====
# 无控制台模式下 sys.stdout/stderr 为 None，库内部写它们会导致崩溃
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

# 必须先于 funasr 导入 subtitle_utils（它会设置 HF 缓存环境变量）
import subtitle_utils
from subtitle_utils import save_subtitle_files, funasr_result_to_segments, clean_funasr_text

from funasr import AutoModel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def transcribe(audio_path, lang, device):
    """转录音频，lang: zh | en | ja | ko | yue
    - 中文: paraformer-zh(带 VAD+逐字时间戳) + ct-punc
    - 英/日/韩/粤: fsmn-vad 分段 + SenseVoiceSmall(多语种、自带标点)
    返回: [{'start': s, 'end': e, 'text': t}, ...]
    """
    if lang == "zh":
        asr = AutoModel(model="paraformer-zh", vad_model="fsmn-vad",
                        disable_update=True, device=device)
        punc = AutoModel(model="ct-punc", disable_update=True, device=device)
        res = asr.generate(input=audio_path, batch_size_s=300)
        segments = funasr_result_to_segments(res, punc)
    else:
        import soundfile as sf

        vad = AutoModel(model="fsmn-vad", disable_update=True, device=device)
        asr = AutoModel(model="iic/SenseVoiceSmall",
                        disable_update=True, device=device)
        vad_res = vad.generate(input=audio_path, max_single_segment_time=60000)
        vad_segs = vad_res[0]["value"] if vad_res else []
        if not vad_segs:
            raise RuntimeError("未检测到语音内容")

        audio, sr = sf.read(audio_path, dtype="float32")
        clips = [audio[int(s * sr / 1000):int(e * sr / 1000)] for s, e in vad_segs]

        res = asr.generate(input=clips, language=lang, use_itn=True, batch_size_s=300)
        segments = []
        for (s_ms, e_ms), item in zip(vad_segs, res):
            text = clean_funasr_text(item.get("text", ""))
            if text:
                segments.append({
                    "start": round(s_ms / 1000.0, 3),
                    "end": round(e_ms / 1000.0, 3),
                    "text": text,
                })

    if not segments:
        raise RuntimeError("未识别到语音内容")
    return segments


def main():
    parser = argparse.ArgumentParser(description="FunASR 转录 temp_audio.wav 生成字幕")
    parser.add_argument("--lang", default="zh",
                        choices=["zh", "en", "ja", "ko", "yue"],
                        help="语言（默认 zh）")
    parser.add_argument("--translate", default=None,
                        choices=["zh", "en", "ja", "ko"],
                        help="转录后用 LLM 翻译为目标语言")
    args = parser.parse_args()

    audio_file = os.path.join(SCRIPT_DIR, "temp_audio.wav")
    if not os.path.exists(audio_file):
        print(f"未找到音频文件: {audio_file}")
        sys.exit(1)

    try:
        import torch
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"
    print(f"设备: {device}, 语言: {args.lang}")

    print("加载 FunASR 模型...")
    segments = transcribe(audio_file, args.lang, device)
    print(f"转录完成，共 {len(segments)} 个片段")

    # 保存 SRT
    srt_path = save_subtitle_files(os.path.join(SCRIPT_DIR, "subtitles"), segments)

    if args.translate:
        import json
        import subprocess
        import tempfile
        print(f"正在用 LLM 翻译为 {args.translate}...")
        fd, in_path = tempfile.mkstemp(suffix=".json", prefix="tr_in_")
        fd2, out_path = tempfile.mkstemp(suffix=".json", prefix="tr_out_")
        os.close(fd)
        os.close(fd2)
        try:
            with open(in_path, "w", encoding="utf-8") as f:
                json.dump([{"text": s["text"]} for s in segments], f,
                          ensure_ascii=False)
            proc = subprocess.run(
                [sys.executable, os.path.join(SCRIPT_DIR, "translate_worker.py"),
                 in_path, args.translate, out_path],
                capture_output=True, text=True, encoding="utf-8",
                errors="replace",
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if proc.returncode != 0:
                raise RuntimeError(f"翻译进程异常退出（代码 {proc.returncode}）")
            with open(out_path, "r", encoding="utf-8") as f:
                translated = json.load(f)
        finally:
            for p in (in_path, out_path):
                try:
                    os.remove(p)
                except OSError:
                    pass
        for seg, t in zip(segments, translated):
            seg["translated_text"] = t
        from subtitle_utils import save_translated_srt
        srt_path = save_translated_srt(
            os.path.join(SCRIPT_DIR, "subtitles"), segments, args.translate)
        print(f"翻译字幕已保存: {srt_path}")

    print("SRT saved to: " + srt_path)
    print("")
    print("=== 字幕内容 ===")
    for seg in segments:
        start = seg["start"]
        end = seg["end"]
        text = seg.get("translated_text") or seg["text"]
        text = text.strip()
        m1, s1 = divmod(start, 60)
        m2, s2 = divmod(end, 60)
        print("[{:02d}:{:05.2f} -> {:02d}:{:05.2f}] {}".format(int(m1), s1, int(m2), s2, text))


if __name__ == "__main__":
    main()
