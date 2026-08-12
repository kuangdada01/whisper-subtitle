"""共享字幕工具:时间戳格式化、SRT 生成、字幕文件保存"""

import json
import os
import re

# ===== 模型缓存目录 =====
# huggingface_hub 在 import 时读取环境变量，必须在任何 transformers/funasr 导入前设置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_CACHE_DIR = os.path.join(SCRIPT_DIR, "models_cache")
os.environ.setdefault("HF_HOME", os.path.join(MODELS_CACHE_DIR, "hf"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

RICH_TEXT_RE = re.compile(r"<\|[^|]*\|>")


def clean_funasr_text(text):
    """剥离 SenseVoice 输出中的 <|zh|><|NEUTRAL|> 等富文本标签"""
    return RICH_TEXT_RE.sub("", text or "").strip()


def format_timestamp(seconds):
    """将秒数格式化为 SRT 时间戳: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return "{:02d}:{:02d}:{:06.3f}".format(hours, minutes, secs).replace(".", ",")


def segments_to_srt(segments):
    """将 segment 列表转换为 SRT 文本"""
    lines = []
    for i, seg in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(
            format_timestamp(seg["start"])
            + " --> "
            + format_timestamp(seg["end"])
        )
        lines.append(seg["text"].strip())
        lines.append("")
    return "\n".join(lines)


def save_subtitle_files(base_path, segments):
    """保存 SRT 文件,返回 SRT 路径"""
    srt_path = base_path + ".srt"
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(segments_to_srt(segments))
    return srt_path


def save_translated_srt(base_path, segments, target_lang):
    """保存翻译后的 SRT：{base}.{target_lang}.srt（时间轴不变，不生成 JSON）"""
    translated = [
        {"start": s["start"], "end": s["end"], "text": s.get("translated_text") or s["text"]}
        for s in segments
    ]
    srt_path = f"{base_path}.{target_lang}.srt"
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(segments_to_srt(translated))
    return srt_path


def load_segments(path):
    """从 .srt 或 .json 读取字幕片段，返回 [{'start','end','text'}, ...]"""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    if ext == ".srt":
        segments = []
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        blocks = re.split(r"\n\s*\n", content.strip())
        for block in blocks:
            lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
            if len(lines) < 2 or "-->" not in lines[1]:
                continue

            def _ts(s):
                parts = s.replace(",", ".").split(":")
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])

            start, end = lines[1].split("-->")
            text = " ".join(lines[2:])
            segments.append({"start": round(_ts(start), 3), "end": round(_ts(end), 3), "text": text})
        return segments
    raise ValueError(f"不支持的文件类型: {path}")


def funasr_result_to_segments(res, punc_model=None, max_line_chars=20, min_pause=0.5):
    """把 FunASR paraformer 输出(逐字时间戳)切成字幕行,可选补标点

    res: model.generate() 的返回列表,每项含 'text'(空格分隔的逐字) 和
        'timestamp'(逐字 [start_ms, end_ms],与逐字 1:1 对齐)
    返回: [{'start': s, 'end': e, 'text': t}, ...]
    """
    lines = []
    for item in res:
        tokens = item["text"].split(" ")
        ts = item.get("timestamp")
        if not ts:
            text = item["text"].strip()
            if text:
                lines.append((text, float(item.get("start", 0)), float(item.get("end", 0))))
            continue
        n = min(len(tokens), len(ts))
        cur_chars, cur_ts = [], []
        for i in range(n):
            if cur_chars and (
                ts[i][0] - ts[i - 1][1] > min_pause * 1000
                or len(cur_chars) >= max_line_chars
            ):
                lines.append(("".join(cur_chars), cur_ts[0][0] / 1000.0, cur_ts[-1][1] / 1000.0))
                cur_chars, cur_ts = [], []
            cur_chars.append(tokens[i])
            cur_ts.append(ts[i])
        if cur_chars:
            lines.append(("".join(cur_chars), cur_ts[0][0] / 1000.0, cur_ts[-1][1] / 1000.0))

    if punc_model is not None and lines:
        texts = punc_model.generate(input=[l[0] for l in lines], batch_size_s=300)
        lines = [(t.get("text", l[0]), l[1], l[2]) for l, t in zip(lines, texts)]

    segments = []
    for text, s, e in lines:
        text = text.strip()
        if text:
            segments.append({"start": round(s, 3), "end": round(e, 3), "text": text})
    return segments
