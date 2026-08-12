"""翻译子进程工作器：由 GUI 在独立进程中启动，加载翻译模型完成翻译

独立进程是为了隔离 bnb 与 funasr 在同一进程内交替加载时触发的崩溃。
用法: python translate_worker.py <输入.srt/.json> <目标语言> <输出.json>
输入: 字幕文件（srt 或 json 片段列表），翻译目标语言 zh/en/ja/ko
输出: JSON 数组（与输入片段等长的译文列表），标准输出打印进度
"""

import json
import os
import sys


def main():
    input_path, target_lang, output_path = sys.argv[1], sys.argv[2], sys.argv[3]

    from subtitle_utils import load_segments
    segments = load_segments(input_path)
    if not segments:
        raise RuntimeError("字幕文件为空或格式无法解析")

    # 为每行构建前文（前 2 句原文），帮助模型结合上下文翻译
    contexts = []
    prev = []
    for s in segments:
        contexts.append("\n".join(prev[-2:]))
        prev.append((s.get("text") or "").strip())
        prev = [p for p in prev if p][-2:]

    import translator
    translated = translator.translate_lines(
        [s["text"] for s in segments], target_lang,
        progress_cb=lambda i, n: print(f"__PROGRESS__ {i} {n}", flush=True),
        contexts=contexts,
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(translated, f, ensure_ascii=False)
    print("__DONE__", flush=True)


if __name__ == "__main__":
    main()
