"""LLM 字幕翻译模块：Qwen3-4B（4bit，约 2.5GB 显存，非思考模式）"""

import os
import re

# 先导入 subtitle_utils 以设置 HF 缓存环境变量（必须在 transformers 导入前）
import subtitle_utils  # noqa: F401

MODEL_ID = os.path.join(subtitle_utils.MODELS_CACHE_DIR, "qwen3-4b-hf")
# 预量化 4bit 缓存（用 make_4bit_cache.py 生成，加载 ~5s、免量化开销）；不存在时回退到 CPU 现量化
CACHE_ID = os.path.join(subtitle_utils.MODELS_CACHE_DIR, "qwen3-4b-bnb4bit")

# 注：曾尝试升级 Qwen3-4B-Instruct-2507，但其对批量+padding 生成存在严重缺陷
# （batch 中长度较短的行会输出异常，实测 padding 必现、非模板/attn 实现可解），
# 批量加速因此不可用；逐行又太慢，故保留 Qwen3-4B。

# Qwen3 偶发输出思考标签，即使 enable_thinking=False 也可能发生：
# 1) 完整思考块 <think>...</think> 或 <|im_start|>think...<|im_end|>
# 2) 模板提示尾部含 <think>\n\n</think>\n\n，模型会重复输出孤立的 </think>
_THINK_RE = re.compile(
    r"<\|im_start\|>think.*?<\|im_end\|>|<think>.*?</think>|<think>|</think>",
    re.S,
)

# 目标语言 -> 系统提示词（字幕翻译专家，只输出译文）
_PROMPTS = {
    "zh": (
        "你是专业的字幕翻译专家。把用户给出的台词翻译成自然流畅的中文。"
        "要求：1) 只输出译文，不要任何解释；"
        "2) 人名、地名、品牌名等专有名词保留原样；"
        "3) 译文口语自然，符合字幕阅读习惯；"
        "4) 保留语气、情绪和标点；"
        "5) 若用户提供前文，结合上下文准确翻译当前句。"
    ),
    "en": (
        "You are a professional subtitle translator. Translate the user's lines into natural English. "
        "Requirements: 1) output only the translation with no explanation; "
        "2) keep proper nouns (names, places, brands) as-is; "
        "3) the translation should sound natural and conversational, fitting subtitles; "
        "4) preserve tone, emotion and punctuation; "
        "5) if prior context is provided, use it to translate the current line accurately."
    ),
    "ja": (
        "あなたはプロの字幕翻訳者です。ユーザーの台詞を自然な日本語に翻訳してください。"
        "要件：1) 翻訳のみを出力し、説明は不要；"
        "2) 人名・地名・ブランド名などの固有名詞はそのまま；"
        "3) 字幕に合う自然で口語的な翻訳に；"
        "4) 語調・感情・句読点を保つ；"
        "5) 前文が提供された場合は、文脈に合わせて正確に翻訳してください。"
    ),
    "ko": (
        "당신은 전문 자막 번역가입니다. 사용자의 대사를 자연스러운 한국어로 번역하세요."
        "요구사항: 1) 번역문만 출력하고 설명하지 마세요;"
        "2) 인명, 지명, 브랜드명 등 고유명사는 그대로 유지하세요;"
        "3) 자막에 어울리는 자연스러운 구어체로 번역하세요;"
        "4) 어조, 감정, 문장 부호를 유지하세요;"
        "5) 앞선 문맥이 주어지면 문맥을 고려해 정확히 번역하세요."
    ),
}

_model = None
_tokenizer = None


def is_loaded():
    return _model is not None


def load():
    """加载模型（4bit 量化，约 2.5GB 显存）；重复调用直接返回"""
    global _model, _tokenizer
    if _model is not None:
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    if os.path.isdir(CACHE_ID):
        # 预量化缓存路径：直接读取量化权重，无量化开销
        _tokenizer = AutoTokenizer.from_pretrained(CACHE_ID)
        _model = AutoModelForCausalLM.from_pretrained(
            CACHE_ID, device_map={"": "cpu"}, low_cpu_mem_usage=True,
        )
    else:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        # 在 CPU 上完成 4bit 量化再整体移入显存：bnb 在 Blackwell（RTX 50 系）上
        # GPU 侧量化内核不稳定（偶发访问冲突）；本路径推理仍 100% 在 GPU 执行
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, quantization_config=quant,
            device_map={"": "cpu"}, low_cpu_mem_usage=True,
        )
    _model.to("cuda")


def unload():
    """释放模型，腾出显存"""
    global _model, _tokenizer
    _model = None
    _tokenizer = None
    try:
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass


BATCH_SIZE = 16


def translate_lines(lines, target_lang, progress_cb=None, contexts=None):
    """批量并行翻译台词，返回翻译后文本列表（与输入等长，空行保持空）

    每批 BATCH_SIZE 行拼成一个 batch 一次生成：摊薄每行的 prefill/内核启动
    固定开销（实测 16 行批量约每行 0.09s，逐行约 0.66s，快约 7 倍）。
    contexts: 与 lines 对齐的前文列表（可选，前 1-2 句原文），帮助模型
              结合上下文翻译指代、省略主语的台词。
    lines: list[str]
    target_lang: zh | en | ja | ko
    """
    if target_lang not in _PROMPTS:
        raise ValueError(f"不支持的目标语言: {target_lang}")

    load()
    import torch

    system = _PROMPTS[target_lang]
    result = []
    total = len(lines)
    done = 0
    for start in range(0, total, BATCH_SIZE):
        batch = lines[start:start + BATCH_SIZE]
        texts = [(t or "").strip() for t in batch]
        ctxs = None if contexts is None else contexts[start:start + BATCH_SIZE]
        msgs = []
        for i, t in enumerate(texts):
            ctx = (ctxs[i] if ctxs else "") or ""
            if ctx and t:
                user_content = f"前文：{ctx}\n当前句：{t}"
            else:
                user_content = t
            msgs.append([
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ])
        prompts = [
            _tokenizer.apply_chat_template(
                m, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            for m in msgs
        ]
        # 左 padding：右 padding 时 pad token 紧贴生成位置，Qwen3-4B 首个采样
        # logits 受其影响，偶发输出 "</think>" 及孤立单词噪声前缀；左 padding 无此问题
        _tokenizer.padding_side = "left"
        enc = _tokenizer(
            prompts, return_tensors="pt", padding=True,
            pad_token_id=_tokenizer.eos_token_id,
        )
        input_ids = enc["input_ids"].to("cuda")
        with torch.no_grad():
            out = _model.generate(
                input_ids=input_ids,
                attention_mask=enc["attention_mask"].to("cuda"),
                max_new_tokens=256,
                do_sample=False,
            )
        dec = _tokenizer.batch_decode(
            out[:, input_ids.shape[1]:], skip_special_tokens=True
        )
        for i, text in enumerate(texts):
            if not text:
                result.append("")
                continue
            resp = (dec[i] or "").strip()
            # 剥离偶发的思考块标签
            resp = _THINK_RE.sub("", resp).strip()
            # 去掉模型偶尔加的前缀/引号
            resp = resp.strip('"').strip()
            result.append(resp)
        done += len(batch)
        if progress_cb:
            progress_cb(done, total)
    return result
