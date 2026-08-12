"""一次性工具：从 bf16 模型生成 NF4 预量化缓存（CPU 量化，规避 Blackwell 量化内核崩溃）

用法: python make_4bit_cache.py <bf16模型目录> <输出缓存目录>
输出目录可直接被 translator.py 的 CACHE_ID 使用，加载 ~5s、免量化开销。
"""

import os
import sys


def main():
    if len(sys.argv) != 3:
        print("用法: python make_4bit_cache.py <bf16模型目录> <输出缓存目录>")
        sys.exit(1)
    model_id, cache_id = sys.argv[1], sys.argv[2]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    print("在 CPU 上加载并量化模型（需约 16GB 内存，请耐心等待）...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=quant,
        device_map={"": "cpu"}, low_cpu_mem_usage=True,
    )
    print("保存量化权重到缓存目录...")
    model.save_pretrained(cache_id, safe_serialization=True)
    print("保存 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(cache_id)
    print("完成:", cache_id)


if __name__ == "__main__":
    main()
