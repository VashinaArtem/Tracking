"""
benchmark_inference.py

Сравнение скорости инференса: vLLM (батч) vs model.chat() (последовательный).
Оба метода запускаются на одних и тех же 100 сэмплах из valid_u.

Usage:
    python benchmark_inference.py --gpu 1
    python benchmark_inference.py --gpu 1 --n_samples 100 --batch_size 16
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

SCRIPT_DIR  = Path(__file__).parent
MEVIS_DIR   = SCRIPT_DIR.parent
VALID_U_DIR = MEVIS_DIR / "data_sets/valid_u"
META_PATH   = VALID_U_DIR / "meta_expressions_v2.json"
IMAGES_DIR  = VALID_U_DIR / "JPEGImages"
MODEL_ID    = "openbmb/MiniCPM-V-4_5"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_samples(n_samples: int, seed: int = 42):
    with open(META_PATH) as f:
        meta = json.load(f)["videos"]

    items = []
    for vid, vdata in meta.items():
        frames = sorted((IMAGES_DIR / vid).glob("*.jpg"))
        for eid, edata in vdata["expressions"].items():
            expr = edata["exp"]
            if frames:
                frame = frames[len(frames) // 2]
                items.append((vid, eid, expr, frame))

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(items), size=min(n_samples, len(items)), replace=False)
    return [items[i] for i in idx]


def build_question(expr: str) -> str:
    return (
        f"Please provide the bounding box coordinate of the region this sentence "
        f"describes: <ref>{expr}</ref>. Return format: <box>x1 y1 x2 y2</box>"
    )


# ---------------------------------------------------------------------------
# vLLM benchmark
# ---------------------------------------------------------------------------

def run_vllm(samples, batch_size: int):
    import gc
    import torch
    from vllm import LLM, SamplingParams
    from vllm.distributed.parallel_state import destroy_model_parallel
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    _stop_names = ["<|im_end|>", "<|endoftext|>"]
    stop_token_ids = list({
        tid for name in _stop_names
        for tid in [tokenizer.convert_tokens_to_ids(name)]
        if isinstance(tid, int) and tid != tokenizer.unk_token_id
    } | {tokenizer.eos_token_id})

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=64,
        stop_token_ids=stop_token_ids,
        skip_special_tokens=False,
    )

    print(f"  Initialising vLLM (device=cuda:0 after CUDA_VISIBLE_DEVICES remap)...")
    t_load = time.time()
    llm = LLM(
        model=MODEL_ID,
        trust_remote_code=True,
        max_model_len=2048,
        gpu_memory_utilization=0.85,
        dtype="bfloat16",
        limit_mm_per_prompt={"image": 1},
    )
    print(f"  vLLM ready in {time.time() - t_load:.1f}s")

    # Подготовка инпутов
    vllm_inputs = []
    for _, _, expr, frame_path in samples:
        question = build_question(expr)
        messages = [{"role": "user", "content": f"(<image>./</image>)\n{question}"}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": False},
        )
        img = Image.open(frame_path).convert("RGB")
        vllm_inputs.append({"prompt": prompt, "multi_modal_data": {"image": img}})

    n = len(vllm_inputs)
    print(f"  Running {n} samples in batches of {batch_size}...")
    t0 = time.time()
    all_outputs = []
    for start in range(0, n, batch_size):
        batch = vllm_inputs[start : start + batch_size]
        outs = llm.generate(batch, sampling_params)
        all_outputs.extend(outs)
    elapsed = time.time() - t0

    # Освобождаем GPU перед запуском model.chat()
    print("  Releasing vLLM GPU memory...")
    del llm
    try:
        destroy_model_parallel()
    except Exception:
        pass
    gc.collect()
    torch.cuda.empty_cache()

    return elapsed, n


# ---------------------------------------------------------------------------
# model.chat() benchmark
# ---------------------------------------------------------------------------

def run_model_chat(samples):
    import torch
    from transformers import AutoModel, AutoTokenizer

    # После CUDA_VISIBLE_DEVICES= единственное устройство — cuda:0
    print(f"  Loading model on cuda:0 (remapped)...")
    t_load = time.time()
    model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    print(f"  model.chat() ready in {time.time() - t_load:.1f}s")

    n = len(samples)
    print(f"  Running {n} samples sequentially...")
    t0 = time.time()
    for _, _, expr, frame_path in samples:
        question = build_question(expr)
        img = Image.open(frame_path).convert("RGB")
        model.chat(
            image=img,
            msgs=[{"role": "user", "content": question}],
            tokenizer=tokenizer,
            sampling=False,
            max_new_tokens=64,
            enable_thinking=False,
        )
    elapsed = time.time() - t0

    return elapsed, n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu",        type=int, default=1)
    parser.add_argument("--n_samples",  type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--skip_chat",  action="store_true",
                        help="Пропустить model.chat() (если нет памяти для двух моделей)")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    print(f"\n{'='*60}")
    print(f"Benchmark: vLLM vs model.chat()  |  n={args.n_samples}  gpu={args.gpu}")
    print(f"{'='*60}\n")

    samples = load_samples(args.n_samples, seed=args.seed)
    print(f"Loaded {len(samples)} samples from valid_u\n")

    # ---- vLLM ----
    print("[1/2] vLLM")
    vllm_elapsed, n = run_vllm(samples, args.batch_size)
    vllm_tps = n / vllm_elapsed

    print(f"\n  vLLM results:")
    print(f"    Total time   : {vllm_elapsed:.1f}s")
    print(f"    Throughput   : {vllm_tps:.1f} samples/s")
    print(f"    Avg latency  : {vllm_elapsed / n * 1000:.0f} ms/sample\n")

    if args.skip_chat:
        print("[2/2] model.chat() skipped (--skip_chat)\n")
        return

    # ---- model.chat() ----
    print("[2/2] model.chat()")
    chat_elapsed, n = run_model_chat(samples)
    chat_tps = n / chat_elapsed

    print(f"\n  model.chat() results:")
    print(f"    Total time   : {chat_elapsed:.1f}s")
    print(f"    Throughput   : {chat_tps:.1f} samples/s")
    print(f"    Avg latency  : {chat_elapsed / n * 1000:.0f} ms/sample\n")

    # ---- Итог ----
    speedup = chat_elapsed / vllm_elapsed
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Samples        : {n}")
    print(f"  vLLM           : {vllm_elapsed:.1f}s  ({vllm_tps:.1f} samp/s)")
    print(f"  model.chat()   : {chat_elapsed:.1f}s  ({chat_tps:.1f} samp/s)")
    print(f"  Speedup (vLLM) : {speedup:.1f}x")
    print("=" * 60)


if __name__ == "__main__":
    main()
