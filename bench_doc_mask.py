#!/usr/bin/env python
"""
bench_doc_mask.py — attention-kernel microbenchmark for the doc_attn_mask feature.

Compares, at skiff and dn4 attention dims:
  (a) SDPA is_causal=True (the production path today)
  (b) FlexAttention with a causal-only BlockMask (flex overhead, no sparsity win)
  (c) FlexAttention with the doc BlockMask (block sparsity from real BOS density)

All variants run compiled (production Attention modules are per-submodule
compiled). Doc lengths are drawn to mimic the packed stream (mean ~1k tokens).
Run on a rig: python bench_doc_mask.py
"""
import math
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from model_v2 import doc_ids_from_tokens, flex_attention, create_block_mask

DEV = "cuda"
BOS = 32000
MEAN_DOC = 1024


def packed_tokens(B, S, seed):
    g = torch.Generator().manual_seed(seed)
    t = torch.randint(0, 30000, (B, S), generator=g)
    for b in range(B):
        pos = int(torch.randint(1, MEAN_DOC, (1,), generator=g))
        while pos < S:
            t[b, pos] = BOS
            pos += max(16, int(MEAN_DOC * (0.25 + 1.5 * torch.rand((), generator=g).item())))
    return t.to(DEV)


def block_mask_for(tokens, doc_masked=True):
    doc = doc_ids_from_tokens(tokens, BOS)

    def mask_mod(b, h, q_idx, kv_idx):
        causal = q_idx >= kv_idx
        if doc_masked:
            return causal & (doc[b, q_idx] == doc[b, kv_idx])
        return causal

    B, S = tokens.shape
    return create_block_mask(mask_mod, B=B, H=None, Q_LEN=S, KV_LEN=S, device=tokens.device)


def bench(fn, *args, iters=30, warmup=8, backward=False):
    for _ in range(warmup):
        out = fn(*args)
        if backward:
            out.sum().backward()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn(*args)
        if backward:
            out.sum().backward()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0  # ms


def run(name, B, S, Hq, Hkv, D):
    print(f"\n--- {name}: B={B} S={S} Hq={Hq} Hkv={Hkv} D={D} (bf16) ---")
    torch.manual_seed(0)
    mk = lambda H: torch.randn(B, H, S, D, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    q, k, v = mk(Hq), mk(Hkv), mk(Hkv)
    tokens = packed_tokens(B, S, seed=1)

    sdpa = torch.compile(lambda q, k, v: F.scaled_dot_product_attention(
        q, k, v, is_causal=True, enable_gqa=True))
    flex = torch.compile(lambda q, k, v, bm: flex_attention(
        q, k, v, block_mask=bm, enable_gqa=True))

    bm_causal = block_mask_for(tokens, doc_masked=False)
    bm_doc = block_mask_for(tokens, doc_masked=True)
    # sparsity: fraction of blocks flex actually computes vs the causal mask
    print(f"  BlockMask density (doc vs causal): {bm_doc.sparsity():.1f}% sparse "
          f"vs {bm_causal.sparsity():.1f}% sparse")

    for tag, backward in (("fwd", False), ("fwd+bwd", True)):
        t_sdpa = bench(sdpa, q, k, v, backward=backward)
        t_fc = bench(flex, q, k, v, bm_causal, backward=backward)
        t_fd = bench(flex, q, k, v, bm_doc, backward=backward)
        print(f"  {tag:8s} sdpa-causal {t_sdpa:7.3f} ms | flex-causal {t_fc:7.3f} ms "
              f"| flex-doc {t_fd:7.3f} ms  ({t_sdpa / t_fd:.2f}x vs sdpa)")

    n = 20
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(n):
        block_mask_for(packed_tokens(B, S, seed=i), doc_masked=True)
    torch.cuda.synchronize()
    print(f"  create_block_mask (eager, incl. token gen): "
          f"{(time.perf_counter() - t0) / n * 1000:.2f} ms/batch")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        sys.exit("CUDA required")
    print(f"torch {torch.__version__} on {torch.cuda.get_device_name(0)}")
    run("skiff attn", B=4, S=2048, Hq=12, Hkv=6, D=64)
    run("dn4 attn", B=4, S=2048, Hq=32, Hkv=16, D=64)
    run("dn4 attn @4k", B=4, S=4096, Hq=32, Hkv=16, D=64)
