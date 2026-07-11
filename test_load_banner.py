#!/usr/bin/env python
"""test_load_banner.py — end-to-end load-path smoke test.

Builds a tiny v4.0 festival checkpoint and runs it through the REAL
load_model_and_tokenizer (both _build_model_from_checkpoint AND the outer
function), asserting the load doesn't raise AND that the load-banner surfaces the
model facts + decision provenance. This exercises the actual runtime path, so it
catches the class of bug that unit-testing the decision LOGIC in isolation missed
(the v4.0->FSDP1 branch gate, the chk_meta-after-del UnboundLocalError).

Needs the llama tokenizer next to the repo (../tokenizers/llama_tokenizer); SKIPS
cleanly if absent. CPU-only.

    python test_load_banner.py
"""
import os, sys, io, tempfile, shutil, dataclasses
from contextlib import redirect_stdout

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
TOK = os.path.normpath(os.path.join(_HERE, "..", "tokenizers", "llama_tokenizer"))
SPECIAL = os.path.normpath(os.path.join(
    _HERE, "..", "..", "notebooks", "datasets", "tokenized", "llama", "tokenizer_config.json"))

if not os.path.exists(TOK):
    print(f"SKIP: llama tokenizer not found at {TOK}")
    sys.exit(0)

import neo_common as nc
from model_v2 import Transformer, ModelArgs


def main():
    torch.manual_seed(0)
    margs = ModelArgs(dim=64, n_layers=4, n_heads=4, n_kv_heads=2, vocab_size=256,
                      max_seq_len=256, dropout=0.0, qk_norm_mode="before_rope",
                      use_keel=True, use_activation_checkpointing=False,
                      tie_word_embeddings=False, bos_token_id=1, rope_theta=500000.0,
                      mtp_enabled=True, swa_enabled=True, swa_window=16,
                      swa_global_interleave=4, doc_attn_mask=True, doc_pos_reset=True)
    model = Transformer(margs).float().eval()

    d = tempfile.mkdtemp()
    try:
        ckpt = os.path.join(d, "model_step_000100.pt")
        torch.save({
            "model": dict(model.state_dict()),
            "config": dataclasses.asdict(margs),
            "step": 100, "total_tokens_processed": 12345678,
            "checkpoint_version": "4.0", "rope_fixed": True,
            "tok_kind": "llama", "tok_path": TOK, "special_tokens": SPECIAL,
            "optimizer_type": "normuon_fsdp2", "max_lr": 3.0e-4,
            "cpu_offload": False, "scs_settings": None,
        }, ckpt)

        buf = io.StringIO()
        err = None
        try:
            with redirect_stdout(buf):
                nc.load_model_and_tokenizer(ckpt, device="cpu", half_precision=False)
        except Exception:
            import traceback
            err = traceback.format_exc()
        out = buf.getvalue()
    finally:
        shutil.rmtree(d, ignore_errors=True)

    print(out)
    if err:
        print("*** LOAD RAISED ***\n" + err)
        sys.exit(1)

    # Every audit-promised line must appear (and the removed duplicates must NOT).
    checks = [
        ("format + model-source gate",
         "Detected FSDP2 checkpoint (v4.0)" in out and "common_fsdp2.model_v2" in out),
        ("generic metadata surfaces rope_fixed", "rope_fixed: True" in out),
        ("generic metadata surfaces optimizer_type", "optimizer_type: normuon_fsdp2" in out),
        ("dead use_adamc gone AND version not duplicated in metadata loop",
         "use_adamc" not in out and "  checkpoint_version:" not in out),
        ("single resolved Tokenizer line", "Tokenizer: kind=llama" in out),
        ("qk_norm provenance", "qk_norm_mode: before_rope (from checkpoint)" in out),
        ("precision logged on the full-precision branch", "full precision (float32)" in out),
        ("doc-mask / SWA / MTP features shown, one per line",
         "doc-mask: on" in out and "SWA: on (window=16" in out and "MTP: on" in out),
        ("params + MTP presence", "params," in out and "MTP module: present" in out),
        ("[rope] always-logs with provenance",
         "[rope] FIXED RoPE" in out and "AUTO from rope_fixed=True" in out),
        ("state dict clean", "State dict loaded cleanly" in out),
    ]
    bad = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print(f"  [{'ok' if ok else 'FAIL'}] {name}")
    print("ALL LOAD-BANNER CHECKS PASS" if not bad else f"*** {len(bad)} CHECK(S) FAILED ***")
    sys.exit(0 if not bad else 1)


if __name__ == "__main__":
    main()
