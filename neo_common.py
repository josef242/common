"""neo_common.py
Shared helpers for both generate_neo.py and chat_neo.py so that the two entry-point
scripts can stay thin while avoiding duplication.

Exports (see __all__):
    • detect_device               – smarter CUDA/MPS/CPU picker with --gpu support
    • load_model_and_tokenizer    – unified loader with optional Accelerate sharding
    • fast_generate / stream_generate – space‑safe, sentencepiece‑aware sampling
    • load_yaml_prompt / load_prompt  – YAML/plain prompt loaders
    • logger & print_and_log      – re‑export of the project’s TCP logger

NOTE: Evaluation utilities (MMLU, HellaSwag, batch-loss helpers) were intentionally
kept inside *generate_neo.py* only, as chat_neo does not need them.

NOTE: This file supports both FSDP1 (v2.0) and FSDP2 (v3.0) checkpoints via dynamic imports.
"""

from __future__ import annotations

import os
import sys
import time
from contextlib import nullcontext
from typing import List, Tuple, Optional

import torch
from torch.nn import functional as F  # noqa: F401 (still used in fast_generate)

# ---------------------------------------------------------------------------
# Project‑local deps
# ---------------------------------------------------------------------------
from tokenizer_abstraction import get_tokenizer, LlamaTokenizerAdapter  # type: ignore
from model_v2 import Transformer, ModelArgs  # type: ignore
import logger                       # TCP logger module bundled with the repo

# Register model_v1 in sys.modules so old FSDP1 checkpoints can unpickle
# (their pickled ModelArgs objects reference 'model_v1' as the source module).
# Skipped silently if saved_model_files/model_v1.py isn't present — only old
# FSDP1 (v2.0) checkpoints need it; FSDP2 (v3.0+) checkpoints work without it.
import importlib, importlib.util
_v1_path = os.path.join(os.path.dirname(__file__), "saved_model_files", "model_v1.py")
if "model_v1" not in sys.modules and os.path.isfile(_v1_path):
    _v1_spec = importlib.util.spec_from_file_location("model_v1", _v1_path)
    if _v1_spec and _v1_spec.loader:
        # Stub out bitsandbytes if missing — only needed for 8-bit training optimizers
        _bnb_stub = "bitsandbytes" not in sys.modules
        if _bnb_stub:
            import types as _types
            sys.modules["bitsandbytes"] = _types.ModuleType("bitsandbytes")
        _v1_mod = importlib.util.module_from_spec(_v1_spec)
        sys.modules["model_v1"] = _v1_mod
        _v1_spec.loader.exec_module(_v1_mod)
        if _bnb_stub:
            del sys.modules["bitsandbytes"]

# Optional Accelerate sharding
try:
    from accelerate import dispatch_model, infer_auto_device_map  # type: ignore
except ImportError:  # keep import‑time light when accelerate is not present
    dispatch_model = infer_auto_device_map = None  # type: ignore

__all__ = [
    "detect_device",
    "load_model_and_tokenizer",
    "stream_generate_kv",
    "CacheState",
    "verify_kv_reuse_parity",
    "generate_with_stats",
    "load_yaml_prompt",
    "load_prompt",
    # logger re‑export
    "logger",
    "print_and_log",
    "trim_messages_inplace",
]

print_and_log = logger.print_and_log  # convenience alias

# ---------------------------------------------------------------------------
# 1. Device helpers
# ---------------------------------------------------------------------------

def detect_device(preferred_gpu: Optional[int] = None) -> str:
    """Pick a CUDA device intelligently or fall back to MPS/CPU."""
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        logger.print_and_log(f"Found {n} CUDA device(s)")

        if preferred_gpu is not None:
            if 0 <= preferred_gpu < n:
                logger.print_and_log(f"Using specified GPU {preferred_gpu}")
                return f"cuda:{preferred_gpu}"
            if preferred_gpu == -1:
                logger.print_and_log(f"Using last GPU ({n-1})")
                return f"cuda:{n-1}"
            logger.print_and_log(f"GPU {preferred_gpu} not available, falling back to 0")
            return "cuda:0"

        # auto‑pick path
        return "cuda:1" if n > 1 else "cuda:0"

    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# ---------------------------------------------------------------------------
# 2. Model & tokenizer loader (unified)
# ---------------------------------------------------------------------------

def create_balanced_device_map(model, n_gpus: int):
    """Create a balanced device map that splits layers evenly across GPUs."""
    device_map = {}
    
    # Get all named modules
    named_modules = dict(model.named_modules())
    
    # Find transformer/layer blocks
    layer_names = []
    for name, module in named_modules.items():
        # Look for transformer blocks/layers
        if any(keyword in name.lower() for keyword in ['layer', 'block', 'transformer']) and \
           not any(keyword in name.lower() for keyword in ['layernorm', 'norm', 'embed']):
            # Check if this is a leaf block (no sub-blocks)
            is_leaf = True
            for other_name in named_modules:
                if other_name.startswith(name + '.') and \
                   any(keyword in other_name.lower() for keyword in ['layer', 'block']):
                    is_leaf = False
                    break
            if is_leaf and '.' in name:  # Not the root module
                layer_names.append(name)
    
    # Sort to ensure consistent ordering
    layer_names = sorted(layer_names)
    logger.print_and_log(f"Found {len(layer_names)} transformer blocks to distribute")
    
    # Put embeddings and early layers on GPU 0
    for name, module in named_modules.items():
        if any(keyword in name.lower() for keyword in ['embed', 'wte', 'wpe', 'tok_embeddings']):
            device_map[name] = 0
    
    # Distribute layers across GPUs
    if layer_names:
        layers_per_gpu = len(layer_names) // n_gpus
        extra_layers = len(layer_names) % n_gpus
        
        current_gpu = 0
        for i, layer_name in enumerate(layer_names):
            device_map[layer_name] = current_gpu
            
            # Check if we should move to next GPU
            layers_on_current = i - (current_gpu * layers_per_gpu) + 1
            if current_gpu < extra_layers:
                threshold = layers_per_gpu + 1
            else:
                threshold = layers_per_gpu
                
            if layers_on_current >= threshold and current_gpu < n_gpus - 1:
                current_gpu += 1
    
    # Put final layers (lm_head, norm) on last GPU
    for name, module in named_modules.items():
        if any(keyword in name.lower() for keyword in ['lm_head', 'output', 'final', 'norm']) and \
           name not in device_map:
            device_map[name] = n_gpus - 1
    
    return device_map

# qk_norm_force defaults to None (auto-detect)
def _build_model_from_checkpoint(checkpoint_path: str, enc, half_precision: bool, qk_norm_mode=None, use_keel=None):
    """Internal helper: construct ``Transformer`` instance and load weights."""

    chk = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)

    # Detect checkpoint version and import appropriate model
    checkpoint_version = chk.get("checkpoint_version", "2.0")  # Default to 2.0 for old checkpoints
    checkpoint_step = chk.get("step", chk.get("iter", "unknown"))

    if _ckpt_version_ge(checkpoint_version, 3.0):
        # FSDP2 checkpoint (v3.0 and up — v4.0 adds festival self-describe +
        # rope_fixed but is the SAME format) - use common_fsdp2 model.
        from model_v2 import Transformer, ModelArgs
        logger.print_and_log(f"Detected FSDP2 checkpoint (v{checkpoint_version}), step: {checkpoint_step} "
                             f"-> common_fsdp2.model_v2 (filter-based config)")
    else:
        # FSDP1 checkpoint - use saved FSDP1 model (has __call__ override for KV cache)
        from common_fsdp1.model_v2 import Transformer, ModelArgs
        logger.print_and_log(f"Detected FSDP1 checkpoint (v{checkpoint_version}), step: {checkpoint_step} "
                             f"-> common_fsdp1.model_v2 (explicit-arg config)")

    # Temp Hack for old checkpoint compatibility
    import model_v2 as _model_v2
    import sys
    sys.modules['model_v1_AdamC'] = _model_v2

    cfg = chk["config"]
    _qk_src = "model default"   # overwritten in the dict-config path below

    # Handle both old (dataclass) and new (dict) checkpoint formats
    if isinstance(cfg, dict):
        # New format - config is a dictionary
        # Backwards compatibility for when inner_dim was called hidden_dim
        if "hidden_dim" in cfg and "inner_dim" not in cfg:
            cfg["inner_dim"] = cfg["hidden_dim"]

        # Determine qk_norm_mode — track PROVENANCE (shown in the banner) since
        # qk_norm changes attention numerics; a CLI override that disagrees with
        # the checkpoint is exactly the case a human needs to see.
        if qk_norm_mode is not None:
            # Explicit override from command line
            resolved_qk_norm_mode = qk_norm_mode if qk_norm_mode != "none" else None
            _qk_src = "CLI override"
        else:
            # Auto-detect from checkpoint
            if "qk_norm_mode" in cfg:
                # New format: mode string directly in config
                resolved_qk_norm_mode = cfg["qk_norm_mode"]
                _qk_src = "from checkpoint"
            elif cfg.get("qk_norm", False):
                # Old format: boolean qk_norm=True → legacy behavior
                resolved_qk_norm_mode = "after_rope_legacy"
                _qk_src = "legacy qk_norm=True coercion"
            else:
                # No QK norm
                resolved_qk_norm_mode = None
                _qk_src = "checkpoint has no qk_norm"

        if _ckpt_version_ge(checkpoint_version, 3.0):
            # FSDP2 (v3.0+): use filter approach - pass all config keys through to
            # ModelArgs (filtered to known fields below). Future-proof, handles
            # MoE + festival params automatically.
            import dataclasses
            model_args = dict(cfg)
            # Festival-feature fallback: checkpoints saved before the whitelist
            # gained doc/swa/mtp fields (early wizard-era) don't self-describe
            # those semantics. Recover them from the newest config_*.yaml sitting
            # next to the checkpoint (the trainer always writes one). Without
            # this, an SWA checkpoint would silently generate FULL-CAUSAL and an
            # MTP module's weights would be dropped on load.
            if 'swa_enabled' not in model_args or 'mtp_enabled' not in model_args:
                try:
                    import glob as _glob
                    import yaml as _yaml
                    _ckdir = os.path.dirname(os.path.abspath(checkpoint_path))
                    _cfgs = sorted(_glob.glob(os.path.join(_ckdir, 'config_*.yaml')))

                    # The trainer's derived-fields block dumps some tuples with
                    # the !!python/tuple tag (e.g. an empty restart_steps), which
                    # yaml.safe_load REFUSES — crashing recovery and silently
                    # dropping SWA/MTP semantics (generation goes full-causal).
                    # Tolerate that ONE tag (construct it as a list — safe, no
                    # arbitrary-object construction, and we only read scalar/dict
                    # festival fields) so the parse survives.
                    class _CfgLoader(_yaml.SafeLoader):
                        pass
                    _CfgLoader.add_constructor(
                        'tag:yaml.org,2002:python/tuple',
                        lambda _ld, _node: _ld.construct_sequence(_node))

                    def _extract(path):
                        with open(path, 'r', encoding='utf-8') as _f:
                            _rc = _yaml.load(_f, Loader=_CfgLoader) or {}
                        _dm = _rc.get('doc_attn_mask') or {}
                        _sw = _rc.get('swa') or {}
                        _mt = _rc.get('mtp') or {}
                        if _dm is True: _dm = {'enabled': True}
                        if _sw is True: _sw = {'enabled': True}
                        if _mt is True: _mt = {'enabled': True}
                        return {
                            'doc_attn_mask': bool(_dm.get('enabled', False)),
                            'doc_pos_reset': bool(_dm.get('reset_positions', False)),
                            # default 1 = SentencePiece BOS (the actual doc
                            # separator in the tokenized shards), matching the
                            # trainer's Settings default — NOT the <|bos|>=32000
                            # special token. Only used if the yaml omits it.
                            'bos_token_id': int(_dm.get('bos_token_id', 1)),
                            'swa_enabled': bool(_sw.get('enabled', False)),
                            'swa_window': int(_sw.get('window', 512)),
                            'swa_global_interleave': int(_sw.get('global_interleave', 4)),
                            'mtp_enabled': bool(_mt.get('enabled', False)),
                        }

                    if _cfgs:
                        _all = {c: _extract(c) for c in _cfgs}
                        _recovered = _all[_cfgs[-1]]
                        if len({tuple(sorted(v.items())) for v in _all.values()}) > 1:
                            # a resumed run dir whose restarts DISAGREE on the
                            # festival flags: the newest yaml may postdate this
                            # checkpoint's semantics — the user must adjudicate
                            logger.print_and_log(
                                f"[warn] {len(_cfgs)} config_*.yaml in the run dir DISAGREE on "
                                f"festival flags — using the NEWEST ({os.path.basename(_cfgs[-1])}). "
                                f"If this checkpoint predates a flag flip, generation semantics "
                                f"will be wrong; verify against the config saved nearest the "
                                f"checkpoint's launch.")
                        model_args.update(_recovered)
                        if any((_recovered['doc_attn_mask'], _recovered['swa_enabled'],
                                _recovered['mtp_enabled'])):
                            logger.print_and_log(
                                f"Festival features recovered from {os.path.basename(_cfgs[-1])}: "
                                f"doc_mask={_recovered['doc_attn_mask']} "
                                f"(reset={_recovered['doc_pos_reset']}, bos={_recovered['bos_token_id']}), "
                                f"swa={_recovered['swa_enabled']} (W={_recovered['swa_window']}, "
                                f"int={_recovered['swa_global_interleave']}), "
                                f"mtp={_recovered['mtp_enabled']}")
                        else:
                            # recovery ran and parsed cleanly but every flag is
                            # off — state that outcome so it's distinguishable
                            # from "recovery never ran".
                            logger.print_and_log(
                                f"Festival features recovered from {os.path.basename(_cfgs[-1])}: "
                                f"all off (doc_mask/swa/mtp=False)")
                    else:
                        # No run config next to the checkpoint (the documented
                        # copy-to-V:/code/ckpt workflow strips it). MTP is
                        # detectable from weights; SWA is NOT — warn LOUDLY.
                        _has_mtp_w = any(
                            k.replace('_orig_mod.', '').startswith('mtp.')
                            for k in chk.get('model', {}))
                        logger.print_and_log(
                            f"[warn] checkpoint config lacks festival-feature fields and NO "
                            f"config_*.yaml sits next to it — cannot recover doc-mask/SWA/MTP "
                            f"semantics. MTP weights {'DETECTED' if _has_mtp_w else 'not found'} "
                            f"in the state dict. If this checkpoint was trained with SWA, "
                            f"generation will be FULL-CAUSAL (silently wrong) — copy the run's "
                            f"config_*.yaml next to the checkpoint and reload.")
                        if _has_mtp_w:
                            model_args['mtp_enabled'] = True
                except Exception as _e:
                    logger.print_and_log(
                        f"[warn] festival-feature recovery from run-dir yaml failed "
                        f"({type(_e).__name__}: {_e}) — if this checkpoint used SWA, "
                        f"generation semantics will be WRONG (full-causal).")
            # Override inference-specific settings
            model_args["ep_degree"] = 1  # single GPU inference
            model_args["use_activation_checkpointing"] = False
            model_args["dropout"] = 0.0
            model_args["qk_norm_mode"] = resolved_qk_norm_mode
            # use_keel: CLI override takes priority, then checkpoint value
            if use_keel is not None:
                model_args["use_keel"] = use_keel
                logger.print_and_log(f"  [use_keel] {use_keel} (CLI override — checkpoint said "
                                     f"{cfg.get('use_keel', 'unset')})")
            # Backwards compat: tie_word_embeddings defaults to True for old checkpoints
            if "tie_word_embeddings" not in model_args:
                model_args["tie_word_embeddings"] = True
                logger.print_and_log("  [tie_word_embeddings] True (ASSUMED — absent from checkpoint config)")
            # Filter to known ModelArgs fields; surface any dropped keys so a
            # schema drift between the checkpoint and the current ModelArgs is
            # visible rather than silent.
            known_fields = {f.name for f in dataclasses.fields(ModelArgs)}
            _dropped = sorted(set(model_args) - known_fields)
            if _dropped:
                logger.print_and_log(f"  [config] {len(_dropped)} checkpoint field(s) not in ModelArgs, "
                                     f"ignored: {_dropped}")
            model_args = {k: v for k, v in model_args.items() if k in known_fields}
            cfg = ModelArgs(**model_args)
        else:
            # FSDP1: explicit parameter list (no MoE support)
            model_args = dict(
                dim=cfg["dim"],
                n_layers=cfg["n_layers"],
                n_heads=cfg["n_heads"],
                n_kv_heads=cfg.get("n_kv_heads", None),
                vocab_size=cfg["vocab_size"],
                inner_dim=cfg.get("inner_dim", None),
                norm_eps=cfg.get("norm_eps", 1e-5),
                max_seq_len=cfg["max_seq_len"],
                dropout=cfg.get("dropout", 0.0),
                pad_id=cfg.get("pad_id", 0),
                use_activation_checkpointing=False,
                qk_norm_mode=resolved_qk_norm_mode,
                tie_word_embeddings=cfg.get("tie_word_embeddings", True),
                rope_theta=cfg.get("rope_theta", 10000.0),
            )
            cfg = ModelArgs(**model_args)
    elif not hasattr(cfg, "vocab_size"):
        # Very old format - pre-ModelArgs checkpoints (always FSDP1)
        logger.print_and_log("[legacy] very old pre-ModelArgs checkpoint: FABRICATING "
                             "vocab_size=32000, rope_theta=10000, n_kv_heads=None (not stored)")
        cfg = ModelArgs(
            dim=cfg["dim"],
            n_layers=cfg["n_layers"],
            n_heads=cfg["n_heads"],
            n_kv_heads=None,
            vocab_size=32000,
            inner_dim=None,
            norm_eps=1e-5,
            max_seq_len=cfg["max_seq_len"],
            dropout=cfg["dropout"],
            rope_theta=10000.0,
        )
        cfg.use_activation_checkpointing = False
    else:
        # Old format - already a model_v1 ModelArgs dataclass; convert to v2 ModelArgs
        import dataclasses as _dc
        _old_fields = {f.name: getattr(cfg, f.name) for f in _dc.fields(cfg) if f.name != "multiple_of"}
        _old_fields["use_activation_checkpointing"] = False
        _old_fields.setdefault("tie_word_embeddings", True)
        if "rope_theta" not in _old_fields:
            _old_fields["rope_theta"] = 10000.0   # v1 default was 10000, v2 default is 500000
            logger.print_and_log("[legacy] old v1 ModelArgs: defaulting rope_theta=10000 "
                                 "(v1 default; NOT read from checkpoint)")
        _known = {f.name for f in _dc.fields(ModelArgs)}
        cfg = ModelArgs(**{k: v for k, v in _old_fields.items() if k in _known})

    cfg.pad_id = enc.pad_id

    # Print model config in a readable format
    logger.print_and_log("Model configuration:")
    logger.print_and_log(f"  dim: {cfg.dim}, n_layers: {cfg.n_layers}, n_heads: {cfg.n_heads}, n_kv_heads: {cfg.n_kv_heads}")
    logger.print_and_log(f"  vocab_size: {cfg.vocab_size}, max_seq_len: {cfg.max_seq_len}")
    logger.print_and_log(f"  inner_dim: {cfg.inner_dim}, norm_eps: {cfg.norm_eps}")
    logger.print_and_log(f"  rope_theta: {getattr(cfg, 'rope_theta', 'N/A')}, qk_norm_mode: {cfg.qk_norm_mode} ({_qk_src})")
    logger.print_and_log(f"  tie_word_embeddings: {getattr(cfg, 'tie_word_embeddings', 'N/A')}, dropout: {cfg.dropout}")
    if getattr(cfg, 'use_keel', False):
        logger.print_and_log(f"  use_keel: {cfg.use_keel}, keel_alpha: {cfg.keel_alpha}")
    if getattr(cfg, 'moe_enabled', False):
        moe_info = f"  MoE: {cfg.moe_num_experts} experts, top-{cfg.moe_top_k}"
        if getattr(cfg, 'moe_num_shared_experts', 0) > 0:
            moe_info += f", {cfg.moe_num_shared_experts} shared"
        n_head_dense = getattr(cfg, 'moe_n_dense_layers', 0)
        n_tail_dense = getattr(cfg, 'moe_n_tail_dense_layers', 0)
        if n_head_dense > 0 or n_tail_dense > 0:
            moe_info += f", dense: {n_head_dense} head + {n_tail_dense} tail"
        if getattr(cfg, 'moe_interleave_step', 1) > 1:
            moe_info += f", interleave={cfg.moe_interleave_step}"
        logger.print_and_log(moe_info)
    # Festival features — these change generation SEMANTICS (windowed vs full-
    # causal, doc isolation, spec-decode availability), so surface them
    # unconditionally from the resolved cfg. An affirmative "none" line is
    # printed when all three are off so their absence is stated, not just missing.
    _festival = []
    if getattr(cfg, 'doc_attn_mask', False):
        _festival.append(f"doc-mask on (reset_positions={getattr(cfg, 'doc_pos_reset', False)}, "
                         f"bos={getattr(cfg, 'bos_token_id', '?')})")
    if getattr(cfg, 'swa_enabled', False):
        _festival.append(f"SWA window={getattr(cfg, 'swa_window', '?')}, "
                        f"interleave={getattr(cfg, 'swa_global_interleave', '?')}")
    if getattr(cfg, 'mtp_enabled', False):
        _festival.append("MTP on (self-speculative decode available)")
    logger.print_and_log(f"  festival: {' | '.join(_festival) if _festival else 'none (all off)'}")
    if getattr(cfg, 'gdn_enabled', False):
        logger.print_and_log(f"  GDN: interleave={getattr(cfg, 'gdn_interleave_step', '?')}, "
                             f"n_heads={getattr(cfg, 'n_gdn_heads', '?')}, "
                             f"head_dim={getattr(cfg, 'gdn_head_dim', '?')}, "
                             f"mode={getattr(cfg, 'gdn_mode', '?')}")
    if getattr(cfg, 'aux_head_layers', None):
        logger.print_and_log(f"  aux heads @ layers {list(cfg.aux_head_layers)}")

    model = Transformer(cfg)

    if half_precision:
        logger.print_and_log("Loading model in half precision (bfloat16)")
        model = model.to(torch.bfloat16)
    else:
        logger.print_and_log("Loading model in full precision (float32)")

    # Clean state‑dict keys (training wrappers)
    state_dict = chk["model"]
    prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(prefix):
            state_dict[k[len(prefix):]] = state_dict.pop(k)

    # EP expert consolidation: when training used Expert Parallel (ep_degree > 1),
    # the main checkpoint only has rank 0's local experts. Load the consolidated
    # ep_experts file which contains ALL experts.
    raw_cfg = chk.get("config", {})
    ep_degree = raw_cfg.get("ep_degree", 1) if isinstance(raw_cfg, dict) else getattr(raw_cfg, "ep_degree", 1)
    if ep_degree > 1:
        step = chk.get("step", 0)
        ckpt_dir = os.path.dirname(checkpoint_path)
        ep_path = os.path.join(ckpt_dir, f"ep_experts_step_{step:06d}.pt")
        if os.path.exists(ep_path):
            logger.print_and_log(f"EP checkpoint (ep_degree={ep_degree}): loading consolidated experts from {os.path.basename(ep_path)}")
            ep_experts = torch.load(ep_path, map_location="cpu", weights_only=True)
            overlaid = 0
            for key, val in ep_experts.items():
                # Clean key prefix if needed
                clean_key = key[len(prefix):] if key.startswith(prefix) else key
                state_dict[clean_key] = val
                overlaid += 1
            del ep_experts
            logger.print_and_log(f"  Overlaid {overlaid} expert parameters")
        else:
            raise FileNotFoundError(
                f"Checkpoint has ep_degree={ep_degree} but missing consolidated experts file: {ep_path}\n"
                f"Re-save from training with the updated save_model to fix."
            )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.print_and_log(f"[warn] Missing {len(missing)} keys in state‑dict: {missing[:10]}")
    if unexpected:
        logger.print_and_log(f"[warn] Unexpected {len(unexpected)} keys in state‑dict: {unexpected[:10]}")
    if not missing and not unexpected:
        logger.print_and_log("State dict loaded cleanly (no missing/unexpected keys)")

    # Model size + MTP-module presence (MTP gates self-speculative decode). Report
    # for EVERY load path — previously params/size were computed only inside the
    # balanced-sharding branch, so a plain single-GPU load showed neither.
    _nparams = sum(p.numel() for p in model.parameters())
    _size_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3
    _mtp_present = getattr(model, 'mtp', None) is not None
    _pstr = f"{_nparams / 1e9:.2f}B" if _nparams >= 1e9 else f"{_nparams / 1e6:.1f}M"
    logger.print_and_log(
        f"Model: {_pstr} params, {_size_gb:.2f} GB "
        f"({'bfloat16' if half_precision else 'float32'}) | "
        f"MTP module: {'present (self-spec decode available)' if _mtp_present else 'absent'}")

    return model, cfg


# Legacy-faithful (corrupted) RoPE policy for ALL load_model_and_tokenizer call
# sites in a tool process. TRI-STATE:
#   None  = AUTO — key off the checkpoint: rope_fixed flag if present, else
#           checkpoint_version >= ROPE_FIX_CHECKPOINT_VERSION means fixed RoPE;
#           anything older (or unmarked) is treated as the pre-2026-07-02
#           CORRUPTED era and gets envelope compat.
#   True  = force legacy/corrupted RoPE (reproduction / --envelope_compat).
#   False = force fixed RoPE (--fixed_rope; e.g. a correct-RoPE checkpoint that
#           still carries an old version number, like the pre-resume wizards).
# Set once after argparse.
ENVELOPE_COMPAT_DEFAULT = None

# Checkpoints at/after this version trained under the RoPE meta-init fix. Older
# ones learned under corrupted freqs tables and must reproduce them at inference.
ROPE_FIX_CHECKPOINT_VERSION = 4.0


def _ckpt_version_ge(ver, target: float) -> bool:
    """True iff a checkpoint_version string/number is >= target (defensive parse;
    unknown/garbage -> False so an unrecognized checkpoint is treated as the
    OLD/corrupted era, matching the safe default)."""
    try:
        return float(ver) >= float(target)
    except (TypeError, ValueError):
        return False

# Process-wide default for stream_generate_kv's spec= when the caller passes
# None: None = AUTO (speculative decode on for MTP checkpoints), False = forced
# classic. Set once after argparse: `if args.no_spec: nc.SPEC_DECODE_DEFAULT = False`.
SPEC_DECODE_DEFAULT = None


def load_model_and_tokenizer(
    checkpoint_path: str,
    device: Optional[str] = None,
    half_precision: bool = False,
    *,
    tok_kind: Optional[str] = None,
    tok_path: Optional[str] = None,
    special_tokens: Optional[str] = None,  # None = auto-detect from checkpoint, or path to JSON file
    shard_strategy: Optional[str] = None,  # 'auto' | 'none' | 'balanced' | HF string
    preferred_gpu: Optional[int] = None,
    max_memory_per_gpu: Optional[str] = None,  # e.g., "14GiB"
    qk_norm_mode: Optional[str] = None,  # None | "before_rope" | "after_rope_legacy" | "after_rope_fixed"
    use_keel: Optional[bool] = None,  # None = auto-detect from checkpoint, True/False = override
    envelope_compat: Optional[bool] = None,  # legacy-faithful RoPE for pre-2026-07-02
    # checkpoints; None -> use the module default (set once by tool CLIs via
    # `nc.ENVELOPE_COMPAT_DEFAULT = args.envelope_compat`, covering every call site)
):
    """Load both tokenizer **and** model; optionally shard across multiple GPUs.

    If tok_kind/tok_path/special_tokens are not specified, they will be auto-detected from the checkpoint.
    """

    if checkpoint_path.endswith(".bin"):
        raise ValueError("Legacy .bin checkpoints not supported – use .pt")

    # Load checkpoint metadata and log it
    chk_meta = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)

    # Log checkpoint metadata GENERICALLY: iterate the actual saved top-level
    # keys (minus large blobs and the ones a dedicated line owns) so new saved
    # fields surface automatically instead of drifting behind a hardcoded
    # whitelist. Owned elsewhere: step/version/iter -> the "Detected ..." line;
    # tok_kind/tok_path -> the "Tokenizer:" line; special_tokens -> the "Loaded
    # special tokens" line.
    _shown_elsewhere = {"model", "config", "step", "iter", "checkpoint_version",
                        "tok_kind", "tok_path", "special_tokens"}
    logger.print_and_log("Checkpoint metadata:")
    for key in sorted(chk_meta.keys()):
        if key in _shown_elsewhere:
            continue
        value = chk_meta.get(key)
        if value is None:
            continue
        display_val = str(value)
        if len(display_val) > 80:
            display_val = display_val[:77] + "..."
        logger.print_and_log(f"  {key}: {display_val}")

    # Auto-detect tokenizer settings from checkpoint if not provided
    special_tokens_source = "cli" if special_tokens else None
    if tok_kind is None:
        tok_kind = chk_meta.get("tok_kind", "llama")
    if tok_path is None:
        tok_path = chk_meta.get("tok_path")
    if special_tokens is None:
        special_tokens = chk_meta.get("special_tokens")
        if special_tokens:
            special_tokens_source = "checkpoint"

    # Resolve relative paths from checkpoint metadata.
    # Training saves paths relative to the training script dir, so they may not
    # resolve from the current working directory. Try:
    #   1. As-is (works if CWD matches training CWD)
    #   2. Relative to the checkpoint file's directory
    ckpt_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    if tok_path and not os.path.isabs(tok_path) and not os.path.exists(tok_path):
        candidate = os.path.normpath(os.path.join(ckpt_dir, tok_path))
        if os.path.exists(candidate):
            logger.print_and_log(f"  tok_path '{tok_path}' re-homed relative to checkpoint dir -> {candidate}")
            tok_path = candidate
    if special_tokens and isinstance(special_tokens, str) and not os.path.isabs(special_tokens) and not os.path.exists(special_tokens):
        candidate = os.path.normpath(os.path.join(ckpt_dir, special_tokens))
        if os.path.exists(candidate):
            logger.print_and_log(f"  special_tokens re-homed relative to checkpoint dir -> {candidate}")
            special_tokens = candidate
    # Single, post-resolution tokenizer line (the raw pre-resolution values are no
    # longer echoed by the metadata loop, which could disagree with what's opened).
    logger.print_and_log(f"Tokenizer: kind={tok_kind}, path={tok_path}")

    # Capture the RoPE-mode keys BEFORE freeing chk_meta — the envelope AUTO
    # resolution below runs after the (memory-hungry) model build, where chk_meta
    # is long gone.
    _ckpt_rope_fixed = chk_meta.get("rope_fixed", None)
    _ckpt_version = chk_meta.get("checkpoint_version", "2.0")

    del chk_meta  # Free memory, will be reloaded in _build_model_from_checkpoint

    if tok_path is None and tok_kind in ("llama", "hf"):
        raise ValueError(
            f"Tokenizer path not found in checkpoint metadata and not specified via CLI.\n"
            f"Use --tok_path to specify the tokenizer location (e.g., --tok_path ../tokenizers/llama_tokenizer)"
        )

    enc = get_tokenizer(tok_kind, path=tok_path, special_tokens=special_tokens)

    # Log special tokens info after tokenizer is created
    if special_tokens_source:
        # Count special tokens (handle both path and list formats)
        if isinstance(special_tokens, list):
            token_count = len(special_tokens)
            token_preview = special_tokens[:3]
        elif isinstance(special_tokens, str):
            # It's a path - load the file to get the actual tokens for display
            from tokenizer_abstraction import _load_special_tokens
            loaded_tokens = _load_special_tokens(special_tokens)
            if loaded_tokens:
                token_count = len(loaded_tokens)
                token_preview = loaded_tokens[:3]
            else:
                token_count = 0
                token_preview = [special_tokens]  # Show path if loading failed
        else:
            token_count = "?"
            token_preview = []
        preview_str = ", ".join(str(t) for t in token_preview)
        if token_count != "?" and token_count > 3:
            preview_str += ", ..."
        logger.print_and_log(f"Loaded special tokens ({special_tokens_source}): {token_count} tokens [{preview_str}]")

    model, cfg = _build_model_from_checkpoint(checkpoint_path, enc, half_precision, qk_norm_mode=qk_norm_mode, use_keel=use_keel)

    # Resolve envelope (legacy/corrupted RoPE) compat. Precedence:
    #   1. explicit per-call arg (True/False),
    #   2. process default ENVELOPE_COMPAT_DEFAULT (True=force legacy /
    #      False=force fixed / None=auto),
    #   3. AUTO from the checkpoint: an explicit rope_fixed flag wins; otherwise
    #      checkpoint_version >= ROPE_FIX_CHECKPOINT_VERSION means fixed RoPE and
    #      anything older/unmarked is the pre-fix CORRUPTED era.
    # Track the decision SOURCE and ALWAYS log the resolved mode — the forced
    # paths (explicit arg / process default from --fixed_rope/--envelope_compat)
    # used to be completely silent, so a forced-fixed load emitted zero RoPE
    # logging and you couldn't confirm the override took effect.
    if envelope_compat is not None:
        _rope_src = "explicit arg"
    else:
        envelope_compat = ENVELOPE_COMPAT_DEFAULT
        _rope_src = "process default (--envelope_compat/--fixed_rope)" if envelope_compat is not None else None
    if envelope_compat is None:
        # AUTO from the checkpoint: explicit rope_fixed wins, else version.
        if _ckpt_rope_fixed is not None:
            envelope_compat = not bool(_ckpt_rope_fixed)
            _rope_src = f"AUTO from rope_fixed={_ckpt_rope_fixed}"
        else:
            envelope_compat = not _ckpt_version_ge(_ckpt_version, ROPE_FIX_CHECKPOINT_VERSION)
            _rope_src = f"AUTO from checkpoint_version={_ckpt_version} (fix @ {ROPE_FIX_CHECKPOINT_VERSION})"
    logger.print_and_log(
        f"[rope] {'LEGACY/corrupted' if envelope_compat else 'FIXED'} RoPE "
        f"({_rope_src}; ckpt rope_fixed={_ckpt_rope_fixed}, v{_ckpt_version}). "
        f"Override with {'--fixed_rope' if envelope_compat else '--envelope_compat'}.")
    if envelope_compat:
        # Legacy-faithful RoPE: every FSDP2 checkpoint trained before the
        # 2026-07-02 meta-init fix learned under CORRUPTED tables (to_empty()
        # destroyed the freqs buffers; the allocator deterministically left
        # cos = zeros and sin = the cos table — attention degraded to a
        # separable cos-envelope). This tool builds CORRECT tables, so legacy
        # checkpoints otherwise generate ~27 mnats off their native
        # distribution (dn4@2k: CE 4.4931 native vs 4.5198 corrected).
        # Reproduce the exact training-time corruption for faithful sampling.
        with torch.no_grad():
            _raw = model._orig_mod if hasattr(model, '_orig_mod') else model
            _ref_cos = _raw.freqs_cos.clone()
            _raw.freqs_sin.copy_(_ref_cos)   # sin <- the cos table
            _raw.freqs_cos.zero_()           # cos <- zeros
        logger.print_and_log(
            "[envelope-compat] RoPE tables set to the LEGACY corruption "
            "(cos=0, sin=cos-table) — faithful sampling for checkpoints trained "
            "before the 2026-07-02 meta-init fix. Do NOT use for wizard-era models.")

    device = device or detect_device(preferred_gpu)

    # --- Optional Accelerate sharding --------------------------------------
    if shard_strategy and shard_strategy != "none" and torch.cuda.device_count() > 1:
        if dispatch_model is None:
            raise ImportError("accelerate not installed – cannot shard model")
        
        # Determine memory limit per GPU
        if max_memory_per_gpu is None:
            # Auto-detect based on GPU memory
            gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if gpu_mem_gb >= 24:  # 3090, 4090, A5000, etc.
                max_memory_per_gpu = "22GiB"
            elif gpu_mem_gb >= 16:  # 4080, A4000, etc.
                max_memory_per_gpu = "14GiB"
            elif gpu_mem_gb >= 12:  # 3060 12GB, 4070Ti, etc.
                max_memory_per_gpu = "10GiB"
            elif gpu_mem_gb >= 10:  # 3080 10GB
                max_memory_per_gpu = "9GiB"
            else:
                max_memory_per_gpu = "7GiB"
            
            logger.print_and_log(f"Auto-detected GPU memory: {gpu_mem_gb:.1f}GB, using {max_memory_per_gpu} per GPU")
        
        # Force balanced sharding if requested
        if shard_strategy == 'balanced':
            # Calculate model size to force splitting
            model_size_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3
            logger.print_and_log(f"Model size: {model_size_gb:.2f}GB, forcing balanced sharding")
            
            # Set memory limits to force splitting across all GPUs
            n_gpus = torch.cuda.device_count()
            forced_limit_gb = model_size_gb / n_gpus * 1.5  # Add 50% overhead for activations
            forced_limit = f"{int(forced_limit_gb)}GiB"
            max_mem = {i: forced_limit for i in range(n_gpus)}
            logger.print_and_log(f"Forcing max {forced_limit} per GPU to ensure balanced sharding")
        else:
            max_mem = {i: max_memory_per_gpu for i in range(torch.cuda.device_count())}
        
        max_mem['cpu'] = "64GiB"  # Allow CPU offloading if needed
        
        # More granular no-split classes
        no_split = ["TransformerBlock", "Block", "ResidualAttentionBlock", "CausalSelfAttention", "MultiHeadAttention"]
        
        logger.print_and_log(f"Inferring device map for model sharding...")
        device_map = infer_auto_device_map(
            model, 
            max_memory=max_mem, 
            no_split_module_classes=no_split,
            dtype=torch.bfloat16 if half_precision else torch.float32
        )
        
        # If still on one device and we want balanced, create custom map
        unique_devices = set(device_map.values())
        if len(unique_devices) == 1 and shard_strategy == 'balanced':
            logger.print_and_log("Auto device map kept model on single GPU, creating balanced map...")
            device_map = create_balanced_device_map(model, torch.cuda.device_count())
        
        # Log the device map for debugging
        logger.print_and_log(f"Device map summary:")
        device_counts = {}
        for module, dev in device_map.items():
            device_counts[dev] = device_counts.get(dev, 0) + 1
        for dev, count in sorted(device_counts.items(), key=lambda x: str(x[0])):
            logger.print_and_log(f"  {dev}: {count} modules")
        
        model = dispatch_model(model, device_map=device_map)
        logger.print_and_log(f"Model sharded across {len(set(device_map.values()))} devices via Accelerate")
    else:
        if shard_strategy and shard_strategy != "none" and torch.cuda.device_count() <= 1:
            logger.print_and_log(f"[shard] strategy={shard_strategy} requested but only "
                                 f"{torch.cuda.device_count()} GPU visible -> loading on single device")
        model.to(device)
        if device.startswith("cuda"):
            # Only compile if not sharding (compilation doesn't work well with sharded models)
            if not (shard_strategy and shard_strategy != "none" and torch.cuda.device_count() > 1):
                torch.compile(model, mode="reduce-overhead", dynamic=True)
                logger.print_and_log(f"Model loaded on {device} (compiled)")
            else:
                logger.print_and_log(f"Model loaded on {device}")
        else:
            logger.print_and_log(f"Model loaded on {device}")

    model.eval()
    return model, enc, cfg

# ---------------------------------------------------------------------------
# 3. Generation utils (space‑safe for SentencePiece)
# ---------------------------------------------------------------------------

# Platform-specific imports handled conditionally
if sys.platform == 'win32':
    import msvcrt
else:
    import select
    try:
        import termios
        import tty
        HAS_TERMIOS = True
    except ImportError:
        HAS_TERMIOS = False

# Add this helper function to neo_common.py:

def check_for_esc():
    """Check if ESC key was pressed."""
    try:
        # Check if we're on Windows
        if sys.platform == 'win32':
            if msvcrt.kbhit():
                key = msvcrt.getch()
                if key == b'\x1b':  # ESC key
                    return True
        else:
            # Unix/Linux/Mac
            if select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.read(1)
                if ord(key) == 27:  # ESC key
                    return True
    except:
        pass
    return False

# For Unix/Linux/Mac, we also need a context manager to handle terminal settings:
class NonBlockingInput:
    """Context manager for non-blocking keyboard input."""
    def __init__(self):
        self.old_settings = None
        
    def __enter__(self):
        if sys.platform != 'win32' and HAS_TERMIOS:
            try:
                self.old_settings = termios.tcgetattr(sys.stdin)
                tty.setcbreak(sys.stdin.fileno())
                # CRITICAL FIX: Flush any leftover characters from stdin
                # before starting generation (prevents spurious early exit)
                self._flush_stdin()
            except:
                pass
        return self
    
    def _flush_stdin(self):
        """Drain any pending input from stdin buffer."""
        try:
            # Keep reading until nothing is available
            while select.select([sys.stdin], [], [], 0)[0]:
                sys.stdin.read(1)
        except:
            pass
    
    def __exit__(self, type, value, traceback):
        if sys.platform != 'win32' and self.old_settings and HAS_TERMIOS:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
                # Also flush on exit to clean up any typed characters
                termios.tcflush(sys.stdin, termios.TCIFLUSH)
            except:
                pass

"""
Fixed stream_generate_kv for SentencePiece tokenizers (LLaMA, etc.)

The issue: SentencePiece encodes word boundaries with a special "▁" character.
When decoding tokens one-at-a-time, the tokenizer loses context and drops spaces.

The fix: Always decode the FULL sequence and print only the new characters (delta).
"""

def _prettify_special_tokens(text: str, role_names: dict) -> str:
    """Replace special tokens with pretty printed versions."""
    assistant_name = role_names.get("assistant", "Assistant")
    user_name = role_names.get("user", "User")

    # Order matters - replace longer tokens first to avoid partial matches
    replacements = [
        ("<|assistant_start|>", f"\n{assistant_name}: "),
        ("<|assistant_end|>", "\n"),
        ("<|user_start|>", f"\n{user_name}: "),
        ("<|user_end|>", "\n"),
        ("<|think|>", "\n[thinking] "),
        ("<|/think|>", " [/thinking]\n"),
        ("<|tool_call|>", "\n[tool] "),
        ("<|/tool_call|>", " [/tool]\n"),
        ("<|tool_result|>", "[result] "),
        ("<|/tool_result|>", " [/result]\n"),
        ("<|bos|>", ""),  # Hide BOS token
    ]

    for old, new in replacements:
        text = text.replace(old, new)

    return text


def _find_safe_print_boundary(text: str) -> int:
    """
    Find the safe boundary in text where we can print without splitting a special token.
    Returns the index up to which we can safely print (and prettify).
    The remainder should be kept in the buffer for the next iteration.
    """
    # All special tokens follow the pattern <|...|>
    # We need to check if text ends with a partial match

    # Check for incomplete special tokens at the end
    # Look for '<' or '<|' that might be the start of an incomplete token
    last_lt = text.rfind('<')
    if last_lt == -1:
        # No '<' found, safe to print everything
        return len(text)

    # Check if there's a complete token after this '<'
    remaining = text[last_lt:]
    if '>' in remaining:
        # There's a '>' after the last '<', so any token is complete
        # But check if there's another '<' after that '>'
        last_gt = remaining.rfind('>')
        if last_gt < len(remaining) - 1:
            # There's content after the last '>', check for another '<'
            after_gt = remaining[last_gt + 1:]
            if '<' in after_gt:
                # There's a '<' after the last '>', this might be incomplete
                return last_lt + last_gt + 1 + after_gt.find('<')
        return len(text)
    else:
        # No '>' after the last '<', this is definitely an incomplete token
        return last_lt


class CacheState:
    """KV-cache reuse POLICY for a model, owned by a caller across generations.

    Enables cross-turn *prefix reuse* in stream_generate_kv: when the new
    prompt shares a token prefix with what is already materialized in the
    model's KV cache, only the divergent suffix needs to be prefilled.

    Soundness (pure-attention models only): K/V at position p is a pure
    function of token p and its prefix (RoPE uses absolute positions), so
    cached K/V for an identical token prefix is bit-identical to a fresh
    prefill. This is UNSOUND for models with Block-AttnRes (cross-block state
    not in the KV cache) or GDN/linear-attention layers (recurrent state not
    in the KV cache), so `reusable` defaults to FALSE — the caller must
    explicitly opt in only for a verified pure-attention checkpoint.

    SWA rolling rings (swa_enabled checkpoints): reuse is APPEND-ONLY once
    more tokens than the smallest ring (min_rolling_cache_len()) have been
    materialized — evicted positions' slots hold FUTURE-position K/V, so
    REWIND re-entry (/rep truncation, history edits) behind the high-water
    mark is unsound after wrap. The generate paths guard this by degrading a
    post-wrap rewind to a full re-prefill (slow, never wrong).

    NOTE: this object holds ONLY policy (`reusable`). The actual record of
    which token IDs are materialized in the cache lives on the MODEL
    (`model.get_cache_ledger()` / `set_cache_ledger()` / `reset_cache_ledger()`),
    co-located with the cache tensors so the two cannot desync. Any cache
    (re)allocation or clear resets that ledger automatically; an empty ledger
    degrades to a full prefill — slow, never wrong.
    """

    def __init__(self, reusable: bool = False):
        # Whether prefix reuse is sound for this model (pure attention).
        # Fail-safe default: OFF unless the caller affirmatively enables it.
        self.reusable = reusable


def _longest_common_prefix_len(a: list[int], b: list[int]) -> int:
    """Length of the longest common prefix of two token-ID lists."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def stream_generate_kv(model, tokenizer, prompt_text, max_new_tokens, context_size,
                       temperature, top_p, display=True, stop_on_eos=False, stop_sequences=None,
                       print_prompt=True, return_stop_info=False,
                       pretty_print=False, role_names=None, cache_state: "CacheState | None" = None,
                       debug_reuse=False, spec=None):
    """
    Generates text using KV Caching for O(N) complexity per token.

    FIXED: Properly handles SentencePiece tokenizers by decoding full sequence
    and printing deltas, preserving spaces correctly.

    Args:
        pretty_print: If True, replace special tokens with readable versions when displaying
        role_names: Dict with "assistant" and "user" keys for pretty print names
        spec: MTP self-speculative decode engine. None (default) = AUTO — on iff
            the checkpoint carries an MTP module; False = classic single-token
            decode; True = require spec (raises without an MTP module).
            The engine is a drop-in: identical features (stop sequences,
            streaming display, pretty print, reuse), greedy outputs
            BIT-IDENTICAL to classic; sampled outputs draw from the identical
            distribution (k=1 speculative sampling is distribution-exact) but
            consume RNG in a different order, so a fixed torch seed produces a
            different — equally valid — trajectory than the classic engine.
    """

    device = next(model.parameters()).device

    _mtp_mod = getattr(model, 'mtp', None)
    if spec is None:
        spec = SPEC_DECODE_DEFAULT   # process-wide override (e.g. a tool's --no_spec flag)
    if spec is True and _mtp_mod is None:
        raise ValueError("spec=True requires an MTP checkpoint (model.mtp is None)")
    use_spec = (_mtp_mod is not None) if spec is None else (bool(spec) and _mtp_mod is not None)

    # 1. Prepare Tokens
    tokens = tokenizer.encode(prompt_text, bos=True, eos=False)
    prompt_len = len(tokens)

    reuse = (cache_state is not None and cache_state.reusable)

    # Bounds check: ensure we don't exceed context size
    if prompt_len >= context_size:
        logger.print_and_log(f"\nError: Prompt length ({prompt_len} tokens) exceeds or equals context size ({context_size} tokens).")
        logger.print_and_log("Please use a shorter prompt or a model with a larger max_seq_len.")
        if reuse:
            model.reset_cache_ledger()  # nothing materialized; don't trust stale prefix
        if return_stop_info:
            return "", {"reason": "error", "detail": "prompt exceeds context", "tokens_generated": 0}
        return ""

    if prompt_len + max_new_tokens > context_size:
        available_tokens = context_size - prompt_len
        logger.print_and_log(f"\nWarning: Requested {max_new_tokens} new tokens, but only {available_tokens} fit within context.")
        logger.print_and_log(f"  Prompt: {prompt_len} tokens, Context size: {context_size} tokens")
        logger.print_and_log(f"  Generation will be limited to {available_tokens} tokens.")
        max_new_tokens = available_tokens

    prompt_ids = tokens.copy()  # Keep a copy for tracking

    # Track ALL generated token IDs (not just the tensor)
    all_token_ids = prompt_ids.copy()
    generated_tokens = []

    # 2. Setup Cache
    bsz = 1
    # The high-water mark this generation may write to.
    total_len = min(context_size, len(all_token_ids) + max_new_tokens)

    # Allocation sizing. For the reuse path we want ONE buffer that survives
    # across turns, but we do NOT want to pay full-context VRAM from turn 1 when
    # the conversation is short. Round the working size up to a block so the
    # buffer grows in coarse steps (each growth reallocates + resets the ledger;
    # coarse steps keep that rare). The non-reuse path keeps the original tight
    # sizing. Either way the idempotent setup_caches() only reallocates when the
    # existing buffer is too small / wrong device-dtype.
    if reuse:
        _BLOCK = 1024
        alloc_len = min(context_size, ((total_len + _BLOCK - 1) // _BLOCK) * _BLOCK)
    else:
        alloc_len = total_len

    # NOTE: start_pos and the suffix tensor are computed AFTER setup_caches()
    # below, because (re)allocation resets the model's cache ledger — the prefix
    # we may reuse depends on whether the allocation survived or was rebuilt.

    # SentencePiece workaround detection
    needs_spm_workaround = isinstance(tokenizer, LlamaTokenizerAdapter)

    # Track generated text for stop sequence checking
    generated_text_so_far = ""
    stop_sequence_hit = None

    # Track why generation stopped
    stop_reason = {"reason": "max_tokens", "detail": None, "tokens_generated": 0}

    # Buffer for delayed printing (to avoid printing stop sequences and to allow
    # special token prettification to work on complete tokens)
    print_buffer = ""
    max_stop_len = max(len(s) for s in stop_sequences) if stop_sequences else 0
    # When pretty_print is enabled, also account for the longest special token that
    # needs to be replaced, so tags don't get split across buffer boundaries
    if pretty_print:
        special_token_lengths = [
            19,  # <|assistant_start|>
            18,  # <|assistant_end|>
            14,  # <|user_start|>
            12,  # <|user_end|>
            16,  # <|/tool_result|>
            15,  # <|tool_result|>
            14,  # <|/tool_call|>
            13,  # <|tool_call|>
            10,  # <|/think|>
            9,   # <|think|>
        ]
        max_stop_len = max(max_stop_len, max(special_token_lengths))

    if needs_spm_workaround:
        # Decode full prompt to establish baseline length
        last_decoded_full = tokenizer.decode(all_token_ids)
        last_decoded_len = len(last_decoded_full)
    else:
        last_decoded_len = 0  # unused off-SPM; defined for the shared emit closure

    # materialized_len = number of cache positions [0, materialized_len) that
    # actually hold valid K/V for prompt_ids+generated tokens. Updated only
    # AFTER each successful forward, so it never counts a sampled-but-not-yet-
    # forwarded token (the C2 stop/EOS off-by-one). This is the true length to
    # record in the model ledger.
    materialized_len = 0
    # Whether we should persist the ledger on exit (set once allocation+prefill
    # are known-consistent; cleared on any inconsistency → safe full prefill).
    persist_ledger = False

    try:
      with torch.no_grad():
        if reuse:
            # Idempotent: keeps the existing allocation (and its cached prefix
            # contents) when it's already big enough and on the right
            # device/dtype; otherwise (re)allocates and resets the ledger.
            # Sized to a rounded-up working length so the buffer survives
            # across turns without paying full-context VRAM up front.
            model.setup_caches(max_batch_size=bsz, max_seq_len=alloc_len)
        else:
            # No reuse requested → original behavior: fresh allocation sized
            # to this generation only.
            model.setup_caches(max_batch_size=bsz, max_seq_len=total_len, force=True)

        # Compute the reusable prefix AFTER allocation. If setup_caches
        # reallocated (grew the buffer / device-dtype change / first alloc),
        # it reset the ledger to [], so the prefix is empty and we full-
        # prefill. If it no-op'd (buffer survived), the ledger still
        # describes the live cache contents and we can reuse them.
        if reuse:
            ledger = model.get_cache_ledger()
            prefix_len = _longest_common_prefix_len(ledger, prompt_ids)
            # Leave >=1 token to forward (need fresh logits to sample from;
            # also handles new-prompt-is-strict-prefix, e.g. /rep truncation).
            start_pos = min(prefix_len, prompt_len - 1)
            # SWA rolling rings: REWIND re-entry (start_pos behind the ledger's
            # high-water mark) is UNSOUND once any ring has wrapped — slots for
            # evicted positions hold FUTURE-position K/V (RoPE-baked), so both
            # cached paths silently attend the wrong timeline (measured 7e-2
            # logit corruption). Append-only reuse stays sound at any length.
            # The wrap test is `>=`, NOT `>`: a spec-engine terminal REJECT
            # forwards a 2-token chunk whose rejected draft is PHYSICALLY written
            # one position PAST the ledger (materialized_len = n_ctx+1 while the
            # chunk write touches slot (n_ctx+1)%Lc), so the physical ring extent
            # is len(ledger)+1. At len(ledger)==Lc that phantom has already
            # evicted slot 0 even though the ledger looks unwrapped — `>` would
            # miss it and serve wrong-timeline K/V on a rewind. `start_pos <
            # len(ledger)` already scopes this to rewinds only (append-only reuse
            # is untouched). Degrade to full re-prefill: slow, never wrong.
            _lc = getattr(model, 'min_rolling_cache_len', lambda: None)()
            if _lc is not None and start_pos < len(ledger) and len(ledger) >= _lc:
                start_pos = 0
            # Spec engine needs the MTP cache in LOCKSTEP with the reused trunk
            # prefix (its own attention runs over all past mtp inputs). If a
            # prior turn ran classic decode (or anything desynced the mtp
            # cache), degrade to full re-prefill so BOTH caches rebuild
            # together — slow, never wrong.
            if use_spec and getattr(model, '_mtp_cache_len', 0) < start_pos:
                start_pos = 0
            if debug_reuse:
                # Observability: how much of the prompt was served from cache vs
                # re-prefilled, AND a hint at WHY reuse was limited this turn:
                #   * ledger==0      → cache was (re)allocated fresh this turn
                #                      (first turn, device/dtype change, or a
                #                      non-preserving realloc). Should be rare now
                #                      that growth preserves contents.
                #   * reused<<ledger → the prompt diverged from the cached tokens
                #                      early (history edited, trimmed from the
                #                      front, or stored text didn't round-trip).
                #   * reused≈ledger  → healthy: only the new tail is re-prefilled.
                reprefill = prompt_len - start_pos
                if len(ledger) == 0:
                    why = "fresh-alloc"
                elif start_pos < len(ledger) - 2:  # diverged before ledger end
                    why = f"diverged@{start_pos}/{len(ledger)} (edit/trim?)"
                else:
                    why = "healthy"
                logger.print_and_log(
                    f"[kv-reuse] prompt={prompt_len} reused={start_pos} "
                    f"reprefill={reprefill} ledger={len(ledger)} -> {why}"
                )
        else:
            start_pos = 0

        # Positions [0, start_pos) are served from existing cache contents;
        # forward only the divergent suffix on the first pass.
        suffix_ids = prompt_ids[start_pos:]
        tokens = torch.tensor(suffix_ids, dtype=torch.long, device=device).unsqueeze(0)
        # Everything in [0, start_pos) is (by construction of the LCP against
        # the ledger) already materialized in the cache.
        materialized_len = start_pos

        if display and print_prompt:
            if pretty_print and role_names:
                print(_prettify_special_tokens(prompt_text, role_names), end="", flush=True)
            else:
                print(prompt_text, end="", flush=True)

        def _sample_stream(next_token_logits):
            """Classic sampling (global RNG) — shared by both decode engines so
            temperature/top-p semantics are identical."""
            if temperature > 0:
                probs = torch.softmax(next_token_logits / temperature, dim=-1)
                if top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    probs[indices_to_remove] = 0
                    probs = probs / probs.sum()
                return int(torch.multinomial(probs, num_samples=1))
            return int(torch.argmax(next_token_logits))

        def _emit_token(next_token_id):
            """Shared per-token pipeline: record, decode, stream with stop-seq
            buffering, EOS check. Returns True when generation must stop.
            Used by BOTH decode engines so classic and spec can never drift —
            extracted verbatim from the classic loop tail. NOTE: emitting
            tokens one-at-a-time through this pipeline is also what makes
            EOS/stop-sequences correct mid-round for the spec engine (an EOS
            draft stops HERE, before its pair partner is ever surfaced)."""
            nonlocal generated_text_so_far, print_buffer, stop_sequence_hit
            nonlocal stop_reason, last_decoded_len

            generated_tokens.append(next_token_id)
            all_token_ids.append(next_token_id)

            if needs_spm_workaround:
                decoded_full = tokenizer.decode(all_token_ids)
                new_text = decoded_full[last_decoded_len:]
                if new_text:
                    generated_text_so_far += new_text
                    print_buffer += new_text
                    last_decoded_len = len(decoded_full)
            else:
                decoded_token = tokenizer.decode([next_token_id])
                generated_text_so_far += decoded_token
                print_buffer += decoded_token

            if stop_sequences:
                for stop_seq in stop_sequences:
                    if stop_seq in generated_text_so_far:
                        stop_sequence_hit = stop_seq
                        generated_text_so_far = generated_text_so_far[:generated_text_so_far.find(stop_seq)]
                        if stop_seq in print_buffer:
                            print_buffer = print_buffer[:print_buffer.find(stop_seq)]
                        break
                if stop_sequence_hit:
                    if display and print_buffer:
                        to_print = print_buffer
                        if pretty_print and role_names:
                            to_print = _prettify_special_tokens(to_print, role_names)
                        print(to_print, end="", flush=True)
                    stop_reason = {"reason": "stop_sequence", "detail": repr(stop_sequence_hit),
                                   "tokens_generated": len(generated_tokens)}
                    return True

            if display and max_stop_len > 0:
                if len(print_buffer) > max_stop_len:
                    candidate = print_buffer[:-max_stop_len]
                    if pretty_print and role_names:
                        safe_len = _find_safe_print_boundary(candidate)
                        safe_to_print = candidate[:safe_len]
                        safe_to_print = _prettify_special_tokens(safe_to_print, role_names)
                        print_buffer = print_buffer[safe_len:]
                    else:
                        safe_to_print = candidate
                        print_buffer = print_buffer[-max_stop_len:]
                    if safe_to_print:
                        print(safe_to_print, end="", flush=True)
            elif display:
                to_print = print_buffer
                if pretty_print and role_names:
                    to_print = _prettify_special_tokens(to_print, role_names)
                print(to_print, end="", flush=True)
                print_buffer = ""

            if stop_on_eos and next_token_id == tokenizer.eos_id:
                stop_reason = {"reason": "eos", "detail": f"token_id={tokenizer.eos_id}",
                               "tokens_generated": len(generated_tokens)}
                return True
            return False

        # 3. Prefill and Generation Loop
        with NonBlockingInput():
            if use_spec and max_new_tokens > 0:
                # ===== SPEC ENGINE (MTP self-speculative decode) =====
                # Drop-in for the classic loop below: same emit pipeline, same
                # sampling semantics (greedy bit-identical; sampled
                # distribution-exact via k=1 speculative sampling). Each round:
                # ONE trunk forward over [cur, draft] + one tiny MTP pass.
                spec_rounds = 0
                spec_accepts = 0

                # trunk suffix prefill (h_pre feeds the MTP backfill)
                logits, h_pre = model.generate_forward(tokens, start_pos, return_h_pre=True)
                materialized_len = start_pos + tokens.shape[1]
                n_ctx = materialized_len
                cur = _sample_stream(logits[0, -1, :])
                _stop = _emit_token(cur)

                # MTP backfill over the forwarded suffix -> first draft.
                # (The reuse guard above forced start_pos=0 whenever the mtp
                # cache wasn't in lockstep, so [0, start_pos) is covered.)
                _nt = torch.tensor([suffix_ids[1:] + [cur]], dtype=torch.long, device=device)
                mtp_logits = model.mtp_decode_chunk(h_pre, _nt, start_pos)
                model._mtp_cache_len = n_ctx
                q_probs = _warp_probs(mtp_logits[0, -1].float(), temperature, top_p)
                draft = int(torch.multinomial(q_probs, 1)) if temperature > 0 else int(q_probs.argmax())

                while not _stop and len(generated_tokens) < max_new_tokens:
                    if check_for_esc():
                        if display:
                            print("\n[Generation interrupted by ESC key]", flush=True)
                        stop_reason = {"reason": "interrupted", "detail": "ESC key",
                                       "tokens_generated": len(generated_tokens)}
                        break
                    if n_ctx + 2 > total_len:
                        # one slot left (or none): finish with classic steps —
                        # emission-count parity with the classic engine
                        if n_ctx < total_len:
                            logits, _hp = model.generate_forward(
                                torch.tensor([[cur]], dtype=torch.long, device=device),
                                n_ctx, return_h_pre=True)
                            materialized_len = n_ctx + 1
                            nxt = _sample_stream(logits[0, -1, :])
                            _stop = _emit_token(nxt)
                            n_ctx += 1
                            cur = nxt
                            if _stop:
                                break
                        stop_reason = {"reason": "context_limit",
                                       "detail": f"reached {context_size} tokens",
                                       "tokens_generated": len(generated_tokens)}
                        break

                    spec_rounds += 1
                    chunk = torch.tensor([[cur, draft]], dtype=torch.long, device=device)
                    logits2, h_pre2 = model.generate_forward(chunk, n_ctx, return_h_pre=True)
                    p1 = _warp_probs(logits2[0, 0].float(), temperature, top_p)

                    # k=1 speculative-sampling verification (shared with spec_generate)
                    ok, reject_tok = _spec_verify_step(p1, q_probs, draft, temperature)

                    if ok:
                        spec_accepts += 1
                        # BOTH chunk rows are now valid materialized positions
                        materialized_len = n_ctx + 2
                        _stop = _emit_token(draft)
                        if _stop:
                            n_ctx += 2
                            break
                        p2 = _warp_probs(logits2[0, 1].float(), temperature, top_p)
                        nxt = int(torch.multinomial(p2, 1)) if temperature > 0 else int(p2.argmax())
                        if len(generated_tokens) < max_new_tokens:
                            _stop = _emit_token(nxt)
                        # MTP backfill both rows; last row drafts t_{n_ctx+3}
                        _nt = torch.tensor([[draft, nxt]], dtype=torch.long, device=device)
                        mtp_logits = model.mtp_decode_chunk(h_pre2, _nt, n_ctx)
                        model._mtp_cache_len = n_ctx + 2
                        q_probs = _warp_probs(mtp_logits[0, -1].float(), temperature, top_p)
                        draft = int(torch.multinomial(q_probs, 1)) if temperature > 0 else int(q_probs.argmax())
                        n_ctx += 2
                        cur = nxt
                    else:
                        # row n_ctx+1 holds the REJECTED draft's K/V: claim only
                        # n_ctx+1 positions (the ledger must never label that
                        # row; the next round's chunk write overwrites it, and
                        # in the interim the banded mask excludes its one
                        # aliased ring slot — review-proven).
                        materialized_len = n_ctx + 1
                        tok_v = reject_tok   # residual sample from _spec_verify_step
                        _stop = _emit_token(tok_v)
                        # MTP backfill row0 only (rejected row's h_pre is invalid)
                        _nt = torch.tensor([[tok_v]], dtype=torch.long, device=device)
                        mtp_logits = model.mtp_decode_chunk(h_pre2[:, :1], _nt, n_ctx)
                        model._mtp_cache_len = n_ctx + 1
                        q_probs = _warp_probs(mtp_logits[0, -1].float(), temperature, top_p)
                        draft = int(torch.multinomial(q_probs, 1)) if temperature > 0 else int(q_probs.argmax())
                        n_ctx += 1
                        cur = tok_v

                stop_reason["spec"] = {
                    "rounds": spec_rounds,
                    "accepted": spec_accepts,
                    "acceptance": (spec_accepts / spec_rounds) if spec_rounds else 0.0,
                }

            for i in range(max_new_tokens if not use_spec else 0):
                # Check for ESC key press
                if check_for_esc():
                    if display:
                        print("\n[Generation interrupted by ESC key]", flush=True)
                    stop_reason = {"reason": "interrupted", "detail": "ESC key", "tokens_generated": i}
                    break

                # Check if we reached context limit
                if start_pos + tokens.shape[1] > context_size:
                    stop_reason = {"reason": "context_limit", "detail": f"reached {context_size} tokens", "tokens_generated": i}
                    break

                # Forward pass (works with both old and new model interface)
                logits, _ = model(tokens, start_pos=start_pos)
                # These suffix positions are now physically in the cache. Record
                # the TRUE materialized length here (after the forward), so a
                # token that gets sampled below but never forwarded (stop/EOS on
                # the next break) is NOT counted. This is the fix for the C2
                # off-by-one that previously corrupted the next turn's reuse.
                materialized_len = start_pos + tokens.shape[1]

                # Select last token logits
                next_token_logits = logits[0, -1, :]

                # Sampling
                if temperature > 0:
                    probs = torch.softmax(next_token_logits / temperature, dim=-1)
                    # Top-p (Nucleus) sampling
                    if top_p < 1.0:
                        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        indices_to_remove = sorted_indices[sorted_indices_to_remove]
                        probs[indices_to_remove] = 0
                        probs = probs / probs.sum()

                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                next_token_id = next_token.item()

                # Store, decode, stream, stop-check — the shared emit pipeline
                _stop = _emit_token(next_token_id)

                # Update for next iteration
                start_pos += tokens.shape[1]
                tokens = next_token.unsqueeze(0)

                if _stop:
                    break

        # Print any remaining buffer (if we didn't hit a stop sequence)
        if display and print_buffer and not stop_sequence_hit:
            to_print = print_buffer
            if pretty_print and role_names:
                to_print = _prettify_special_tokens(to_print, role_names)
            print(to_print, end="", flush=True)

        # Update tokens_generated for max_tokens case (loop completed naturally)
        if stop_reason["reason"] == "max_tokens":
            stop_reason["tokens_generated"] = len(generated_tokens)
            stop_reason["detail"] = f"reached limit of {max_new_tokens}"

        # Generation completed without an exception → the ledger is consistent
        # with what we physically forwarded, so it's safe to persist on the
        # reuse path. (For non-reuse we clear the cache in finally instead.)
        persist_ledger = True

    except Exception as _gen_exc:
        # Make a reuse-path failure VISIBLE rather than silently degrading.
        # The most likely culprit is a CUDA OOM while (re)allocating or growing
        # the KV cache — common on tightly-packed multi-GPU balanced shards,
        # where the held full-context cache isn't in the sharding budget. We
        # log a clear, attributable warning and re-raise so the caller still
        # handles it; the `finally` below resets the ledger so the next turn is
        # safe.
        if reuse:
            logger.print_and_log(
                f"\n[kv-reuse] generation failed ({type(_gen_exc).__name__}: {_gen_exc}). "
                f"KV-cache reuse state has been reset; the next turn will full-prefill. "
                f"If this is a CUDA OOM on a sharded model, reduce context/gen size or "
                f"run on fewer/larger GPUs."
            )
        raise
    finally:
        # C6: this runs even if the forward raised (OOM, etc.). On the reuse
        # path we KEEP the allocation across turns, but the ledger must only
        # describe what was actually materialized:
        #   * clean completion → record all_token_ids[:materialized_len], i.e.
        #     the forwarded prefix MINUS any sampled-but-unforwarded stop/EOS
        #     token (C2). materialized_len is exact because it's bumped only
        #     after each successful forward.
        #   * exception mid-forward → don't trust partial writes; reset ledger.
        if reuse:
            if persist_ledger:
                model.set_cache_ledger(all_token_ids[:materialized_len])
            else:
                model.reset_cache_ledger()
            # Keep the allocation (don't clear) so it survives to the next turn.
        else:
            # No reuse: original behavior — free the cache between calls.
            model.clear_caches()

    # Return the generated text
    # If we tracked it for stop sequences, use that (already trimmed correctly)
    if stop_sequences or needs_spm_workaround:
        generated_text = generated_text_so_far
    else:
        # Fallback: decode generated tokens
        generated_text = tokenizer.decode(generated_tokens)

    if return_stop_info:
        return generated_text, stop_reason
    else:
        return generated_text


def verify_kv_reuse_parity(model, tokenizer, context_size,
                           base_text="The quick brown fox jumps over the lazy dog. ",
                           turn1_suffix="It was a bright cold day in April. ",
                           turn2_suffix="The clocks were striking thirteen. ",
                           gen_steps=8, atol=1e-3, verbose=True):
    """Prove cross-turn KV-cache prefix reuse is bit-exact (within fp noise).

    Strategy: simulate two turns that share a long token prefix, then compare
    the model's next-token logits produced via the REUSE path against a fresh
    FULL-prefill reference at every generated step. If reused-prefix K/V were
    wrong (e.g. the C1 mask bug, or a C2/C4 ledger desync), the logits diverge
    immediately and we report the first failing step.

    This is deterministic (argmax greedy, no RNG), so it catches correctness
    regressions that temperature>0 smoke testing would mask.

    Returns (ok: bool, detail: str).
    """
    import neo_common as _self  # for CacheState

    device = next(model.parameters()).device

    def _greedy_ids_and_first_logits(prompt_text, cache_state):
        """Generate gen_steps tokens greedily via the model's cached path,
        honoring cache_state (reuse or not). Returns (generated_ids,
        logits_per_step) where logits_per_step[k] is the full next-token logit
        vector the model produced at generation step k. Mirrors the core of
        stream_generate_kv but headless and id-level."""
        ids = tokenizer.encode(prompt_text, bos=True, eos=False)
        prompt_len = len(ids)
        reuse = cache_state is not None and cache_state.reusable
        bsz = 1
        total_len = min(context_size, prompt_len + gen_steps)
        if reuse:
            _BLOCK = 1024
            alloc_len = min(context_size, ((total_len + _BLOCK - 1) // _BLOCK) * _BLOCK)
        else:
            alloc_len = total_len

        gen_ids = []
        step_logits = []
        try:
            with torch.no_grad():
                if reuse:
                    model.setup_caches(max_batch_size=bsz, max_seq_len=alloc_len)
                    ledger = model.get_cache_ledger()
                    p = _longest_common_prefix_len(ledger, ids)
                    start_pos = min(p, prompt_len - 1)
                    # mirror stream_generate_kv's SWA rolling-ring rewind guard
                    # (>= not >: covers the spec-reject phantom write one past the
                    # ledger — see the primary guard's note above)
                    _lc = getattr(model, 'min_rolling_cache_len', lambda: None)()
                    if _lc is not None and start_pos < len(ledger) and len(ledger) >= _lc:
                        start_pos = 0
                else:
                    model.setup_caches(max_batch_size=bsz, max_seq_len=total_len, force=True)
                    start_pos = 0

                cur = torch.tensor(ids[start_pos:], dtype=torch.long, device=device).unsqueeze(0)
                materialized = start_pos
                all_ids = list(ids)
                for _ in range(gen_steps):
                    if start_pos + cur.shape[1] > context_size:
                        break
                    logits, _u = model(cur, start_pos=start_pos)
                    materialized = start_pos + cur.shape[1]
                    nxt_logits = logits[0, -1, :].float()
                    step_logits.append(nxt_logits)
                    nxt = int(torch.argmax(nxt_logits).item())
                    gen_ids.append(nxt)
                    all_ids.append(nxt)
                    start_pos += cur.shape[1]
                    cur = torch.tensor([[nxt]], dtype=torch.long, device=device)
            if reuse:
                model.set_cache_ledger(all_ids[:materialized])
            else:
                model.clear_caches()
        except Exception:
            if reuse:
                model.reset_cache_ledger()
            else:
                model.clear_caches()
            raise
        return gen_ids, step_logits

    prompt1 = base_text + turn1_suffix
    prompt2 = base_text + turn1_suffix + turn2_suffix  # shares prompt1 as a prefix

    # Reference: reuse OFF, fresh full prefill of the turn-2 prompt.
    ref_state = _self.CacheState(reusable=False)
    ref_ids, ref_logits = _greedy_ids_and_first_logits(prompt2, ref_state)

    # Reuse path: turn 1 populates the ledger, turn 2 exercises prefix reuse.
    model.reset_cache_ledger()
    reuse_state = _self.CacheState(reusable=True)
    _greedy_ids_and_first_logits(prompt1, reuse_state)          # warm the cache
    reuse_ids, reuse_logits = _greedy_ids_and_first_logits(prompt2, reuse_state)

    # Compare per-step next-token logits and the resulting greedy ids.
    n = min(len(ref_logits), len(reuse_logits))
    max_abs = 0.0
    first_bad = None
    for k in range(n):
        d = (ref_logits[k] - reuse_logits[k]).abs().max().item()
        max_abs = max(max_abs, d)
        if (ref_ids[k] != reuse_ids[k] or d > atol) and first_bad is None:
            first_bad = (k, ref_ids[k], reuse_ids[k], d)

    ok = first_bad is None and ref_ids[:n] == reuse_ids[:n]
    if ok:
        detail = (f"PASS: reuse matches full prefill over {n} steps "
                  f"(max |Δlogit|={max_abs:.2e}, atol={atol:g}). "
                  f"ids={reuse_ids[:n]}")
    else:
        k, ri, ui, d = first_bad
        detail = (f"FAIL at step {k}: ref_id={ri} reuse_id={ui} "
                  f"|Δlogit|={d:.2e} (max over run {max_abs:.2e}). "
                  f"ref_ids={ref_ids[:n]} reuse_ids={reuse_ids[:n]}")

    # Leave the cache clean so a live session isn't polluted by the probe.
    model.clear_caches()

    if verbose:
        logger.print_and_log("[kv-reuse parity] " + detail)
    return ok, detail


def _warp_probs(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    """Logits -> sampling distribution under temperature + top-p.
    temperature == 0 returns a one-hot argmax distribution, so the speculative
    accept/reject math below degrades EXACTLY to greedy verification."""
    if temperature <= 0:
        p = torch.zeros_like(logits)
        p[logits.argmax()] = 1.0
        return p
    probs = torch.softmax(logits / temperature, dim=-1)
    if top_p < 1.0:
        sp, si = torch.sort(probs, descending=True)
        cum = torch.cumsum(sp, dim=-1)
        remove = cum > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        probs = probs.clone()
        probs[si[remove]] = 0.0
        probs = probs / probs.sum()
    return probs


def _spec_verify_step(p1, q_probs, draft, temperature, generator=None):
    """k=1 speculative-sampling verification of ONE draft token — the SINGLE
    SOURCE OF TRUTH shared by spec_generate and stream_generate_kv's inlined
    engine (extracted so the two can never drift on this safety-critical math).

    Given the target distribution p1 (warped trunk logits at the verified
    position) and the draft distribution q_probs (warped MTP logits) that
    `draft` was sampled from, returns (accept, emit):
        accept=True,  emit=None  -> caller emits `draft`;
        accept=False, emit=tok   -> caller emits `tok`, a residual draw from
                                    normalize(max(0, p1 - q_probs)).
    The accept/residual pair makes the emitted-token distribution EXACTLY p1 for
    ANY draft law q_probs — the speculative-sampling guarantee (Leviathan et al.
    2023; Chen et al. 2023). temperature<=0 degrades to greedy verification
    (p1, q_probs are one-hot): accept iff draft==argmax(p1), else emit
    argmax(p1) — bit-exact greedy.

    RNG is consumed in the SAME order as ordinary sampling — at most one uniform
    (the accept Bernoulli) then one categorical (only on reject) — so seeded
    trajectories are identical whether or not this is factored out."""
    if temperature <= 0:
        am = int(p1.argmax())
        return (am == draft), (None if am == draft else am)
    q_d = float(q_probs[draft])
    p_d = float(p1[draft])
    r = float(torch.rand((), generator=generator, device=p1.device))
    if q_d > 0 and r < min(1.0, p_d / q_d):
        return True, None
    resid = torch.clamp(p1 - q_probs, min=0.0)
    s = float(resid.sum())
    probs = resid / s if s > 0 else p1
    return False, int(torch.multinomial(probs, 1, generator=generator))


def spec_generate(model, tokenizer, prompt_text, max_new_tokens, context_size,
                  temperature=0.0, top_p=1.0, stop_on_eos=False, seed=None,
                  prompt_ids=None, on_token=None):
    """SELF-SPECULATIVE decoding via the model's own MTP module (DeepSeek-style).

    Each round runs ONE trunk forward over [current_token, draft_token] (a
    2-token chunk costs ~the same as 1 in the memory-bound decode regime) and
    one tiny MTP-block pass. The draft is verified against the trunk's own
    distribution with the standard k=1 speculative-sampling rule, so the output
    distribution equals ordinary (warped) sampling:
      accept draft B with prob min(1, p(B)/q(B));
      on reject, sample from normalize(max(0, p - q)).
    temperature=0 degrades exactly to greedy verification (one-hot p and q).
    Throughput: tokens/trunk-forward = 1 + acceptance_rate.

    Returns dict: text, token_ids, tokens_generated, rounds, accepted,
    acceptance_rate, accept_bits (per-round — the 'lookahead meter'),
    stop_reason, tok_s.
    """
    mtp = getattr(model, 'mtp', None)
    if mtp is None:
        raise ValueError("spec_generate requires an MTP checkpoint (model.mtp is None) — "
                         "use generate_with_stats / stream_generate_kv instead")
    device = next(model.parameters()).device

    if prompt_ids is None:
        prompt_ids = tokenizer.encode(prompt_text, bos=True, eos=False)
    prompt_len = len(prompt_ids)
    if prompt_len >= context_size:
        return {"text": "", "token_ids": [], "tokens_generated": 0, "rounds": 0,
                "accepted": 0, "acceptance_rate": 0.0, "accept_bits": [],
                "stop_reason": "error_prompt_too_long", "tok_s": 0.0}
    max_new_tokens = min(max_new_tokens, context_size - prompt_len)
    eos_id = getattr(tokenizer, 'eos_id', None)

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))

    def _sample(p):
        if temperature <= 0:
            return int(p.argmax())
        return int(torch.multinomial(p, 1, generator=generator))

    total_len = prompt_len + max_new_tokens
    emitted = []
    rounds = 0
    accepted = 0
    accept_bits = []
    stop_reason = "max_tokens"
    t0 = time.time()

    with torch.no_grad():
        model.setup_caches(max_batch_size=1, max_seq_len=total_len)
        toks = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)

        # ---- trunk prefill (captures pre-norm h for the MTP backfill) ----
        logits, h_pre = model.generate_forward(toks, 0, return_h_pre=True)
        cur = _sample(_warp_probs(logits[0, -1].float(), temperature, top_p))
        emitted.append(cur)
        if on_token: on_token(cur)

        # ---- MTP backfill over the whole prompt -> first draft ----
        nt = torch.tensor(prompt_ids[1:] + [cur], dtype=torch.long,
                          device=device).unsqueeze(0)
        mtp_logits = model.mtp_decode_chunk(h_pre, nt, 0)
        q_probs = _warp_probs(mtp_logits[0, -1].float(), temperature, top_p)
        draft = _sample(q_probs)

        n_ctx = prompt_len  # trunk+mtp caches materialized through position n_ctx-1

        while len(emitted) < max_new_tokens:
            if n_ctx + 2 > total_len:
                stop_reason = "context_limit"
                break
            if stop_on_eos and eos_id is not None and emitted and emitted[-1] == eos_id:
                stop_reason = "eos"
                break
            rounds += 1
            chunk = torch.tensor([[cur, draft]], dtype=torch.long, device=device)
            logits2, h_pre2 = model.generate_forward(chunk, n_ctx, return_h_pre=True)
            p1 = _warp_probs(logits2[0, 0].float(), temperature, top_p)

            # k=1 speculative-sampling verification (shared with the stream engine)
            ok, reject_tok = _spec_verify_step(p1, q_probs, draft, temperature,
                                               generator=generator)

            if ok:
                accepted += 1
                accept_bits.append(1)
                emitted.append(draft)
                if on_token: on_token(draft)
                if stop_on_eos and eos_id is not None and draft == eos_id:
                    # EOS accepted as FIRST of the pair: stop HERE. Without this,
                    # the loop speculates to the token cap, writing post-EOS
                    # positions into the caches while the post-loop truncation
                    # shrinks `emitted` — the ledger would then UNDERSTATE the
                    # materialized high-water mark and re-arm the wrong-timeline
                    # ring-reuse corruption the rewind guard cannot see
                    # (start_pos == len(ledger) looks like clean append).
                    # Both chunk rows (cur, draft) ARE materialized -> n_ctx += 2
                    # keeps the ledger arithmetic exact.
                    n_ctx += 2
                    stop_reason = "eos"
                    break
                p2 = _warp_probs(logits2[0, 1].float(), temperature, top_p)
                nxt = _sample(p2)
                if len(emitted) < max_new_tokens:
                    emitted.append(nxt)
                    if on_token: on_token(nxt)
                    if stop_on_eos and eos_id is not None and nxt == eos_id:
                        # EOS as second of the pair: nxt IS emitted but never
                        # forwarded — n_ctx += 2 covers exactly the materialized
                        # rows (cur, draft); the ledger slice excludes nxt. Guard
                        # this by `nxt was appended`: if the draft already hit the
                        # token cap, nxt is dropped and the turn ends via
                        # max_tokens — reporting "eos" for a token we never
                        # emitted would make stop_reason lie (last token != eos).
                        n_ctx += 2
                        stop_reason = "eos"
                        break
                # MTP backfill BOTH rows; row1 (position n_ctx+1) drafts t_{n_ctx+3}
                nt = torch.tensor([[draft, nxt]], dtype=torch.long, device=device)
                mtp_logits = model.mtp_decode_chunk(h_pre2, nt, n_ctx)
                q_probs = _warp_probs(mtp_logits[0, -1].float(), temperature, top_p)
                draft_new = _sample(q_probs)
                n_ctx += 2
                cur = nxt
                draft = draft_new
            else:
                accept_bits.append(0)
                # residual sample (from _spec_verify_step) keeps the output exact
                tok_v = reject_tok
                emitted.append(tok_v)
                if on_token: on_token(tok_v)
                # trunk position n_ctx+1 holds the REJECTED draft's K/V — the next
                # round's chunk write at n_ctx+1 overwrites it (ring and full
                # caches both overwrite by position; nothing reads it before then).
                # MTP backfill row0 only (h_pre of the rejected row is invalid).
                nt = torch.tensor([[tok_v]], dtype=torch.long, device=device)
                mtp_logits = model.mtp_decode_chunk(h_pre2[:, :1], nt, n_ctx)
                q_probs = _warp_probs(mtp_logits[0, -1].float(), temperature, top_p)
                draft = _sample(q_probs)
                n_ctx += 1
                cur = tok_v

        # LEDGER SNAPSHOT BEFORE ANY TRUNCATION (set_cache_ledger's contract:
        # exactly the ids physically forwarded into the cache — defense in
        # depth against any future truncation path desyncing ledger vs cache).
        materialized_ids = list(prompt_ids) + emitted[:max(0, n_ctx - prompt_len)]

        if stop_on_eos and eos_id is not None and eos_id in emitted:
            emitted = emitted[:emitted.index(eos_id) + 1]
            stop_reason = "eos"

    dt = max(time.time() - t0, 1e-9)
    model.set_cache_ledger(materialized_ids)
    return {
        "text": tokenizer.decode(emitted),
        "token_ids": emitted,
        "tokens_generated": len(emitted),
        "rounds": rounds,
        "accepted": accepted,
        "acceptance_rate": (accepted / rounds) if rounds else 0.0,
        "accept_bits": accept_bits,
        "stop_reason": stop_reason,
        "tok_s": len(emitted) / dt,
    }


def generate_with_stats(model, tokenizer, prompt_text, max_new_tokens,
                        context_size, temperature=0.7, top_p=0.9,
                        stop_on_eos=False, seed=None,
                        progress_prefix=None, progress_every=8):
    """Sibling of stream_generate_kv: silent, non-interactive, captures
    per-token entropy from the model's raw (T=1) next-token distribution.

    Used by the coherence sweep to get an intrinsic "how uncertain was the
    model at each step" signal alongside the generated text.

    Args:
        seed: optional int — if provided, sampling uses a per-call torch
            Generator seeded with this value. Passing the same seed at every
            checkpoint makes the sampling trajectory reproducible, so metric
            drift is attributable to the model and not the RNG.
        progress_prefix: optional string — if non-None, a live progress line
            is printed on stdout every `progress_every` tokens, carriage-
            returned so it overwrites in place. Caller is responsible for
            finishing the line (newline or overprint) after the call returns.
        progress_every: int — print progress every N generated tokens.

    Returns:
        dict with keys:
            text              : str  — decoded generation (excludes prompt)
            token_ids         : list[int]
            token_strings     : list[str]  — each id decoded in isolation
            per_token_entropy : list[float]  — raw-softmax entropy in nats
            tokens_generated  : int
            stop_reason       : str — "max_tokens" | "eos" | "context_limit"
    """
    device = next(model.parameters()).device

    prompt_ids = tokenizer.encode(prompt_text, bos=True, eos=False)
    prompt_len = len(prompt_ids)
    if prompt_len >= context_size:
        return {
            "text": "",
            "token_ids": [],
            "token_strings": [],
            "per_token_entropy": [],
            "tokens_generated": 0,
            "stop_reason": "error_prompt_too_long",
        }
    if prompt_len + max_new_tokens > context_size:
        max_new_tokens = context_size - prompt_len

    tokens = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    all_token_ids = list(prompt_ids)
    generated_tokens = []
    per_token_entropy = []

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))

    bsz = 1
    total_len = min(context_size, prompt_len + max_new_tokens)
    stop_reason = "max_tokens"

    gen_t0 = time.time()

    with torch.no_grad():
        model.setup_caches(max_batch_size=bsz, max_seq_len=total_len)
        start_pos = 0

        for i in range(max_new_tokens):
            if start_pos + tokens.shape[1] > context_size:
                stop_reason = "context_limit"
                break

            logits, _ = model(tokens, start_pos=start_pos)
            next_token_logits = logits[0, -1, :].float()

            # Intrinsic entropy: raw model distribution (T=1, no top-p).
            log_probs = torch.log_softmax(next_token_logits, dim=-1)
            probs_raw = torch.exp(log_probs)
            ent = -(probs_raw * log_probs).sum().item()
            per_token_entropy.append(ent)

            # Sampling distribution: temperature + top-p (as in stream_generate_kv).
            if temperature > 0:
                probs = torch.softmax(next_token_logits / temperature, dim=-1)
                if top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    sorted_remove = cumulative_probs > top_p
                    sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
                    sorted_remove[..., 0] = 0
                    to_remove = sorted_indices[sorted_remove]
                    probs[to_remove] = 0
                    probs = probs / probs.sum()
                if generator is not None:
                    next_token = torch.multinomial(probs, num_samples=1, generator=generator)
                else:
                    next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            next_token_id = int(next_token.item())
            generated_tokens.append(next_token_id)
            all_token_ids.append(next_token_id)

            start_pos += tokens.shape[1]
            tokens = next_token.unsqueeze(0)

            # Live token-level progress (overwrites single line).
            if progress_prefix is not None:
                done = i + 1
                if done % progress_every == 0 or done == max_new_tokens:
                    elapsed = time.time() - gen_t0
                    rate = done / elapsed if elapsed > 0 else 0.0
                    remaining = max_new_tokens - done
                    eta_s = int(remaining / rate) if rate > 0 else 0
                    eta_m, eta_r = divmod(eta_s, 60)
                    print(
                        f"\r{progress_prefix} tok {done}/{max_new_tokens} "
                        f"({rate:.1f} tok/s) ETA {eta_m}m{eta_r:02d}s   ",
                        end="", flush=True,
                    )

            if stop_on_eos and next_token_id == tokenizer.eos_id:
                stop_reason = "eos"
                break

    model.clear_caches()

    # Decode: full-sequence delta (handles SPM spacing correctly), then each
    # generated token in isolation for classifier use.
    prompt_text_decoded = tokenizer.decode(prompt_ids)
    full_text = tokenizer.decode(all_token_ids)
    gen_text = full_text[len(prompt_text_decoded):] if full_text.startswith(prompt_text_decoded) \
               else tokenizer.decode(generated_tokens)
    token_strings = [tokenizer.decode([tid]) for tid in generated_tokens]

    return {
        "text": gen_text,
        "token_ids": generated_tokens,
        "token_strings": token_strings,
        "per_token_entropy": per_token_entropy,
        "tokens_generated": len(generated_tokens),
        "stop_reason": stop_reason,
    }


# ---------------------------------------------------------------------------
# 4. Prompt loaders
# ---------------------------------------------------------------------------

import yaml  # local import after torch to keep import order clean

def load_yaml_prompt(path: str, users: List[str]):
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    prompt = data["prompt"]
    ai_name = data.get("ai_name", "Assistant")
    seed = data.get("seed", -1)

    prompt = prompt.replace("{{char}}", ai_name).replace("{{user}}", users[0]).replace("{{nl}}", "\n")
    print_and_log(f"Loaded prompt '{ai_name}' (seed {seed})")
    return prompt, ai_name, seed

def load_prompt(path: str, users: List[str]):
    if not os.path.exists(path):
        return None, "Assistant", -1  # -1 for seed if not specified
    if path.endswith(".yaml"):
        prompt, ai_name, seed = load_yaml_prompt(path, users)
        return prompt, ai_name, seed
    with open(path, "r", encoding="utf-8") as fh:
        prompt = fh.read()
    return prompt, "Assistant", -1  # -1 for seed if not specified


def load_yaml_chat_prompt(path: str, users: List[str]):
    """
    Load a YAML prompt file with chat format support.

    Supports two formats:

    Format 1 (inline conversations in prompt):
        ai_name: Sam
        prompt: |-
          Your name is {{char}}...
          {{char}}: "Hello!"
          {{user}}: "Hi there."

    Format 2 (separate conversations list):
        ai_name: Sam
        prompt: "Your name is {{char}}..."
        conversations:
          - role: "{{char}}"
            content: "Hello!"
          - role: "{{user}}"
            content: "Hi there."

    Returns:
        (system_prompt, conversations, ai_name, seed)
        where conversations is a list of {"role": "user"|"assistant", "content": "..."}
    """
    if not os.path.exists(path):
        return None, None, "Assistant", -1

    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    ai_name = data.get("ai_name", "Assistant")
    seed = data.get("seed", -1)
    user_name = users[0] if users else "User"

    conversations = []

    # Check if we have a separate conversations list (Format 2)
    if data.get("conversations"):
        # Format 2: separate conversations list
        system_prompt = data.get("prompt", "")
        system_prompt = system_prompt.replace("{{char}}", ai_name).replace("{{user}}", user_name).replace("{{nl}}", "\n")

        for msg in data["conversations"]:
            role = msg.get("role", "")
            content = msg.get("content", "")

            # Map role names
            if role == "{{char}}" or role == ai_name:
                role = "assistant"
            elif role == "{{user}}" or role == user_name:
                role = "user"

            # Apply placeholder replacements to content
            content = content.replace("{{char}}", ai_name).replace("{{user}}", user_name).replace("{{nl}}", "\n")
            conversations.append({"role": role, "content": content})
    else:
        # Format 1: inline conversations in prompt text
        # Parse lines looking for "{{char}}:" or "{{user}}:" patterns
        raw_prompt = data.get("prompt", "")
        raw_prompt = raw_prompt.replace("{{nl}}", "\n")

        system_lines = []
        char_prefix = "{{char}}:"
        user_prefix = "{{user}}:"

        for line in raw_prompt.split("\n"):
            stripped = line.strip()

            if stripped.startswith(char_prefix):
                # Assistant message
                content = stripped[len(char_prefix):].strip()
                content = content.replace("{{char}}", ai_name).replace("{{user}}", user_name)
                conversations.append({"role": "assistant", "content": content})
            elif stripped.startswith(user_prefix):
                # User message
                content = stripped[len(user_prefix):].strip()
                content = content.replace("{{char}}", ai_name).replace("{{user}}", user_name)
                conversations.append({"role": "user", "content": content})
            else:
                # System prompt line (before any conversation starts)
                if not conversations:
                    system_lines.append(line)
                # After conversations start, ignore non-prefixed lines (or could append to last message)

        system_prompt = "\n".join(system_lines)
        system_prompt = system_prompt.replace("{{char}}", ai_name).replace("{{user}}", user_name)

    print_and_log(f"Loaded chat prompt '{ai_name}' with {len(conversations)} messages (seed {seed})")
    return system_prompt, conversations, ai_name, seed


def render_chat_for_completion(system_prompt: str, conversations: list, add_generation_prompt: bool = True) -> str:
    """
    Render a conversation in chat format with special tokens.

    Matches training format (pre_tokenize_conversations.py):
    - System prompt is merged into the first user message with \\n\\n separator
    - Format: <|bos|><|user_start|>{system}\\n\\n{user_msg}<|user_end|>
              <|assistant_start|>{msg}<|assistant_end|>...

    Args:
        system_prompt: The system/instruction prompt (merged into first user turn)
        conversations: List of {"role": "user"|"assistant", "content": "..."}
        add_generation_prompt: If True and last message is from user, append <|assistant_start|>

    Returns:
        Rendered string ready for tokenization
    """
    parts = ["<|bos|>"]

    # Merge system prompt into first user message (matches training format)
    system_merged = False

    if system_prompt and conversations and conversations[0]["role"] == "user":
        # Merge system + first user message
        merged_content = system_prompt + "\n\n" + conversations[0]["content"]
        parts.append(f"<|user_start|>{merged_content}<|user_end|>")
        system_merged = True
    elif system_prompt:
        # System alone (no user message follows) - treat as user turn
        parts.append(f"<|user_start|>{system_prompt}<|user_end|>")

    # Render each conversation turn (skip first if already merged)
    for i, msg in enumerate(conversations):
        if i == 0 and system_merged:
            continue

        role = msg["role"]
        content = msg["content"]

        if role == "assistant":
            parts.append(f"<|assistant_start|>{content}<|assistant_end|>")
        elif role == "user":
            parts.append(f"<|user_start|>{content}<|user_end|>")

    # Add generation prompt if requested and last message was from user
    if add_generation_prompt:
        if not conversations or conversations[-1]["role"] == "user":
            parts.append("<|assistant_start|>")

    return "".join(parts)


def pretty_print_chat(system_prompt: str, conversations: list, ai_name: str = "Assistant", user_name: str = "User"):
    """
    Pretty print a chat conversation for display.

    Example output:
        ┌─ SYSTEM ─────────────────────────────────────────
        │ Your name is Sam. You are a very intelligent...
        │
        ├─ SAM ────────────────────────────────────────────
        │ "What can I assist you with this evening?"
        │
        ├─ JOSEF ──────────────────────────────────────────
        │ "I just have a few questions for you."
        │
        └─ SAM (generating...) ────────────────────────────
    """
    width = 50
    lines = []

    def format_block(label: str, content: str, is_first: bool = False, is_generating: bool = False):
        # Header line
        prefix = "┌" if is_first else "├"
        suffix = " (generating...)" if is_generating else ""
        header = f"{prefix}─ {label}{suffix} "
        header += "─" * max(0, width - len(header))
        lines.append(header)

        # Content lines (if any)
        if content:
            for line in content.split("\n"):
                lines.append(f"│ {line}")
            lines.append("│")

    # System prompt
    if system_prompt:
        format_block("SYSTEM", system_prompt, is_first=True)

    # Conversation turns
    for i, msg in enumerate(conversations):
        role = msg["role"]
        content = msg["content"]
        label = ai_name.upper() if role == "assistant" else user_name.upper()
        is_first = (i == 0 and not system_prompt)
        format_block(label, content, is_first=is_first)

    # Generation prompt
    last_role = conversations[-1]["role"] if conversations else "user"
    if last_role == "user":
        label = ai_name.upper()
        lines.append(f"└─ {label} (generating...) " + "─" * max(0, width - len(label) - 20))
    else:
        # End the box
        lines.append("└" + "─" * width)

    return "\n".join(lines)


# ----------------------------------------------------------------------
#  trim-helper — keeps messages inside context window
# ----------------------------------------------------------------------
def trim_messages_inplace(messages: list[str],
                          enc,
                          context_len: int,
                          max_new_tokens: int) -> int:
    """
    Mutates *messages* by deleting earliest “middle” turns (index 1, then 2…)
    until the tokenised length fits within `context_len - max_new_tokens`.

    Returns the number of messages removed.
    """
    deleted = 0
    while (
        len(enc.encode("".join(messages), bos=True, eos=False))
        > context_len - max_new_tokens
        and len(messages) > 1          # keep the system prompt
    ):
        del messages[1]
        deleted += 1
    return deleted
