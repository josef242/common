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

    if checkpoint_version == "3.0":
        # FSDP2 checkpoint - use common_fsdp2 model (no __call__ override)
        from model_v2 import Transformer, ModelArgs
        logger.print_and_log(f"Detected FSDP2 checkpoint (v{checkpoint_version}), step: {checkpoint_step}")
    else:
        # FSDP1 checkpoint - use saved FSDP1 model (has __call__ override for KV cache)
        from common_fsdp1.model_v2 import Transformer, ModelArgs
        logger.print_and_log(f"Detected FSDP1 checkpoint (v{checkpoint_version}), step: {checkpoint_step}")

    # Temp Hack for old checkpoint compatibility
    import model_v2 as _model_v2
    import sys
    sys.modules['model_v1_AdamC'] = _model_v2

    cfg = chk["config"]

    # Handle both old (dataclass) and new (dict) checkpoint formats
    if isinstance(cfg, dict):
        # New format - config is a dictionary
        # Backwards compatibility for when inner_dim was called hidden_dim
        if "hidden_dim" in cfg and "inner_dim" not in cfg:
            cfg["inner_dim"] = cfg["hidden_dim"]

        # Determine qk_norm_mode
        if qk_norm_mode is not None:
            # Explicit override from command line
            resolved_qk_norm_mode = qk_norm_mode if qk_norm_mode != "none" else None
        else:
            # Auto-detect from checkpoint
            if "qk_norm_mode" in cfg:
                # New format: mode string directly in config
                resolved_qk_norm_mode = cfg["qk_norm_mode"]
            elif cfg.get("qk_norm", False):
                # Old format: boolean qk_norm=True → legacy behavior
                resolved_qk_norm_mode = "after_rope_legacy"
                logger.print_and_log("Detected old qk_norm=True, using 'after_rope_legacy' mode")
            else:
                # No QK norm
                resolved_qk_norm_mode = None

        if checkpoint_version == "3.0":
            # FSDP2: use filter approach - pass all config keys through to ModelArgs
            # This is future-proof and automatically handles MoE parameters
            import dataclasses
            model_args = dict(cfg)
            # Override inference-specific settings
            model_args["ep_degree"] = 1  # single GPU inference
            model_args["use_activation_checkpointing"] = False
            model_args["dropout"] = 0.0
            model_args["qk_norm_mode"] = resolved_qk_norm_mode
            # use_keel: CLI override takes priority, then checkpoint value
            if use_keel is not None:
                model_args["use_keel"] = use_keel
            # Backwards compat: tie_word_embeddings defaults to True for old checkpoints
            model_args.setdefault("tie_word_embeddings", True)
            # Filter to known ModelArgs fields
            known_fields = {f.name for f in dataclasses.fields(ModelArgs)}
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
        _old_fields.setdefault("rope_theta", 10000.0)  # v1 default was 10000, v2 default is 500000
        _known = {f.name for f in _dc.fields(ModelArgs)}
        cfg = ModelArgs(**{k: v for k, v in _old_fields.items() if k in _known})

    cfg.pad_id = enc.pad_id

    # Print model config in a readable format
    logger.print_and_log("Model configuration:")
    logger.print_and_log(f"  dim: {cfg.dim}, n_layers: {cfg.n_layers}, n_heads: {cfg.n_heads}, n_kv_heads: {cfg.n_kv_heads}")
    logger.print_and_log(f"  vocab_size: {cfg.vocab_size}, max_seq_len: {cfg.max_seq_len}")
    logger.print_and_log(f"  inner_dim: {cfg.inner_dim}, norm_eps: {cfg.norm_eps}")
    logger.print_and_log(f"  rope_theta: {getattr(cfg, 'rope_theta', 'N/A')}, qk_norm_mode: {cfg.qk_norm_mode}")
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

    model = Transformer(cfg)

    if half_precision:
        logger.print_and_log("Loading model in half precision (bfloat16)")
        model = model.to(torch.bfloat16)

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

    return model, cfg


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
):
    """Load both tokenizer **and** model; optionally shard across multiple GPUs.

    If tok_kind/tok_path/special_tokens are not specified, they will be auto-detected from the checkpoint.
    """

    if checkpoint_path.endswith(".bin"):
        raise ValueError("Legacy .bin checkpoints not supported – use .pt")

    # Load checkpoint metadata and log it
    chk_meta = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)

    # Log all checkpoint metadata
    logger.print_and_log("Checkpoint metadata:")
    # Keys that are actually saved in checkpoints (from train_mara.py)
    metadata_keys = ["step", "total_tokens_processed", "checkpoint_version",
                     "tok_kind", "tok_path", "special_tokens",
                     "optimizer_type", "use_adamc", "max_lr"]
    for key in metadata_keys:
        value = chk_meta.get(key)
        if value is not None:
            # Truncate long values for display
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
            tok_path = candidate
    if special_tokens and isinstance(special_tokens, str) and not os.path.isabs(special_tokens) and not os.path.exists(special_tokens):
        candidate = os.path.normpath(os.path.join(ckpt_dir, special_tokens))
        if os.path.exists(candidate):
            special_tokens = candidate

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


def stream_generate_kv(model, tokenizer, prompt_text, max_new_tokens, context_size,
                       temperature, top_p, display=True, stop_on_eos=False, stop_sequences=None,
                       print_prompt=True, return_stop_info=False,
                       pretty_print=False, role_names=None):
    """
    Generates text using KV Caching for O(N) complexity per token.

    FIXED: Properly handles SentencePiece tokenizers by decoding full sequence
    and printing deltas, preserving spaces correctly.

    Args:
        pretty_print: If True, replace special tokens with readable versions when displaying
        role_names: Dict with "assistant" and "user" keys for pretty print names
    """

    device = next(model.parameters()).device

    # 1. Prepare Tokens
    tokens = tokenizer.encode(prompt_text, bos=True, eos=False)
    prompt_len = len(tokens)

    # Bounds check: ensure we don't exceed context size
    if prompt_len >= context_size:
        logger.print_and_log(f"\nError: Prompt length ({prompt_len} tokens) exceeds or equals context size ({context_size} tokens).")
        logger.print_and_log("Please use a shorter prompt or a model with a larger max_seq_len.")
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
    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    # Track ALL generated token IDs (not just the tensor)
    all_token_ids = prompt_ids.copy()
    generated_tokens = []

    # 2. Setup Cache
    bsz = 1
    total_len = min(context_size, len(all_token_ids) + max_new_tokens)

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

    with torch.no_grad():
        model.setup_caches(max_batch_size=bsz, max_seq_len=total_len)

        start_pos = 0

        if display and print_prompt:
            if pretty_print and role_names:
                print(_prettify_special_tokens(prompt_text, role_names), end="", flush=True)
            else:
                print(prompt_text, end="", flush=True)

        # 3. Prefill and Generation Loop
        with NonBlockingInput():
            for i in range(max_new_tokens):
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
                
                # Store the generated token
                generated_tokens.append(next_token_id)
                all_token_ids.append(next_token_id)

                # ===== FIXED: Decode and print with SentencePiece workaround =====
                # Always decode to track generated text (needed for stop sequences)
                if needs_spm_workaround:
                    # Decode the FULL sequence to preserve spacing context
                    decoded_full = tokenizer.decode(all_token_ids)
                    new_text = decoded_full[last_decoded_len:]
                    if new_text:
                        generated_text_so_far += new_text
                        print_buffer += new_text
                        last_decoded_len = len(decoded_full)
                else:
                    # Non-SentencePiece tokenizers can decode token-by-token
                    decoded_token = tokenizer.decode([next_token_id])
                    generated_text_so_far += decoded_token
                    print_buffer += decoded_token

                # Check for stop sequences
                if stop_sequences:
                    for stop_seq in stop_sequences:
                        if stop_seq in generated_text_so_far:
                            stop_sequence_hit = stop_seq
                            # Trim the generated text at the stop sequence
                            generated_text_so_far = generated_text_so_far[:generated_text_so_far.find(stop_seq)]
                            # Also trim print buffer - only print up to stop sequence
                            if stop_seq in print_buffer:
                                print_buffer = print_buffer[:print_buffer.find(stop_seq)]
                            break
                    if stop_sequence_hit:
                        # Print remaining safe buffer before breaking
                        if display and print_buffer:
                            to_print = print_buffer
                            if pretty_print and role_names:
                                to_print = _prettify_special_tokens(to_print, role_names)
                            print(to_print, end="", flush=True)
                        stop_reason = {"reason": "stop_sequence", "detail": repr(stop_sequence_hit), "tokens_generated": i + 1}
                        break

                # Buffered printing: only print text that's safe (beyond stop sequence length
                # and not splitting special tokens when pretty_print is enabled)
                if display and max_stop_len > 0:
                    if len(print_buffer) > max_stop_len:
                        # Determine how much we can safely print
                        candidate = print_buffer[:-max_stop_len]

                        # If pretty_print is enabled, find safe boundary that doesn't split tokens
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
                    # No stop sequences, print immediately
                    to_print = print_buffer
                    if pretty_print and role_names:
                        to_print = _prettify_special_tokens(to_print, role_names)
                    print(to_print, end="", flush=True)
                    print_buffer = ""

                # Update for next iteration
                start_pos += tokens.shape[1]
                tokens = next_token.unsqueeze(0)

                # Stop condition
                if stop_on_eos and next_token_id == tokenizer.eos_id:
                    stop_reason = {"reason": "eos", "detail": f"token_id={tokenizer.eos_id}", "tokens_generated": i + 1}
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

    # Clean up memory
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
