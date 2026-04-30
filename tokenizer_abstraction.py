# tokenizer_abstraction.py
"""
A *uniform* interface that hides whether we are using:
  • the legacy in-house ``Tokenizer`` class (tokenizer.py), or
  • any Hugging-Face AutoTokenizer (including custom SentencePiece/
    GPT-2/LLaMA models), or
  • an OpenAI-style *tiktoken* encoding (e.g. ``cl100k_base``), or
  • Claude's numeric tokenizer with R2L grouping and NUM_START tokens.

Core API each adapter must expose
---------------------------------
    encode(text: str, bos: bool = False, eos: bool = False) -> List[int]
    decode(ids:  List[int]) -> str
    pad_id: int
    eos_id: int  <-- NEW

Optional helpers for zero-copy batching
---------------------------------------
    encode_to_uint16(text, add_bos=True) -> np.ndarray[uint16]
    encode_to_uint32(text, add_bos=True) -> np.ndarray[uint32]

Implemented adapters
--------------------
* LlamaTokenizerAdapter – wrapper around the original C++/Python Tokenizer
* HFTokenizerAdapter    – generic wrapper for any 🤗 AutoTokenizer
* TikTokenAdapter       – wrapper for OpenAI's *tiktoken* encodings
* ClaudeTokenizerAdapter – wrapper for Claude's numeric tokenizer
* get_tokenizer         – factory that selects the right adapter
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Sequence
import numpy as np

###############################################################################
# Base protocol – what the rest of the code expects
###############################################################################
class BaseTokenizer:  # noqa: D101 – simple interface
    pad_id: int

    # ---------- NEW: Mandatory EOS ID -------------------------------------
    @property
    def eos_id(self) -> int:
        raise NotImplementedError

    @property
    def bos_id(self) -> int:
        raise NotImplementedError

    # ---------- mandatory -------------------------------------------------
    def encode(
        self, text: str, *, bos: bool = False, eos: bool = False
    ) -> List[int]:
        raise NotImplementedError

    def decode(self, ids: Sequence[int]) -> str:
        raise NotImplementedError

    # ---------- optional helpers for dataloaders --------------------------
    def encode_to_uint16(
        self, text: str, *, add_bos: bool = True
    ) -> np.ndarray:
        """Return a uint16 tensor (fails if vocab ≥ 65 536)."""
        raise NotImplementedError

    def encode_to_uint32(
        self, text: str, *, add_bos: bool = True
    ) -> np.ndarray:
        """Return a uint32 tensor (no practical vocab limit)."""
        raise NotImplementedError
    
    # ------- new: uniform vocab length ---------------------------------
    def __len__(self) -> int:
        """Return total number of token IDs (incl. specials)."""
        raise NotImplementedError

    @property
    def auto_added_tokens(self) -> int:
        """Number of tokens automatically added by the adapter (e.g., <pad>).

        Used for vocab size verification: if a config expects N tokens but
        we have N + auto_added_tokens, that's expected and not a mismatch.
        """
        return 0  # Default: no auto-added tokens

    @property
    def base_vocab_size(self) -> int:
        """Return the base vocabulary size before any special tokens were added."""
        raise NotImplementedError

    @property
    def special_token_ids(self) -> dict:
        """Return mapping of custom special token string → ID (excludes auto-added tokens like <pad>)."""
        return {}


###############################################################################
# Legacy LLaMA tokenizer adapter
###############################################################################
class LlamaTokenizerAdapter(BaseTokenizer):
    """Adapter for the original in-house `tokenizer.Tokenizer` class."""

    def __init__(self, model_path: str | Path, special_tokens: Optional[List[str]] = None):
        from llama_tokenizer import Tokenizer  # local import keeps CLI snappy

        model_path = Path(model_path).expanduser()
        self._t = Tokenizer(model_path)
        self._base_vocab_size = self._t.get_vocab_size()

        # Add custom special tokens (IDs start after base vocab)
        self._special_token_to_id: dict[str, int] = {}
        self._id_to_special_token: dict[int, str] = {}
        if special_tokens:
            next_id = self._base_vocab_size
            for token in special_tokens:
                if token not in self._special_token_to_id:
                    self._special_token_to_id[token] = next_id
                    self._id_to_special_token[next_id] = token
                    next_id += 1

        # Build regex for splitting on special tokens
        if self._special_token_to_id:
            # Escape special regex chars and join with |
            escaped = [re.escape(t) for t in self._special_token_to_id.keys()]
            self._special_pattern = re.compile("(" + "|".join(escaped) + ")")
        else:
            self._special_pattern = None

    # ---------- required API ---------------------------------------------
    @property
    def pad_id(self) -> int:        # always non-negative
        pid = self._t.pad_id
        return pid if pid >= 0 else self._t.eos_id   # or 0

    @property
    def eos_id(self) -> int:
        return self._t.eos_id

    @property
    def bos_id(self) -> int:
        return self._t.bos_id

    def encode(self, text: str, *, bos: bool = False, eos: bool = False):
        """Encode text, handling special tokens."""
        if not self._special_pattern:
            # No special tokens, use base encoder
            return self._t.encode(text, bos=bos, eos=eos)

        # Split on special tokens
        parts = self._special_pattern.split(text)
        ids = []

        if bos:
            ids.append(self._t.bos_id)

        for part in parts:
            if not part:
                continue
            if part in self._special_token_to_id:
                ids.append(self._special_token_to_id[part])
            else:
                ids.extend(self._t.encode(part, bos=False, eos=False))

        if eos:
            ids.append(self._t.eos_id)

        return ids

    def decode(self, ids: Sequence[int]):
        """Decode IDs, handling special tokens.

        SentencePiece uses ▁ (U+2581) to represent word-boundary spaces and
        strips the leading ▁ from the first piece in each decode() call.
        When we split on custom special tokens, non-first chunks lose their
        leading space.  Fix: check if the chunk's first token starts with ▁
        in SentencePiece's vocabulary and re-add the space if decode stripped it.
        """
        if not self._id_to_special_token:
            return self._t.decode(ids)

        # Decode in chunks, replacing special token IDs with their strings
        result = []
        chunk = []
        is_first_chunk = True
        for id_ in ids:
            if id_ in self._id_to_special_token:
                # Flush any pending regular tokens
                if chunk:
                    decoded = self._t.decode(chunk)
                    if not is_first_chunk and chunk:
                        # SentencePiece strips the leading ▁ from the first
                        # piece in each decode call.  For non-first chunks,
                        # check if the first token's piece starts with ▁ and
                        # re-add the space that decode() stripped.
                        piece = self._t.sp_model.id_to_piece(chunk[0])
                        if piece.startswith('▁') and not decoded.startswith(' '):
                            decoded = ' ' + decoded
                    result.append(decoded)
                    is_first_chunk = False
                    chunk = []
                result.append(self._id_to_special_token[id_])
            else:
                chunk.append(id_)

        # Flush remaining
        if chunk:
            decoded = self._t.decode(chunk)
            if not is_first_chunk and chunk:
                piece = self._t.sp_model.id_to_piece(chunk[0])
                if piece.startswith('▁') and not decoded.startswith(' '):
                    decoded = ' ' + decoded
            result.append(decoded)

        return "".join(result)

    # ---------- extras ----------------------------------------------------
    def _encode_core(self, text: str, *, add_bos: bool):
        return self.encode(text, bos=add_bos, eos=False)

    def encode_to_uint16(self, text: str, *, add_bos: bool = True):
        ids = self._encode_core(text, add_bos=add_bos)
        assert len(self) < 65_536, "vocab too large for uint16"
        return np.asarray(ids, dtype=np.uint16)

    def encode_to_uint32(self, text: str, *, add_bos: bool = True):
        ids = self._encode_core(text, add_bos=add_bos)
        return np.asarray(ids, dtype=np.uint32)

    def __len__(self):
        return self._base_vocab_size + len(self._special_token_to_id)

    @property
    def base_vocab_size(self) -> int:
        return self._base_vocab_size

    @property
    def special_token_ids(self) -> dict:
        return dict(self._special_token_to_id)

###############################################################################
# Generic HF AutoTokenizer adapter
###############################################################################
class HFTokenizerAdapter(BaseTokenizer):
    """Wrap any Hugging-Face *AutoTokenizer* (fast or python)."""

    def __init__(self, model_path: str | Path, *, use_fast: bool = True):
        from transformers import AutoTokenizer

        model_path = Path(model_path).expanduser()
        self._t = AutoTokenizer.from_pretrained(
            model_path, use_fast=use_fast, legacy=False
        )

        # Ensure PAD exists
        if self._t.pad_token is None:
            self._t.add_special_tokens({"pad_token": "<pad>"})

        # Map BOS → EOS when only EOS is defined (GPT-2 style)
        if self._t.bos_token_id is None and self._t.eos_token_id is not None:
            self._t.add_special_tokens({"bos_token": self._t.eos_token})

        # Lift model_max_length to avoid warnings
        self._t.model_max_length = 1_000_000_000
        self._t.init_kwargs["model_max_length"] = 1_000_000_000

    # ---------- required API ---------------------------------------------
    @property
    def pad_id(self):
        return int(self._t.pad_token_id)

    @property
    def eos_id(self):
        return int(self._t.eos_token_id)

    @property
    def bos_id(self):
        return int(self._t.bos_token_id) if self._t.bos_token_id is not None else -1

    def encode(self, text: str, *, bos: bool = False, eos: bool = False):
        ids = self._t.encode(text, add_special_tokens=False)
        if bos and self._t.bos_token_id is not None:
            ids.insert(0, self._t.bos_token_id)
        if eos and self._t.eos_token_id is not None:
            ids.append(self._t.eos_token_id)
        return ids

    def decode(self, ids: Sequence[int], skip_special: bool = False):
        """By default **keep** special tokens (matches TikTokenAdapter)."""
        return self._t.decode(ids, skip_special_tokens=skip_special)

    # ---------- extras ----------------------------------------------------
    def _encode_core(self, text: str, *, add_bos: bool):
        ids = self._t.encode(text, add_special_tokens=False)
        if add_bos and self._t.bos_token_id is not None:
            ids.insert(0, self._t.bos_token_id)
        return ids

    def encode_to_uint16(self, text: str, *, add_bos: bool = True):
        ids = self._encode_core(text, add_bos=add_bos)
        assert len(self._t) < 65_536, "vocab too large for uint16"
        return np.asarray(ids, dtype=np.uint16)

    def encode_to_uint32(self, text: str, *, add_bos: bool = True):
        ids = self._encode_core(text, add_bos=add_bos)
        return np.asarray(ids, dtype=np.uint32)

    def __len__(self):
        return len(self._t)             # HF tokenizers override __len__


###############################################################################
# TikToken adapter – OpenAI *tiktoken* encodings
###############################################################################
class TikTokenAdapter(BaseTokenizer):
    """Adapter around ``tiktoken`` (GPT-style byte-level BPE)."""

    def __init__(self, name: str = "cl100k_base", special_tokens: Optional[List[str]] = None):
        import tiktoken

        self._enc = tiktoken.get_encoding(name)
        self._base_n_vocab = self._enc.n_vocab  # Store before any modifications

        # BOS / EOS (GPT uses the same token for both)
        self._bos_token = "<|endoftext|>"
        self._bos_id = self._enc.encode(
            self._bos_token, allowed_special={self._bos_token}
        )[0]
        self._eos_id = self._bos_id

        # Collect all special tokens to add
        new_special_tokens = {}
        self._custom_special_ids = {}  # Track custom tokens (not <pad>)
        next_id = self._enc.n_vocab
        self._auto_added_pad = False  # Track if we auto-added pad

        # Add PAD if missing
        if "<pad>" not in self._enc._special_tokens:  # type: ignore[attr-defined]
            new_special_tokens["<pad>"] = next_id
            next_id += 1
            self._auto_added_pad = True

        # Add custom special tokens (e.g., conversation markers, tool tokens)
        if special_tokens:
            for token in special_tokens:
                if token not in self._enc._special_tokens and token not in new_special_tokens:
                    new_special_tokens[token] = next_id
                    self._custom_special_ids[token] = next_id
                    next_id += 1

        # Rebuild encoding if we have new tokens
        if new_special_tokens:
            self._enc = tiktoken.Encoding(
                name=f"{name}-extended",
                pat_str=self._enc._pat_str,  # type: ignore[attr-defined]
                mergeable_ranks=self._enc._mergeable_ranks,  # type: ignore[attr-defined]
                special_tokens={
                    **self._enc._special_tokens,
                    **new_special_tokens,
                },  # type: ignore[attr-defined]
            )

        self._pad_id = self._enc.encode("<pad>", allowed_special={"<pad>"})[0]

    # ---------- required API ---------------------------------------------
    @property
    def pad_id(self):
        return self._pad_id

    @property
    def eos_id(self):
        return self._eos_id

    @property
    def bos_id(self):
        return self._bos_id

    def encode(self, text: str, *, bos: bool = False, eos: bool = False):
        ids = self._enc.encode(
            text, allowed_special="all", disallowed_special=()  # type: ignore[attr-defined]
        )
        if bos:
            ids.insert(0, self._bos_id)
        if eos:
            ids.append(self._eos_id)
        return ids

    def decode(self, ids: Sequence[int]):
        return self._enc.decode(ids)

    # ---------- extras ----------------------------------------------------
    def _encode_core(self, text: str, *, add_bos: bool):
        ids = self._enc.encode(
            text, allowed_special="all", disallowed_special=()  # type: ignore[attr-defined]
        )
        if add_bos:
            ids.insert(0, self._bos_id)
        return ids

    def encode_to_uint16(self, text: str, *, add_bos: bool = True):
        ids = self._encode_core(text, add_bos=add_bos)
        assert self._enc.n_vocab < 65_536, "vocab too large for uint16"
        return np.asarray(ids, dtype=np.uint16)

    def encode_to_uint32(self, text: str, *, add_bos: bool = True):
        ids = self._encode_core(text, add_bos=add_bos)
        return np.asarray(ids, dtype=np.uint32)

    def __len__(self):
        return max(self._enc._special_tokens.values(), default=-1) + 1  # type: ignore[attr-defined]

    @property
    def auto_added_tokens(self) -> int:
        """Number of tokens automatically added (e.g., <pad>)."""
        return 1 if self._auto_added_pad else 0

    @property
    def base_vocab_size(self) -> int:
        return self._base_n_vocab

    @property
    def special_token_ids(self) -> dict:
        return dict(self._custom_special_ids)


###############################################################################
# Claude's Numeric Tokenizer (embedded)
###############################################################################
_SEPARATORS = {",", "_", "'"}   # characters we keep as tokens in numbers

class ClaudeTokenizer:
    """
    Claude-numeric tokenizer wrapper that adds Claude-3-style number logic
    (R2L 3-digit grouping + "mystery" number-start token) on top of an 
    existing HF PreTrainedTokenizerFast.
    """
    NUM_START = "<NUM_START>"
    NUM_START_WITH_SPACE = "Ġ<NUM_START>"  # Version with space prefix

    def __init__(self, tokenizer):
        """Initialize with an existing HF tokenizer instance."""
        self._t = tokenizer
        
        # Add both versions of NUM_START token if not present
        special_tokens_to_add = []
        vocab = self._t.get_vocab()
        
        if self.NUM_START not in vocab:
            special_tokens_to_add.append(self.NUM_START)
        if self.NUM_START_WITH_SPACE not in vocab:
            special_tokens_to_add.append(self.NUM_START_WITH_SPACE)
            
        if special_tokens_to_add:
            self._t.add_special_tokens(
                {"additional_special_tokens": special_tokens_to_add}
            )

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """Load from pretrained model path."""
        from transformers import PreTrainedTokenizerFast
        tokenizer = PreTrainedTokenizerFast.from_pretrained(*args, **kwargs)
        return cls(tokenizer)

    # Delegate common tokenizer attributes
    def __getattr__(self, name):
        return getattr(self._t, name)

    def __len__(self):
        return len(self._t)

    # ------------------------------------------------------------- #
    # helpers
    # ------------------------------------------------------------- #
    @staticmethod
    def _group_r2l(digits: str) -> list[str]:
        """Right-to-left 3-digit grouping for a *pure* digit string."""
        # For 1234567: length=7, 7%3=1, so first group is 1 digit
        # Result: ['1', '234', '567']
        first = len(digits) % 3
        if first == 0:
            first = 3
        groups = [digits[:first]] + [
            digits[i : i + 3] for i in range(first, len(digits), 3)
        ]
        return groups

    @classmethod
    def _format_number_parts(cls, raw: str) -> list[str]:
        """
        Split number into parts according to R2L grouping.
        Handles separators by attaching them to the preceding group.
        """
        parts = []
        current_digits = []
        
        for char in raw:
            if char in _SEPARATORS:
                # Process accumulated digits
                if current_digits:
                    digit_str = ''.join(current_digits)
                    parts.extend(cls._group_r2l(digit_str))
                    current_digits = []
                # Attach separator to last part
                if parts:
                    parts[-1] += char
            elif char.isdigit():
                current_digits.append(char)
        
        # Process any remaining digits
        if current_digits:
            digit_str = ''.join(current_digits)
            parts.extend(cls._group_r2l(digit_str))
        
        return parts

    @classmethod
    def _inject_nums(cls, text: str) -> str:
        """
        Process text to inject NUM_START tokens before numbers.
        Uses Ġ<NUM_START> when there's a space before the number.
        Numbers directly attached to letters/digits do NOT get NUM_START.
        """
        # Pattern that matches:
        # - integers with optional separators (1,234 or 1_234)
        # - decimals (1.5, 3.14159)
        # - numbers with both (1,234.56)
        # But NOT multiple dots (IP addresses like 192.168.1.1)
        pattern = re.compile(r'\d+(?:[,_\']\d+)*(?:\.\d+)?')
        
        result = []
        last_end = 0
        
        for match in pattern.finditer(text):
            start, end = match.span()
            number = match.group()
            
            # Check what comes before this number
            if start > 0:
                char_before = text[start - 1]
                # Skip if preceded by alphanumeric, underscore, or period
                # (to avoid splitting IP addresses or version numbers)
                if char_before.isalnum() or char_before in '_.' :
                    # Don't add NUM_START, just add the text as-is
                    result.append(text[last_end:end])
                    last_end = end
                    continue
            
            # Add everything before the number
            prefix = text[last_end:start]
            result.append(prefix)
            
            # Check if we should use space version of NUM_START
            if prefix.endswith(' '):
                # Remove the trailing space from prefix
                result[-1] = prefix[:-1]
                # Use Ġ<NUM_START>
                result.append(cls.NUM_START_WITH_SPACE)
            else:
                # Use regular NUM_START
                result.append(cls.NUM_START)
            
            # Add the number as-is
            result.append(number)
            
            last_end = end
        
        # Add any remaining text
        result.append(text[last_end:])
        
        return ''.join(result)

    def tokenize(self, text, **kw):
        # Pre-process text to inject NUM_START tokens
        processed = self._inject_nums(text)
        
        # Let the base tokenizer handle it
        tokens = self._t.tokenize(processed, **kw)
        
        # Post-process to enforce R2L grouping
        result = []
        i = 0
        
        while i < len(tokens):
            token = tokens[i]
            
            # Keep NUM_START tokens as-is
            if token in [self.NUM_START, self.NUM_START_WITH_SPACE]:
                result.append(token)
                i += 1
                # The next tokens should be a number - process them specially
                continue
            
            # Check if this is a digit token that needs R2L grouping
            clean_token = token.lstrip('Ġ')
            if clean_token.isdigit() and len(result) > 0 and result[-1] in [self.NUM_START, self.NUM_START_WITH_SPACE]:
                # This is the start of a number after NUM_START
                # Collect all consecutive digit/decimal tokens
                has_space_prefix = token.startswith('Ġ')
                digit_sequence = clean_token
                j = i + 1
                
                # Check if this is part of a decimal number
                has_decimal = False
                decimal_part = ""
                
                # Look for decimal point
                if j < len(tokens) and tokens[j] == '.':
                    # Check if followed by digits
                    if j + 1 < len(tokens) and tokens[j + 1].lstrip('Ġ').isdigit():
                        has_decimal = True
                        j += 1  # Skip the period
                        # Collect decimal digits
                        while j < len(tokens) and tokens[j].lstrip('Ġ').isdigit():
                            decimal_part += tokens[j].lstrip('Ġ')
                            j += 1
                else:
                    # No decimal, collect remaining integer digits
                    while j < len(tokens) and tokens[j].lstrip('Ġ').isdigit():
                        digit_sequence += tokens[j].lstrip('Ġ')
                        j += 1
                
                # Apply R2L grouping only to the integer part
                groups = self._group_r2l(digit_sequence)
                
                # Add the groups
                result.extend(groups)
                
                # Add decimal part if present
                if has_decimal:
                    result.append('.')
                    result.append(decimal_part)
                
                i = j
                continue
            
            # For other tokens, check if they're part of a multi-token number
            if clean_token.isdigit() and i > 0 and tokens[i-1].lstrip('Ġ').isdigit():
                # Skip this - it was already processed as part of a number
                i += 1
                continue
            
            # Regular token, add as-is
            result.append(token)
            i += 1
        
        return result

    def __call__(self, text, **kw):
        return self._t(self._inject_nums(text), **kw)

    def encode(self, text, **kw):
        return self._t.encode(self._inject_nums(text), **kw)

    # Original decode - shows <NUM_START> tokens as-is
    #def decode(self, ids, **kw):
    #    return self._t.decode(ids, **kw)

    def decode(self, ids, **kw):
        # By default, we will strip <NUM_START> from the decoded text
        text = self._t.decode(ids, **kw)
        # Remove NUM_START tokens from the decoded text
        text = text.replace(self.NUM_START, "").replace(self.NUM_START_WITH_SPACE, " ")
        return text

    def batch_encode_plus(self, batch_text_or_text_pairs, **kw):
        batch = [self._inject_nums(t) for t in batch_text_or_text_pairs]
        return self._t.batch_encode_plus(batch, **kw)


###############################################################################
# Claude tokenizer adapter – Numeric tokenizer with R2L grouping
###############################################################################
class ClaudeTokenizerAdapter(BaseTokenizer):
    """Adapter for Claude's numeric tokenizer with R2L 3-digit grouping."""

    def __init__(self, model_path: str | Path):
        from transformers import PreTrainedTokenizerFast

        model_path = Path(model_path).expanduser()
        base_tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
        self._t = ClaudeTokenizer(base_tokenizer)

        # Ensure PAD exists
        if self._t.pad_token is None:
            self._t.add_special_tokens({"pad_token": "<pad>"})

        # Map BOS → EOS when only EOS is defined
        if self._t.bos_token_id is None and self._t.eos_token_id is not None:
            self._t.add_special_tokens({"bos_token": self._t.eos_token})

    # ---------- required API ---------------------------------------------
    @property
    def pad_id(self):
        return int(self._t.pad_token_id)

    @property
    def eos_id(self):
        return int(self._t.eos_token_id)

    @property
    def bos_id(self):
        return int(self._t.bos_token_id) if self._t.bos_token_id is not None else -1

    def encode(self, text: str, *, bos: bool = False, eos: bool = False):
        # ClaudeTokenizer's encode method already handles the numeric preprocessing
        ids = self._t.encode(text, add_special_tokens=False)
        if bos and self._t.bos_token_id is not None:
            ids.insert(0, self._t.bos_token_id)
        if eos and self._t.eos_token_id is not None:
            ids.append(self._t.eos_token_id)
        return ids

    def decode(self, ids: Sequence[int], skip_special: bool = False):
        """By default **keep** special tokens."""
        return self._t.decode(ids, skip_special_tokens=skip_special)

    # ---------- extras ----------------------------------------------------
    def _encode_core(self, text: str, *, add_bos: bool):
        ids = self._t.encode(text, add_special_tokens=False)
        if add_bos and self._t.bos_token_id is not None:
            ids.insert(0, self._t.bos_token_id)
        return ids

    def encode_to_uint16(self, text: str, *, add_bos: bool = True):
        ids = self._encode_core(text, add_bos=add_bos)
        assert len(self._t) < 65_536, "vocab too large for uint16"
        return np.asarray(ids, dtype=np.uint16)

    def encode_to_uint32(self, text: str, *, add_bos: bool = True):
        ids = self._encode_core(text, add_bos=add_bos)
        return np.asarray(ids, dtype=np.uint32)

    def __len__(self):
        return len(self._t)


###############################################################################
# Factory – pick the right adapter for the caller
###############################################################################
def _load_special_tokens(special_tokens_config) -> Optional[List[str]]:
    """Load special tokens from config (path to JSON file, list, or dict).

    Supports multiple formats:
    - List of token strings: ["<|bos|>", "<|user_start|>", ...]
    - Dict mapping token to ID: {"<|bos|>": 100277, ...} (keys are extracted)
    - Path to JSON file containing either format above, or:
      - {"special_tokens": [...]} or {"special_tokens": {...}}
      - {"tokens": [...]}
    """
    import json

    if special_tokens_config is None:
        return None

    # If it's already a list, return it
    if isinstance(special_tokens_config, list):
        return special_tokens_config

    # If it's a dict (token -> ID mapping), extract just the keys
    if isinstance(special_tokens_config, dict):
        return list(special_tokens_config.keys())

    # If it's a string, treat as path to JSON file
    if isinstance(special_tokens_config, (str, Path)):
        path = Path(special_tokens_config).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Special tokens file not found: {path}")

        with open(path) as f:
            config = json.load(f)

        # Support various formats
        if isinstance(config, list):
            return config
        elif isinstance(config, dict):
            if "special_tokens" in config:
                st = config["special_tokens"]
                # special_tokens can be a list or a dict
                return list(st.keys()) if isinstance(st, dict) else st
            elif "tokens" in config:
                return config["tokens"]
            else:
                raise ValueError(
                    f"Special tokens JSON must contain 'special_tokens' or 'tokens' key, "
                    f"or be a list. Got keys: {list(config.keys())}"
                )

    return None


def get_tokenizer(
    kind: str = "llama",
    path: Optional[str | Path] = None,
    special_tokens: Optional[str | Path | List[str] | dict] = None,
) -> BaseTokenizer:
    """Return a *BaseTokenizer* based on *kind* or explicit *path*.

    Parameters
    ----------
    kind
        • "llama"               → legacy Tokenizer (unless *path* overrides)
        • "tiktoken"/"cl100k"   → OpenAI *tiktoken* encodings
        • "claude"              → Claude numeric tokenizer (requires *path*)
        • "hf" / other string   → treat as 🤗 model-id or local directory
    path
        Optional filesystem path or model-id.  Ignored for "llama" unless
        you want to override the default tiktoken encoding name.
        Required for "claude" to specify the tokenizer location.
    special_tokens
        Optional special tokens to add to the tokenizer. Can be:
        - A list of token strings, e.g. ["<|bos|>", "<|user_start|>", ...]
        - A dict mapping token strings to IDs (keys are used)
        - A path to a JSON file containing {"special_tokens": [...]} or a list
        Currently only supported for tiktoken and llama adapters.
    """
    kind_lc = kind.lower()

    tokens_list = _load_special_tokens(special_tokens)

    # ----- legacy llama ---------------------------------------------------
    tokenizer: BaseTokenizer
    if kind_lc == "llama":
        if path is None:
            raise ValueError("llama requires a model path. Usually ../tokenizers/llama_tokenizer")
        tokenizer = LlamaTokenizerAdapter(path, special_tokens=tokens_list)

    # ----- tiktoken encodings --------------------------------------------
    elif kind_lc in {"tiktoken", "cl100k", "o200k", "p50k", "r50k"}:
        encoding_name = (
            path
            or (f"{kind_lc}_base" if kind_lc != "tiktoken" else "cl100k_base")
        )
        tokenizer = TikTokenAdapter(name=str(encoding_name), special_tokens=tokens_list)

    # ----- claude numeric tokenizer --------------------------------------
    elif kind_lc == "claude":
        if path is None:
            raise ValueError(
                "ClaudeTokenizer requires a model path. "
                "Usage: get_tokenizer('claude', path='path/to/claude-tokenizer')"
            )
        if tokens_list:
            print(f"[warning] special_tokens not yet supported for claude tokenizer, ignoring")
        tokenizer = ClaudeTokenizerAdapter(path)

    # ----- hf / llama_hf / anything else ---------------------------------
    else:
        model_path = path or kind  # bare directory or model-id
        if tokens_list:
            print(f"[warning] special_tokens not yet supported for HF tokenizer, ignoring")
        tokenizer = HFTokenizerAdapter(model_path)

    return tokenizer