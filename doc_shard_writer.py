"""DocShardWriter — streaming, document-aligned .npy shard writer.

The single home for the lab's shard "sensibilities", so trees are born correct
instead of needing a rechunk pass (see V:\\code\\tools\\docs\\
PRETOKENIZE_ALIGNED_WRITER_PLAN.md and mara_fsdp2/tools/rechunk_doc_aligned.py,
whose output format and manifest schema this matches):

  - every shard starts with BOS and contains only WHOLE documents (a doc that
    doesn't fit the remaining buffer flushes the buffer first; a doc larger
    than the cap becomes its own oversized shard, warned)
  - val holdout at generation: the first ~N tokens of documents (in arrival
    order — with parallel tokenization that is completion order, i.e. the
    approximate corpus head) are routed to the val split
  - shard-count coprimality (train split): at close(), while
    gcd(S, coprime) != 1, the largest train shard is split at the document
    boundary nearest its midpoint; the new half is appended at the next free
    index (generated trees have approximate doc order anyway)
  - undersized tails are merged into the previous shard (the dataloader skips
    shards below B*T+1 tokens; min_shard keeps a wide margin)
  - manifest_{split}.json with per-shard blake2b16 — verifiable with
    `rechunk_doc_aligned.py --verify-only [--deep]`, one auditor for both
    generated and rebuilt trees
  - crash-safe: .tmp-uuid + atomic rename (a crash loses only the in-memory
    buffer, always whole documents); resume continues numbering per split and
    counts existing val tokens toward the holdout

BOS comes from the caller's tokenizer object (`tokenizer.bos_id`) — never a
hand-typed literal (bos-audit rule).
"""

import hashlib
import json
import math
import os
import uuid

import numpy as np


def _npy_len(path):
    """Token count of a 1-D .npy without retaining a file descriptor."""
    with open(path, "rb") as f:
        version = np.lib.format.read_magic(f)
        if version == (1, 0):
            shape, _, _ = np.lib.format.read_array_header_1_0(f)
        elif version == (2, 0):
            shape, _, _ = np.lib.format.read_array_header_2_0(f)
        else:
            a = np.load(path, mmap_mode="r")
            shape = a.shape
            del a
    return int(shape[0])


class DocShardWriter:
    def __init__(self, outdir, label, bos_id, cap=100_000_000, min_shard=5_000_000,
                 coprime=6, val_holdout=0, dtype="uint16", log=print):
        if bos_id is None or int(bos_id) < 0:
            raise ValueError(f"bos_id={bos_id!r}: writer needs the tokenizer's real BOS "
                             f"(tokenizer.bos_id) — doc alignment is meaningless without one")
        self.outdir, self.label = outdir, label
        self.bos = int(bos_id)
        self.cap, self.min_shard, self.coprime = int(cap), int(min_shard), int(coprime)
        self.val_holdout = int(val_holdout or 0)
        self.log = log
        os.makedirs(outdir, exist_ok=True)
        self._cleanup_tmp()

        np_dtype = {"uint16": np.uint16, "uint32": np.uint32}.get(dtype, dtype)
        self.np_dtype = np.dtype(np_dtype)
        self.buf = np.empty((self.cap,), dtype=self.np_dtype)
        self.used = 0
        self.buf_docs = 0
        self.entries = {"train": [], "val": []}   # this run's shards, per split
        self.oversized = 0

        # resume: continue numbering per split; count existing val tokens
        self.index = {sp: self._next_index(sp) for sp in ("train", "val")}
        pre_val = sum(_npy_len(os.path.join(outdir, f))
                      for f in self._existing(("val",)))
        self._val_written = pre_val
        self.split = ("val" if self.val_holdout and self._val_written < self.val_holdout
                      else "train")
        if pre_val:
            self.log(f"[writer] resume: {pre_val:,} val tokens already on disk")
        if 0 < self.val_holdout < self.min_shard:
            self.log(f"[writer] WARNING: val holdout {self.val_holdout:,} < min_shard "
                     f"{self.min_shard:,} — the dataloader may skip a shard this small")

    # ── public API ───────────────────────────────────────────────────────
    def add_doc(self, arr):
        """Add ONE whole tokenized document (must start with BOS)."""
        n = len(arr)
        if n == 0:
            return
        if int(arr[0]) != self.bos:
            raise ValueError(
                f"document does not start with BOS id {self.bos} (got {int(arr[0])}) — "
                f"tokenize with add_bos=True, or the bos_id passed to the writer is wrong")
        if n > self.cap:
            self._flush()
            self._write_shard(np.ascontiguousarray(arr, dtype=self.np_dtype),
                              int((np.asarray(arr) == self.bos).sum()))
            self.oversized += 1
            self.log(f"[writer] WARNING: {n:,}-token document exceeds cap — "
                     f"written as its own oversized shard")
        else:
            if self.used + n > self.cap:
                self._flush()
            self.buf[self.used:self.used + n] = arr
            self.used += n
            self.buf_docs += 1
        if self.split == "val" and self._val_written + self.used >= self.val_holdout:
            self._flush()
            self.split = "train"

    add = add_doc   # drop-in for callers of the legacy ShardWriter.add

    def close(self):
        """Flush, absorb pre-existing shards (file-level resume), merge
        undersized tail, coprime-finalize on the TOTAL count, write manifests."""
        self._flush()
        self._absorb_preexisting()   # must precede coprime: resume runs would
        for sp in ("train", "val"):  # otherwise enforce it on this run's count only
            self.entries[sp].sort(key=lambda e: e["file"])
        self._merge_small_tail("train")
        self._merge_small_tail("val")
        self._adjust_coprime()
        summary = {}
        for sp in ("train", "val"):
            if not self.entries[sp]:
                continue
            ents = sorted(self.entries[sp], key=lambda e: e["file"])
            man = {
                "group": self.label, "split": sp, "bos": self.bos, "cap": self.cap,
                "coprime_base": self.coprime,
                "shard_count": len(ents),
                "coprime_ok": (math.gcd(len(ents), self.coprime) == 1
                               if sp == "train" else True),
                "val_holdout_tokens": self.val_holdout,
                "generator": "DocShardWriter",
                "doc_order": "arrival (completion order under parallel tokenization)",
                "tokens": sum(e["tokens"] for e in ents),
                "docs": sum(e["docs"] for e in ents),
                "dtype": str(self.np_dtype),
                "shards": ents,
            }
            mp = os.path.join(self.outdir, f"manifest_{sp}.json")
            with open(mp + ".tmp", "w") as f:
                json.dump(man, f, indent=1)
            os.replace(mp + ".tmp", mp)
            summary[sp] = {k: man[k] for k in ("shard_count", "tokens", "docs", "coprime_ok")}
            self.log(f"[writer] {self.label}/{sp}: {man['shard_count']} shards, "
                     f"{man['tokens']:,} tokens, {man['docs']:,} docs, "
                     f"coprime_ok={man['coprime_ok']}")
        return summary

    # ── shard IO ─────────────────────────────────────────────────────────
    def _flush(self):
        if self.used:
            self._write_shard(self.buf[:self.used].copy(), self.buf_docs)
            self.used = 0
            self.buf_docs = 0

    def _write_shard(self, arr, docs):
        sp = self.split
        name = f"{self.label}_{sp}_{self.index[sp]:06d}.npy"
        self._write_file(name, arr)
        self.entries[sp].append({
            "file": name, "tokens": int(len(arr)), "docs": int(docs),
            "blake2b16": hashlib.blake2b(arr.tobytes(), digest_size=16).hexdigest(),
        })
        self.index[sp] += 1
        if sp == "val":
            self._val_written += len(arr)
        self.log(f"[write] {name}  ({len(arr):,} tokens, {docs:,} docs)")

    def _write_file(self, name, arr):
        final = os.path.join(self.outdir, name)
        tmp = final + f".tmp-{uuid.uuid4().hex}"
        with open(tmp, "wb") as f:
            np.save(f, arr)
        os.replace(tmp, final)

    # ── close()-time fixups ─────────────────────────────────────────────
    def _merge_small_tail(self, sp):
        ents = self.entries[sp]
        if len(ents) >= 2 and ents[-1]["tokens"] < self.min_shard:
            tail, prev = ents.pop(), ents[-1]
            a = np.concatenate([np.load(os.path.join(self.outdir, prev["file"])),
                                np.load(os.path.join(self.outdir, tail["file"]))])
            self._write_file(prev["file"], a)
            os.remove(os.path.join(self.outdir, tail["file"]))
            self.index[sp] -= 1
            prev.update(tokens=int(len(a)), docs=prev["docs"] + tail["docs"],
                        blake2b16=hashlib.blake2b(a.tobytes(), digest_size=16).hexdigest())
            self.log(f"[writer] merged {tail['tokens']:,}-token tail into {prev['file']}")

    def _adjust_coprime(self):
        ents = self.entries["train"]
        guard = 0
        while ents and math.gcd(len(ents), self.coprime) != 1 and guard < 8:
            guard += 1
            for ent in sorted(ents, key=lambda e: -e["tokens"]):
                a = np.load(os.path.join(self.outdir, ent["file"]))
                bpos = np.flatnonzero(a == self.bos)
                interior = bpos[bpos > 0]
                if interior.size == 0:
                    continue
                cut = int(interior[np.argmin(np.abs(interior - len(a) // 2))])
                head, tail = a[:cut], a[cut:]
                new_name = f"{self.label}_train_{self.index['train']:06d}.npy"
                self._write_file(new_name, np.ascontiguousarray(tail))
                self._write_file(ent["file"], np.ascontiguousarray(head))
                head_docs = int((head == self.bos).sum())
                ents.append({
                    "file": new_name, "tokens": int(len(tail)),
                    "docs": ent["docs"] - head_docs,
                    "blake2b16": hashlib.blake2b(
                        np.ascontiguousarray(tail).tobytes(), digest_size=16).hexdigest(),
                })
                ent.update(tokens=int(len(head)), docs=head_docs,
                           blake2b16=hashlib.blake2b(
                               np.ascontiguousarray(head).tobytes(), digest_size=16).hexdigest())
                self.index["train"] += 1
                self.log(f"[writer] orbit fix: split {ent['file']} at a doc boundary "
                         f"-> S={len(ents)}")
                break
            else:
                self.log("[writer] WARNING: no splittable shard; count left non-coprime")
                return

    def _absorb_preexisting(self):
        """Manifest entries for shards from PREVIOUS runs (file-level resume):
        load each, count docs, checksum — so the manifest covers the whole dir."""
        for sp in ("train", "val"):
            known = {e["file"] for e in self.entries[sp]}
            for name in self._existing((sp,)):
                if name in known:
                    continue
                a = np.load(os.path.join(self.outdir, name))
                if len(a) and int(a[0]) != self.bos:
                    self.log(f"[writer] WARNING: pre-existing {name} does not start "
                             f"with BOS (legacy river shard?) — recorded as-is")
                self.entries[sp].append({
                    "file": name, "tokens": int(len(a)),
                    "docs": int((a == self.bos).sum()),
                    "blake2b16": hashlib.blake2b(a.tobytes(), digest_size=16).hexdigest(),
                })

    # ── filesystem helpers ───────────────────────────────────────────────
    def _existing(self, splits):
        out = []
        for f in sorted(os.listdir(self.outdir)):
            for sp in splits:
                if f.startswith(f"{self.label}_{sp}_") and f.endswith(".npy"):
                    out.append(f)
        return out

    def _next_index(self, sp):
        mx = -1
        for f in self._existing((sp,)):
            try:
                mx = max(mx, int(f[:-4].rsplit("_", 1)[1]))
            except ValueError:
                pass
        return mx + 1

    def _cleanup_tmp(self):
        for f in os.listdir(self.outdir):
            if ".tmp-" in f:
                try:
                    os.remove(os.path.join(self.outdir, f))
                    self.log(f"[cleanup] removed stale temp file {f}")
                except OSError:
                    pass
