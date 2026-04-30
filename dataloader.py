# dataloader.py
# DataLoader with one-shard-per-group caching and robust dataset management

import os
import sys
import glob
import json
import time
import hashlib
import datetime
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass, field
import numpy as np
import torch
import torch.distributed as dist

common_path = '../common_fsdp2'
if common_path not in sys.path:
    sys.path.insert(0, common_path)

import logger

def rank_prefix(rank):
    return f"[R{rank}]   ·"


# =============================================================================
# Data Mix Schedule - for data annealing with linear interpolation
# =============================================================================

class DataMixSchedule:
    """
    Linear interpolation between data mix schedule points for data annealing.

    Allows smoothly transitioning from one data mix to another during training.
    For example, start with 70% stories, gradually shift to more instruction data.

    Config format:
        data_schedule:
          - [0,     {stories: 70, ao3: 30}]
          - [10000, {stories: 55, ao3: 25, mid_smoltalk: 10, mid_mmlu: 10}]
          - [20000, {stories: 40, ao3: 15, mid_smoltalk: 20, mid_mmlu: 25}]
    """

    def __init__(self, schedule: List[Tuple[int, Dict[str, float]]]):
        """
        Args:
            schedule: List of [step, {group_name: percentage, ...}] pairs.
                      Percentages don't need to sum to 100 - they'll be normalized.
        """
        if not schedule:
            raise ValueError("Data schedule cannot be empty")

        self.points = sorted(schedule, key=lambda x: x[0])

        # Collect all group names mentioned anywhere in schedule
        self.all_groups: set = set()
        for _, mix in self.points:
            self.all_groups.update(mix.keys())

        # Cache for avoiding redundant calculations
        self._last_step = -1
        self._last_mix = None

    def get_mix_at_step(self, step: int) -> Dict[str, float]:
        """
        Get interpolated percentages at a given step.

        Returns normalized percentages that sum to 100%.
        Groups not present at a step have 0%.
        """
        # Cache hit - avoid recalculation
        if step == self._last_step and self._last_mix is not None:
            return self._last_mix

        result = self._interpolate(step)
        self._last_step = step
        self._last_mix = result
        return result

    def _interpolate(self, step: int) -> Dict[str, float]:
        """Perform the actual interpolation."""
        # Before first point: use first point
        if step <= self.points[0][0]:
            return self._normalize(self.points[0][1])

        # After last point: use last point
        if step >= self.points[-1][0]:
            return self._normalize(self.points[-1][1])

        # Find surrounding points and interpolate
        for i in range(len(self.points) - 1):
            step_a, mix_a = self.points[i]
            step_b, mix_b = self.points[i + 1]

            if step_a <= step < step_b:
                # Linear interpolation factor
                t = (step - step_a) / (step_b - step_a)

                interpolated = {}
                for group in self.all_groups:
                    pct_a = mix_a.get(group, 0.0)
                    pct_b = mix_b.get(group, 0.0)
                    interpolated[group] = pct_a + t * (pct_b - pct_a)

                return self._normalize(interpolated)

        # Fallback (shouldn't reach here)
        return self._normalize(self.points[-1][1])

    def _normalize(self, mix: Dict[str, float]) -> Dict[str, float]:
        """Normalize percentages to sum to exactly 100%."""
        total = sum(mix.values())
        if total <= 0:
            # All zeros - distribute equally (edge case)
            n = len(self.all_groups)
            return {k: 100.0 / n for k in self.all_groups}

        # Include all groups, defaulting to 0 if not in mix
        result = {}
        for group in self.all_groups:
            result[group] = mix.get(group, 0.0) * 100.0 / total
        return result

    def get_initial_groups(self, step: int = 0) -> List[Tuple[str, float]]:
        """
        Get all groups with their percentages at `step`, suitable for dataloader init.

        Groups that appear anywhere in the schedule are included, with their
        interpolated percentage at the given step (which may be 0% if they
        haven't ramped in yet, or 0% if they've already ramped out).

        On a fresh run, callers should pass step=0 (the default). On resume,
        they should pass the resume step so that active_groups reflects the
        real state at resume time — this is what lets set_state() correctly
        restore shard positions for groups that are in-ramp mid-training.
        """
        initial_mix = self.get_mix_at_step(step)
        return [(name, pct) for name, pct in sorted(initial_mix.items())]

    def get_schedule_summary(self) -> str:
        """Return a human-readable summary of the schedule."""
        lines = ["Data Mix Schedule:"]
        for step, mix in self.points:
            mix_str = ", ".join(f"{k}: {v:.1f}%" for k, v in sorted(mix.items()) if v > 0)
            lines.append(f"  Step {step:>6d}: {mix_str}")
        return "\n".join(lines)

    def get_current_phase(self, step: int) -> Tuple[int, int, float]:
        """
        Get info about current schedule phase.

        Returns:
            (phase_start_step, phase_end_step, progress_within_phase)
        """
        if step <= self.points[0][0]:
            return (0, self.points[0][0], 0.0)

        if step >= self.points[-1][0]:
            return (self.points[-1][0], self.points[-1][0], 1.0)

        for i in range(len(self.points) - 1):
            step_a, _ = self.points[i]
            step_b, _ = self.points[i + 1]

            if step_a <= step < step_b:
                progress = (step - step_a) / (step_b - step_a)
                return (step_a, step_b, progress)

        return (self.points[-1][0], self.points[-1][0], 1.0)

    @classmethod
    def from_groups(cls, groups: List[Tuple[str, Any]]) -> Optional['DataMixSchedule']:
        """Build a DataMixSchedule from extended groups format.

        Each group is (name, weight) where weight is either:
          - A static number:  20.0
          - A schedule:       [[0, 5.0], [5000, 0.0]]

        Returns None if all groups are static (no schedule needed).
        """
        has_schedule = any(isinstance(w, list) for _, w in groups)
        if not has_schedule:
            return None

        # Collect all unique steps from scheduled groups (always include 0)
        all_steps = {0}
        group_schedules = {}  # name -> sorted list of (step, value)
        for name, weight in groups:
            if isinstance(weight, list):
                waypoints = sorted([(int(s), float(v)) for s, v in weight])
                group_schedules[name] = waypoints
                all_steps.update(s for s, _ in waypoints)
            else:
                group_schedules[name] = [(0, float(weight))]

        all_steps = sorted(all_steps)

        # Build schedule points by interpolating each group at each step
        schedule = []
        for step in all_steps:
            mix = {}
            for name, waypoints in group_schedules.items():
                mix[name] = cls._interp_waypoints(waypoints, step)
            schedule.append((step, mix))

        return cls(schedule)

    @staticmethod
    def _interp_waypoints(waypoints: List[Tuple[int, float]], step: int) -> float:
        """Linearly interpolate a single group's waypoints at a given step."""
        if len(waypoints) == 1:
            return waypoints[0][1]
        if step <= waypoints[0][0]:
            return waypoints[0][1]
        if step >= waypoints[-1][0]:
            return waypoints[-1][1]
        for i in range(len(waypoints) - 1):
            s0, v0 = waypoints[i]
            s1, v1 = waypoints[i + 1]
            if s0 <= step <= s1:
                t = (step - s0) / (s1 - s0)
                return v0 + t * (v1 - v0)
        return waypoints[-1][1]

def broadcast_object(obj: Any, src: int = 0) -> Any:
    """Broadcast any picklable python object to every rank."""
    rank = dist.get_rank()
    
    # Debug: Check size of object being broadcast (only if logger is available)
    if rank == src:
        try:
            import pickle
            pickled = pickle.dumps(obj)
            size_mb = len(pickled) / (1024 * 1024)
        except:
            pass
        objects = [obj]
    else:
        objects = [None]
    
    try:
        dev = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}") \
            if dist.get_backend() == "nccl" else torch.device("cpu")
        dist.broadcast_object_list(objects, src=src, device=dev)
    except Exception as e:
        logger.print_and_log(f"{rank_prefix(rank)} Broadcast failed with error: {e}", r0_only=False)
        raise
    
    return objects[0]


@dataclass
class ShardState:
    """Tracks the state of a single loaded shard."""
    path: str
    group_idx: int
    tokens: torch.Tensor
    position: int = 0
    
    @property
    def tokens_remaining(self) -> int:
        return len(self.tokens) - self.position
    
    @property
    def is_exhausted(self) -> bool:
        return self.position >= len(self.tokens)


@dataclass
class DatasetGroup:
    """Represents a single dataset group with its shards and metadata."""
    name: str
    percentage: float  # Target percentage (0-100, 0 means deprecated)
    shards: List[str]
    num_tokens: int
    # Historical tracking (never resets except on 'hard' reset)
    historical_tokens_served: int = 0
    # Window tracking (resets on 'soft' reset for percentage changes)
    window_tokens_served: int = 0
    # Shard management
    rank_positions: List[int] = field(default_factory=list)  # Per-rank positions in shard list
    loaded_shard: Optional[ShardState] = None  # Currently loaded shard for this group
    
    @property
    def is_active(self) -> bool:
        """Check if this group is active (non-zero percentage)."""
        return self.percentage > 0.0
    
    @property
    def is_deprecated(self) -> bool:
        """Check if this group is deprecated (zero percentage but has history)."""
        return self.percentage == 0.0 and self.historical_tokens_served > 0


class PercentageDataLoader:
    """
    DataLoader with one-shard-per-group caching and robust dataset management.
    
    Key features:
    - Maintains exactly one shard per active group in memory
    - Supports changing percentages mid-training without catch-up behavior
    - Handles 0% groups (deprecated datasets) gracefully
    - Supports adding new datasets mid-training
    - Never re-serves data that has already been seen
    """
    
    def __init__(
        self,
        B: int,
        T: int,
        rank: int,
        world_size: int,
        split: str,
        data_root: str,
        groups: List[Tuple[str, float]],  # [(group_name, percentage), ...]
        validation: bool = False,
        cache_file: Optional[str] = "token_counts.json",
        skip_shard_init: bool = False,
        data_schedule: Optional[DataMixSchedule] = None,
        resume_step: Optional[int] = None,
    ):
        """Initialize the DataLoader with one-shard-per-group caching.

        Args:
            skip_shard_init: If True, skip initial shard loading. Use this when
                resuming training - shards will be loaded later via set_state().
                This avoids loading shards twice during resume.
            resume_step: If provided, the loader describes itself as initialized
                at this step (rather than step 0). Callers resuming from a
                checkpoint should pass the resume step so that the `groups`
                parameter — which is expected to carry the mix *at that step* —
                is correctly reflected in logs and in `active_groups`. This is
                what lets `set_state` restore shard positions for groups that
                are mid-ramp at resume time. Leave as None for fresh runs.
        """
        assert split in {'train', 'val'}

        loader_label = "val" if validation else "train"
        logger.print_and_log(f"{rank_prefix(rank)} Starting DataLoader [{loader_label}] @ {data_root}", r0_only=True)

        self.B = B
        self.T = T
        self.rank = rank
        self.world_size = world_size
        self.split = split
        self.validation = validation
        self.data_root = data_root
        self.window_start_tokens = 0  # Initialize this early
        self.data_schedule = data_schedule  # Optional schedule for data annealing
        # Step the loader is being initialized at. On a fresh run this is 0;
        # on a resume it's the resume step. Used for step-accurate logging of
        # the schedule state and for describing ramps relative to "now".
        self.init_step = resume_step if resume_step is not None else 0
        self._is_resuming = resume_step is not None

        self._fixed_val_group = None  # when set, validation stays on this group
        
        # Validate percentages (only active groups must sum to 100%)
        active_groups = [(name, pct) for name, pct in groups if pct > 0]
        if active_groups:
            total_pct = sum(pct for _, pct in active_groups)
            if abs(total_pct - 100.0) > 0.01:
                raise ValueError(
                    f"Active group percentages must sum to 100%, got {total_pct:.2f}% "
                    f"(excluding {len(groups) - len(active_groups)} groups at 0%)"
                )
        
        # Initialize groups (broadcast from rank 0 for consistency)
        if rank == 0:
            logger.print_and_log(f"{rank_prefix(self.rank)} Discovering groups...")
            discovered_groups = self._discover_groups(groups, cache_file)
            # Convert to simpler format for broadcasting
            groups_data = []
            for g in discovered_groups:
                groups_data.append({
                    'name': g.name,
                    'percentage': g.percentage,
                    'shards': g.shards,
                    'num_tokens': g.num_tokens,
                    'historical_tokens_served': g.historical_tokens_served,
                    'window_tokens_served': g.window_tokens_served,
                })
            logger.print_and_log(f"{rank_prefix(self.rank)} Broadcasting {len(groups_data)} groups")
        else:
            groups_data = None
        
        # Broadcast the simplified data
        groups_data = broadcast_object(groups_data, src=0)

        # logger.print_and_log(f"{rank_prefix(self.rank)} Groups broadcast complete, received {len(groups_data)} groups", r0_only=False)

        # Reconstruct DatasetGroup objects from broadcast data
        self.groups = []
        for gd in groups_data:
            group = DatasetGroup(
                name=gd['name'],
                percentage=gd['percentage'],
                shards=gd['shards'],
                num_tokens=gd['num_tokens'],
                historical_tokens_served=gd['historical_tokens_served'],
                window_tokens_served=gd['window_tokens_served'],
                rank_positions=[0] * world_size,
                loaded_shard=None
            )
            self.groups.append(group)

        # logger.print_and_log(f"{rank_prefix(self.rank)} Reconstructed {len(self.groups)} DatasetGroup objects", r0_only=False)

        # Create name -> group mapping
        self.group_map = {g.name: g for g in self.groups}
        
        # Separate active and deprecated groups for efficient access
        self.active_groups = [g for g in self.groups if g.is_active]
        self.deprecated_groups = [g for g in self.groups if not g.is_active]
        
        # Calculate total tokens
        self.total_tokens = sum(g.num_tokens for g in self.groups)
        self.active_tokens = sum(g.num_tokens for g in self.groups if g.is_active)  # Active only
        
        # Log configuration
        self._log_configuration()

        # Initialize state
        self.current_group_idx = 0

        # Load initial shards (only for active groups) unless skipped for resume
        if skip_shard_init:
            logger.print_and_log(f"{rank_prefix(self.rank)} Skipping shard init (will load via set_state)", r0_only=True)
        else:
            self._initialize_shards()

        # logger.print_and_log(f"{rank_prefix(self.rank)} Shard initialization complete", r0_only=False)

    def group_names(self, *, active_only: bool = False) -> List[str]:
        """
        Return group names in a deterministic order.

        Args:
            active_only: If True, include only groups with percentage > 0.
        """
        
        if active_only:
            return [g.name for g in self.active_groups]
        return [g.name for g in self.groups]

    def _manifest_and_fingerprint(self, shards):
        """
        Build a compact manifest from file stats and a deterministic SHA256 fingerprint.
        Uses relpath (stable), size, and integer mtime — O(#files) and very fast.
        """
        h = hashlib.sha256()
        total_bytes = 0
        for p in sorted(shards):
            st = os.stat(p)
            rel = os.path.relpath(p, self.data_root)
            size = st.st_size
            mtime = int(st.st_mtime)
            total_bytes += size
            # feed stable, newline-delimited records into the hash
            h.update(rel.encode('utf-8')); h.update(b'|')
            h.update(str(size).encode('ascii')); h.update(b'|')
            h.update(str(mtime).encode('ascii')); h.update(b'\n')
        fingerprint = h.hexdigest()
        manifest = {
            "num_files": len(shards),
            "total_bytes": total_bytes,
            "fingerprint": fingerprint,
        }
        return manifest, fingerprint

    def _log_configuration(self):
        """Log the current configuration with clear active/scheduled/deprecated status.

        On a fresh run (init_step == 0) this describes step-0 state. On a
        resume (init_step == resume_step) it describes the state *at the
        resume step*, so the labels match what will actually be served from
        the next batch onwards.
        """
        rp = rank_prefix(self.rank)
        has_schedule = self.data_schedule is not None
        step = self.init_step

        header_suffix = f" (resuming at step {step:,d})" if self._is_resuming else ""
        logger.print_and_log(f"{rp} DataLoader initialized with {len(self.groups)} groups{header_suffix}:")

        # Log schedule summary if present (using dataloader's prefix style)
        if has_schedule:
            logger.print_and_log(f"{rp} Data annealing schedule ({len(self.data_schedule.points)} phases):")
            for sp_step, mix in self.data_schedule.points:
                marker = "  <-- now" if self._is_resuming and sp_step == step else ""
                mix_str = ", ".join(f"{k}: {v:.0f}%" for k, v in sorted(mix.items()) if v > 0)
                logger.print_and_log(f"{rp}   Step {sp_step:>6,d}: {mix_str}{marker}")

        # Three-bucket classification:
        #   1. Active now (pct > 0 at init_step)
        #   2. Scheduled but currently 0% (appears in the schedule, not active this step)
        #   3. Truly deprecated (not in schedule at all but has shards / history)
        scheduled_names = set(self.data_schedule.all_groups) if has_schedule else set()

        active = self.active_groups
        scheduled_inactive = [g for g in self.deprecated_groups if g.name in scheduled_names]
        truly_deprecated = [g for g in self.deprecated_groups if g.name not in scheduled_names]

        if active:
            step_suffix = f" at step {step:,d}" if self._is_resuming else ""
            logger.print_and_log(f"{rp} Active groups{step_suffix}:")
            for g in active:
                ramp_info = self._describe_ramp(g.name, step) if has_schedule else ""
                ramp_suffix = f"  [{ramp_info}]" if ramp_info else ""
                logger.print_and_log(
                    f"{rp}           {g.name:20s}: {g.percentage:5.1f}% "
                    f"({g.num_tokens/1e9:.2f}B tokens, {len(g.shards)} shards){ramp_suffix}"
                )

        if scheduled_inactive:
            logger.print_and_log(f"{rp} Scheduled groups (currently 0% at step {step:,d}):")
            for g in scheduled_inactive:
                ramp_info = self._describe_ramp(g.name, step)
                logger.print_and_log(
                    f"{rp}           {g.name:20s}: {g.num_tokens/1e9:.2f}B tokens, "
                    f"{len(g.shards)} shards — {ramp_info}"
                )

        if truly_deprecated:
            logger.print_and_log(f"{rp} Deprecated groups (0% — not in schedule):")
            for g in truly_deprecated:
                history_str = (
                    f"{g.historical_tokens_served/1e9:.2f}B served historically"
                    if g.historical_tokens_served > 0 else "no history"
                )
                logger.print_and_log(
                    f"{rp}           {g.name:20s}: {history_str} "
                    f"({g.num_tokens/1e9:.2f}B total, {len(g.shards)} shards)"
                )

        logger.print_and_log(f"{rp} Total tokens: {self.total_tokens/1e9:.2f}B")
        logger.print_and_log(f"{rp} Active tokens: {self.active_tokens/1e9:.2f}B")
        logger.print_and_log(f"{rp} Memory usage: max {len(self.active_groups)} shards (one per active group)")

    def _describe_ramp(self, group_name: str, current_step: int) -> str:
        """Describe a group's schedule trajectory relative to `current_step`.

        Shows the current interpolated percentage and, if the schedule has any
        upcoming change for this group, the next target and when it lands.
        Also notes recent past transitions so resumes have a complete picture.
        """
        if self.data_schedule is None:
            return ""

        points = self.data_schedule.points  # sorted by step
        current_pct = self.data_schedule.get_mix_at_step(current_step).get(group_name, 0.0)

        # Find the most recent past point where pct was distinct from current
        prev_change = None  # (step, pct)
        for sp_step, mix in points:
            if sp_step > current_step:
                break
            pct = mix.get(group_name, 0.0)
            if abs(pct - current_pct) > 0.01:
                prev_change = (sp_step, pct)

        # Find the next future point where pct becomes distinct from current
        next_change = None  # (step, pct)
        for sp_step, mix in points:
            if sp_step <= current_step:
                continue
            pct = mix.get(group_name, 0.0)
            if abs(pct - current_pct) > 0.01:
                next_change = (sp_step, pct)
                break

        parts = [f"currently {current_pct:.1f}%"]
        if prev_change is not None:
            parts.append(f"was {prev_change[1]:.0f}% @ step {prev_change[0]:,d}")
        if next_change is not None:
            direction = "→" if next_change[1] != current_pct else "="
            parts.append(f"{direction} {next_change[1]:.0f}% by step {next_change[0]:,d}")
        else:
            parts.append("stable")
        return ", ".join(parts)

    def _discover_groups(self, config_groups, cache_file) -> List[DatasetGroup]:
        groups = []
        cache_path = os.path.join(self.data_root, cache_file) if cache_file else None
        cache = self._load_cache(cache_path) if cache_path else {}
        cache_updated = False

        for name, percentage in config_groups:
            group_dir = os.path.join(self.data_root, name)
            pattern = f"*_{self.split}_*.npy"
            shards = sorted(glob.glob(os.path.join(group_dir, pattern)))

            if not shards:
                if percentage > 0:
                    raise ValueError(f"No shards found for active group '{name}' with pattern {pattern}")
                else:
                    logger.print_and_log(f"{rank_prefix(self.rank)} Warning: No shards for deprecated group '{name}' (OK since 0%)")
                    continue

            cache_key = f"{name}_{self.split}"
            manifest, current_fp = self._manifest_and_fingerprint(shards)

            entry = cache.get(cache_key)
            if entry:
                cached_tokens = int(entry.get("token_count", 0))
                cached_fp = entry.get("fingerprint")
                if cached_fp == current_fp and cached_tokens > 0:
                    num_tokens = cached_tokens
                    logger.print_and_log(
                        f"{rank_prefix(self.rank)} Loaded from cache: {name} = {num_tokens/1e9:.2f}B tokens (fp match)"
                    )
                else:
                    logger.print_and_log(
                        f"{rank_prefix(self.rank)} Fingerprint changed for {name} (or no fp). Rescanning {len(shards)} shards…"
                    )
                    num_tokens = self._count_tokens(shards)
                    cache[cache_key] = {
                        "token_count": num_tokens,
                        "fingerprint": current_fp,
                        "manifest": manifest,
                    }
                    cache_updated = True
            else:
                logger.print_and_log(f"{rank_prefix(self.rank)} Scanning {name}: {len(shards)} shards…")
                num_tokens = self._count_tokens(shards)
                cache[cache_key] = {
                    "token_count": num_tokens,
                    "fingerprint": current_fp,
                    "manifest": manifest,
                }
                cache_updated = True

            groups.append(DatasetGroup(
                name=name,
                percentage=percentage,
                shards=shards,
                num_tokens=num_tokens
            ))

        if cache_updated and cache_path and not self.validation:
            self._save_cache(cache_path, cache)

        return groups
    
    def _count_tokens(self, shards: List[str]) -> int:
        """Count total tokens in a list of shards."""
        total = 0
        next_milestone = 5_000_000_000  # Log every 5B tokens
        
        for shard_path in shards:
            arr = np.load(shard_path, mmap_mode='r')
            total += arr.shape[0]
            
            # Log progress at milestones
            if total >= next_milestone:
                logger.print_and_log(f"{rank_prefix(self.rank)}   --> {shard_path} : {total/1e9:6.2f} B tokens")
                next_milestone += 5_000_000_000

        logger.print_and_log(f"{rank_prefix(self.rank)}  Scanned {total/1e6:.1f}M tokens")
        return total
    
    def _load_cache(self, path: str) -> Dict:
        """Load token count cache with optional metadata (backward compatible)."""
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                result = {}
                if isinstance(data, dict):
                    for key, value in data.items():
                        if key in ('version', 'generated_at'):
                            continue
                        # v2: bare int or float
                        if isinstance(value, (int, float)):
                            result[key] = {
                                "token_count": int(value),
                                "fingerprint": None,
                                "manifest": None,
                            }
                        # v3/v4+: dict style
                        elif isinstance(value, dict):
                            if "token_count" in value:
                                result[key] = {
                                    "token_count": int(value["token_count"]),
                                    "fingerprint": value.get("fingerprint"),
                                    "manifest": value.get("manifest"),
                                }
                return result
            except Exception as e:
                logger.print_and_log(f"Warning: Failed to load cache: {e}")
        return {}

    def _save_cache(self, path: str, cache: Dict):
        """Save token count cache with metadata."""
        try:
            # `cache` is expected to be key -> dict(token_count, fingerprint, manifest)
            output = {"version": "4.0", "generated_at": datetime.datetime.now().isoformat()}
            output.update(cache)
            with open(path, 'w') as f:
                json.dump(output, f, indent=2)
            logger.print_and_log(f"Updated cache: {path}")
        except Exception as e:
            logger.print_and_log(f"Warning: Failed to save cache: {e}")


    def _initialize_shards(self):
        """Initialize by loading one shard per active group."""
        # logger.print_and_log(f"{rank_prefix(self.rank)} Entering _initialize_shards, active_groups={len(self.active_groups)}", r0_only=False)

        if not self.active_groups:
            logger.print_and_log(f"{rank_prefix(self.rank)} Warning: No active groups to load shards from!", r0_only=False)
            return
        
        for group in self.active_groups:
            # logger.print_and_log(f"{rank_prefix(self.rank)} Loading shard for group {group.name}", r0_only=False)
            if self.validation and group != self.active_groups[0]:
                # For validation, only load first group initially
                break
            # Load a shard for this group on this rank
            self._load_next_shard_for_group(self.groups.index(group))
        
        # Set initial group based on rank distribution (only among active groups)
        if self.validation:
            self.current_group_idx = self.groups.index(self.active_groups[0])
        else:
            # Distribute ranks across active groups for diverse start
            cumsum = np.cumsum([0] + [g.percentage for g in self.active_groups])
            rank_position = (self.rank / self.world_size) * 100
            active_idx = np.searchsorted(cumsum[1:], rank_position)
            active_idx = min(active_idx, len(self.active_groups) - 1)
            self.current_group_idx = self.groups.index(self.active_groups[active_idx])
    
    def _load_next_shard_for_group(self, group_idx: int) -> Optional[ShardState]:
        """Load the next shard for a specific group, replacing any existing one."""
        group = self.groups[group_idx]
        
        # Don't load shards for inactive groups
        if not group.is_active:
            logger.print_and_log(f"{rank_prefix(self.rank)} Skipping shard load for inactive group {group.name} (0%)", r0_only=False)
            return None
        
        # Get this rank's next shard from this group
        my_position = group.rank_positions[self.rank]
        shard_idx = (self.rank + my_position * self.world_size) % len(group.shards)
        shard_path = group.shards[shard_idx]
        
        # Load the shard
        tokens = self._load_shard_tokens(shard_path)
        
        # Skip if shard is too small
        min_tokens = self.B * self.T + 1
        if len(tokens) < min_tokens:
            logger.print_and_log(f"{rank_prefix(self.rank)} Skipping small shard {os.path.basename(shard_path)} ({len(tokens)} < {min_tokens})",r0_only=False)
            # Increment position and try next shard
            group.rank_positions[self.rank] = (my_position + 1) % len(group.shards)
            return self._load_next_shard_for_group(group_idx)
        
        # Create new shard state (replaces any existing one)
        shard_state = ShardState(
            path=shard_path,
            group_idx=group_idx,
            tokens=tokens,
            position=0
        )
        
        # Replace the group's loaded shard
        old_shard = group.loaded_shard
        group.loaded_shard = shard_state
        
        # Increment this rank's position for next time
        group.rank_positions[self.rank] = (my_position + 1) % len(group.shards)
        
        #if not self.validation:
        #    if old_shard:
        #        logger.print_and_log(f"{rank_prefix(self.rank)} Replaced {group.name}/{os.path.basename(old_shard.path)} (used {old_shard.position}/{len(old_shard.tokens)} tokens)",r0_only=False)
        #    logger.print_and_log(f"{rank_prefix(self.rank)} Loaded {group.name}/{os.path.basename(shard_path)} ({len(tokens):,} tokens)", r0_only=False)

        return shard_state
    
    # Retry budget for shard-load OSError (NAS outage, NFS reconnect, etc.)
    # Total budget is paced by exponential backoff capped at NAS_RETRY_MAX_SLEEP_S.
    # Sized to outlive a ~15 min NAS firmware reboot but stay under typical NCCL
    # timeouts. Tune via environment variable MARA_NAS_RETRY_BUDGET_S if needed.
    NAS_RETRY_BUDGET_S = float(os.environ.get("MARA_NAS_RETRY_BUDGET_S", 1200))  # 20 min
    NAS_RETRY_INITIAL_SLEEP_S = 5.0
    NAS_RETRY_MAX_SLEEP_S = 300.0

    def _load_shard_tokens(self, path: str) -> torch.Tensor:
        """Load tokens from a shard file.

        Retries on transient OSError (NAS outage, NFS reconnect) with exponential
        backoff up to NAS_RETRY_BUDGET_S before giving up. FileNotFoundError is
        treated as non-transient and raised immediately.
        """
        # get the filename from the path
        filename = os.path.basename(path)
        if not self.validation:
            logger.print_and_log(f"{rank_prefix(self.rank)} New data shard: {filename}", r0_only=False)

        start_ts = time.monotonic()
        sleep_s = self.NAS_RETRY_INITIAL_SLEEP_S
        attempt = 0
        while True:
            attempt += 1
            try:
                arr = np.load(path, mmap_mode='r')
                # Force the mmap to be fully materialized here so any deferred
                # I/O failure surfaces inside this try block (astype triggers
                # the read for uint16; uint32 path needs an explicit copy).
                if arr.dtype == np.uint16:
                    arr = arr.astype(np.int32)
                elif arr.dtype == np.uint32:
                    arr = np.ascontiguousarray(arr, dtype=np.int32)
                else:
                    raise ValueError(f"Unsupported dtype {arr.dtype} in {path}")
                break  # success
            except FileNotFoundError:
                raise  # genuine missing file — not transient
            except OSError as e:
                elapsed = time.monotonic() - start_ts
                if elapsed + sleep_s > self.NAS_RETRY_BUDGET_S:
                    logger.print_and_log(
                        f"{rank_prefix(self.rank)} FATAL: shard load failed after "
                        f"{elapsed:.0f}s / {self.NAS_RETRY_BUDGET_S:.0f}s budget "
                        f"({attempt} attempts): {path} ({type(e).__name__}: {e})",
                        r0_only=False,
                    )
                    raise
                logger.print_and_log(
                    f"{rank_prefix(self.rank)} I/O error loading shard "
                    f"(attempt {attempt}, {type(e).__name__}: {e}); "
                    f"retrying in {sleep_s:.0f}s "
                    f"(elapsed {elapsed:.0f}s / {self.NAS_RETRY_BUDGET_S:.0f}s): {path}",
                    r0_only=False,
                )
                time.sleep(sleep_s)
                sleep_s = min(sleep_s * 2, self.NAS_RETRY_MAX_SLEEP_S)

        tensor = torch.from_numpy(arr).long()

        return tensor
    
    def _calculate_deficits(self) -> np.ndarray:
        """Calculate deficit for each active group using window tokens."""
        if not self.active_groups:
            return np.array([])
        
        # Only calculate deficits for active groups
        total_window = max(1, sum(g.window_tokens_served for g in self.active_groups))
        
        deficits = []
        for g in self.active_groups:
            target_ratio = g.percentage / 100.0
            actual_ratio = g.window_tokens_served / total_window
            deficit = target_ratio - actual_ratio
            deficits.append(deficit)
        
        return np.array(deficits)

    def set_val_group(self, group_name: Optional[str], *, eval_iters: Optional[int] = None) -> None:
        """
        Validation helper.
        • group_name=str  -> lock to that group.
        • group_name=None -> unlock (go back to validation cycling).
        When locked, if the group has only 1 shard (or fewer shards than ranks),
        each rank reads a disjoint contiguous block from the shard so validation
        isn’t duplicated across ranks.
        """
        if group_name is None:
            self._fixed_val_group = None
            return

        if group_name not in self.group_map:
            raise KeyError(f"Unknown group '{group_name}'")

        g = self.group_map[group_name]
        idx = self.groups.index(g)
        self._fixed_val_group = idx
        self.current_group_idx = idx

        # Force a fresh shard with cursor=0 on this rank
        g.rank_positions[self.rank] = 0
        self._load_next_shard_for_group(idx)
        shard = g.loaded_shard
        B, T, W = self.B, self.T, self.world_size

        # Partition the shard into per-rank contiguous blocks of full batches
        total_batches = (len(shard.tokens) - 1) // (B * T)
        if eval_iters is not None and eval_iters * W <= total_batches:
            # exact block per rank, sized to your eval_iters
            start_batch = self.rank * eval_iters
        else:
            # even split of the shard into W blocks
            batches_per_rank = max(1, total_batches // W)
            start_batch = self.rank * batches_per_rank

        shard.position = start_batch * (B * T)
    
    def _select_next_group(self) -> int:
        """Select the next active group based on deficit (train) or validation mode."""
        if not self.active_groups:
            raise RuntimeError("No active groups available to select from!")

        # NEW: fixed validation group -> always stay on it
        if self.validation and self._fixed_val_group is not None:
            return self._fixed_val_group

        if self.validation:
            # existing behavior: sequential cycle across active groups
            current_active_idx = next(
                (i for i, g in enumerate(self.active_groups)
                if self.groups.index(g) == self.current_group_idx),
                0
            )
            next_active_idx = (current_active_idx + 1) % len(self.active_groups)
            return self.groups.index(self.active_groups[next_active_idx])

        # training: deficit-based
        deficits = self._calculate_deficits()
        active_idx = int(np.argmax(deficits))
        return self.groups.index(self.active_groups[active_idx])

    
    def update_percentages(self, new_groups: List[Tuple[str, float]]):
        """
        Update group percentages and reset window to avoid catch-up behavior.
        Handles new groups and deprecated groups gracefully.
        
        Args:
            new_groups: List of (group_name, new_percentage) tuples
        """
        # Validate that active percentages sum to 100%
        active_new = [(name, pct) for name, pct in new_groups if pct > 0]
        if active_new:
            total_pct = sum(pct for _, pct in active_new)
            if abs(total_pct - 100.0) > 0.01:
                raise ValueError(
                    f"Active percentages must sum to 100%, got {total_pct:.2f}% "
                    f"(excluding {len(new_groups) - len(active_new)} groups at 0%)"
                )
        
        # Create mapping of new percentages
        new_pct_map = {name: pct for name, pct in new_groups}
        
        # Check for new groups that need to be added
        new_group_names = set(new_pct_map.keys()) - set(self.group_map.keys())
        if new_group_names:
            logger.print_and_log(f"{rank_prefix(self.rank)} Adding new groups: {', '.join(new_group_names)}", r0_only=False)
            # Discover and add new groups
            new_groups_to_add = [(name, new_pct_map[name]) for name in new_group_names]
            if self.rank == 0:
                discovered = self._discover_groups(new_groups_to_add, None)
                # Convert to simpler format for broadcasting
                discovered_data = []
                for g in discovered:
                    discovered_data.append({
                        'name': g.name,
                        'percentage': g.percentage,
                        'shards': g.shards,
                        'num_tokens': g.num_tokens,
                        'historical_tokens_served': 0,
                        'window_tokens_served': 0,
                    })
            else:
                discovered_data = None
            discovered_data = broadcast_object(discovered_data, src=0)
            
            # Reconstruct and add to our groups list
            for gd in discovered_data:
                g = DatasetGroup(
                    name=gd['name'],
                    percentage=gd['percentage'],
                    shards=gd['shards'],
                    num_tokens=gd['num_tokens'],
                    historical_tokens_served=gd['historical_tokens_served'],
                    window_tokens_served=gd['window_tokens_served'],
                    rank_positions=[0] * self.world_size,
                    loaded_shard=None
                )
                self.groups.append(g)
                self.group_map[g.name] = g
                if g.is_active:
                    # Load shard for new active group
                    self._load_next_shard_for_group(len(self.groups) - 1)
        # Update percentages and log changes
        logger.print_and_log(f"{rank_prefix(self.rank)} Updating data mix percentages:", r0_only=False)
        for g in self.groups:
            old_pct = g.percentage
            new_pct = new_pct_map.get(g.name, 0.0)  # Default to 0% if not in new config
            
            if abs(old_pct - new_pct) > 0.01:
                if old_pct > 0 and new_pct == 0:
                    logger.print_and_log(f"{rank_prefix(self.rank)}  {g.name}: {old_pct:.1f}% → DEPRECATED (0%)", r0_only=False)
                    # Unload shard to save memory
                    if g.loaded_shard:
                        logger.print_and_log(f"{rank_prefix(self.rank)}    └─ Unloading shard to save memory", r0_only=False)
                        g.loaded_shard = None
                elif old_pct == 0 and new_pct > 0:
                    logger.print_and_log(f"{rank_prefix(self.rank)}  {g.name}: REACTIVATED → {new_pct:.1f}%", r0_only=False)
                    # Load a shard for reactivated group
                    self._load_next_shard_for_group(self.groups.index(g))
                else:
                    logger.print_and_log(f"{rank_prefix(self.rank)}  {g.name}: {old_pct:.1f}% → {new_pct:.1f}%", r0_only=False)

            g.percentage = new_pct
        
        # Update active/deprecated group lists
        self.active_groups = [g for g in self.groups if g.is_active]
        self.deprecated_groups = [g for g in self.groups if not g.is_active]
        # recalc active tokens
        self.active_tokens = sum(g.num_tokens for g in self.groups if g.is_active)  # Active only
        
        # Ensure we have a valid current group
        if self.groups[self.current_group_idx] not in self.active_groups:
            if self.active_groups:
                self.current_group_idx = self.groups.index(self.active_groups[0])
            else:
                raise RuntimeError("No active groups after percentage update!")
        
        # Perform soft reset to start fresh window
        total_historical = sum(g.historical_tokens_served for g in self.groups)
        logger.print_and_log(f"{rank_prefix(self.rank)} Starting new percentage window at {total_historical/1e9:.2f}B tokens", r0_only=False)
        self._soft_reset()
    
    def _soft_reset(self):
        """Reset window counters for new percentage targeting without affecting positions."""
        # Record where the new window starts
        self.window_start_tokens = sum(g.historical_tokens_served for g in self.groups)

        # Reset window counters for all groups
        for group in self.groups:
            group.window_tokens_served = 0

        logger.print_and_log(f"{rank_prefix(self.rank)} Soft reset: Window counters cleared, continuing from current shard positions", r0_only=False)

    def reset_window(self):
        """Reset window counters without logging. Useful for testing."""
        self.window_start_tokens = sum(g.historical_tokens_served for g in self.groups)
        for group in self.groups:
            group.window_tokens_served = 0

    def set_percentages_silent(self, new_percentages: Dict[str, float]):
        """
        Lightweight percentage update for data annealing - designed to be called every step.

        Unlike update_percentages(), this does NOT:
        - Reset window counters (deficit tracking continues smoothly)
        - Log changes (would spam logs)
        - Discover new groups (all groups must be initialized upfront)

        Handles:
        - Groups transitioning 0% → >0% (loads shard on demand)
        - Groups transitioning >0% → 0% (keeps shard in memory for potential reuse)
        - Updating active_groups list

        Args:
            new_percentages: Dict of {group_name: percentage} - should sum to ~100%
        """
        for group in self.groups:
            old_pct = group.percentage
            new_pct = new_percentages.get(group.name, 0.0)
            group.percentage = new_pct

            # Handle shard loading for newly activated groups
            if old_pct == 0 and new_pct > 0 and not group.loaded_shard:
                self._load_next_shard_for_group(self.groups.index(group))

        # Update active/deprecated lists
        self.active_groups = [g for g in self.groups if g.is_active]
        self.deprecated_groups = [g for g in self.groups if not g.is_active]

        # Recalculate active tokens
        self.active_tokens = sum(g.num_tokens for g in self.groups if g.is_active)

        # Ensure current group is still valid
        if self.active_groups and self.groups[self.current_group_idx] not in self.active_groups:
            self.current_group_idx = self.groups.index(self.active_groups[0])

    def next_batch(self, step: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the next batch of training data.

        Args:
            step: Current training step. If provided and data_schedule is set,
                  the data mix percentages will be updated based on the schedule.
        """
        # Apply data schedule if configured (step must be provided)
        if self.data_schedule is not None and step is not None:
            current_mix = self.data_schedule.get_mix_at_step(step)
            self.set_percentages_silent(current_mix)

        if not self.active_groups:
            raise RuntimeError("No active groups to serve batches from!")
        
        B, T = self.B, self.T
        
        # Ensure current group is active
        current_group = self.groups[self.current_group_idx]
        if not current_group.is_active:
            # This shouldn't happen, but handle gracefully
            logger.print_and_log(f"{rank_prefix(self.rank)} Warning: Current group {current_group.name} is inactive, switching to active group", r0_only=False)
            self.current_group_idx = self._select_next_group()
            current_group = self.groups[self.current_group_idx]
        
        # Check if current group's shard needs refresh
        if not current_group.loaded_shard or current_group.loaded_shard.tokens_remaining < B * T + 1:
            if current_group.loaded_shard and current_group.loaded_shard.tokens_remaining > 0:
                # Log that we're abandoning some tokens (this should be rare)
                logger.print_and_log(f"{rank_prefix(self.rank)} Abandoning {current_group.loaded_shard.tokens_remaining} tokens from {current_group.name} shard (too few for batch)", r0_only=False)
            self._load_next_shard_for_group(self.current_group_idx)
        
        # Extract batch from current group's shard
        shard = current_group.loaded_shard
        start = shard.position
        end = start + B * T + 1
        tokens = shard.tokens[start:end]
        x = tokens[:-1].view(B, T)
        y = tokens[1:].view(B, T)
        
        # Update statistics (both historical and window)
        batch_tokens = B * T
        shard.position += batch_tokens
        current_group.historical_tokens_served += batch_tokens
        current_group.window_tokens_served += batch_tokens
        
        # Select next group for next batch (based on deficit)
        self.current_group_idx = self._select_next_group()
        
        # Ensure next group has a shard loaded
        next_group = self.groups[self.current_group_idx]
        if not next_group.loaded_shard or next_group.loaded_shard.is_exhausted:
            self._load_next_shard_for_group(self.current_group_idx)
        
        return x, y
    
    def _schedule_fingerprint(self) -> str:
        """Return a stable hash of the current data schedule.

        Two schedules hash to the same fingerprint iff their waypoints and
        per-group percentages are identical (to 6 decimal places). Used to
        detect schedule changes across resume boundaries so we can soft-reset
        `window_tokens_served` and avoid the deficit-stiffness bias where a
        tiny target drift past the resume step causes the selector to starve
        falling-target groups and stack rising-target groups.

        Returns 'none' when there is no schedule (static-mix training).
        """
        if self.data_schedule is None:
            return "none"

        normalized = []
        for step, mix in self.data_schedule.points:
            mix_sorted = sorted(
                (name, round(float(pct), 6)) for name, pct in mix.items()
            )
            normalized.append([int(step), mix_sorted])

        serialized = json.dumps(normalized, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]

    def get_state(self) -> Dict:
        """Get state for checkpointing."""
        shard_states = {}
        for group in self.groups:
            if group.loaded_shard:
                shard_states[group.name] = {
                    'path': group.loaded_shard.path,
                    'position': group.loaded_shard.position
                }

        return {
            'version': '5.1',
            'window_start_tokens': self.window_start_tokens,
            'current_group_idx': self.current_group_idx,
            'schedule_fingerprint': self._schedule_fingerprint(),
            'group_states': {
                g.name: {
                    'percentage': g.percentage,
                    'rank_positions': g.rank_positions,
                    'historical_tokens_served': g.historical_tokens_served,
                    'window_tokens_served': g.window_tokens_served
                }
                for g in self.groups
            },
            'shard_states': shard_states
        }
    
    def set_state(self, state: Dict):
        """Restore state from checkpoint, handling new/removed groups gracefully."""
        # Restore window tracking
        self.window_start_tokens = state.get('window_start_tokens', 0)
        
        # Restore per-group state
        group_states = state.get('group_states', {})
        for group in self.groups:
            if group.name in group_states:
                gs = group_states[group.name]
                # Note: We keep current percentage from config, not from state
                # This allows changing percentages between runs
                group.rank_positions = gs.get('rank_positions', [0] * self.world_size)
                group.historical_tokens_served = gs.get('historical_tokens_served', 0)
                group.window_tokens_served = gs.get('window_tokens_served', 0)
                
                # Log if this was a previously deprecated group that's now active
                saved_pct = gs.get('percentage', 0)
                if saved_pct == 0 and group.percentage > 0:
                    logger.print_and_log(f"{rank_prefix(self.rank)} Group {group.name} reactivated: was 0%, now {group.percentage:.1f}%", r0_only=False)
            else:
                # New group not in checkpoint
                logger.print_and_log(
                    f"{rank_prefix(self.rank)} New group {group.name} ({group.percentage:.1f}%) - starting from beginning", r0_only=False)
                group.rank_positions = [0] * self.world_size
                group.historical_tokens_served = 0
                group.window_tokens_served = 0
        
        # Update active/deprecated lists based on current config
        self.active_groups = [g for g in self.groups if g.is_active]
        self.deprecated_groups = [g for g in self.groups if not g.is_active]

        # IMPORTANT: Recalculate active tokens after updating groups
        self.active_tokens = sum(g.num_tokens for g in self.groups if g.is_active)
        self.total_tokens = sum(g.num_tokens for g in self.groups)
        
        # Try to restore shard states for currently-active groups.
        #
        # Correctness note: this loop relies on `self.active_groups` already
        # reflecting the resume step — which is why `__init__` now takes a
        # `resume_step` parameter and seeds per-group percentages from the
        # schedule at that step. If the loader were constructed with step-0
        # percentages but the checkpoint was taken mid-ramp, any group that
        # is 0% at step 0 but >0% at resume time would silently have its
        # shard state dropped here. We catch that case defensively below by
        # warning about any `shard_states` entries we did not end up restoring.
        shard_states = state.get('shard_states', {})
        restored_shard_names: set = set()
        for group in self.active_groups:
            if group.name in shard_states:
                ss = shard_states[group.name]
                shard_path = ss['path']
                if os.path.exists(shard_path):
                    try:
                        tokens = self._load_shard_tokens(shard_path)
                        group.loaded_shard = ShardState(
                            path=shard_path,
                            group_idx=self.groups.index(group),
                            tokens=tokens,
                            position=ss['position']
                        )
                        logger.print_and_log(f"{rank_prefix(self.rank)} Restored {group.name} shard: {os.path.basename(shard_path)} @ position {ss['position']}",r0_only=False)
                        restored_shard_names.add(group.name)
                    except Exception as e:
                        logger.print_and_log(f"{rank_prefix(self.rank)} Failed to restore {group.name} shard: {e}",r0_only=False)
                        self._load_next_shard_for_group(self.groups.index(group))
                else:
                    logger.print_and_log(
                        f"{rank_prefix(self.rank)} Shard not found for {group.name}, loading fresh",
                        r0_only=False
                    )
                    self._load_next_shard_for_group(self.groups.index(group))
            else:
                # No saved state for this group, load fresh if active
                if group.is_active:
                    self._load_next_shard_for_group(self.groups.index(group))

        # Defensive: warn about any saved shard states we did not restore.
        # The common cause is a construction-time mismatch where the loader's
        # percentages reflect a different step than the checkpoint's.
        for saved_name in shard_states.keys():
            if saved_name in restored_shard_names:
                continue
            g = self.group_map.get(saved_name)
            if g is None:
                logger.print_and_log(
                    f"{rank_prefix(self.rank)} WARNING: checkpoint had shard state for '{saved_name}' "
                    f"but that group is no longer in the current config",
                    r0_only=False,
                )
            else:
                logger.print_and_log(
                    f"{rank_prefix(self.rank)} WARNING: checkpoint had shard state for '{saved_name}' "
                    f"(path={os.path.basename(shard_states[saved_name].get('path',''))}, "
                    f"pos={shard_states[saved_name].get('position',0):,}) "
                    f"but it was not restored — current pct is {g.percentage:.2f}% at init_step {self.init_step:,d}. "
                    f"If this group should be active, verify that resume_step was passed to the DataLoader.",
                    r0_only=False,
                )
        
        # Restore or set current group index
        saved_idx = state.get('current_group_idx', 0)
        if saved_idx < len(self.groups) and self.groups[saved_idx].is_active:
            self.current_group_idx = saved_idx
        elif self.active_groups:
            self.current_group_idx = self.groups.index(self.active_groups[0])
        else:
            raise RuntimeError("No active groups available after state restoration!")

        # Schedule-change detection: if the checkpoint was saved under a
        # different data schedule than we're resuming into, the cumulative
        # `window_tokens_served` is at equilibrium with the *old* targets,
        # which interacts catastrophically with the deficit-based selector
        # when the new schedule drifts past the resume step. Symptom: the
        # selector degenerates into a "sign of drift" filter and starves
        # falling-target groups while stacking rising-target ones, producing
        # a mix radically different from the current target. Fix: detect the
        # schedule change and soft-reset `window_tokens_served` so the
        # selector rebuilds its deficit tracking from scratch against the
        # new schedule. Historical counters and shard positions are preserved.
        saved_fp = state.get('schedule_fingerprint')
        current_fp = self._schedule_fingerprint()
        if saved_fp is None:
            # Old checkpoint with no fingerprint — can't detect change.
            # Don't force a reset; log a hint so the user knows their old
            # checkpoints can't benefit from this protection.
            if self.data_schedule is not None:
                logger.print_and_log(
                    f"{rank_prefix(self.rank)} Checkpoint has no schedule fingerprint "
                    f"(pre-v5.1 state). Schedule-change detection disabled for this resume.",
                    r0_only=True,
                )
        elif saved_fp != current_fp:
            logger.print_and_log(
                f"{rank_prefix(self.rank)} Data schedule changed across resume "
                f"(checkpoint fingerprint={saved_fp}, current={current_fp}). "
                f"Performing auto soft-reset of window_tokens_served to avoid "
                f"deficit-stiffness bias under the new schedule.",
                r0_only=True,
            )
            self._soft_reset()
    
    def reset(self, mode: str = 'continue'):
        """
        Reset the dataloader state.
        
        Args:
            mode: Reset mode
                - 'continue': Normal resume, no reset (default)
                - 'soft': Reset window only (for percentage changes)
                - 'hard': Reset everything except positions
        """
        if mode == 'continue':
            # No reset, just continue
            logger.print_and_log(f"{rank_prefix(self.rank)} Continue mode: No reset performed")
            return
        
        elif mode == 'soft':
            # Soft reset: Clear window counters only
            self._soft_reset()
        
        elif mode == 'hard':
            # Hard reset: Clear all counters but keep positions
            logger.print_and_log(f"{rank_prefix(self.rank)} Hard reset: Clearing all token counters")
            for group in self.groups:
                group.historical_tokens_served = 0
                group.window_tokens_served = 0
            self.window_start_tokens = 0
        
        else:
            raise ValueError(f"Unknown reset mode: {mode}")
    
    # Compatibility properties and methods
    @property
    def current_shard(self) -> Optional[str]:
        """Current shard path for compatibility."""
        if self.current_group_idx < len(self.groups):
            group = self.groups[self.current_group_idx]
            return group.loaded_shard.path if group.loaded_shard else None
        return None
    
    @property
    def current_position(self) -> Optional[int]:
        """Current token position for compatibility."""
        if self.current_group_idx < len(self.groups):
            group = self.groups[self.current_group_idx]
            return group.loaded_shard.position if group.loaded_shard else None
        return None
    
    def current_shard_info(self) -> str:
        """Get descriptive string of current shard."""
        if self.current_group_idx >= len(self.groups):
            return "No valid group selected"
        
        group = self.groups[self.current_group_idx]
        if not group.loaded_shard:
            return f"No shard loaded for {group.name}"
        
        shard = group.loaded_shard
        shard_name = os.path.basename(shard.path)
        num_tokens = len(shard.tokens)
        
        # Calculate percentages (use window for current targeting)
        if self.active_groups:
            total_window = max(1, sum(g.window_tokens_served for g in self.active_groups))
            window_pct = (group.window_tokens_served / total_window) * 100 if group.is_active else 0
        else:
            window_pct = 0
        
        # Show all loaded shards
        loaded_groups = [g.name for g in self.groups if g.loaded_shard and not g.loaded_shard.is_exhausted]
        cache_info = f"[Loaded: {', '.join(loaded_groups)}]"
        
        status = "DEPRECATED" if not group.is_active else f"window: {window_pct:.1f}%"
        
        return (
            f"{group.name}/{shard_name} @ {shard.position:,}/{num_tokens:,} "
            f"({status}, target: {group.percentage:.1f}%) {cache_info}"
        )
    
    def get_statistics(self) -> Dict:
        """Get current mixing statistics and shard status."""
        # Calculate both historical and window statistics
        total_historical = max(1, sum(g.historical_tokens_served for g in self.groups))
        total_window_active = max(1, sum(g.window_tokens_served for g in self.active_groups))
        
        stats = {
            'total_historical_tokens': total_historical,
            'total_window_tokens': total_window_active,
            'window_start_tokens': self.window_start_tokens,
            'loaded_shards': sum(1 for g in self.groups if g.loaded_shard),
            'active_groups': len(self.active_groups),
            'deprecated_groups': len(self.deprecated_groups),
            'groups': {}
        }
        
        # Active groups statistics
        for g in self.active_groups:
            historical_pct = (g.historical_tokens_served / total_historical) * 100
            window_pct = (g.window_tokens_served / total_window_active) * 100
            
            group_stats = {
                'status': 'active',
                'target_percentage': g.percentage,
                'historical_percentage': historical_pct,
                'window_percentage': window_pct,
                'window_deviation': window_pct - g.percentage,
                'historical_tokens_served': g.historical_tokens_served,
                'window_tokens_served': g.window_tokens_served
            }
            
            # Add shard info if loaded
            if g.loaded_shard:
                group_stats['loaded_shard'] = {
                    'name': os.path.basename(g.loaded_shard.path),
                    'position': g.loaded_shard.position,
                    'total_tokens': len(g.loaded_shard.tokens),
                    'percent_used': (g.loaded_shard.position / len(g.loaded_shard.tokens)) * 100,
                    'tokens_remaining': g.loaded_shard.tokens_remaining
                }
            else:
                group_stats['loaded_shard'] = None
            
            stats['groups'][g.name] = group_stats
        
        # Deprecated groups statistics
        for g in self.deprecated_groups:
            historical_pct = (g.historical_tokens_served / total_historical) * 100 if total_historical > 0 else 0
            
            stats['groups'][g.name] = {
                'status': 'deprecated',
                'target_percentage': 0.0,
                'historical_percentage': historical_pct,
                'window_percentage': 0.0,
                'window_deviation': 0.0,
                'historical_tokens_served': g.historical_tokens_served,
                'window_tokens_served': 0,
                'loaded_shard': None  # Deprecated groups don't have loaded shards
            }
        
        return stats
    
    def get_epoch_progress(self) -> Dict:
        """Calculate epoch progress for each group."""
        progress = {}
        
        for group in self.groups:
            if group.num_tokens > 0:
                # Calculate how many times we've gone through this dataset
                epochs_completed = group.historical_tokens_served / group.num_tokens
                
                # Current position within the current epoch (0.0 to 1.0)
                current_epoch_progress = (group.historical_tokens_served % group.num_tokens) / group.num_tokens
                
                progress[group.name] = {
                    'epochs_completed': epochs_completed,
                    'current_epoch_progress': current_epoch_progress * 100,  # as percentage
                    'tokens_served': group.historical_tokens_served,
                    'total_tokens': group.num_tokens,
                    'status': 'active' if group.is_active else 'deprecated'
                }
            else:
                progress[group.name] = {
                    'epochs_completed': 0,
                    'current_epoch_progress': 0,
                    'tokens_served': 0,
                    'total_tokens': 0,
                    'status': 'active' if group.is_active else 'deprecated'
                }
        
        return progress
    
    # === Distributed aggregation helpers (global across ranks) ===
    def _dist_device(self):
        if dist.is_available() and dist.is_initialized():
            try:
                if dist.get_backend() == "nccl" and torch.cuda.is_available():
                    return torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
            except Exception:
                pass
        return torch.device("cpu")

    def _all_reduce_tensor(self, t: torch.Tensor) -> torch.Tensor:
        if not (dist.is_available() and dist.is_initialized()):
            return t
        device = self._dist_device()
        t_dev = t.to(device)
        dist.all_reduce(t_dev, op=dist.ReduceOp.SUM)
        return t_dev.to("cpu")

    def _all_reduce_scalar(self, value: int) -> int:
        if not (dist.is_available() and dist.is_initialized()):
            return int(value)
        device = self._dist_device()
        t = torch.tensor([int(value)], dtype=torch.long, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return int(t.item())

    def _all_reduce_group_counters(self):
        """Return (hist_all, win_all) reduced across ranks for ALL groups in order."""
        hist = torch.tensor([g.historical_tokens_served for g in self.groups], dtype=torch.long)
        win  = torch.tensor([g.window_tokens_served for g in self.groups], dtype=torch.long)
        hist = self._all_reduce_tensor(hist)
        win  = self._all_reduce_tensor(win)
        return hist.tolist(), win.tolist()

    def _global_epoch_progress_from_hist(self, hist_all):
        """Compute epoch progress dict using globally reduced historical counts."""
        progress = {}
        for g, hist in zip(self.groups, hist_all):
            if g.num_tokens > 0:
                epochs_completed = hist / g.num_tokens
                current_epoch_progress = (hist % g.num_tokens) / g.num_tokens
                progress[g.name] = {
                    'epochs_completed': epochs_completed,
                    'current_epoch_progress': current_epoch_progress * 100,
                    'tokens_served': hist,
                    'total_tokens': g.num_tokens,
                    'status': 'active' if g.is_active else 'deprecated'
                }
            else:
                progress[g.name] = {
                    'epochs_completed': 0,
                    'current_epoch_progress': 0,
                    'tokens_served': hist,
                    'total_tokens': 0,
                    'status': 'active' if g.is_active else 'deprecated'
                }
        return progress

    def log_schedule_status(self, step: int, ddp_rank: int, log_fn):
        """Log current data mix schedule status if schedule is configured.

        Args:
            step: Current training step
            ddp_rank: Only logs on rank 0
            log_fn: Logging function (e.g., logger.print_and_log)
        """
        if self.data_schedule is None or ddp_rank != 0:
            return

        current_mix = self.data_schedule.get_mix_at_step(step)
        phase_start, phase_end, phase_progress = self.data_schedule.get_current_phase(step)

        # Only show groups with >0.5% allocation
        mix_str = " | ".join(f"{k}: {v:.1f}%" for k, v in sorted(current_mix.items()) if v > 0.5)
        log_fn(f"Data Mix @ step {step}: {mix_str}")
        if phase_end > phase_start:
            log_fn(f"  Schedule phase: {phase_start} → {phase_end} ({phase_progress*100:.1f}% complete)")

    def log_detailed_dataloader_status(self, step, ddp_rank):
        """Log global (all-rank) status including epoch progress and sampling accuracy.

        This performs an all-reduce over per-group counters so the pretty table,
        deviation, weighted epochs, and the "Total tokens processed" line reflect
        **global** values across all ranks.
        """
        # --- Everyone participates in the reductions ---
        hist_all, win_all = self._all_reduce_group_counters()
        total_hist_global = max(1, sum(hist_all))
        # Only active groups contribute to window percentages
        active_indices = [i for i, g in enumerate(self.groups) if g.is_active]
        total_window_global = max(1, sum(win_all[i] for i in active_indices)) if active_indices else 1

        # Build a stats dict mirroring get_statistics(), but using GLOBAL counters
        stats = {
            'total_historical_tokens': total_hist_global,
            'total_window_tokens': total_window_global,
            'window_start_tokens': self.window_start_tokens,   # local marker; informational only
            'loaded_shards': sum(1 for g in self.groups if g.loaded_shard),  # local; not printed
            'active_groups': len([g for g in self.groups if g.is_active]),
            'deprecated_groups': len([g for g in self.groups if not g.is_active]),
            'groups': {}
        }

        for i, g in enumerate(self.groups):
            hist = hist_all[i]
            win = win_all[i]
            if g.is_active:
                hist_pct = (hist / total_hist_global) * 100
                win_pct = (win / total_window_global) * 100
                group_stats = {
                    'status': 'active',
                    'target_percentage': g.percentage,
                    'historical_percentage': hist_pct,
                    'window_percentage': win_pct,
                    'window_deviation': win_pct - g.percentage,
                    'historical_tokens_served': hist,
                    'window_tokens_served': win
                }
            else:
                group_stats = {
                    'status': 'deprecated',
                    'target_percentage': 0.0,
                    'historical_percentage': (hist / total_hist_global) * 100,
                    'window_percentage': 0.0,
                    'window_deviation': 0.0,
                    'historical_tokens_served': hist,
                    'window_tokens_served': 0
                }
            # Add shard info if available (local info, purely informational)
            if g.loaded_shard:
                shard = g.loaded_shard
                shard_name = os.path.basename(shard.path)
                shard_info = {
                    'shard': shard_name,
                    'position': shard.position,
                    'tokens_remaining': shard.tokens_remaining
                }
                group_stats['current_shard'] = shard_info
            stats['groups'][g.name] = group_stats

        # Epoch progress from GLOBAL historical counts
        epoch_progress = self._global_epoch_progress_from_hist(hist_all)

        # --- Only rank 0 prints ---
        if ddp_rank != 0:
            return

        logger.print_and_log("╔" + "═" * 94 + "╗")
        logger.print_and_log(f"║ DATALOADER STATUS - STEP {step:<58}          ║")
        logger.print_and_log("╠" + "═" * 94 + "╣")

        # Header
        logger.print_and_log("║ Dataset            │ Epoch  │ Progress                 │ Target  │ Actual  │ Deviation       ║")
        logger.print_and_log("╟────────────────────┼────────┼──────────────────────────┼─────────┼─────────┼─────────────────╢")

        for name in sorted(stats['groups'].keys()):
            info = stats['groups'][name]
            epoch_info = epoch_progress[name]

            if info['status'] == 'active':
                epochs = epoch_info['epochs_completed']
                progress_pct = epoch_info['current_epoch_progress']

                # Progress bar
                bar_length = 22
                filled = int(progress_pct / 100 * bar_length)
                bar = '█' * filled + '░' * (bar_length - filled)

                # Deviation indicator
                dev = info['window_deviation']
                if abs(dev) < 1.0:
                    dev_indicator = "✓ OK"
                elif abs(dev) < 3.0:
                    dev_indicator = "⚠ WARN"
                else:
                    dev_indicator = "✗ HIGH"

                logger.print_and_log(
                    f"║ {name:<18} │ {epochs:>6.2f} │ [{bar}] │ {info['target_percentage']:>6.1f}% │ {info['window_percentage']:>6.1f}% │ {dev:>6.1f}% {dev_indicator:<7} ║"
                )

        logger.print_and_log("╚" + "═" * 94 + "╝")

        # Summary statistics (GLOBAL)
        logger.print_and_log(f"Total tokens processed: {stats['total_historical_tokens']/1e9:.3f}B")
        # Weighted average epochs (GLOBAL, weighted by dataset token sizes across active groups)
        total_dataset_tokens = sum(self.groups[i].num_tokens for i in active_indices)
        if total_dataset_tokens > 0:
            weighted_epochs = sum(
                (epoch_progress[self.groups[i].name]['epochs_completed'] * self.groups[i].num_tokens)
                for i in active_indices
            ) / total_dataset_tokens
            logger.print_and_log(f"Weighted average epochs: {weighted_epochs:.3f}")

    @property
    def global_tokens_served(self) -> int:
        """Total tokens served historically (for compatibility)."""
        return sum(g.historical_tokens_served for g in self.groups)