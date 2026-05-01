from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openfl.utils.types.User import User

from typing import List

import math
import random
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from torch.utils.data import Dataset


from openfl.ml.partition_spec import UserPartitionSpec


# Wraps a Subset and rewrites labels on read via user.flip_map.
# Image untouched; only the returned label changes. Used to simulate
# malicious users that mislabel their training data.
class FlippedLabelDataset(Dataset):
    def __init__(self, dataset, user):
        self.dataset = dataset
        self.user = user

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        # Fall back to original label if not in the flip map.
        return image, self.user.flip_map.get(int(label), int(label))


class DataPartition:
    # allow_overlap=True + replication_factor>1 lets the same sample land under multiple users.
    # Factor=1.5 means each sample appears in ~1.5 users on average. Disjoint mode keeps factor=1.
    # per_user_specs (optional): switches to PerUserSpecStrategy. Otherwise StratifiedGlobalStrategy.
    def __init__(
        self,
        validation_split: float = 0.1,
        seed: int = 42,
        allow_overlap: bool = False,
        replication_factor: float = 1.0,
        per_user_specs: Optional[Dict[int, UserPartitionSpec]] = None,
    ):
        if not 0 <= validation_split < 1:
            raise ValueError("validation_split must be in the range [0, 1)")
        if replication_factor < 1.0:
            raise ValueError("replication_factor must be >= 1.0")
        if replication_factor > 1.0 and not allow_overlap:
            raise ValueError("replication_factor > 1.0 requires allow_overlap=True")

        self.validation_split = validation_split
        self.seed = seed
        self.allow_overlap = bool(allow_overlap)
        self.replication_factor = float(replication_factor)
        self.per_user_specs = per_user_specs or None

        if self.per_user_specs:
            self.strategy: PartitionStrategy = PerUserSpecStrategy(self)
        else:
            self.strategy = StratifiedGlobalStrategy(self)

    # Public entry point. Delegates to the active strategy. Backward
    # compatible with prior signature (users, labels) -> {user_id: {...}}.

    def split_by_label(self, users: List[User], labels):
        labels = self.normalize_labels(labels)
        return self.strategy.split_by_label(users, labels)

    # ---- shared helpers used by both strategies ----

    # Destructive filter: drops samples whose label is not in only_labels.
    # Kept for backward compat with the legacy global path; in per_user mode
    # the strategy already restricts indices to only_labels so calling this
    # is a no-op.
    def filter_indices_by_label(self, indices, all_labels, only_labels):
        only_labels_set = set(only_labels)
        return [i for i in indices if int(all_labels[i]) in only_labels_set]

    def apply_flip_map(self, dataset, user):
        return FlippedLabelDataset(dataset, user)

    def split_train_val(self, assigned_ids):
        if not assigned_ids:
            return [], []

        # Single-sample users cannot be split; everything goes to train.
        if self.validation_split == 0 or len(assigned_ids) == 1:
            return list(assigned_ids), []

        # Force at least 1 train and 1 val sample when possible, so
        # downstream loaders never see an empty split.
        val_size = int(len(assigned_ids) * self.validation_split)
        val_size = max(1, min(val_size, len(assigned_ids) - 1))

        val_ids = list(assigned_ids[:val_size])
        train_ids = list(assigned_ids[val_size:])
        return train_ids, val_ids

    def get_percent(self, user):
        if hasattr(user, "data_percent"):
            return float(user.data_percent)
        raise ValueError(
            f"User {user.get_id_or_address()} is missing data_percent"
        )

    # torchvision datasets expose targets as tensor / ndarray / list.
    # Convert to plain Python ints so dict keys hash consistently.
    def normalize_labels(self, labels):
        if hasattr(labels, "tolist"):
            return labels.tolist()
        return [label.item() if hasattr(label, "item") else label for label in labels]


class PartitionStrategy(ABC):
    def __init__(self, partition: DataPartition):
        self.partition = partition

    @abstractmethod
    def split_by_label(self, users: List["User"], labels) -> dict:
        ...


class StratifiedGlobalStrategy(PartitionStrategy):
    # Stratified partition of `labels` across `users`, weighted by each
    # user's data_percent.
    #
    # When allow_overlap=False (default): partition is disjoint, each
    # sample appears under exactly one user.
    # When allow_overlap=True: each class bucket is duplicated by
    # replication_factor before slicing. The same sample_id may now end
    # up under multiple users, but is deduplicated within a single user
    # (no user trains on the same image twice).
    def split_by_label(self, users, labels):
        self.validate_percentages(users)

        # Fixed seed -> deterministic split across runs.
        rng = random.Random(self.partition.seed)

        # Bucket sample indices by class so we can slice each class
        # proportionally; this is what makes the split stratified.
        ids_by_label: Dict[int, List[int]] = {}
        for sample_id, label in enumerate(labels):
            ids_by_label.setdefault(label, []).append(sample_id)

        assigned_ids_by_user = {user.get_id_or_address(): [] for user in users}

        # Per class: shuffle, optionally inflate by replication_factor,
        # then hand out contiguous chunks sized by data_percent.
        # Without overlap the start cursor advances over a disjoint pool;
        # with overlap the inflated pool contains duplicates so different
        # users' chunks can collide on the same sample_id.
        for label, sample_ids in ids_by_label.items():
            pool = self.build_pool(sample_ids, rng)
            counts = self.get_counts_by_percent(users, len(pool))
            start = 0

            for user, count in zip(users, counts):
                user_id = user.get_id_or_address()
                assigned_ids_by_user[user_id].extend(pool[start:start + count])
                start += count

        # Dedup within user (overlap mode can produce within-user dupes
        # from the inflated pool) and mix classes so batches stay diverse.
        for user_id, sample_ids in assigned_ids_by_user.items():
            if self.partition.allow_overlap:
                sample_ids = self.dedupe_preserve_order(sample_ids)
            rng.shuffle(sample_ids)
            assigned_ids_by_user[user_id] = sample_ids

        return self.build_user_splits(users, assigned_ids_by_user)

    # Build the per-class slicing pool. Disjoint mode: shuffled bucket
    # as-is. Overlap mode: bucket repeated ceil(factor) times, truncated
    # to ceil(len * factor), shuffled. Truncation lets non-integer
    # factors (e.g. 1.5) work without distorting class balance.
    def build_pool(self, sample_ids, rng):
        if not self.partition.allow_overlap or self.partition.replication_factor == 1.0:
            shuffled = list(sample_ids)
            rng.shuffle(shuffled)
            return shuffled

        target_size = math.ceil(len(sample_ids) * self.partition.replication_factor)
        repeats = math.ceil(self.partition.replication_factor)
        pool = list(sample_ids) * repeats
        pool = pool[:target_size]
        rng.shuffle(pool)
        return pool

    # Removes duplicate sample_ids while keeping first-seen order.
    # Used in overlap mode so a single user never trains on the same image twice.
    def dedupe_preserve_order(self, ids):
        seen = set()
        out = []
        for sample_id in ids:
            if sample_id in seen:
                continue
            seen.add(sample_id)
            out.append(sample_id)
        return out

    def build_user_splits(self, users, assigned_ids_by_user):
        user_splits = {}

        for user in users:
            user_id = user.get_id_or_address()
            assigned_ids = list(assigned_ids_by_user[user_id])
            train_ids, val_ids = self.partition.split_train_val(assigned_ids)
            user_splits[user_id] = {
                "data_percent": self.partition.get_percent(user),
                "num_samples": len(assigned_ids),
                "train_ids": train_ids,
                "val_ids": val_ids,
                "train_samples": len(train_ids),
                "val_samples": len(val_ids),
            }

        return user_splits

    # Largest-remainder method: floor each share, then hand out the
    # leftover samples one-by-one to users with the biggest fractional
    # remainder. Guarantees sum(counts) == dataset_size and minimises
    # rounding error vs. simple floor-and-discard.
    def get_counts_by_percent(self, users, dataset_size):
        if dataset_size == 0:
            return []

        total_percent = sum(self.partition.get_percent(user) for user in users)
        raw_counts = [
            dataset_size * self.partition.get_percent(user) / total_percent
            for user in users
        ]
        sample_counts = [math.floor(count) for count in raw_counts]
        leftovers = dataset_size - sum(sample_counts)
        remainders = [raw - floor for raw, floor in zip(raw_counts, sample_counts)]

        # -1 sentinel ensures each user receives at most one leftover
        # per pass; without it the same user would win every iteration.
        for _ in range(leftovers):
            best_index = max(range(len(remainders)), key=remainders.__getitem__)
            sample_counts[best_index] += 1
            remainders[best_index] = -1

        return sample_counts

    # Hard requirement: shares must cover the full dataset.
    # Sub-100 totals would silently drop samples; over-100 would
    # over-allocate and break the disjoint-partition invariant.
    def validate_percentages(self, users):
        percents = [self.partition.get_percent(user) for user in users]
        if not math.isclose(sum(percents), 100.0, abs_tol=1e-9):
            raise ValueError("Total data_percent must equal 100")


class PerUserSpecStrategy(PartitionStrategy):
    # Per-user strategy: each user takes a fair stratified share of every
    # class equal to data_percent of the (rep-inflated) class pool, then:
    #   1. only_labels (if set) acts as a hard whitelist; classes outside
    #      are dropped entirely.
    #   2. label_distribution (if set) acts as a per-class retention factor
    #      in [0, 1]: weight 1.0 keeps the full fair share, 0.5 keeps half,
    #      0.0 drops the class. Classes not mentioned default to 1.0 (full).
    #
    # data_percent stays the same mental model regardless of skew; a user
    # with only_labels=[4,9] and pct=10% effectively trains on ~2% of the
    # dataset. label_distribution lets a user further sub-sample within
    # their kept classes without affecting other users' allocations.
    #
    # No per-class supply check is needed: sum(pct) <= 100 guarantees the
    # per-class cursor never exceeds the inflated pool.
    def split_by_label(self, users, labels):
        rng = random.Random(self.partition.seed)
        specs = self.partition.per_user_specs or {}

        # Resolve user -> spec via user.number (the data-user index).
        user_to_spec = {}
        for user in users:
            spec = self._resolve_user_spec(user, specs)
            user_to_spec[user.get_id_or_address()] = spec

        # Bucket samples by class.
        ids_by_label: Dict[int, List[int]] = {}
        for sample_id, label in enumerate(labels):
            ids_by_label.setdefault(int(label), []).append(sample_id)

        all_classes = sorted(ids_by_label.keys())

        # Per-class shuffled pool (with replication if overlap). Cursor
        # advances as users take their fair share; users iterate in
        # deterministic spec.user_index order.
        pools = {cls: self._build_class_pool(ids_by_label[cls], rng) for cls in all_classes}
        cursors = {cls: 0 for cls in all_classes}

        ordered_users = sorted(
            users,
            key=lambda u: user_to_spec[u.get_id_or_address()].user_index,
        )

        assigned_by_class: Dict[str, Dict[int, List[int]]] = {
            user.get_id_or_address(): {cls: [] for cls in all_classes} for user in users
        }

        for user in ordered_users:
            user_id = user.get_id_or_address()
            spec = user_to_spec[user_id]
            whitelist = self._resolve_whitelist(spec, all_classes)
            retention = self._resolve_retention(spec)
            pct = spec.data_percent / 100.0

            for cls in all_classes:
                count = int(round(len(pools[cls]) * pct))
                chunk = pools[cls][cursors[cls]:cursors[cls] + count]
                cursors[cls] += count
                if cls not in whitelist:
                    continue
                keep = int(round(len(chunk) * retention.get(cls, 1.0)))
                if keep <= 0:
                    continue
                assigned_by_class[user_id][cls] = chunk[:keep]

        # Stratified train/val split per class, then concatenate + shuffle.
        return self._build_splits_stratified(users, user_to_spec, assigned_by_class, all_classes, rng)

    # only_labels is the hard whitelist. Without it, every class is allowed
    # through (label_distribution still trims via retention factors).
    def _resolve_whitelist(self, spec: UserPartitionSpec, all_classes) -> set:
        if spec.only_labels is not None:
            return {int(cls) for cls in spec.only_labels}
        return set(all_classes)

    # Retention factor per class (0..1). Classes not in label_distribution
    # default to 1.0 (keep full fair share).
    def _resolve_retention(self, spec: UserPartitionSpec) -> Dict[int, float]:
        if spec.label_distribution is None:
            return {}
        return {int(cls): float(weight) for cls, weight in spec.label_distribution.items()}

    # Picks the spec for a given user. Prefers a spec already attached to
    # the User by the runner (`user.partition_spec`); otherwise falls back
    # to specs indexed by `user.number` or `user.get_id_or_address()` so
    # this strategy can be exercised in tests without the full User class.
    def _resolve_user_spec(self, user, specs):
        attached = getattr(user, "partition_spec", None)
        if attached is not None:
            return attached
        index = getattr(user, "number", None)
        if index is not None and index in specs:
            return specs[index]
        if user.get_id_or_address() in specs:
            return specs[user.get_id_or_address()]
        raise ValueError(
            f"PerUserSpecStrategy: no spec found for user {user.get_id_or_address()} "
            f"(number={index})"
        )

    def _build_class_pool(self, sample_ids, rng):
        if not self.partition.allow_overlap or self.partition.replication_factor == 1.0:
            shuffled = list(sample_ids)
            rng.shuffle(shuffled)
            return shuffled

        target_size = math.floor(len(sample_ids) * self.partition.replication_factor)
        repeats = math.ceil(self.partition.replication_factor)
        pool = list(sample_ids) * repeats
        pool = pool[:target_size]
        rng.shuffle(pool)
        return pool

    # Stratified train/val split: split per class, then concat. Guarantees
    # the val split keeps the same class distribution as train within a user.
    def _build_splits_stratified(self, users, user_to_spec, assigned_by_class, all_classes, rng):
        user_splits = {}

        for user in users:
            user_id = user.get_id_or_address()
            spec = user_to_spec[user_id]

            train_ids: List[int] = []
            val_ids: List[int] = []

            for cls in all_classes:
                ids = assigned_by_class[user_id].get(cls, [])
                if not ids:
                    continue
                cls_train, cls_val = self.partition.split_train_val(ids)
                train_ids.extend(cls_train)
                val_ids.extend(cls_val)

            # Within-user dedup in case overlap-mode pool produced dupes.
            if self.partition.allow_overlap:
                train_ids = self._dedupe_preserve_order(train_ids)
                val_ids = self._dedupe_preserve_order(val_ids)
                # If a sample landed in both train and val for this user,
                # keep it in train and drop from val.
                train_set = set(train_ids)
                val_ids = [i for i in val_ids if i not in train_set]

            rng.shuffle(train_ids)
            rng.shuffle(val_ids)

            num_samples = len(train_ids) + len(val_ids)
            user_splits[user_id] = {
                "data_percent": spec.data_percent,
                "num_samples": num_samples,
                "train_ids": train_ids,
                "val_ids": val_ids,
                "train_samples": len(train_ids),
                "val_samples": len(val_ids),
            }

        return user_splits

    def _dedupe_preserve_order(self, ids):
        seen = set()
        out = []
        for sample_id in ids:
            if sample_id in seen:
                continue
            seen.add(sample_id)
            out.append(sample_id)
        return out
