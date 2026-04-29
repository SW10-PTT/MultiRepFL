from typing import List

import math
import random
from torch.utils.data import Dataset

from openfl.utils.types import User


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
    def __init__(self, validation_split=0.1, seed=42, allow_overlap=False, replication_factor=1.0):
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

    # Stratified partition of `labels` across `users`, weighted by each
    # user's data_percent.
    #
    # When allow_overlap=False (default): partition is disjoint, each
    # sample appears under exactly one user.
    # When allow_overlap=True: each class bucket is duplicated by
    # replication_factor before slicing. The same sample_id may now end
    # up under multiple users, but is deduplicated within a single user
    # (no user trains on the same image twice).

    def split_by_label(self, users: List["User"], labels):
        labels = self.normalize_labels(labels)
        self.validate_percentages(users)

        # Fixed seed -> deterministic split across runs.
        rng = random.Random(self.seed)

        # Bucket sample indices by class so we can slice each class
        # proportionally; this is what makes the split stratified.
        ids_by_label = {}
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
                assigned_ids_by_user[user_id].extend(sample_ids[start:start + count])
                start += count

        # Dedup within user (overlap mode can produce within-user dupes
        # from the inflated pool) and mix classes so batches stay diverse.
        for user_id, sample_ids in assigned_ids_by_user.items():
            if self.allow_overlap:
                sample_ids = self.dedupe_preserve_order(sample_ids)
            rng.shuffle(sample_ids)
            assigned_ids_by_user[user_id] = sample_ids

        return self.build_user_splits(users, assigned_ids_by_user)

    # Build the per-class slicing pool. Disjoint mode: shuffled bucket
    # as-is. Overlap mode: bucket repeated ceil(factor) times, truncated
    # to ceil(len * factor), shuffled. Truncation lets non-integer
    # factors (e.g. 1.5) work without distorting class balance.
    def build_pool(self, sample_ids, rng):
        if not self.allow_overlap or self.replication_factor == 1.0:
            shuffled = list(sample_ids)
            rng.shuffle(shuffled)
            return shuffled

        target_size = math.ceil(len(sample_ids) * self.replication_factor)
        repeats = math.ceil(self.replication_factor)
        pool = list(sample_ids) * repeats
        pool = pool[:target_size]
        rng.shuffle(pool)
        return pool

    def dedupe_preserve_order(self, ids):
        seen = set()
        out = []
        for sample_id in ids:
            if sample_id in seen:
                continue
            seen.add(sample_id)
            out.append(sample_id)
        return out

    # Destructive filter: drops samples whose label is not in only_labels.
    # TODO: User ends up with fewer samples than data_percent implies; no
    # rebalancing happens to compensate. Consider if this is a problem and if so, how to address it.
    def filter_indices_by_label(self, indices, all_labels, only_labels):
        only_labels_set = set(only_labels)
        return [i for i in indices if int(all_labels[i]) in only_labels_set]

    def apply_flip_map(self, dataset, user):
        return FlippedLabelDataset(dataset, user)

    def build_user_splits(self, users, assigned_ids_by_user):
        user_splits = {}

        for user in users:
            user_id = user.get_id_or_address()
            assigned_ids = list(assigned_ids_by_user[user_id])
            train_ids, val_ids = self.split_train_val(assigned_ids)
            user_splits[user_id] = {
                "data_percent": self.get_percent(user),
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

        total_percent = sum(self.get_percent(user) for user in users)
        raw_counts = [
            dataset_size * self.get_percent(user) / total_percent
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

    # Hard requirement: shares must cover the full dataset.
    # Sub-100 totals would silently drop samples; over-100 would
    # over-allocate and break the disjoint-partition invariant.
    def validate_percentages(self, users):
        percents = [self.get_percent(user) for user in users]
        if not math.isclose(sum(percents), 100.0, abs_tol=1e-9):
            raise ValueError("Total data_percent must equal 100")

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
