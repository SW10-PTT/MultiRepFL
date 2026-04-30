from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openfl.utils.types.User import User

from typing import List

import math
import random
from torch.utils.data import Dataset


class FlippedLabelDataset(Dataset):
    def __init__(self, dataset, user):
        self.dataset = dataset
        self.user = user

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return image, self.user.flip_map.get(int(label), int(label))


class DataPartition:
    def __init__(self, validation_split=0.1, seed=42):
        if not 0 <= validation_split < 1:
            raise ValueError("validation_split must be in the range [0, 1)")

        self.validation_split = validation_split
        self.seed = seed

    def split_by_label(self, users: List[User], labels):
        labels = self.normalize_labels(labels)
        self.validate_percentages(users)

        rng = random.Random(self.seed)
        ids_by_label = {}
        for sample_id, label in enumerate(labels):
            ids_by_label.setdefault(label, []).append(sample_id)

        assigned_ids_by_user = {user.get_id_or_address(): [] for user in users}

        for label, sample_ids in ids_by_label.items():
            rng.shuffle(sample_ids)
            counts = self.get_counts_by_percent(users, len(sample_ids))
            start = 0

            for user, count in zip(users, counts):
                user_id = user.get_id_or_address()
                assigned_ids_by_user[user_id].extend(sample_ids[start:start + count])
                start += count

        for sample_ids in assigned_ids_by_user.values():
            rng.shuffle(sample_ids)

        return self.build_user_splits(users, assigned_ids_by_user)

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

        for _ in range(leftovers):
            best_index = max(range(len(remainders)), key=remainders.__getitem__)
            sample_counts[best_index] += 1
            remainders[best_index] = -1

        return sample_counts

    def split_train_val(self, assigned_ids):
        if not assigned_ids:
            return [], []

        if self.validation_split == 0 or len(assigned_ids) == 1:
            return list(assigned_ids), []

        val_size = int(len(assigned_ids) * self.validation_split)
        val_size = max(1, min(val_size, len(assigned_ids) - 1))

        val_ids = list(assigned_ids[:val_size])
        train_ids = list(assigned_ids[val_size:])
        return train_ids, val_ids

    def validate_percentages(self, users):
        percents = [self.get_percent(user) for user in users]
        if not math.isclose(sum(percents), 100.0, abs_tol=1e-9):
            raise ValueError("Total data_percent must equal 100")

    def get_percent(self, user: User):
        if hasattr(user, "data_percent"):
            return float(user.data_percent)
        raise ValueError(
            f"User {user.get_id_or_address()} is missing data_percent"
        )

    def normalize_labels(self, labels):
        if hasattr(labels, "tolist"):
            return labels.tolist()
        return [label.item() if hasattr(label, "item") else label for label in labels]
