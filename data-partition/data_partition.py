import math
import random


class DataPartition:
    def __init__(self, validation_split=0.1, seed=42):
        if not 0 <= validation_split < 1:
            raise ValueError("validation_split must be in the range [0, 1)")

        self.validation_split = validation_split
        self.seed = seed

    def get_num_participants(self, users):
        return len(list(users))

    def build_plan(self, users, dataset_size):
        users = list(users)
        self._validate_users(users, dataset_size)

        indices = list(range(dataset_size))
        random.Random(self.seed).shuffle(indices)

        sample_counts = self._resolve_sample_counts(users, dataset_size)
        plan = {}
        start = 0

        for order, user in enumerate(users):
            user_indices = indices[start:start + sample_counts[order]]
            start += sample_counts[order]

            train_indices, val_indices = self._split_train_val(user_indices)
            plan[user.id] = {
                "data_percent": float(user.data_percent),
                "num_samples": len(user_indices),
                "train_indices": train_indices,
                "val_indices": val_indices,
                "train_samples": len(train_indices),
                "val_samples": len(val_indices),
            }

        return plan

    def _validate_users(self, users, dataset_size):
        if dataset_size < 0:
            raise ValueError("dataset_size must be non-negative")

        if not users:
            raise ValueError("users must contain at least one entry")

        duplicate_ids = {
            user.id for user in users
            if sum(1 for other in users if other.id == user.id) > 1
        }
        if duplicate_ids:
            raise ValueError(f"Duplicate user ids found: {sorted(duplicate_ids)}")

        missing_percent = [user.id for user in users if not hasattr(user, "data_percent")]
        if missing_percent:
            raise ValueError(f"Users missing data_percent: {missing_percent}")

        non_positive = [user.id for user in users if user.data_percent <= 0]
        if non_positive:
            raise ValueError(f"data_percent must be positive for users: {non_positive}")

        total_percent = sum(float(user.data_percent) for user in users)
        if not math.isclose(total_percent, 100.0, abs_tol=1e-9):
            raise ValueError(
                f"Total data_percent must equal 100, got {total_percent}"
            )

    def _resolve_sample_counts(self, users, dataset_size):
        raw_counts = [
            dataset_size * float(user.data_percent) / 100.0
            for user in users
        ]
        sample_counts = [math.floor(count) for count in raw_counts]

        leftovers = dataset_size - sum(sample_counts)
        remainders = sorted(
            range(len(users)),
            key=lambda idx: (raw_counts[idx] - sample_counts[idx], -idx),
            reverse=True,
        )

        for idx in remainders[:leftovers]:
            sample_counts[idx] += 1

        return sample_counts

    def _split_train_val(self, user_indices):
        if not user_indices:
            return [], []

        if self.validation_split == 0 or len(user_indices) == 1:
            return list(user_indices), []

        val_size = int(len(user_indices) * self.validation_split)
        val_size = max(1, min(val_size, len(user_indices) - 1))

        val_indices = list(user_indices[:val_size])
        train_indices = list(user_indices[val_size:])
        return train_indices, val_indices
