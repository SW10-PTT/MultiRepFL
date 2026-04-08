import math
import random

"""
Validation Split is a flot number between 0 and 1.
it is the amout of data a users will set aside for validation. For example
if it is 0.1, than 10% of the data is assigned for validation and 90% is for traning.
"""

class DataPartition:
    def __init__(self, validation_split=0.1, seed=42):
        if not 0 <= validation_split < 1:
            raise ValueError("validation_split must be in the range [0, 1)")

        self.validation_split = validation_split
        self.seed = seed

    def get_num_participants(self, users):
        count = 0
        for _user in users:
            count += 1
        return count

    def split_dataset(self, users, dataset_size):
        users = list(users)
        self.validate_users(users, dataset_size)

        sample_ids = list(range(dataset_size))
        random.Random(self.seed).shuffle(sample_ids)

        sample_counts = self._resolve_sample_counts(users, dataset_size)
        user_splits = {}
        start = 0

        for order in range(len(users)):
            user = users[order]
            assigned_ids = sample_ids[start:start + sample_counts[order]]
            start += sample_counts[order]

            train_ids, val_ids = self._split_train_val(assigned_ids)
            user_id = self._get_user_id(user)
            data_percent = self._get_data_percent(user)
            user_splits[user_id] = {
                "data_percent": data_percent,
                "num_samples": len(assigned_ids),
                "train_ids": train_ids,
                "val_ids": val_ids,
                "train_samples": len(train_ids),
                "val_samples": len(val_ids),
            }

        return user_splits

    def assign_to_users(self, users, user_splits):
        for user in users:
            user_id = self._get_user_id(user)
            if user_id not in user_splits:
                raise KeyError(f"User id {user_id} not found in user_splits")

            user_split = user_splits[user_id]
            user.train_ids = list(user_split["train_ids"])
            user.val_ids = list(user_split["val_ids"])
            user.num_samples = int(user_split["num_samples"])

    def validate_users(self, users, dataset_size):
        if dataset_size < 0:
            raise ValueError("dataset_size must be non-negative")

        if not users:
            raise ValueError("users must contain at least one entry")

        seen_user_ids = []
        for user in users:
            user_id = self._get_user_id(user)
            seen_user_ids.append(user_id)

        duplicate_ids = []
        for user_id in seen_user_ids:
            if seen_user_ids.count(user_id) > 1 and user_id not in duplicate_ids:
                duplicate_ids.append(user_id)
        duplicate_ids.sort()
        if duplicate_ids:
            raise ValueError(f"Duplicate user ids found: {duplicate_ids}")

        missing_percent = []
        for user in users:
            if self._resolve_data_percent(user) is None:
                missing_percent.append(self._get_user_id(user))
        if missing_percent:
            raise ValueError(
                "Users missing data_percent/dataSplit: "
                f"{missing_percent}"
            )

        non_positive = []
        for user in users:
            if self._get_data_percent(user) <= 0:
                non_positive.append(self._get_user_id(user))
        if non_positive:
            raise ValueError(
                "data_percent/dataSplit must be positive for users: "
                f"{non_positive}"
            )

        total_percent = 0.0
        for user in users:
            total_percent += self._get_data_percent(user)
        if not math.isclose(total_percent, 100.0, abs_tol=1e-9):
            raise ValueError(
                "Total data_percent/dataSplit must equal 100, "
                f"got {total_percent}"
            )

    def _resolve_sample_counts(self, users, dataset_size):
        raw_counts = []
        for user in users:
            raw_count = dataset_size * self._get_data_percent(user) / 100.0
            raw_counts.append(raw_count)

        sample_counts = []
        for count in raw_counts:
            sample_counts.append(math.floor(count))

        assigned_count = 0
        for count in sample_counts:
            assigned_count += count

        leftovers = dataset_size - assigned_count

        remainders = []
        for index in range(len(users)):
            remainder = raw_counts[index] - sample_counts[index]
            remainders.append(remainder)

        for _ in range(leftovers):
            best_index = 0
            best_remainder = remainders[0]

            for index in range(1, len(remainders)):
                if remainders[index] > best_remainder:
                    best_index = index
                    best_remainder = remainders[index]

            sample_counts[best_index] += 1
            remainders[best_index] = -1

        return sample_counts

    def _split_train_val(self, assigned_ids):
        if not assigned_ids:
            return [], []

        if self.validation_split == 0 or len(assigned_ids) == 1:
            return list(assigned_ids), []

        val_size = int(len(assigned_ids) * self.validation_split)
        val_size = max(1, min(val_size, len(assigned_ids) - 1))

        val_ids = list(assigned_ids[:val_size])
        train_ids = list(assigned_ids[val_size:])
        return train_ids, val_ids

    def _get_user_id(self, user):
        if hasattr(user, "id"):
            return user.id
        if hasattr(user, "number"):
            return user.number
        raise ValueError("User is missing id/number attribute")

    def _resolve_data_percent(self, user):
        if hasattr(user, "data_percent"):
            return user.data_percent
        if hasattr(user, "dataSplit"):
            return user.dataSplit
        return None

    def _get_data_percent(self, user):
        data_percent = self._resolve_data_percent(user)
        if data_percent is None:
            raise ValueError(
                f"User {self._get_user_id(user)} is missing data_percent/dataSplit"
            )
        return float(data_percent)
