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

    def split_mnist_same_share_per_label(self, users, labels):
        users = list(users)
        labels = self._normalize_labels(labels)
        self._validate_percentages(users)

        rng = random.Random(self.seed)
        ids_by_label = {}
        for sample_id, label in enumerate(labels):
            ids_by_label.setdefault(label, []).append(sample_id)

        assigned_ids_by_user = {self._get_user_id(user): [] for user in users}

        for label, sample_ids in ids_by_label.items():
            rng.shuffle(sample_ids)
            eligible_users = self._get_users_for_label(users, label)
            counts = self._get_counts_by_percent(eligible_users, len(sample_ids))
            start = 0

            for user, count in zip(eligible_users, counts):
                user_id = self._get_user_id(user)
                assigned_ids_by_user[user_id].extend(sample_ids[start:start + count])
                start += count

        for sample_ids in assigned_ids_by_user.values():
            rng.shuffle(sample_ids)

        return self._build_user_splits(users, assigned_ids_by_user)

    def assign_to_users(self, users, user_splits):
        for user in users:
            user_id = self._get_user_id(user)
            user_split = user_splits[user_id]
            user.train_ids = list(user_split["train_ids"])
            user.val_ids = list(user_split["val_ids"])
            user.num_samples = int(user_split["num_samples"])

    def _build_user_splits(self, users, assigned_ids_by_user):
        user_splits = {}

        for user in users:
            user_id = self._get_user_id(user)
            assigned_ids = list(assigned_ids_by_user[user_id])
            train_ids, val_ids = self._split_train_val(assigned_ids)
            user_splits[user_id] = {
                "data_percent": self._get_percent(user),
                "num_samples": len(assigned_ids),
                "train_ids": train_ids,
                "val_ids": val_ids,
                "train_samples": len(train_ids),
                "val_samples": len(val_ids),
            }

        return user_splits

    def _get_counts_by_percent(self, users, dataset_size):
        if dataset_size == 0:
            return []

        total_percent = sum(self._get_percent(user) for user in users)
        raw_counts = [
            dataset_size * self._get_percent(user) / total_percent
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

    def _get_users_for_label(self, users, label):
        # If a user has only_labels, they only receive those labels.
        # If nobody asked for this label, we keep default behavior and share it to all users.
        allowed_users = [
            user for user in users
            if self._user_accepts_label(user, label)
        ]
        if allowed_users:
            return allowed_users
        return users

    def _user_accepts_label(self, user, label):
        only_labels = getattr(user, "only_labels", None)
        if only_labels is None:
            return True
        return label in only_labels

    def _validate_percentages(self, users):
        percents = [self._get_percent(user) for user in users]
        if not math.isclose(sum(percents), 100.0, abs_tol=1e-9):
            raise ValueError("Total data_percent/dataSplit must equal 100")

    def _get_user_id(self, user):
        if hasattr(user, "id"):
            return user.id
        if hasattr(user, "number"):
            return user.number
        raise ValueError("User is missing id/number attribute")

    def _get_percent(self, user):
        if hasattr(user, "data_percent"):
            return float(user.data_percent)
        if hasattr(user, "dataSplit"):
            return float(user.dataSplit)
        raise ValueError(
            f"User {self._get_user_id(user)} is missing data_percent/dataSplit"
        )

    def _normalize_labels(self, labels):
        if labels is None:
            raise ValueError("labels must be provided for mnist")

        if hasattr(labels, "tolist"):
            labels = labels.tolist()
        else:
            labels = list(labels)

        return [
            label.item() if hasattr(label, "item") else label
            for label in labels
        ]
