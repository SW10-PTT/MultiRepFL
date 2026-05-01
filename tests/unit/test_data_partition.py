import pytest

from openfl.ml.data_partition import DataPartition


# Stand-in for User so tests don't pull the full User class
# (which transitively imports torch, web3, contract artifacts, etc.).
class FakeUser:
    def __init__(self, uid, pct):
        self.id = uid
        self.address = None
        self.data_percent = pct

    def get_id_or_address(self):
        return self.id


def make_users(n):
    return [FakeUser(i, 100.0 / n) for i in range(n)]


def make_labels(n_per_class=100, n_classes=10):
    return [i % n_classes for i in range(n_per_class * n_classes)]


def collect_ids(splits, user):
    return splits[user.id]["train_ids"] + splits[user.id]["val_ids"]


def test_disjoint_covers_dataset():
    users = make_users(5)
    labels = make_labels()
    splits = DataPartition(seed=1).split_by_label(users, labels)
    all_ids = [i for u in users for i in collect_ids(splits, u)]
    assert len(all_ids) == len(labels)
    assert len(set(all_ids)) == len(labels)


def test_disjoint_users_share_no_ids():
    users = make_users(5)
    labels = make_labels()
    splits = DataPartition(seed=1).split_by_label(users, labels)
    sets = [set(collect_ids(splits, u)) for u in users]
    # Pairwise check: every (i, j) with i<j must have no shared IDs.
    for i, a in enumerate(sets):
        for b in sets[i + 1:]:
            assert a.isdisjoint(b)


def test_overlap_increases_total_samples():
    users = make_users(5)
    labels = make_labels()
    p = DataPartition(seed=1, allow_overlap=True, replication_factor=2.0)
    splits = p.split_by_label(users, labels)
    total = sum(len(collect_ids(splits, u)) for u in users)
    # Lower bound: must exceed disjoint total (else overlap does not work).
    # Upper bound: factor=2 inflates pool to 2N; within-user dedup can shrink but never grow past 2N.
    assert total > len(labels)
    assert total <= 2 * len(labels)


def test_overlap_users_share_some_ids():
    users = make_users(5)
    labels = make_labels()
    p = DataPartition(seed=1, allow_overlap=True, replication_factor=2.0)
    splits = p.split_by_label(users, labels)
    sets = [set(collect_ids(splits, u)) for u in users]
    # At least one user pair must share IDs; otherwise overlap is structurally absent.
    pair_overlaps = [len(a & b) for i, a in enumerate(sets) for b in sets[i + 1:]]
    assert any(x > 0 for x in pair_overlaps)


def test_within_user_dedup_in_overlap_mode():
    # Factor 1.8 makes per-user duplicate samples likely, so dedup logic gets exercised. (Dedup logic = a user never trains on the same image twice)
    users = make_users(4)
    labels = make_labels()
    p = DataPartition(seed=1, allow_overlap=True, replication_factor=1.8)
    splits = p.split_by_label(users, labels)
    for u in users:
        ids = collect_ids(splits, u)
        assert len(ids) == len(set(ids))


def test_determinism_same_seed():
    users = make_users(4)
    labels = make_labels()
    a = DataPartition(seed=42).split_by_label(users, labels)
    b = DataPartition(seed=42).split_by_label(users, labels)
    assert a == b


def test_different_seed_changes_split():
    users = make_users(4)
    labels = make_labels()
    a = DataPartition(seed=1).split_by_label(users, labels)
    b = DataPartition(seed=2).split_by_label(users, labels)
    assert a != b


def test_stratified_class_balance():
    users = make_users(4)
    labels = make_labels(n_per_class=100, n_classes=10)
    splits = DataPartition(seed=1).split_by_label(users, labels)
    for u in users:
        per_class = [0] * 10
        for i in collect_ids(splits, u):
            per_class[labels[i]] += 1
        # Each user has 25 per class with equal data_percent. Allow small rounding slack.
        assert all(20 <= c <= 30 for c in per_class), per_class


def test_validation_factor_below_one():
    with pytest.raises(ValueError, match="replication_factor must be >= 1.0"):
        DataPartition(replication_factor=0.5)


def test_validation_factor_without_overlap():
    with pytest.raises(ValueError, match="replication_factor > 1.0 requires allow_overlap=True"):
        DataPartition(replication_factor=1.5, allow_overlap=False)


def test_overlap_mode_factor_one_still_disjoint_in_practice():
    # allow_overlap=True with factor=1.0 should behave like disjoint mode (no inflation).
    users = make_users(5)
    labels = make_labels()
    p = DataPartition(seed=1, allow_overlap=True, replication_factor=1.0)
    splits = p.split_by_label(users, labels)
    total = sum(len(collect_ids(splits, u)) for u in users)
    assert total == len(labels)


def test_uneven_data_percent_respected():
    # 50/30/20 split must approximate the configured shares despite per-class rounding (largest-remainder).
    users = [FakeUser(0, 50.0), FakeUser(1, 30.0), FakeUser(2, 20.0)]
    labels = make_labels(n_per_class=100, n_classes=10)
    splits = DataPartition(seed=1).split_by_label(users, labels)
    sizes = [len(collect_ids(splits, u)) for u in users]
    assert abs(sizes[0] - 500) <= 5
    assert abs(sizes[1] - 300) <= 5
    assert abs(sizes[2] - 200) <= 5
