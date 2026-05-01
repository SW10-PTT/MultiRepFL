import json
from collections import Counter

import pytest

from openfl.ml.data_partition import DataPartition
from openfl.ml.partition_spec import UserPartitionSpec, load_partition_specs


# Stand-in for User so tests don't pull the full User class
# (which transitively imports torch, web3, contract artifacts, etc.).
class FakeUser:
    def __init__(self, uid, pct, partition_spec=None):
        self.id = uid
        self.address = None
        self.data_percent = pct
        self.partition_spec = partition_spec
        self.number = uid

    def get_id_or_address(self):
        return self.id


def make_labels(n_per_class=100, n_classes=10):
    return [i % n_classes for i in range(n_per_class * n_classes)]


def class_counts(labels, ids):
    return Counter(labels[i] for i in ids)


def collect_ids(splits, user):
    return splits[user.id]["train_ids"] + splits[user.id]["val_ids"]


# ---- UserPartitionSpec validation ----

def test_spec_rejects_zero_or_excessive_data_percent():
    with pytest.raises(ValueError, match="data_percent"):
        UserPartitionSpec(user_index=0, data_percent=0)
    with pytest.raises(ValueError, match="data_percent"):
        UserPartitionSpec(user_index=0, data_percent=150)


def test_spec_rejects_empty_label_distribution():
    with pytest.raises(ValueError, match="label_distribution cannot be empty"):
        UserPartitionSpec(user_index=0, data_percent=10, label_distribution={})


def test_spec_rejects_non_positive_weight():
    with pytest.raises(ValueError, match="weights must be > 0"):
        UserPartitionSpec(user_index=0, data_percent=10, label_distribution={0: 0.0})


def test_spec_rejects_distribution_outside_only_labels():
    with pytest.raises(ValueError, match="not in only_labels"):
        UserPartitionSpec(
            user_index=0,
            data_percent=10,
            only_labels=[0, 1],
            label_distribution={2: 1.0},
        )


# ---- JSON loader ----

def test_load_partition_specs_from_file(tmp_path):
    payload = {
        "users": [
            {"user_index": 0, "data_percent": 50.0, "label_distribution": {"0": 1.0, "1": 1.0}},
            {"user_index": 1, "data_percent": 50.0, "only_labels": [2, 3], "flip_map": {"2": 3}},
        ]
    }
    path = tmp_path / "partitions.json"
    path.write_text(json.dumps(payload))

    specs = load_partition_specs(str(path))
    assert set(specs.keys()) == {0, 1}
    assert specs[0].label_distribution == {0: 1.0, 1: 1.0}
    assert specs[1].only_labels == [2, 3]
    assert specs[1].flip_map == {2: 3}


def test_load_partition_specs_from_dict_keys():
    payload = {
        "0": {"data_percent": 60.0},
        "1": {"data_percent": 40.0, "only_labels": [0]},
    }
    specs = load_partition_specs(payload)
    assert specs[0].user_index == 0
    assert specs[0].data_percent == 60.0
    assert specs[1].only_labels == [0]


# ---- PerUserSpecStrategy partitioning ----

def test_per_user_distribution_matches_weights():
    # 1000 samples (100 per class). Budgets sized so per-class demand stays within supply:
    #   user 0: 10% = 100 samples, 70/30 over L4/L9 -> 70 of L4, 30 of L9
    #   user 1: 10% = 100 samples, 30/70 over L4/L9 -> 30 of L4, 70 of L9
    #   user 2: 20% = 200 samples, stratified across non-{4,9} (8 classes, ~25 each)
    spec0 = UserPartitionSpec(user_index=0, data_percent=10.0, label_distribution={4: 0.7, 9: 0.3})
    spec1 = UserPartitionSpec(user_index=1, data_percent=10.0, label_distribution={4: 0.3, 9: 0.7})
    spec2 = UserPartitionSpec(user_index=2, data_percent=20.0, only_labels=[0, 1, 2, 3, 5, 6, 7, 8])
    users = [
        FakeUser(0, 10.0, spec0),
        FakeUser(1, 10.0, spec1),
        FakeUser(2, 20.0, spec2),
    ]
    labels = make_labels(n_per_class=100, n_classes=10)

    partitioner = DataPartition(seed=1, per_user_specs={0: spec0, 1: spec1, 2: spec2})
    splits = partitioner.split_by_label(users, labels)

    counts0 = class_counts(labels, collect_ids(splits, users[0]))
    counts1 = class_counts(labels, collect_ids(splits, users[1]))
    counts2 = class_counts(labels, collect_ids(splits, users[2]))

    assert sum(counts0.values()) == 100
    assert sum(counts1.values()) == 100
    assert sum(counts2.values()) == 200

    # User 0 ~ 70% L4, 30% L9 (allow 1-sample rounding slack).
    assert abs(counts0[4] - 70) <= 1
    assert abs(counts0[9] - 30) <= 1
    # User 2 has 0 of restricted labels.
    assert counts2[4] == 0
    assert counts2[9] == 0


def test_per_user_disjoint_no_overlap():
    spec0 = UserPartitionSpec(user_index=0, data_percent=40.0)
    spec1 = UserPartitionSpec(user_index=1, data_percent=40.0)
    users = [FakeUser(0, 40.0, spec0), FakeUser(1, 40.0, spec1)]
    labels = make_labels()

    partitioner = DataPartition(seed=1, per_user_specs={0: spec0, 1: spec1})
    splits = partitioner.split_by_label(users, labels)

    a = set(collect_ids(splits, users[0]))
    b = set(collect_ids(splits, users[1]))
    assert a.isdisjoint(b)


def test_per_user_class_supply_conflict():
    # Total dataset has 100 samples of label 4. Two users each demand 80 of L4.
    spec0 = UserPartitionSpec(user_index=0, data_percent=8.0, label_distribution={4: 1.0})
    spec1 = UserPartitionSpec(user_index=1, data_percent=8.0, label_distribution={4: 1.0})
    users = [FakeUser(0, 8.0, spec0), FakeUser(1, 8.0, spec1)]
    labels = make_labels(n_per_class=100, n_classes=10)

    partitioner = DataPartition(seed=1, per_user_specs={0: spec0, 1: spec1})
    with pytest.raises(ValueError, match="per-class budget conflict"):
        partitioner.split_by_label(users, labels)


def test_per_user_determinism():
    # 1000 samples (100 per class). Each user takes 10% (100 samples) split
    # over disjoint label pairs, so per-class demand stays within supply.
    spec0 = UserPartitionSpec(user_index=0, data_percent=10.0, label_distribution={0: 0.5, 1: 0.5})
    spec1 = UserPartitionSpec(user_index=1, data_percent=10.0, label_distribution={2: 0.5, 3: 0.5})
    users = [FakeUser(0, 10.0, spec0), FakeUser(1, 10.0, spec1)]
    labels = make_labels()

    a = DataPartition(seed=42, per_user_specs={0: spec0, 1: spec1}).split_by_label(users, labels)
    b = DataPartition(seed=42, per_user_specs={0: spec0, 1: spec1}).split_by_label(users, labels)
    assert a == b


def test_per_user_seed_changes_split():
    spec0 = UserPartitionSpec(user_index=0, data_percent=50.0)
    spec1 = UserPartitionSpec(user_index=1, data_percent=50.0)
    users = [FakeUser(0, 50.0, spec0), FakeUser(1, 50.0, spec1)]
    labels = make_labels()

    a = DataPartition(seed=1, per_user_specs={0: spec0, 1: spec1}).split_by_label(users, labels)
    b = DataPartition(seed=2, per_user_specs={0: spec0, 1: spec1}).split_by_label(users, labels)
    assert a != b


def test_per_user_stratified_val_split():
    # 60% of dataset, evenly split across L0/L1/L2. Each class slice should
    # be split per the val_split ratio independently, so val keeps the
    # same class distribution as train.
    spec = UserPartitionSpec(
        user_index=0,
        data_percent=60.0,
        label_distribution={0: 1.0, 1: 1.0, 2: 1.0},
    )
    users = [FakeUser(0, 60.0, spec)]
    labels = make_labels(n_per_class=100, n_classes=3)

    partitioner = DataPartition(validation_split=0.2, seed=1, per_user_specs={0: spec})
    splits = partitioner.split_by_label(users, labels)
    user_split = splits[0]

    train_counts = class_counts(labels, user_split["train_ids"])
    val_counts = class_counts(labels, user_split["val_ids"])

    # Each class must contribute to BOTH splits — that is what stratified means.
    for cls in (0, 1, 2):
        assert train_counts[cls] > 0
        assert val_counts[cls] > 0
    # And the per-class counts should be roughly even (within 1 of each other).
    assert max(val_counts.values()) - min(val_counts.values()) <= 1


def test_per_user_overlap_mode_allows_double_subscription():
    # Per-user budget scales by replication_factor in overlap mode, mirroring
    # global-strategy semantics. Each user's data_percent=4% becomes
    # 4% * 2 = 8% -> 80 of L4. Combined demand 160 fits within rep-inflated
    # supply 200, but exceeds the 100-sample base supply, so users MUST share.
    spec0 = UserPartitionSpec(user_index=0, data_percent=4.0, label_distribution={4: 1.0})
    spec1 = UserPartitionSpec(user_index=1, data_percent=4.0, label_distribution={4: 1.0})
    users = [FakeUser(0, 4.0, spec0), FakeUser(1, 4.0, spec1)]
    labels = make_labels(n_per_class=100, n_classes=10)

    partitioner = DataPartition(
        seed=1,
        allow_overlap=True,
        replication_factor=2.0,
        per_user_specs={0: spec0, 1: spec1},
    )
    splits = partitioner.split_by_label(users, labels)
    counts0 = class_counts(labels, collect_ids(splits, users[0]))
    counts1 = class_counts(labels, collect_ids(splits, users[1]))
    assert counts0[4] > 0 and counts1[4] > 0
    # Combined demand 160 with base supply 100 forces at least 60 shared.
    shared = set(collect_ids(splits, users[0])) & set(collect_ids(splits, users[1]))
    assert len(shared) > 0


def test_per_user_only_labels_filter_in_strategy():
    # 1000 samples (100 per class). 20% budget = 200 over 3 allowed classes
    # is 66-67 per class, safely under the 100/class supply.
    spec0 = UserPartitionSpec(user_index=0, data_percent=20.0, only_labels=[0, 1, 2])
    spec1 = UserPartitionSpec(user_index=1, data_percent=20.0, only_labels=[3, 4, 5])
    users = [FakeUser(0, 20.0, spec0), FakeUser(1, 20.0, spec1)]
    labels = make_labels(n_per_class=100, n_classes=10)

    partitioner = DataPartition(seed=1, per_user_specs={0: spec0, 1: spec1})
    splits = partitioner.split_by_label(users, labels)

    counts0 = class_counts(labels, collect_ids(splits, users[0]))
    counts1 = class_counts(labels, collect_ids(splits, users[1]))
    assert set(counts0.keys()) <= {0, 1, 2}
    assert set(counts1.keys()) <= {3, 4, 5}


# ---- Backward compat: existing global strategy still passes ----

def test_global_strategy_unaffected_by_new_path():
    users = [FakeUser(0, 50.0), FakeUser(1, 50.0)]
    labels = make_labels()
    splits = DataPartition(seed=1).split_by_label(users, labels)
    a = set(collect_ids(splits, users[0]))
    b = set(collect_ids(splits, users[1]))
    assert a.isdisjoint(b)
    assert len(a) + len(b) == len(labels)
