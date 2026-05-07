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


def test_spec_rejects_weight_outside_zero_one():
    with pytest.raises(ValueError, match=r"weights must be in \[0, 1\]"):
        UserPartitionSpec(user_index=0, data_percent=10, label_distribution={0: -0.1})
    with pytest.raises(ValueError, match=r"weights must be in \[0, 1\]"):
        UserPartitionSpec(user_index=0, data_percent=10, label_distribution={0: 1.5})


def test_spec_accepts_zero_weight_as_drop():
    # 0 means "drop this class entirely" — equivalent to excluding it via
    # only_labels but reachable from inside label_distribution alone.
    spec = UserPartitionSpec(user_index=0, data_percent=10, label_distribution={3: 0.0})
    assert spec.label_distribution == {3: 0.0}


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
            {"user_index": "alpha-guid", "data_percent": 50.0, "label_distribution": {"0": 1.0, "1": 1.0}},
            {"user_index": "beta-guid", "data_percent": 50.0, "only_labels": [2, 3], "flip_map": {"2": 3}},
        ]
    }
    path = tmp_path / "partitions.json"
    path.write_text(json.dumps(payload))

    specs = load_partition_specs(str(path))
    assert set(specs.keys()) == {"alpha-guid", "beta-guid"}
    assert specs["alpha-guid"].label_distribution == {0: 1.0, 1: 1.0}
    assert specs["beta-guid"].only_labels == [2, 3]
    assert specs["beta-guid"].flip_map == {2: 3}


def test_load_partition_specs_from_dict_keys():
    payload = {
        "0": {"data_percent": 60.0},
        "1": {"data_percent": 40.0, "only_labels": [0]},
    }
    specs = load_partition_specs(payload)
    assert specs["0"].user_index == "0"
    assert specs["0"].data_percent == 60.0
    assert specs["1"].only_labels == [0]


def test_load_partition_specs_accepts_guid_keys():
    payload = {
        "550e8400-e29b-41d4-a716-446655440000": {"data_percent": 30.0},
        "6ba7b810-9dad-11d1-80b4-00c04fd430c8": {"data_percent": 70.0, "only_labels": [3]},
    }
    specs = load_partition_specs(payload)
    assert set(specs.keys()) == {
        "550e8400-e29b-41d4-a716-446655440000",
        "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
    }
    assert specs["6ba7b810-9dad-11d1-80b4-00c04fd430c8"].only_labels == [3]


# ---- PerUserSpecStrategy partitioning ----

def test_per_user_fair_share_then_filter():
    # Fair share: each user takes pct% of every class. only_labels drops
    # classes outside the list; label_distribution applies a per-class
    # retention factor in [0, 1] to the fair share.
    #   user 0: 10% pct, only_labels=[4,9] -> ~10 of L4, ~10 of L9
    #   user 1: 10% pct, label_distribution={4:0.5, 9:0.0} -> ~5 of L4, 0 of L9,
    #           full ~10 of every other class (unmentioned defaults to 1.0)
    #   user 2: 20% pct, only_labels=[0,1,2,3,5,6,7,8] -> ~20 per kept class
    spec0 = UserPartitionSpec(user_index=0, data_percent=10.0, only_labels=[4, 9])
    spec1 = UserPartitionSpec(user_index=1, data_percent=10.0, label_distribution={4: 0.5, 9: 0.0})
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

    # User 0: only_labels limits to {4, 9}, ~10 each.
    assert set(counts0.keys()) <= {4, 9}
    assert abs(counts0[4] - 10) <= 1
    assert abs(counts0[9] - 10) <= 1

    # User 1: retention 0.5 on L4, 0.0 on L9, 1.0 on the rest.
    assert abs(counts1[4] - 5) <= 1
    assert counts1[9] == 0
    for cls in (0, 1, 2, 3, 5, 6, 7, 8):
        assert abs(counts1[cls] - 10) <= 1

    # User 2: only_labels filters L4 and L9 out, ~20 per kept class.
    assert counts2[4] == 0
    assert counts2[9] == 0
    for cls in (0, 1, 2, 3, 5, 6, 7, 8):
        assert abs(counts2[cls] - 20) <= 1


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


def test_per_user_no_class_supply_conflict_under_fair_share():
    # Fair-share-then-filter never raises a per-class conflict: each user's
    # per-class demand is at most pct/100 of the class pool, so sum(pct)<=100
    # is the only constraint. Two users restricted to L4 each keep 8 of L4.
    spec0 = UserPartitionSpec(user_index=0, data_percent=8.0, only_labels=[4])
    spec1 = UserPartitionSpec(user_index=1, data_percent=8.0, only_labels=[4])
    users = [FakeUser(0, 8.0, spec0), FakeUser(1, 8.0, spec1)]
    labels = make_labels(n_per_class=100, n_classes=10)

    partitioner = DataPartition(seed=1, per_user_specs={0: spec0, 1: spec1})
    splits = partitioner.split_by_label(users, labels)
    counts0 = class_counts(labels, collect_ids(splits, users[0]))
    counts1 = class_counts(labels, collect_ids(splits, users[1]))
    assert set(counts0.keys()) <= {4}
    assert set(counts1.keys()) <= {4}
    assert counts0[4] == 8
    assert counts1[4] == 8


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
    # 60% of dataset across all 3 classes (whitelist covers everything).
    # Each class slice should be split per the val_split ratio independently,
    # so val keeps the same class distribution as train.
    spec = UserPartitionSpec(
        user_index=0,
        data_percent=60.0,
        only_labels=[0, 1, 2],
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


def test_per_user_overlap_mode_shares_samples():
    # Overlap mode pulls fair-share allocations from the rep-inflated class
    # pool, so two users on the same class can land on the same sample_id.
    # 50% pcts pulling only L4 from a rep=2 pool of 200 = 2 concatenated
    # shuffles of 100. Each user's 100-sample chunk is one full shuffle, so
    # they end up with overlapping base ids.
    spec0 = UserPartitionSpec(user_index=0, data_percent=50.0, only_labels=[4])
    spec1 = UserPartitionSpec(user_index=1, data_percent=50.0, only_labels=[4])
    users = [FakeUser(0, 50.0, spec0), FakeUser(1, 50.0, spec1)]
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
    assert set(counts0.keys()) <= {4}
    assert set(counts1.keys()) <= {4}
    # Combined demand 120 of L4 with only 100 distinct base ids forces overlap.
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
