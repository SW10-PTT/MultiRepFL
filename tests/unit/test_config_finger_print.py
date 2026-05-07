from types import SimpleNamespace

from experiment.experiment_configuration import ExperimentConfiguration


# get_finger_print iterates participants and reads .finger_print on each.
# Return a plain list of stub participants — no web3/contracts dependency needed.
def make_participants(*participant_fingerprints):
    return [SimpleNamespace(finger_print=fp) for fp in participant_fingerprints]


def make_config(**overrides):
    defaults = dict(
        number_of_good_contributors=1,
        number_of_bad_contributors=0,
        number_of_freerider_contributors=0,
        seed=42,
        allow_overlap=True,
        replication_factor=2.0,
    )
    defaults.update(overrides)
    return ExperimentConfiguration(**defaults)


def test_finger_print_is_deterministic():
    cfg = make_config()
    stub = make_participants("hash_a", "hash_b", "hash_c")
    assert cfg.get_finger_print(stub) == cfg.get_finger_print(stub)


def test_finger_print_order_invariant_across_participants():
    # Sorted internally, so participant order in the list must not affect the hash.
    cfg = make_config()
    a = make_participants("hash_a", "hash_b", "hash_c")
    b = make_participants("hash_c", "hash_a", "hash_b")
    assert cfg.get_finger_print(a) == cfg.get_finger_print(b)


def test_finger_print_changes_with_seed():
    a = make_config(seed=1).get_finger_print(make_participants("h1"))
    b = make_config(seed=2).get_finger_print(make_participants("h1"))
    assert a != b


def test_finger_print_changes_with_allow_overlap():
    a = make_config(allow_overlap=True, replication_factor=2.0).get_finger_print(make_participants("h1"))
    b = make_config(allow_overlap=False, replication_factor=1.0).get_finger_print(make_participants("h1"))
    assert a != b


def test_finger_print_changes_with_replication_factor():
    a = make_config(allow_overlap=True, replication_factor=1.5).get_finger_print(make_participants("h1"))
    b = make_config(allow_overlap=True, replication_factor=2.0).get_finger_print(make_participants("h1"))
    assert a != b


def test_finger_print_changes_with_user_seeds():
    a = make_config(user_seeds={0: 100}).get_finger_print(make_participants("h1"))
    b = make_config(user_seeds={0: 200}).get_finger_print(make_participants("h1"))
    assert a != b


def test_finger_print_user_seeds_order_invariant():
    # Insertion order of user_seeds dict must not affect the hash.
    a = make_config(user_seeds={2: 200, 0: 100}).get_finger_print(make_participants("h1"))
    b = make_config(user_seeds={0: 100, 2: 200}).get_finger_print(make_participants("h1"))
    assert a == b


def test_finger_print_changes_with_participants():
    cfg = make_config()
    a = cfg.get_finger_print(make_participants("h1", "h2"))
    b = cfg.get_finger_print(make_participants("h1", "h3"))
    assert a != b
