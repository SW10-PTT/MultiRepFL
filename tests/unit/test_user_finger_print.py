from openfl.utils.types.Attitude import Attitude
from openfl.utils.types.User import User


# User.__init__ pulls in collateral RNG and color setup that aren't relevant here.
# Build via __new__ and only set the attrs finger_print actually reads.
def make_user(**overrides):
    u = User.__new__(User)
    u.futureAttitude = Attitude.Honest
    u.attitudeSwitch = 1
    u.min_collateral = 0
    u.max_collateral = 0
    u.data_percent = 50.0
    u.only_labels = None
    u.flip_map = {}
    u.seed = 42
    for key, value in overrides.items():
        setattr(u, key, value)
    return u


def test_finger_print_is_deterministic():
    a = make_user()
    b = make_user()
    assert a.finger_print == b.finger_print


def test_finger_print_changes_with_seed():
    a = make_user(seed=1)
    b = make_user(seed=2)
    assert a.finger_print != b.finger_print


def test_finger_print_changes_with_data_percent():
    a = make_user(data_percent=50.0)
    b = make_user(data_percent=50.00000001)
    # Rounded to 8 decimals so trivially small drift survives, but real changes shift the hash.
    assert a.finger_print != b.finger_print


def test_finger_print_order_invariant_flip_map():
    # dict insertion order must not affect the hash.
    a = make_user(flip_map={4: 9, 2: 5})
    b = make_user(flip_map={2: 5, 4: 9})
    assert a.finger_print == b.finger_print


def test_finger_print_order_invariant_only_labels():
    # only_labels order must not affect the hash.
    a = make_user(only_labels=[3, 1, 2])
    b = make_user(only_labels=[1, 2, 3])
    assert a.finger_print == b.finger_print


def test_finger_print_changes_with_flip_map_content():
    a = make_user(flip_map={4: 9})
    b = make_user(flip_map={4: 8})
    assert a.finger_print != b.finger_print
