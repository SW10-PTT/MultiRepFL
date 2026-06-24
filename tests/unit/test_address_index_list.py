"""Unit tests for the AddressIndexList custom collection.

AddressIndexList wraps a fixed-width numpy vector and lets callers index it
either positionally (int) or by a participant address/id (via a lookup dict).
It is a pure, side-effect-free value type — no web3/torch — so these tests
construct it directly.

NOTE: the int-key path of ``__setitem__`` is a known latent bug (it writes the
positional slot and then *also* falls through to an address lookup, raising
KeyError). Per the agreed plan we do NOT exercise that path here; it is written
up in the bug report instead. Writes are therefore only tested via address keys.
"""

import numpy as np
import pytest

from openfl.utils.types.AddressIndexList import AddressIndexList


def test_requires_exactly_one_source_none():
    # Neither participants nor an external address list -> ambiguous, must raise.
    with pytest.raises(TypeError):
        AddressIndexList()


def test_rejects_both_sources(make_participant_stub):
    # Supplying both inputs is contradictory and must raise.
    participants = [make_participant_stub(0)]
    with pytest.raises(TypeError):
        AddressIndexList(participants=participants, external_address_list={"0xAddr0": 0})


def test_initialises_zeroed_vector_from_participants(make_participant_stub):
    participants = [make_participant_stub(i) for i in range(3)]
    lst = AddressIndexList(participants=participants)

    # Fresh list starts all-zero with one slot per participant.
    assert lst.get_as_normal_int() == [0, 0, 0]


def test_getitem_by_positional_index_and_by_address(make_participant_stub):
    participants = [make_participant_stub(i) for i in range(3)]
    lst = AddressIndexList(participants=participants)

    # Seed slot 1 through the address key, then read it back both ways.
    lst[participants[1].address] = 7

    assert int(lst[1]) == 7                          # positional read
    assert int(lst[participants[1].address]) == 7    # address read


def test_setitem_by_address_clamps_to_dtype_max(make_participant_stub):
    # uint8 caps at 255; min(value, max) clamps the UPPER bound only.
    participants = [make_participant_stub(i) for i in range(2)]
    lst = AddressIndexList(participants=participants, np_int_type=np.uint8)

    lst[participants[0].address] = 300   # above uint8 max
    lst[participants[1].address] = 42    # within range

    assert int(lst[0]) == 255
    assert int(lst[1]) == 42


def test_unknown_address_raises_keyerror(make_participant_stub):
    participants = [make_participant_stub(i) for i in range(2)]
    lst = AddressIndexList(participants=participants)

    with pytest.raises(KeyError):
        _ = lst["0xNotAParticipant"]


def test_get_as_normal_int_scalar_for_key(make_participant_stub):
    participants = [make_participant_stub(i) for i in range(2)]
    lst = AddressIndexList(participants=participants)
    lst[participants[0].address] = 5

    value = lst.get_as_normal_int(participants[0].address)
    # Returns a native python int, not a numpy scalar.
    assert value == 5
    assert isinstance(value, int)


def test_external_address_list_uses_labels(make_participant_stub):
    # EvaluationData builds these from an id-keyed dict + id->label map; in that
    # configuration the label lookup resolves (idx->id->label).
    participants = [make_participant_stub(i, label=f"Worker {i}") for i in range(2)]
    id_to_idx = {p.id: i for i, p in enumerate(participants)}
    id_to_label = {p.id: p.display_label() for p in participants}

    lst = AddressIndexList(external_address_list=id_to_idx, id_to_label=id_to_label)

    assert lst._label(0) == "Worker 0"
    assert lst._full_label(1) == "Worker 1 (1)"


def test_str_empty_list_is_marked_empty():
    lst = AddressIndexList(external_address_list={})
    assert str(lst) == "(empty)"


def test_str_and_csv_render_values(make_participant_stub):
    participants = [make_participant_stub(i, label=f"W{i}") for i in range(2)]
    id_to_idx = {p.id: i for i, p in enumerate(participants)}
    id_to_label = {p.id: p.display_label() for p in participants}
    lst = AddressIndexList(external_address_list=id_to_idx, id_to_label=id_to_label)
    # Seed the backing array directly: in the id-keyed configuration every key
    # is an int, which would route through the buggy int-key __setitem__ path
    # (see bug report). Rendering is what we're exercising here, not the setter.
    lst._list[0] = 9

    rendered = str(lst)
    csv = lst.to_csv_cell()

    assert "W0" in rendered
    assert csv.startswith("[") and csv.endswith("]")
    assert "(W0 (0), 9)" in csv
