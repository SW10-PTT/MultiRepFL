"""Unit tests for the AddressIndexMatrix custom collection.

AddressIndexMatrix is an N×N numpy matrix indexable positionally (int) or by
participant id (via a lookup dict). Used for the feedback/accuracy/loss matrices
in EvaluationData. Pure value type — constructed directly in these tests.

Unlike AddressIndexList, the matrix's ``__setitem__`` has a proper int/else
split, so both the positional and id-lookup write paths are safe to exercise.
"""

import numpy as np
import pytest

from openfl.utils.types.AddressIndexMatrix import AddressIndexMatrix


def _string_id_participants(make_participant_stub, n):
    # String ids force the id-lookup branch (an int id would route through the
    # positional branch instead).
    return [make_participant_stub(i, id=f"u{i}", label=f"W{i}") for i in range(n)]


def test_requires_exactly_one_source_none():
    with pytest.raises(TypeError):
        AddressIndexMatrix()


def test_rejects_both_sources(make_participant_stub):
    participants = [make_participant_stub(0, id="u0")]
    with pytest.raises(TypeError):
        AddressIndexMatrix(participants=participants, external_address_list={"u0": 0})


def test_initialises_square_zero_matrix(make_participant_stub):
    participants = _string_id_participants(make_participant_stub, 3)
    mat = AddressIndexMatrix(participants=participants)

    normal = mat.get_as_normal_int()
    assert normal.shape == (3, 3)
    assert normal.sum() == 0


def test_setitem_getitem_by_id_lookup(make_participant_stub):
    participants = _string_id_participants(make_participant_stub, 3)
    mat = AddressIndexMatrix(participants=participants)

    mat[("u0", "u2")] = 5

    # id-keyed read and the equivalent positional read agree.
    assert int(mat[("u0", "u2")]) == 5
    assert int(mat[(0, 2)]) == 5


def test_getitem_row_by_id_and_by_index(make_participant_stub):
    participants = _string_id_participants(make_participant_stub, 2)
    mat = AddressIndexMatrix(participants=participants)
    mat[("u1", "u0")] = 3

    row_by_id = mat["u1"]
    row_by_index = mat[1]

    assert list(row_by_id) == list(row_by_index)
    assert int(row_by_id[0]) == 3


def test_setitem_clamps_upper_bound_only_for_signed_dtype(make_participant_stub):
    # int8 range is [-128, 127]; min(value, 127) clamps the top but lets
    # negative feedback values through unchanged.
    participants = _string_id_participants(make_participant_stub, 2)
    mat = AddressIndexMatrix(participants=participants, np_int_type=np.int8)

    mat[(0, 1)] = 200    # above int8 max -> clamped
    mat[(1, 0)] = -5     # negative -> passes through
    mat[(0, 0)] = 50     # within range

    assert int(mat[(0, 1)]) == 127
    assert int(mat[(1, 0)]) == -5
    assert int(mat[(0, 0)]) == 50


def test_mixed_key_types_raise_typeerror(make_participant_stub):
    participants = _string_id_participants(make_participant_stub, 2)
    mat = AddressIndexMatrix(participants=participants)

    with pytest.raises(TypeError):
        _ = mat[(0, "u1")]   # int giver, str receiver -> disallowed


def test_get_user_address_returns_id(make_participant_stub):
    participants = _string_id_participants(make_participant_stub, 3)
    mat = AddressIndexMatrix(participants=participants)
    assert mat.get_user_address(2) == "u2"


def test_get_as_normal_int_for_scalar_cell(make_participant_stub):
    participants = _string_id_participants(make_participant_stub, 2)
    mat = AddressIndexMatrix(participants=participants)
    mat[(0, 1)] = 8

    assert int(mat.get_as_normal_int((0, 1))) == 8


def test_str_empty_matrix_is_marked_empty():
    mat = AddressIndexMatrix(external_address_list={})
    assert str(mat) == "(empty)"


def test_str_and_csv_render_nonzero_cells(make_participant_stub):
    participants = _string_id_participants(make_participant_stub, 2)
    mat = AddressIndexMatrix(participants=participants)
    mat[("u0", "u1")] = 4

    rendered = str(mat)
    csv = mat.to_csv_cell()

    # Header/labels present; only the non-zero cell appears in the CSV cell.
    assert "W0" in rendered
    assert csv.startswith("[") and csv.endswith("]")
    # Only the single non-zero cell is emitted (zero cells are skipped).
    assert csv == "[(W0 (u0),W1 (u1),4)]"
