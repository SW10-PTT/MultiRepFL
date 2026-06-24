"""Unit tests for pure value types: Attitude, TaskType / TrainingSpecs*,
EvaluationData, and the Colors helpers.

These carry no web3/torch state. The TrainingSpecs tests deliberately pin the
positional tuple ordering of ``to_solidity_*`` because those tuples are the ABI
contract with the Solidity side — a reordering here would silently corrupt
on-chain deploys.
"""

import numpy as np
import pytest

from openfl.utils.types.Attitude import Attitude
from openfl.utils.types import Colors
from openfl.utils.types.AddressIndexList import AddressIndexList
from openfl.utils.types.AddressIndexMatrix import AddressIndexMatrix
from openfl.utils.types.EvaluationData import EvaluationData
from openfl.utils.types.TrainingSpecsJobListing import (
    TaskType,
    TrainingSpecsJobListing,
    TrainingSpecsChallenge,
)


# --------------------------------------------------------------------------- #
# Attitude
# --------------------------------------------------------------------------- #
def test_attitude_ordinal_values():
    # Ordinals must match the Solidity-facing assumptions (Honest=0 ... Inactive=3).
    assert (Attitude.Honest.value, Attitude.FreeRider.value,
            Attitude.Malicious.value, Attitude.Inactive.value) == (0, 1, 2, 3)


def test_from_string_passthrough_enum_member():
    assert Attitude.from_string(Attitude.Malicious) is Attitude.Malicious


@pytest.mark.parametrize("text,expected", [
    ("Honest", Attitude.Honest),
    ("FreeRider", Attitude.FreeRider),
    ("Malicious", Attitude.Malicious),
    ("Inactive", Attitude.Inactive),
    ("Attitude.FreeRider", Attitude.FreeRider),  # dotted form is stripped to the tail
])
def test_from_string_parses_names(text, expected):
    assert Attitude.from_string(text) is expected


def test_from_string_invalid_raises():
    with pytest.raises(ValueError):
        Attitude.from_string("Saboteur")


# --------------------------------------------------------------------------- #
# TaskType
# --------------------------------------------------------------------------- #
def test_tasktype_is_intenum_with_expected_ordinals():
    assert TaskType.template == 0
    assert TaskType.MNIST == 5
    assert TaskType.CIFAR10 == 6
    assert int(TaskType.IMDB) == 8


def test_from_dataset_name_none_is_template():
    assert TaskType.from_dataset_name(None) is TaskType.template


@pytest.mark.parametrize("name,expected", [
    ("MNIST", TaskType.MNIST),
    ("mnist", TaskType.MNIST),
    ("CIFAR-10", TaskType.CIFAR10),
    ("cifar_10", TaskType.CIFAR10),
    ("Fashion MNIST", TaskType.FashionMNIST),
    ("imdb", TaskType.IMDB),
])
def test_from_dataset_name_normalises_separators_and_case(name, expected):
    assert TaskType.from_dataset_name(name) is expected


def test_from_dataset_name_unknown_falls_back_to_template():
    assert TaskType.from_dataset_name("svhn") is TaskType.template


# --------------------------------------------------------------------------- #
# TrainingSpecsJobListing / TrainingSpecsChallenge — Solidity tuple contract
# --------------------------------------------------------------------------- #
def _make_job_spec():
    return TrainingSpecsJobListing(
        modelHash=b"hash",
        min_collateral=10,
        max_collateral=20,
        manager_address="0xMgr",
        reward=5,
        min_rounds=3,
        punishfactor=2,
        punishfactorContrib=4,
        freeriderPenalty=1,
        taskType=6,
    )


def test_to_solidity_job_tuple_order():
    spec = _make_job_spec()
    assert spec.to_solidity_job() == (
        b"hash", 10, 20, "0xMgr", 5, 3, 2, 4, 1, 6,
    )


def test_to_challenge_propagates_fields_and_sets_challenge_specifics():
    spec = _make_job_spec()
    challenge = spec.to_challenge(
        contribution_score_strategy="dotproduct",
        outlier_detection=True,
        joblisting_address="0xJob",
        loss_tolerance_pct=0.2,
    )

    assert isinstance(challenge, TrainingSpecsChallenge)
    # Carried over from the job spec.
    assert challenge.modelHash == b"hash"
    assert challenge.taskType == 6
    assert challenge.reward == 5
    # Challenge-only fields.
    assert challenge.contribution_score_strategy == "dotproduct"
    assert challenge.outlier_detection is True
    assert challenge.joblisting_address == "0xJob"
    assert challenge.loss_tolerance_pct == 0.2


def test_to_solidity_challenge_tuple_order_with_default_tunables():
    challenge = _make_job_spec().to_challenge(
        contribution_score_strategy="naive",
        outlier_detection=False,
        joblisting_address="0xJob",
    )
    assert challenge.to_solidity_challenge() == (
        b"hash", 10, 20, "0xMgr", 5, 3, 2, 4, 1, 6, "0xJob",
        int(2e17),  # tr_alpha default
        int(2e17),  # tr_n_blend default
        2,          # tr_n_0 default
        5,          # tr_lambda default
        int(2e17),  # tr_integrity_learning_rate default
        2,          # tr_gain_cap_multiplier default
    )


# --------------------------------------------------------------------------- #
# EvaluationData
# --------------------------------------------------------------------------- #
def test_new_builds_matrices_and_get_reads_feedback(make_participant_stub):
    participants = [make_participant_stub(i, id=f"u{i}", label=f"W{i}") for i in range(2)]
    ed = EvaluationData.new(participants)

    # Positional write avoids the int-id setter; row 0 == votes from u0.
    ed.feedback_matrix[(0, 1)] = 2

    votes = ed.get("u0")
    assert votes.feedback == {"u0": 0, "u1": 2}
    # accuracy/loss matrices were built (not None) -> dicts of defaults.
    assert votes.accuracy == {"u0": 0, "u1": 0}
    assert votes.loss == {"u0": 0, "u1": 0}
    assert ed.get_user_id(1) == "u1"


def test_get_handles_optional_none_matrices():
    # When accuracy/loss matrices are absent, get() must short-circuit to None
    # rather than indexing a missing matrix.
    id_to_idx = {"a": 0, "b": 1}
    feedback = AddressIndexMatrix(external_address_list=id_to_idx, np_int_type=np.int8)
    feedback[(0, 1)] = -1
    prev_acc = AddressIndexList(external_address_list=id_to_idx)
    prev_loss = AddressIndexList(external_address_list=id_to_idx)

    ed = EvaluationData(
        id_to_idx=id_to_idx,
        feedback_matrix=feedback,
        accuracy_matrix=None,
        loss_matrix=None,
        prev_accuracies=prev_acc,
        prev_losses=prev_loss,
    )

    votes = ed.get("a")
    assert votes.feedback == {"a": 0, "b": -1}
    assert votes.accuracy is None
    assert votes.loss is None
    assert votes.prev_accuracy == 0
    assert votes.prev_loss == 0


# --------------------------------------------------------------------------- #
# Colors
# --------------------------------------------------------------------------- #
def test_get_color_special_attitudes():
    assert Colors.get_color(0, "bad") == Colors.bad_c
    assert Colors.get_color(0, "freerider") == Colors.free_c


def test_get_color_none_index_returns_none():
    assert Colors.get_color(None, "good") is None


def test_get_color_valid_index_returns_palette_entry():
    color = Colors.get_color(0, "good")
    assert isinstance(color, str) and color.startswith("#")


def test_get_color_out_of_range_index_returns_none():
    assert Colors.get_color(10_000, "good") is None


@pytest.mark.parametrize("fn", [Colors.green, Colors.gb, Colors.rb, Colors.b, Colors.red, Colors.yellow])
def test_color_text_helpers_wrap_input(fn):
    # termcolor wraps the text in ANSI codes but the original text survives.
    assert "hello" in fn("hello")
