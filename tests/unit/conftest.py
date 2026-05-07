import os
import random
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pytest
import types



from openfl.contracts.FLChallenge import FLChallenge


@pytest.fixture
def mock_w3():
    """Mocks a web3 connection to Ethereum."""
    w3 = MagicMock()

    mock_receipt = MagicMock()
    mock_receipt.gasUsed = 21000
    mock_receipt.transactionHash = b'\x00' * 32
    mock_receipt.logs = []

    mock_receipt.__getitem__.side_effect = lambda x: getattr(mock_receipt, x)

    w3.eth.get_transaction_count.return_value = 10
    w3.eth.get_balance.return_value = 1000000000000000000
    w3.eth.wait_for_transaction_receipt.return_value = mock_receipt

    w3.to_checksum_address.side_effect = lambda x: x
    return w3


@pytest.fixture
def mock_contract():
    """Mocks a Solidity Smart Contract."""
    contract = MagicMock()
    contract.functions.register.return_value.transact.return_value = b"\x01" * 32
    contract.functions.feedback.return_value.transact.return_value = b"\x02" * 32
    contract.functions.closeRound.return_value.transact.return_value = b"\x03" * 32
    contract.functions.rewardLeft.return_value.call.return_value = 5000
    return contract


@pytest.fixture
def mock_participants():
    """Create a list of 3 dummy participants."""
    users = []
    for i in range(3):
        user = MagicMock()
        user.address = f"0xAddressUser{i}"
        user.privateKey = f"privateKey{i}"
        user.collateral = 1000
        user.isRegistered = False
        user.attitude = "honest"
        user.cheater = []
        user.id = i
        user.hashedModel = b'hash'
        # display_label is consumed by f-string format specs (e.g. ":<12") and a
        # bare MagicMock can't be __format__'d — pin it to a real string.
        user.display_label.return_value = f"user{i}"

        # Give unique secrets to ensure filtering tests work correctly
        user.secret = 100 + i

        users.append(user)
    return users


@pytest.fixture
def mock_participants_with_values():
    """Create a list of 6 dummy participants with randomly generated accuracies and losses."""
    users = []
    for i in range(6):
        user = MagicMock()
        user.address = f"0xAddressUser{i}"
        user.privateKey = f"privateKey{i}"
        user.collateral = 1000
        user.isRegistered = False
        user.attitude = "honest"
        user.cheater = []
        user.id = i
        user.hashedModel = b'hash'
        user.secret = 100 + i

        # Randomly generated accuracies (60-99) and losses (5-20)
        # with one random outlier injected
        normal_accuracies = [random.randint(60, 99) for _ in range(4)]
        normal_losses     = [random.randint(5, 20)  for _ in range(4)]

        outlier_idx = random.randint(0, 4)
        normal_accuracies.insert(outlier_idx, 0)        # outlier: accuracy = 0
        normal_losses.insert(outlier_idx, 10000)        # outlier: loss = 10000

        user._accuracies = normal_accuracies
        user._losses     = normal_losses

        users.append(user)
    return users


@pytest.fixture
def fl_challenge(request, mock_w3, mock_contract, mock_participants, monkeypatch):
    # FLChallenge.__init__ now triggers blockchain deploy via globals.w3. Bypass
    # __init__ via __new__ and wire only the fields the tests need. Patch
    # openfl.api.globals.w3/fork plus the FLChallenge module's bare `fork` and
    # `w3` references used in give_feedback. Note: `openfl.contracts.FLChallenge`
    # via `import` resolves to the re-exported class (see openfl/contracts/__init__.py),
    # so reach into sys.modules to get the actual module.
    import sys
    fl_challenge_module = sys.modules["openfl.contracts.FLChallenge"]
    from openfl.api import globals as openfl_globals

    monkeypatch.setattr(openfl_globals, "w3", mock_w3, raising=False)
    monkeypatch.setattr(openfl_globals, "fork", True, raising=False)
    monkeypatch.setattr(fl_challenge_module, "fork", True, raising=False)
    monkeypatch.setattr(fl_challenge_module, "w3", mock_w3, raising=False)

    challenge = FLChallenge.__new__(FLChallenge)
    challenge.contract = mock_contract
    challenge.contractAddress = "0xModelAddress"
    # Legacy alias: tests written against old field name; production code uses .contract.
    challenge.model = mock_contract

    pytorch_model = MagicMock()
    pytorch_model.participants = mock_participants
    pytorch_model.round = 1
    pytorch_model.disqualified = []
    challenge.pytorch_model = pytorch_model

    challenge.MIN_BUY_IN = 100
    challenge.MAX_BUY_IN = 1000
    challenge.REWARD = 500
    challenge.MIN_ROUNDS = 3
    challenge.PUNISHMENT_FACTOR = 0.5
    challenge.PUNISHMENT_FACTOR_CONTRIB = 3
    challenge.FREERIDER_FACTOR = 0.1
    challenge.scores = []
    challenge.gas_feedback = []
    challenge.gas_register = []
    challenge.gas_slot = []
    challenge.gas_weights = []
    challenge.gas_close = []
    challenge.gas_deploy = []
    challenge.gas_exit = []
    challenge.txHashes = []
    challenge._reward_balance = []
    challenge._punishments = []
    challenge.disqualifiedUserEvents = []
    challenge.writeTxProgress = 0

    experiment_config = getattr(
        request, "param", SimpleNamespace(
            contribution_score_strategy="dotproduct",
            use_outlier_detection=False,
        )
    )
    challenge.experiment_config = experiment_config
    challenge._contribution_score_strategy = experiment_config.contribution_score_strategy
    challenge.contribution_score_strategy = experiment_config.contribution_score_strategy
    challenge.use_outlier_detection = experiment_config.use_outlier_detection
    challenge.loss_tolerance_pct = 0.05

    challenge._contribution_score_calculators = {
        "dotproduct": challenge._calculate_scores_dotproduct,
        "naive": challenge._calculate_scores_naive,
        "accuracy_loss": challenge._calculate_scores_accuracy_loss,
        "accuracy_only": challenge._calculate_scores_accuracy_only,
        "loss_only": challenge._calculate_scores_loss_only,
        "loss_tolerance_aware": challenge._calculate_scores_loss_tolerance_aware,
        "loss_tolerance_snap": challenge._calculate_scores_loss_tolerance_snap,
    }

    challenge.config = SimpleNamespace(WAIT_DELAY=86400, FEEDBACK_ROUND_TIMEOUT=0, CONTRIBUTION_ROUND_TIMEOUT=0)
    challenge.writer = MagicMock()
    challenge._logger = MagicMock()
    challenge.w3 = mock_w3

    challenge.get_global_reputation_of_user = MagicMock(return_value=1000)
    challenge.build_tx = MagicMock(return_value={"gas": 100000, "gasPrice": 1, "nonce": 1})
    challenge.build_non_fork_tx = MagicMock(return_value={"gas": 100000, "nonce": 1})
    challenge._log_receipt = MagicMock()
    challenge._log_warning = MagicMock()
    challenge._log_contribution_scores = MagicMock()

    yield challenge
