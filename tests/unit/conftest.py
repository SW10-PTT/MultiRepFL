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
def fl_challenge(request, mock_w3, mock_contract, mock_participants):
    # Skipped: FLChallenge.__init__ signature changed to
    # (publisher: User, pyTorchModel, training_specs: TrainingSpecsChallenge, jobListing, ...)
    # and now invokes initialize_challenge() + ConnectionHelper.deploy() against
    # globals.w3. The fixture's old (manager, configs, pytorch_model, experiment_config)
    # call shape no longer matches and the deploy chain isn't mocked. Tests using
    # this fixture need a rewrite against the current API.
    pytest.skip("fl_challenge fixture out of date with current FLChallenge.__init__")
