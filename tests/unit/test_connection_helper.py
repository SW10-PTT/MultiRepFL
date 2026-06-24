"""Unit tests for the pure / mockable surface of ConnectionHelper.

ConnectionHelper wraps a web3 node. The connection-bootstrap paths
(initiate_rpc/initiate_connection) shell out to Linux tooling and poll a live
RPC, so they stay out of scope here. These tests cover the transaction-building
and receipt-processing helpers, driving them through a mocked ``globals.w3`` and
``globals.fork`` — the established pattern in this repo.

Only the fork code path of transact is tested; the non-fork path calls
``super().build_non_fork_tx`` (super() is object) and is a known bug — see the
bug report.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import openfl.api.globals as g
from openfl.api.ConnectionHelper import ConnectionHelper


ZERO_ADDR = "0x0000000000000000000000000000000000000000"


@pytest.fixture
def w3(monkeypatch):
    """A mocked web3 connection wired onto globals.w3 (fork mode by default)."""
    mock = MagicMock()
    mock.to_checksum_address.side_effect = lambda a: a
    monkeypatch.setattr(g, "w3", mock)
    monkeypatch.setattr(g, "fork", True)
    return mock


# --------------------------------------------------------------------------- #
# build_tx
# --------------------------------------------------------------------------- #
def test_build_tx_checksums_and_shapes_dict(w3):
    w3.to_checksum_address.side_effect = lambda a: a.upper()
    ch = ConnectionHelper()

    tx = ch.build_tx("0xfrom", "0xto", 5)

    assert tx == {"from": "0XFROM", "to": "0XTO", "value": 5}


def test_build_tx_rejects_zero_to_address(w3):
    ch = ConnectionHelper()
    with pytest.raises(AssertionError):
        ch.build_tx("0xfrom", ZERO_ADDR)


# --------------------------------------------------------------------------- #
# build_non_fork_tx
# --------------------------------------------------------------------------- #
@pytest.fixture
def ch_non_fork(monkeypatch, w3):
    w3.eth.chain_id = 1337
    w3.to_wei.side_effect = lambda v, unit: v
    w3.from_wei.side_effect = lambda v, unit: v
    w3.eth.get_balance.side_effect = lambda addr: 10 ** 30  # plenty, no warning
    return ConnectionHelper()


def test_build_non_fork_tx_data_branch(ch_non_fork):
    tx = ch_non_fork.build_non_fork_tx("0xa", nonce=7, to="0xb", data=b"\x01\x02")
    assert tx["data"] == b"\x01\x02"
    assert tx["to"] == "0xb"
    assert tx["chainId"] == 1337
    assert tx["gas"] == 5_000_000          # default gas budget
    assert tx["nonce"] == 7


def test_build_non_fork_tx_to_branch_without_data(ch_non_fork):
    tx = ch_non_fork.build_non_fork_tx("0xa", nonce=1, to="0xb")
    assert tx["to"] == "0xb"
    assert "data" not in tx


def test_build_non_fork_tx_default_branch_has_no_to(ch_non_fork):
    tx = ch_non_fork.build_non_fork_tx("0xa", nonce=1)
    assert "to" not in tx
    assert tx["from"] == "0xa"


def test_build_non_fork_tx_honours_gas_override(ch_non_fork):
    tx = ch_non_fork.build_non_fork_tx("0xa", nonce=1, gas_limit=123)
    assert tx["gas"] == 123


def test_build_non_fork_tx_warns_but_returns_on_low_balance(w3, capsys):
    w3.eth.chain_id = 1337
    w3.to_wei.side_effect = lambda v, unit: v
    w3.from_wei.side_effect = lambda v, unit: v
    w3.eth.get_balance.side_effect = lambda addr: 0  # below estimated gas cost
    ch = ConnectionHelper()

    result = ch.build_non_fork_tx("0xa", nonce=2)

    assert result["nonce"] == 2
    assert "Warning" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# batch_process_receipt
# --------------------------------------------------------------------------- #
def test_batch_process_receipt_records_gas_and_returns_events(w3, monkeypatch):
    add_gas = MagicMock()
    monkeypatch.setattr(g, "add_gas_usage", add_gas)
    ch = ConnectionHelper()
    receipt = {"gasUsed": 21000, "status": 1}

    out = ch.batch_process_receipt(receipt, [], "gas.tag", "0xAccount")

    assert out == (receipt, {})
    add_gas.assert_called_once_with("gas.tag", 21000, "0xAccount")


def test_batch_process_receipt_raises_on_failed_status(w3, monkeypatch):
    monkeypatch.setattr(g, "add_gas_usage", MagicMock())
    ch = ConnectionHelper()
    receipt = {"gasUsed": 1, "status": 0, "transactionHash": b"\xab" * 32}

    with pytest.raises(RuntimeError) as exc:
        ch.batch_process_receipt(receipt, [], "gas.tag", "0xAccount", func_name="register")

    # Failure message surfaces the func name and the hex tx hash.
    assert "register" in str(exc.value)
    assert (b"\xab" * 32).hex() in str(exc.value)


# --------------------------------------------------------------------------- #
# get_events
# --------------------------------------------------------------------------- #
def test_get_events_empty_names_returns_empty_dict(w3):
    ch = ConnectionHelper()
    assert ch.get_events(MagicMock(), []) == {}


def test_get_events_filters_logs_by_signature(w3):
    ch = ConnectionHelper()

    event_obj = MagicMock()
    event_obj.abi = {"inputs": [{"type": "uint256"}]}
    event_obj.process_log.return_value = {"args": {"foo": 1}}
    ch.contract = MagicMock()
    ch.contract.events.MyEvent.return_value = event_obj

    w3.keccak.return_value.hex.return_value = "0xsig"

    matching = MagicMock()
    matching.__getitem__.return_value = [SimpleNamespace(hex=lambda: "0xsig")]
    nonmatching = MagicMock()
    nonmatching.__getitem__.return_value = [SimpleNamespace(hex=lambda: "0xother")]
    receipt = MagicMock()
    receipt.logs = [matching, nonmatching]

    result = ch.get_events(receipt, ["MyEvent"])

    # Only the log whose topic[0] matches the event signature is decoded.
    assert result == {"MyEvent": [{"foo": 1}]}


# --------------------------------------------------------------------------- #
# transact (fork path) + gas-type guard
# --------------------------------------------------------------------------- #
def test_transact_raw_rejects_non_string_gas_type(w3):
    ch = ConnectionHelper()
    ch.contract = MagicMock()
    with pytest.raises(Exception):
        ch.transact_raw_addreses("register", "0xacc", "pk", 0, [], 123)


def test_transact_fork_path_sends_and_processes_receipt(w3, monkeypatch):
    monkeypatch.setattr(g, "add_gas_usage", MagicMock())
    ch = ConnectionHelper()
    ch.contract = MagicMock()
    ch.contract.address = "0xModel"
    ch.contract.functions.register.return_value.transact.return_value = b"\x01" * 32

    receipt = {"gasUsed": 21000, "status": 1}
    w3.eth.wait_for_transaction_receipt.return_value = receipt

    account = SimpleNamespace(address="0xAccount", privateKey="pk")
    out = ch.transact("register", account, 100, [], "gas.register", "arg0")

    assert out == (receipt, {})
    # The contract function was invoked with the forwarded arg and a fork tx.
    ch.contract.functions.register.assert_called_once_with("arg0")
    ch.contract.functions.register.return_value.transact.assert_called_once()


# --------------------------------------------------------------------------- #
# deploy (classmethod-style; first positional arg is the factory)
# --------------------------------------------------------------------------- #
class _Receipt(dict):
    """web3 receipts behave as dicts AND expose .contractAddress."""
    def __init__(self, data, contract_address):
        super().__init__(data)
        self.contractAddress = contract_address


def test_deploy_fork_returns_contract_and_receipt(w3):
    factory = MagicMock()
    factory.abi = []
    factory.constructor.return_value.transact.return_value = b"\xaa" * 32

    receipt = _Receipt({"status": 1, "gasUsed": 50000}, "0x" + "ab" * 20)
    w3.eth.wait_for_transaction_receipt.return_value = receipt
    w3.eth.contract.return_value = "deployed-contract"

    sender = SimpleNamespace(address="0xSender")
    contract, returned_receipt = ConnectionHelper.deploy(factory, [1, 2], sender)

    assert contract == "deployed-contract"
    assert returned_receipt is receipt
    factory.constructor.assert_called_once_with(1, 2)


def test_deploy_raises_on_failed_status(w3):
    factory = MagicMock()
    factory.constructor.return_value.transact.return_value = b"\xaa" * 32
    receipt = _Receipt({"status": 0}, "0x" + "ab" * 20)
    w3.eth.wait_for_transaction_receipt.return_value = receipt

    sender = SimpleNamespace(address="0xSender")
    with pytest.raises(RuntimeError):
        ConnectionHelper.deploy(factory, [], sender)
