"""Unit tests for the web3 glue helpers in openfl.utils.W3Helper and the gas
accounting in openfl.api.globals.

Both modules lean on the process-global ``globals.w3`` / ``globals.gas_used``.
Every test snapshots/replaces that state via monkeypatch so nothing leaks across
tests, and no real RPC connection is ever made (Web3/Account are stubbed).

NOTE on add_gas_usage: the leaf-append logic currently sits *inside* the
keys[:-1] loop, so single-segment keys record nothing and 3+-segment keys
double-record. Those are real bugs (see bug report) and are NOT exercised here;
only the two-segment case (where buggy == intended) and the type-guard error
paths are tested.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import openfl.api.globals as g
import openfl.utils.W3Helper as w3h


# --------------------------------------------------------------------------- #
# W3Helper.get_w3 / get_RPC_Endpoint
# --------------------------------------------------------------------------- #
def test_get_w3_returns_existing_connection(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(g, "w3", sentinel)
    assert w3h.get_w3() is sentinel  # no new Web3 constructed


def test_get_w3_constructs_and_caches_when_absent(monkeypatch):
    monkeypatch.setattr(g, "w3", None)
    monkeypatch.setattr(w3h, "require_env_var", lambda key: "http://rpc.example:8545")

    class FakeWeb3:
        @staticmethod
        def HTTPProvider(url):
            return ("provider", url)

        def __init__(self, provider):
            self.provider = provider

    monkeypatch.setattr(w3h, "Web3", FakeWeb3)

    result = w3h.get_w3()

    assert isinstance(result, FakeWeb3)
    assert result.provider == ("provider", "http://rpc.example:8545")
    assert g.w3 is result  # cached back onto the module global


def test_get_rpc_endpoint_reads_env(monkeypatch):
    seen = {}

    def fake_require(key):
        seen["key"] = key
        return "http://x"

    monkeypatch.setattr(w3h, "require_env_var", fake_require)
    assert w3h.get_RPC_Endpoint() == "http://x"
    assert seen["key"] == "RPC_URL"


# --------------------------------------------------------------------------- #
# W3Helper.get_PRIVKEYS
# --------------------------------------------------------------------------- #
def test_get_privkeys_returns_none_in_fork_mode(monkeypatch):
    monkeypatch.setattr(g, "w3", object())  # get_w3 short-circuits
    assert w3h.get_PRIVKEYS(SimpleNamespace(fork=True)) is None


def test_get_privkeys_loads_accounts_in_non_fork_mode(monkeypatch):
    monkeypatch.setattr(g, "w3", object())  # avoid constructing a real Web3
    monkeypatch.setattr(w3h, "require_env_var", lambda key: "key1\nkey2")
    monkeypatch.setattr(
        w3h, "Account",
        SimpleNamespace(from_key=lambda k: SimpleNamespace(_private_key=f"raw-{k}", address=f"addr-{k}")),
    )

    result = w3h.get_PRIVKEYS(SimpleNamespace(fork=False))

    assert len(result) == 2
    assert result[0].privateKey == "raw-key1"
    assert result[0].address == "addr-key1"
    assert result[1].address == "addr-key2"


# --------------------------------------------------------------------------- #
# W3Helper.get_account_RPC
# --------------------------------------------------------------------------- #
def test_get_account_rpc_fork_uses_node_accounts(monkeypatch):
    w3 = MagicMock()
    w3.eth.accounts = ["0xNode0", "0xNode1"]
    w3.to_checksum_address.side_effect = lambda a: a
    monkeypatch.setattr(g, "w3", w3)

    address, private_key = w3h.get_account_RPC(1, fork=True)

    assert address == "0xNode1"
    assert private_key is None


def test_get_account_rpc_non_fork_uses_supplied_accounts(monkeypatch):
    w3 = MagicMock()
    w3.to_checksum_address.side_effect = lambda a: a
    monkeypatch.setattr(g, "w3", w3)
    accounts = [SimpleNamespace(address="0xA", privateKey="pkA")]

    address, private_key = w3h.get_account_RPC(0, fork=False, accounts=accounts)

    assert address == "0xA"
    assert private_key == "pkA"


# --------------------------------------------------------------------------- #
# globals.add_gas_usage (safe paths only)
# --------------------------------------------------------------------------- #
def test_add_gas_usage_two_segment_key_records_tuple(monkeypatch):
    monkeypatch.setattr(g, "gas_used", {})

    g.add_gas_usage("register.user", 21000, "0xUser")
    g.add_gas_usage("register.user", 22000, "0xUser2")

    assert g.gas_used == {
        "register": {"user": [("0xUser", 21000), ("0xUser2", 22000)]}
    }


def test_add_gas_usage_raises_when_branch_is_not_a_dict(monkeypatch):
    monkeypatch.setattr(g, "gas_used", {"register": "not-a-dict"})
    with pytest.raises(TypeError):
        g.add_gas_usage("register.user", 1, "0xUser")


def test_add_gas_usage_raises_when_leaf_is_not_a_list(monkeypatch):
    monkeypatch.setattr(g, "gas_used", {"register": {"user": "not-a-list"}})
    with pytest.raises(TypeError):
        g.add_gas_usage("register.user", 1, "0xUser")
