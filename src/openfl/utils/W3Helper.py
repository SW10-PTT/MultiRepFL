from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

from eth_account import Account
from web3 import Web3

# Imported only for type hints; skipped at runtime to avoid import errors when not on sys.path.
if TYPE_CHECKING:
    from experiment_configuration import ExperimentConfiguration
from openfl.utils import require_env_var
from openfl.utils.printer import log
from openfl.api import globals


def get_w3():
    if globals.w3 is None:
        globals.w3 = Web3(Web3.HTTPProvider(get_RPC_Endpoint()))
    return globals.w3

def get_RPC_Endpoint():
    return require_env_var("RPC_URL")

def get_PRIVKEYS(experiment_config: ExperimentConfiguration):
    globals.w3 = get_w3()

    if experiment_config.fork == False:

        raw_keys = require_env_var("PRIVATE_KEYS")
        privKeys = [k.strip() for k in raw_keys.splitlines() if k.strip()]

        # Convert to Web3 Account objects
        loaded_accounts = [Account.from_key(k) for k in privKeys]

        # Wrap for compatibility with older code expecting `.privateKey`
        PRIVKEYS = [
            SimpleNamespace(privateKey=acc._private_key, address=acc.address)
            for acc in loaded_accounts
        ]

        log("setup_env", f"Loaded {len(PRIVKEYS)} private keys.")
    else:
        PRIVKEYS = None

    return PRIVKEYS

def get_account_RPC(ix: int, fork: bool, accounts = None):
    w3 = get_w3()
    if fork:
        address = w3.to_checksum_address(w3.eth.accounts[ix])
        private_key = None
        return address, private_key

    address = w3.to_checksum_address(accounts[ix].address)
    private_key = accounts[ix].privateKey
    return address, private_key