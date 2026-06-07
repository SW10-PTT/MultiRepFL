"""
Launch and manage a local blockchain node (Anvil or Ganache) for FL experiments.

Usage:
    from experiment.blockchain_launcher import start
    start("anvil")   # or "ganache"

Sets os.environ["RPC_URL"] to the node's HTTP endpoint.
Registers atexit + SIGTERM handlers to terminate the process and delete temp files.
"""

import atexit
import os
import shlex
import shutil
import signal
import socket
import subprocess
import tempfile
import time

from openfl.utils.printer import log

_NUM_ACCOUNTS = 30
_PORT_START = 8545
_PORT_MAX = 8645
_STARTUP_TIMEOUT = 30  # seconds

_ANVIL_CACHE_BASE = os.path.expanduser("~/foundry/tmp")

_proc: subprocess.Popen | None = None
_tmpdir: str | None = None


def _is_port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def _wait_for_rpc(port: int, timeout: int = _STARTUP_TIMEOUT) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except OSError:
            time.sleep(0.3)
    return False


def _cleanup() -> None:
    global _proc, _tmpdir
    if _proc is not None:
        log("setup_env", f"Stopping blockchain node (pid={_proc.pid})")
        try:
            _proc.terminate()
            _proc.wait(timeout=5)
        except Exception:
            try:
                _proc.kill()
            except Exception:
                pass
        _proc = None
    if _tmpdir is not None:
        shutil.rmtree(_tmpdir, ignore_errors=True)
        log("setup_env", f"Removed blockchain tmp dir: {_tmpdir}")
        _tmpdir = None


def _install_cleanup_handlers() -> None:
    atexit.register(_cleanup)

    def _sigterm_handler(signum, frame):
        _cleanup()
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        os.kill(os.getpid(), signal.SIGTERM)

    signal.signal(signal.SIGTERM, _sigterm_handler)


_ACCOUNT_BALANCE_ETH = 100_000_000

def _build_cmd(mode: str, port: int, tmpdir: str) -> list[str]:
    if mode == "anvil":
        return [
            "anvil",
            "--accounts", str(_NUM_ACCOUNTS),
            "--port", str(port),
            "--balance", str(_ACCOUNT_BALANCE_ETH),
            "--max-persisted-states", "100",
            "--cache-path", tmpdir,
        ]
    # ganache — try modern ganache (v7+) first, fall back to legacy ganache-cli
    if shutil.which("ganache"):
        return [
            "ganache",
            f"--wallet.totalAccounts={_NUM_ACCOUNTS}",
            f"--wallet.defaultBalance={_ACCOUNT_BALANCE_ETH}",
            f"--server.port={port}",
            f"--database.dbPath={tmpdir}",
        ]
    if shutil.which("ganache-cli"):
        return [
            "ganache-cli",
            "--accounts", str(_NUM_ACCOUNTS),
            "--defaultBalanceEther", str(_ACCOUNT_BALANCE_ETH),
            "--port", str(port),
            "--db", tmpdir,
        ]
    raise RuntimeError("Neither 'ganache' nor 'ganache-cli' found in PATH")


def start(mode: str) -> str:
    """Start a local blockchain node and return its RPC URL.

    mode: 'anvil' or 'ganache'
    Scans ports starting at 8545 until a free one is found.
    Sets os.environ['RPC_URL'] and registers cleanup on exit.
    """
    global _proc, _tmpdir

    if mode not in ("anvil", "ganache"):
        raise ValueError(f"Unknown blockchain mode: {mode!r}")

    if mode == "anvil":
        os.makedirs(_ANVIL_CACHE_BASE, exist_ok=True)
        tmpdir = tempfile.mkdtemp(prefix="anvil-", dir=_ANVIL_CACHE_BASE)
    else:
        tmpdir = tempfile.mkdtemp(prefix="fl_ganache_")

    for port in range(_PORT_START, _PORT_MAX + 1):
        if not _is_port_free(port):
            log("setup_env", f"Port {port} busy, trying next...")
            continue

        cmd = _build_cmd(mode, port, tmpdir)
        log("setup_env", f"Starting {mode} on port {port}: {shlex.join(cmd)}")
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if _wait_for_rpc(port):
            _proc = proc
            _tmpdir = tmpdir
            _install_cleanup_handlers()
            url = f"http://127.0.0.1:{port}"
            os.environ["RPC_URL"] = url
            log("setup_env", f"{mode} ready at {url}")
            return url

        log("setup_env", f"{mode} did not respond on port {port}, trying next...")
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    if tmpdir:
        shutil.rmtree(tmpdir, ignore_errors=True)
    raise RuntimeError(
        f"Could not start {mode} on any port between {_PORT_START} and {_PORT_MAX}"
    )
