import importlib.util
import json
import sys
import types
from pathlib import Path


def test_compile_contracts_runs_with_stubs(tmp_path, monkeypatch):
    # Stub solcx so the script doesn't actually invoke a Solidity compiler.
    class FakeSolcx:
        def __init__(self):
            self.installed = []
            self.version = None
            self.compiled_config = None

        def install_solc(self, version):
            self.installed.append(version)

        def set_solc_version(self, version):
            self.version = version

        def compile_standard(self, config):
            self.compiled_config = config
            # Mirror the script's expected output shape: each .sol file maps to
            # the contract names it defines, each with abi + evm.bytecode.object.
            return {
                "contracts": {
                    "OpenFLManager.sol": {
                        "OpenFLManager": {"abi": [{"name": "mgr"}], "evm": {"bytecode": {"object": "aa"}}},
                    },
                    "OpenFLChallenge.sol": {
                        "OpenFLChallenge": {"abi": [{"name": "ch"}], "evm": {"bytecode": {"object": "bb"}}},
                    },
                    "JobListing.sol": {
                        "JobListing": {"abi": [{"name": "jl"}], "evm": {"bytecode": {"object": "cc"}}},
                    },
                },
            }

    fake = FakeSolcx()
    solcx_stub = types.ModuleType("solcx")
    solcx_stub.install_solc = fake.install_solc
    solcx_stub.set_solc_version = fake.set_solc_version
    solcx_stub.compile_standard = fake.compile_standard
    solcx_stub.get_installed_solc_versions = lambda: list(fake.installed)
    monkeypatch.setitem(sys.modules, "solcx", solcx_stub)

    project_root = Path(__file__).resolve().parents[2]
    src_path = project_root / "scripts" / "compile_contracts.py"
    code = src_path.read_text(encoding="utf-8")

    # Place the script inside tmp_path/scripts so its parents[1] points at tmp_path,
    # which is where it expects to find ./contracts and where it writes ./artifacts.
    module_path = tmp_path / "scripts" / "compile_contracts.py"
    module_path.parent.mkdir(parents=True, exist_ok=True)
    module_path.write_text(code, encoding="utf-8")

    contracts_dir = tmp_path / "contracts"
    contracts_dir.mkdir()
    for name in ("OpenFLManager.sol", "OpenFLChallenge.sol", "JobListing.sol", "Types.sol", "Clones.sol"):
        (contracts_dir / name).write_text("pragma solidity ^0.8.9;", encoding="utf-8")

    spec = importlib.util.spec_from_file_location("tmp_compile_contracts", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["tmp_compile_contracts"] = module
    module.__file__ = str(module_path)
    spec.loader.exec_module(module)

    # Verify the solcx stubs were invoked with the expected version pin.
    assert fake.installed == ["0.8.9"]
    assert fake.version == "0.8.9"
    # Verify all five sources reached compile_standard.
    assert set(fake.compiled_config["sources"].keys()) == {
        "OpenFLManager.sol", "OpenFLChallenge.sol", "JobListing.sol", "Types.sol", "Clones.sol",
    }

    # Verify the artifact files the script writes.
    build_dir = tmp_path / "artifacts" / "bytecode"
    assert build_dir.exists()
    for f in (
        "manager_abi.json", "manager_bytecode.bin",
        "model_abi.json", "model_bytecode.bin",
        "job_listing_abi.json", "job_listing_bytecode.bin",
        "abi_model.py",
    ):
        assert (build_dir / f).exists(), f"expected artifact {f} to be written"

    # Spot-check ABI content round-trips through json.
    assert json.loads((build_dir / "manager_abi.json").read_text(encoding="utf-8")) == [{"name": "mgr"}]
    assert (build_dir / "manager_bytecode.bin").read_text(encoding="utf-8") == "aa"
    assert (build_dir / "model_bytecode.bin").read_text(encoding="utf-8") == "bb"
    assert (build_dir / "job_listing_bytecode.bin").read_text(encoding="utf-8") == "cc"
