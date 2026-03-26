from solcx import install_solc, set_solc_version, compile_standard, get_installed_solc_versions
from pathlib import Path
import json

print(get_installed_solc_versions())

# 1) Ensure exact compiler
install_solc("0.8.9")
set_solc_version("0.8.9")

# 2) Load sources
root = Path(__file__).parents[1]
contracts_dir = root / "contracts"
sources = {
    "OpenFLManager.sol": {"content": (contracts_dir / "OpenFLManager.sol").read_text(encoding="utf-8")},
    "OpenFLChallenge.sol":   {"content": (contracts_dir / "OpenFLChallenge.sol").read_text(encoding="utf-8")},
    "JobListing.sol":   {"content": (contracts_dir / "JobListing.sol").read_text(encoding="utf-8")},
    "Types.sol":   {"content": (contracts_dir / "Types.sol").read_text(encoding="utf-8")},
    "Clones.sol":   {"content": (contracts_dir / "Clones.sol").read_text(encoding="utf-8")},
}

# 3) Compile
compiled = compile_standard({
    "language": "Solidity",
    "sources": sources,
    "settings": {
        "optimizer": {"enabled": True, "runs": 200},
        "outputSelection": {"*": {"*": ["abi","evm.bytecode.object"]}}
    }
})

contracts = [
    ("Manager", compiled["contracts"]["OpenFLManager.sol"]["OpenFLManager"]),
    ("JobListing", compiled["contracts"]["JobListing.sol"]["JobListing"]),
    ("OpenFLChallenge", compiled["contracts"]["OpenFLChallenge.sol"]["OpenFLChallenge"]),
]

for name, data in contracts:
    bytecode = data["evm"]["bytecode"]["object"]
    size_bytes = len(bytecode) // 2
    size_kb = size_bytes / 1024

    print(f"{name}: {size_bytes} bytes ({size_kb:.2f} KB)")

# 4) Extract artifacts
mgr = compiled["contracts"]["OpenFLManager.sol"]["OpenFLManager"]
mdl = compiled["contracts"]["OpenFLChallenge.sol"]["OpenFLChallenge"]
jls = compiled["contracts"]["JobListing.sol"]["JobListing"]

build = root / "artifacts" / "bytecode"
build.mkdir(parents=True, exist_ok=True)

# IMPORTANT: abi.txt should be JSON, because Python should json.load it later
(Path(build / "manager_abi.json")).write_text(json.dumps(mgr["abi"], separators=(",",":")), encoding="utf-8")
(Path(build / "manager_bytecode.bin")).write_text(mgr["evm"]["bytecode"]["object"], encoding="utf-8")

(Path(build / "model_abi.json")).write_text(json.dumps(mdl["abi"], separators=(",",":")), encoding="utf-8")
(Path(build / "model_bytecode.bin")).write_text(mdl["evm"]["bytecode"]["object"], encoding="utf-8")

(Path(build / "job_listing_abi.json")).write_text(json.dumps(jls["abi"], separators=(",",":")), encoding="utf-8")
(Path(build / "job_listing_bytecode.bin")).write_text(jls["evm"]["bytecode"]["object"], encoding="utf-8")

# Write ABI as a Python variable file
abi_py_file = build / "abi_model.py"

with open(abi_py_file, "w", encoding="utf-8") as f:
    f.write("# Auto-generated OpenFLChallenge ABI\n")
    # Dump ABI as a JSON string (triple quotes)
    f.write("import json\n\n")
    f.write("OPEN_FL_MODEL_ABI = json.loads('''\n")
    f.write(json.dumps(mdl["abi"], indent=2))  # JSON string inside triple quotes
    f.write("\n''')\n")



print("Artifacts written to build/: manager_abi.json, OPEN_FL_MODEL_ABI.py, manager_bytecode.bin, model_abi.json, model_bytecode.bin, job_listing_abi.json, job_listing_bytecode.bin")
