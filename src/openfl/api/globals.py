from enum import IntFlag, auto

class ReplayMode(IntFlag):
    Record = auto()
    PlayBack = auto()
    HardPlayBack = auto()
    _actively_replaying = auto() # internal use, do not use

fork = True
w3 = None
reuse_runs: ReplayMode = ReplayMode.Record# | ReplayMode.PlayBack | ReplayMode.HardPlayBack
gas_used = {}
repo_dir = "runs"
progress = 0
min_free_ram_gb: float | None = None  # lowest free RAM seen during current task
fp_data_cache: dict = {}   # fingerprint hash → raw data dict used to compute it
fp_user_labels: dict = {}  # participant finger_print → display name



def add_gas_usage(gas_type: str, amount: int, user_addr) -> None:
    global gas_used
    keys = gas_type.split(".")
    d = gas_used

    for key in keys[:-1]:
        if key in d:
            if not isinstance(d[key], dict):
                raise TypeError(
                    f"Cannot create nested key under '{key}': "
                    f"expected dict, found {type(d[key]).__name__}"
                )
        else:
            d[key] = {}

        d = d[key]

        leaf = keys[-1]

        if leaf in d:
            if not isinstance(d[leaf], list):
                raise TypeError(
                    f"Cannot append to '{leaf}': "
                    f"expected list, found {type(d[leaf]).__name__}"
                )
        else:
            d[leaf] = []

        d[leaf].append((user_addr, amount))