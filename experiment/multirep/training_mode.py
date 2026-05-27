from enum import Enum


class TrainingMode(Enum):
    """Controls where FL experiment training is executed.

    LOCAL  — run the experiment locally (default behaviour, unchanged).
    REMOTE — submit the config to the remote experiment API, poll until done,
             download the result tarball, and replay it as a RunRepo.
    MIXED  — probabilistic blend driven by how many completed remote runs
             already exist for the experiment fingerprint:

             Let  n = len(runs at /runs/by-fingerprint/{fp})
                  T = MIXED_RUN_THRESHOLD  (constant in multirep.py)

             • n > T  → pick a random existing remote run (always)
             • n ≤ T  → with probability n/T pick a random existing run,
                        otherwise submit a new remote run
    """

    LOCAL = "local"
    REMOTE = "remote"
    MIXED = "mixed"