from enum import Enum


class TrainingMode(Enum):
    """Controls where FL experiment training is executed.

    LOCAL  — run the experiment locally.
    REMOTE — submit to the remote API, poll, download and replay.
             Falls back to LOCAL if the remote run fails for any reason.
             If remote_pool_size is set on the preset, a pool of that length
             is built from existing runs for the fingerprint; a random slot is
             picked and reused if non-empty, otherwise a new run is submitted.
    """

    LOCAL = "local"
    REMOTE = "remote"
