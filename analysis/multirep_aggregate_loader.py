"""Discover and load *multiple* multirep experiments for aggregate graphing.

Directory layout expected (see experiment/data/FinishedRuns):

    <root>/
      EXP-globalrep-avg-distribution-5-task-dataset-switch/
        sessions/<run-1>.tar.gz
        sessions/<run-2>.tar.gz
      EXP-multirep-avg-distribution-5-task-dataset-switch/
        sessions/<run-1>.tar.gz
        ...

Each directory directly under <root> is one *experiment*; every ``*.tar.gz``
found beneath it (recursively) is one *run* of that experiment.  Runs of the
same experiment are averaged together by the plotting layer.

Experiments are paired by name: the ``globalrep`` and ``multirep`` variants of
otherwise-identically-named experiments form an :class:`ExperimentPair`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from analysis.multirep_loader import MultirepSession, load_session_from_tarball

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATASET_TT = {"mnist": 5, "cifar-10": 6, "cifar10": 6}

# task_type ints (mirror TrainingSpecsJobListing.TaskType / multirep_plots).
MNIST_TT = 5
CIFAR_TT = 6
TASK_TYPE_LABELS = {MNIST_TT: "MNIST", CIFAR_TT: "CIFAR-10"}
DATASET_TT = {"mnist": MNIST_TT, "cifar-10": CIFAR_TT, "cifar10": CIFAR_TT}

SYSTEM_TOKENS = ("globalrep", "multirep")


@dataclass
class ExperimentRuns:
    """All runs of a single experiment (one FinishedRuns subfolder)."""

    name: str                       # folder name, e.g. EXP-multirep-avg-...
    system: str                     # 'multirep' | 'globalrep'
    pair_key: str                   # name with the system token removed
    sessions: list[MultirepSession] = field(default_factory=list)

    @property
    def n_runs(self) -> int:
        return len(self.sessions)

    # --- combined tables across runs (each tagged with a 0-based ``run`` id) ---

    def global_accuracy(self) -> pd.DataFrame:
        """Concatenated per-round accuracy/loss across runs, tagged with ``run``."""
        frames = []
        for r, s in enumerate(self.sessions):
            ga = s.global_accuracy
            if ga is None or ga.empty:
                continue
            ga = ga.copy()
            ga["run"] = r
            frames.append(ga)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def reputation_timeline(self) -> pd.DataFrame:
        """Concatenated reputation timeline across runs, tagged with ``run``."""
        frames = []
        for r, s in enumerate(self.sessions):
            rep = s.reputation_timeline
            if rep is None or rep.empty:
                continue
            rep = rep.copy()
            rep["run"] = r
            frames.append(rep)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def iter_task_users(self):
        """Yield (run, task_index, dataset, task_type, users_df) for every task
        whose embedded run_data carries a non-empty ``users`` table."""
        for r, s in enumerate(self.sessions):
            for t in s.tasks:
                rd = t.get("run_data")
                if not rd:
                    continue
                u = rd.get("users")
                if u is None or not hasattr(u, "columns") or u.empty:
                    continue
                if "state" not in u.columns:
                    continue
                yield r, t["task_index"], t.get("dataset"), t.get("task_type"), u


@dataclass
class ExperimentPair:
    """A globalrep + multirep pair sharing the same base name."""

    key: str
    label: str
    globalrep: ExperimentRuns | None = None
    multirep: ExperimentRuns | None = None

    def is_complete(self) -> bool:
        return self.globalrep is not None and self.multirep is not None

    def items(self):
        """Yield (system, ExperimentRuns) for whichever sides exist."""
        if self.globalrep is not None:
            yield "globalrep", self.globalrep
        if self.multirep is not None:
            yield "multirep", self.multirep


def _detect_system(folder_name: str, session: MultirepSession | None) -> str:
    low = folder_name.lower()
    for tok in SYSTEM_TOKENS:
        if tok in low:
            return tok
    # Fall back to the preset flag if the name is uninformative.
    if session is not None and session.preset.get("global_rep_only"):
        return "globalrep"
    return "multirep"


def _pair_key(folder_name: str, system: str) -> str:
    """Folder name with the system token stripped, normalised for matching.

    'EXP-multirep-avg-distribution-5-task' -> 'exp-avg-distribution-5-task'.
    Note: extra variant tokens (e.g. 'noqvalue') are *kept*, so a variant
    experiment will not accidentally pair with the plain one.
    """
    key = folder_name.lower().replace(system, "")
    while "--" in key:
        key = key.replace("--", "-")
    return key.strip("-")


def _pretty_label(pair_key: str) -> str:
    return pair_key.removeprefix("exp-").replace("-", " ").strip()


def load_experiment(folder: Path) -> ExperimentRuns | None:
    """Load every run tarball under *folder* into one ExperimentRuns, or None."""
    folder = Path(folder)
    tarballs = sorted(folder.rglob("*.tar.gz"))
    if not tarballs:
        return None

    sessions: list[MultirepSession] = []
    for tb in tarballs:
        try:
            sessions.append(load_session_from_tarball(tb))
        except Exception as exc:  # noqa: BLE001 - keep loading the rest
            print(f"  [warn] failed to load {tb.name}: {exc}")
    if not sessions:
        return None

    system = _detect_system(folder.name, sessions[0])
    return ExperimentRuns(
        name=folder.name,
        system=system,
        pair_key=_pair_key(folder.name, system),
        sessions=sessions,
    )


def discover_experiments(root: Path) -> list[ExperimentRuns]:
    """Load every experiment directory directly under *root*."""
    root = Path(root)
    experiments = []
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        exp = load_experiment(sub)
        if exp is not None:
            print(f"  loaded {exp.name}  [{exp.system}, {exp.n_runs} run(s)]")
            experiments.append(exp)
    return experiments


def build_pairs(experiments: list[ExperimentRuns]) -> list[ExperimentPair]:
    """Group experiments into globalrep/multirep pairs by their pair_key."""
    pairs: dict[str, ExperimentPair] = {}
    for exp in experiments:
        pair = pairs.get(exp.pair_key)
        if pair is None:
            pair = ExperimentPair(key=exp.pair_key, label=_pretty_label(exp.pair_key))
            pairs[exp.pair_key] = pair
        if exp.system == "globalrep":
            pair.globalrep = exp
        else:
            pair.multirep = exp
    return list(pairs.values())


def load_partition_data_percent(exp: ExperimentRuns) -> dict[str, dict[int, float]]:
    """Return {participant_name: {task_type: data_percent}} from the experiment's
    partition file, or {} if it can't be located.  Matching downstream is by name
    (stable across blockchain instances).
    """
    if not exp.sessions:
        return {}
    pf = exp.sessions[0].preset.get("partition_file")
    if not pf:
        return {}
    path = Path(pf)
    if not path.is_absolute():
        path = _REPO_ROOT / pf
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    entries = payload.get("presets") or payload.get("users") or payload
    entries = entries if isinstance(entries, list) else list(entries.values())
    out: dict[str, dict[int, float]] = {}
    for e in entries:
        if not isinstance(e, dict):
            continue
        name = e.get("name")
        datasets = e.get("datasets", {})
        if not name or not isinstance(datasets, dict):
            continue
        pct: dict[int, float] = {}
        for ds_name, spec in datasets.items():
            tt = _DATASET_TT.get(str(ds_name).strip().lower().replace(".", "-"))
            if tt is not None and isinstance(spec, dict) and "data_percent" in spec:
                pct[tt] = float(spec["data_percent"])
        if pct:
            out[name] = pct
    return out


def find_experiment(experiments: list[ExperimentRuns], *needles: str) -> ExperimentRuns | None:
    """Return the first experiment whose name contains all *needles* (case-insensitive)."""
    for exp in experiments:
        low = exp.name.lower()
        if all(n.lower() in low for n in needles):
            return exp
    return None
