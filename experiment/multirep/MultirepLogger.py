import pickle
import shutil
import tarfile
import uuid
from pathlib import Path
from typing import List

import pandas as pd

_WAD = 10 ** 18


class MultirepLogger:
    """Accumulates reputation snapshots and task metadata across a multirep session.

    Saves a single session.pkl containing:
      - reputation_timeline: DataFrame with one row per (task, user), capturing
        pre-task and post-task rep state, selection decision, and confidence.
      - tasks: list of per-task metadata dicts, each optionally embedding the
        RunData tables from that task's individual pkl so the session pickle is
        fully self-contained.
    """

    def __init__(self, session_id: str, preset_name: str, session_timestamp: str, preset_dict: dict):
        self.session_id = session_id
        self.preset_name = preset_name
        self.session_timestamp = session_timestamp
        self.preset_dict = preset_dict

        self._rep_rows: list = []
        self._task_entries: list = []

    def log_task(
        self,
        task_index: int,
        dataset: str,
        task_type: int,
        fingerprint: str,
        was_cached: bool,
        users,
        selected_users,
        pre_state: dict,
        scores: dict,
        post_confidence: dict | None = None,
        post_k: dict | None = None,
        post_running_mean: dict | None = None,
        post_m2: dict | None = None,
        pkl_path: str | None = None,
        run_data: dict | None = None,
    ) -> None:
        """Record one task's worth of data.

        pre_state  : {address: {"tr": int, "gir": int, "q": int, "balance": int}}  (all WAD)
        scores     : {address: int}  (WAD-scaled selection score)
        post_*     : {address: float/int}  (already normalised where relevant)
        run_data   : dict of DataFrames from the individual task pkl tables, or None
        """
        selected_set = {u.address for u in selected_users}

        for user in users:
            addr = user.address
            pre = pre_state.get(addr, {})

            name = (
                user.partition_spec.name
                if (user.partition_spec and user.partition_spec.name)
                else f"User {user.number}"
            )
            behavior = (
                user.futureAttitude.name.lower()
                if hasattr(user.futureAttitude, "name")
                else str(user.futureAttitude)
            )

            self._rep_rows.append({
                "task_index":        task_index,
                "dataset":           dataset,
                "task_type":         task_type,
                "fingerprint":       fingerprint,
                "user_name":         name,
                "user_address":      addr,
                "guid":              user.guid,
                "behavior":          behavior,
                "was_selected":      addr in selected_set,
                "was_cached":        was_cached,
                # pre-task state (values used for the selection decision)
                "tr_pre":            pre.get("tr", 0) / _WAD,
                "tr_all_pre":        {tt: v / _WAD for tt, v in pre.get("tr_all", {}).items()},
                "gir_pre":           pre.get("gir", 0) / _WAD,
                "q_pre":             pre.get("q", 0) / _WAD,
                "q_all_pre":         {tt: v / _WAD for tt, v in pre.get("q_all", {}).items()},
                "balance_pre":       pre.get("balance", 0) / _WAD,
                "selection_score":   scores.get(addr, 0) / _WAD,
                # post-task state
                "tr_post":           user.task_rep.get(task_type, 0) / _WAD,
                "tr_all_post":       {tt: v / _WAD for tt, v in user.task_rep.items()},
                "gir_post":          user.global_integrity_rep / _WAD,
                "q_post":            user.q_value.get(task_type, 0) / _WAD,
                "q_all_post":        {tt: v / _WAD for tt, v in user.q_value.items()},
                "balance_post":      user.balance / _WAD,
                "total_contrib_post": user.total_contrib_score / _WAD,
                # reputation internals (for selected users)
                "confidence":        (post_confidence or {}).get(addr),
                "k":                 (post_k or {}).get(addr),
                "running_c_mean":    (post_running_mean or {}).get(addr),
                "m2":                (post_m2 or {}).get(addr),
            })

        self._task_entries.append({
            "task_index": task_index,
            "dataset":    dataset,
            "task_type":  task_type,
            "fingerprint": fingerprint,
            "was_cached": was_cached,
            "pkl_path":   pkl_path,
            "run_data":   run_data,
        })

    def _build_global_accuracy(self) -> pd.DataFrame:
        """Concatenate per-round global accuracy from every task into one DataFrame.

        Columns: task_index, dataset, round, objective_global_accuracy,
                 objective_global_loss, reward_pool, punishment_pool.
        Only tasks with non-empty run_data are included.
        """
        frames = []
        want = ["round", "round_time", "objective_global_accuracy",
                "objective_global_loss", "reward_pool", "punishment_pool"]
        for t in self._task_entries:
            rd = t.get("run_data")
            if not rd:
                continue
            gdf = rd.get("global")
            if gdf is None or not hasattr(gdf, "empty") or gdf.empty:
                continue
            cols = [c for c in want if c in gdf.columns]
            if "objective_global_accuracy" not in cols:
                continue
            frame = gdf[cols].copy()
            frame["task_index"] = t["task_index"]
            frame["dataset"]    = t["dataset"]
            frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def save(self, path: Path) -> None:
        path = Path(path)
        payload = {
            "session_id":           self.session_id,
            "preset_name":          self.preset_name,
            "session_timestamp":    self.session_timestamp,
            "preset":               self.preset_dict,
            "reputation_timeline":  pd.DataFrame(self._rep_rows),
            "global_accuracy":      self._build_global_accuracy(),
            "tasks":                self._task_entries,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def pack_session_tarball(session_dir: Path) -> Path:
    """Create <session_dir>.tar.gz next to session_dir containing all its contents."""
    session_dir = Path(session_dir)
    archive_path = session_dir.parent / f"{session_dir.name}.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(session_dir, arcname=session_dir.name)
    return archive_path


def copy_remote_task_files(src_dir: Path, task_dir: Path) -> None:
    """Copy csv/pkl/json files from a downloaded remote run dir into task_dir."""
    src_dir = Path(src_dir)
    task_dir = Path(task_dir)
    if not src_dir.exists():
        return
    for f in src_dir.iterdir():
        if f.is_file() and f.suffix in {".csv", ".pkl", ".json"}:
            shutil.copy2(f, task_dir / f.name)


def load_task_pkl_tables(pkl_path: Path) -> dict | None:
    """Load the 'tables' dict from an individual task pkl, or None on error."""
    try:
        with open(pkl_path, "rb") as f:
            payload = pickle.load(f)
        return payload.get("tables")
    except Exception:
        return None
