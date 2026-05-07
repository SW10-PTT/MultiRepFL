from __future__ import annotations
from typing import TYPE_CHECKING, Optional

from experiment.print_config import DEFAULT_ENABLED_PRINTS_CONFIG#, PRINTS_SILENT, PRINTS_MINIMAL, PRINTS_NORMAL, PRINTS_DEBUG
if TYPE_CHECKING:
    from openfl.utils.types.User import User
    from openfl.contracts.FLChallenge import FLChallenge
import hashlib
import json

import math
from openfl.ml.partition_spec import (
    ANY_DATASET,
    load_dataset_partition_specs,
    normalize_dataset_name,
)
from openfl.utils.types.Attitude import Attitude

from openfl.utils.types.TrainingSpecsJobListing import TrainingSpecsJobListing


VALID_PARTITION_STRATEGIES = ("global", "per_user")

class ExperimentConfiguration:
    def __init__(self,
                 name=None,
                 dataset="MNIST",
                 number_of_good_contributors=4,
                 number_of_bad_contributors=1,
                 number_of_freerider_contributors=1,
                 number_of_inactive_contributors=0,
                 reward=int(1e18),
                 minimum_rounds=5,
                 min_buy_in=int(1e18),
                 max_buy_in=int(1e18),
                 standard_buy_in=int(1e18),
                 epochs=1,
                 batch_size=32, #  Try increase perhaps
                 punish_factor=3,
                 punish_factor_contrib=3,
                 first_round_fee=50, # Percentage of buy-in to charge as fee in first round
                 fork=True,
                 use_outlier_detection = True,
                 contribution_score_strategy="loss_only", # Options: dotproduct, naive, accuracy_loss, accuracy_only, loss_only, loss_tolerance_aware, loss_tolerance_snap
                 loss_tolerance_pct=0.05, # ε = pct * avg_prev_loss; only used by loss_tolerance_* strategies
                 freerider_noise_scale=0.1,
                 freerider_start_round=3,
                 malicious_noise_scale=1.0,
                 malicious_start_round=3,
                 number_of_participants=8,
                 force_merge_all=False,
                 data_percentages=None,
                 label_rules=None,
                 enabled_prints=None,
                 seed=123,
                 user_seeds=None,
                 allow_overlap=False,
                 replication_factor=1.0,
                 partition_strategy="per_user", # Options: global, per_user
                 per_user_partitions="experiment/partitions/example.json"): # Path to JSON file with per-user partition specs; see example.json for format. Or None. Example: "experiment/partitions/example.json"

        self.name = name
        self.dataset = dataset

        self.fork = fork

        # Apply scaling only if we’re on Sepolia (fork = False)
        if not fork:
            scale = 0.005  # scale down
            reward = int(reward * scale)
            min_buy_in = int(min_buy_in * scale)
            max_buy_in = int(max_buy_in * scale)
            standard_buy_in = int(standard_buy_in * scale)

        # Store everything
        self.number_of_participants =number_of_participants
        self.reward = reward
        self.minimum_rounds = minimum_rounds
        self.min_buy_in = min_buy_in
        self.max_buy_in = max_buy_in
        self.standard_buy_in = standard_buy_in
        self.epochs = epochs
        self.batch_size = batch_size
        self.punish_factor = punish_factor
        self.punish_factor_contrib = punish_factor_contrib
        self.first_round_fee = first_round_fee
        self.contribution_score_strategy = contribution_score_strategy
        self.loss_tolerance_pct = float(loss_tolerance_pct)
        if self.loss_tolerance_pct < 0:
            raise ValueError("loss_tolerance_pct must be >= 0")
        self.use_outlier_detection = use_outlier_detection
        self.freerider_noise_scale = freerider_noise_scale
        self.freerider_start_round = freerider_start_round
        self.malicious_start_round = malicious_start_round
        self.malicious_noise_scale = malicious_noise_scale
        self.force_merge_all = force_merge_all
        self.enabled_prints = (
            set(enabled_prints) if enabled_prints is not None
            else set(DEFAULT_ENABLED_PRINTS_CONFIG)
        )
        # Master seed drives the partition; per-user seeds are derived from it for independent RNG streams.
        # allow_overlap+replication_factor control whether participants can share dataset samples.
        self.seed = int(seed)
        self.user_seeds = self._resolve_user_seeds(user_seeds)
        self.allow_overlap = bool(allow_overlap)
        self.replication_factor = float(replication_factor)
        if self.replication_factor < 1.0:
            raise ValueError("replication_factor must be >= 1.0")
        if self.replication_factor > 1.0 and not self.allow_overlap:
            raise ValueError("replication_factor > 1.0 requires allow_overlap=True")

        # Toggle between the legacy stratified-global partitioner and the
        # spec-driven per-user partitioner. In per_user mode the spec drives
        # both data shares and per-user behavior (Honest/Malicious/Free-rider),
        # so the contributor counts are derived from the spec and the
        # number_of_*_contributors kwargs are ignored. data_percentages and
        # label_rules are forbidden in per_user mode.
        if partition_strategy not in VALID_PARTITION_STRATEGIES:
            raise ValueError(
                f"partition_strategy must be one of {VALID_PARTITION_STRATEGIES}, got {partition_strategy!r}"
            )
        self.partition_strategy = partition_strategy
        self.per_user_partitions = self._resolve_per_user_partitions(per_user_partitions)

        if self.partition_strategy == "per_user":
            if not self.per_user_partitions:
                raise ValueError(
                    "partition_strategy='per_user' requires per_user_partitions to be provided"
                )
            if data_percentages is not None:
                raise ValueError(
                    "data_percentages is not allowed when partition_strategy='per_user'; "
                    "set data_percent inside the per-user spec instead"
                )
            if label_rules is not None:
                raise ValueError(
                    "label_rules is not allowed when partition_strategy='per_user'; "
                    "set only_labels/flip_map inside the per-user spec instead"
                )
            self._validate_per_user_index_set()
            self._refresh_counts_from_specs()
            self.data_percentages = []
            self.label_rules = {}
            self._validate_per_user_partitions()
        else:
            self.number_of_good_contributors = number_of_good_contributors
            self.number_of_bad_contributors = number_of_bad_contributors
            self.number_of_freerider_contributors = number_of_freerider_contributors
            self.number_of_inactive_contributors = number_of_inactive_contributors
            self.data_percentages = self._resolve_data_percentages(data_percentages)
            self.label_rules = self._resolve_label_rules(label_rules)


    def get_training_specs(self, manager_address, model_hash) -> TrainingSpecsJobListing:
        return TrainingSpecsJobListing(model_hash, self.min_buy_in, self.max_buy_in, manager_address, self.reward, self.minimum_rounds, self.punish_factor, self.punish_factor_contrib, self.first_round_fee, 1) # Todo: Tasktype

    @property
    def number_of_contributors(self):
        return (self.number_of_good_contributors +
                self.number_of_bad_contributors +
                self.number_of_freerider_contributors +
                self.number_of_inactive_contributors)

    @property
    def number_of_data_users(self):
        return (self.number_of_good_contributors +
                self.number_of_bad_contributors +
                self.number_of_freerider_contributors)

    def _resolve_data_percentages(self, data_percentages):
        # make equal split
        if data_percentages is None:
            equal_percent = 100.0 / self.number_of_data_users
            return [equal_percent] * self.number_of_data_users

        data_percentages = [float(percent) for percent in data_percentages]
        if len(data_percentages) != self.number_of_data_users:
            raise ValueError("data_percentages must match the number of configured users")
        if not math.isclose(sum(data_percentages), 100.0, abs_tol=1e-9):
            raise ValueError("data_percentages must sum to 100")

        return data_percentages

    def _resolve_per_user_partitions(self, per_user_partitions):
        if per_user_partitions is None:
            return {}
        return load_dataset_partition_specs(per_user_partitions)

    # All datasets must list the same set of user indices. Per-dataset behavior
    # may still differ (e.g. user 0 honest on MNIST, malicious on CIFAR-10),
    # but the participant roster is fixed.
    def _validate_per_user_index_set(self):
        reference_indices: Optional[set[str]] = None
        reference_label: Optional[str] = None
        for dataset_key, specs in self.per_user_partitions.items():
            label = "default" if dataset_key == ANY_DATASET else dataset_key
            indices = set(specs.keys())
            if reference_indices is None:
                reference_indices = indices
                reference_label = label
                continue
            if indices != reference_indices:
                missing = reference_indices - indices
                extra = indices - reference_indices
                raise ValueError(
                    f"per_user_partitions: dataset {label!r} has a different user_index "
                    f"set than {reference_label!r} (missing={sorted(missing)}, "
                    f"extra={sorted(extra)}); every dataset must list the same users"
                )

    # Per-user mode counts depend on the active dataset since behavior is
    # per-dataset. Recomputes the four contributor counts from the spec for
    # self.dataset. Idempotent — call again whenever self.dataset changes.
    def _refresh_counts_from_specs(self):
        specs = self.get_partition_specs(self.dataset)
        counts = {
            Attitude.Honest: 0,
            Attitude.Malicious: 0,
            Attitude.FreeRider: 0,
            Attitude.Inactive: 0,
        }
        for spec in specs.values():
            counts[spec.behavior] += 1
        self.number_of_good_contributors = counts[Attitude.Honest]
        self.number_of_bad_contributors = counts[Attitude.Malicious]
        self.number_of_freerider_contributors = counts[Attitude.FreeRider]
        self.number_of_inactive_contributors = counts[Attitude.Inactive]

    # The runner sets the active dataset name late (it's normalised from a CLI
    # arg). In per_user mode, contributor counts depend on that dataset, so the
    # runner must call this after it knows the final name. No-op for global.
    def refresh_for_dataset(self, dataset_name: str) -> None:
        self.dataset = dataset_name
        if self.partition_strategy == "per_user":
            self._refresh_counts_from_specs()

    # {user_index: Attitude} for the active dataset, used by the experiment
    # runner to assign attitudes when partition_strategy='per_user'.
    def get_behaviors_per_user(self) -> dict[str, Attitude]:
        if self.partition_strategy != "per_user":
            raise RuntimeError(
                "get_behaviors_per_user is only valid when partition_strategy='per_user'"
            )
        specs = self.get_partition_specs(self.dataset)
        return {user_index: spec.behavior for user_index, spec in specs.items()}

    # Lookup specs for a given dataset. Falls back to the wildcard bucket
    # (legacy single-dataset JSON) when no dataset-specific entry exists.
    def get_partition_specs(self, dataset_name=None):
        if not self.per_user_partitions:
            return {}
        key = normalize_dataset_name(dataset_name if dataset_name is not None else self.dataset)
        if key in self.per_user_partitions:
            return self.per_user_partitions[key]
        if ANY_DATASET in self.per_user_partitions:
            return self.per_user_partitions[ANY_DATASET]
        raise KeyError(
            f"per_user_partitions has no entry for dataset {key!r}; "
            f"available: {sorted(self.per_user_partitions.keys())}"
        )

    # Fail-fast validation for the per_user strategy. With fair-share-then-
    # filter semantics, the only invariant left is sum(data_percent) <= 100;
    # per-class allocation can't overflow once that holds, since each user's
    # fair share is at most pct/100 of every class pool. Validates every
    # dataset entry independently so a bad profile is caught up front.
    # Index-set consistency across datasets is checked by
    # _validate_per_user_index_set; here we only verify uniqueness/budget.
    # user_index keys are opaque strings (GUIDs or numeric strings); the
    # runner iterates them in sorted order for deterministic account
    # allocation, so no positional/density invariant is required.
    def _validate_per_user_partitions(self):
        if not self.per_user_partitions:
            raise ValueError(
                "partition_strategy='per_user' requires per_user_partitions to be provided"
            )

        budget_cap = 100.0

        for dataset_key, specs in self.per_user_partitions.items():
            label = "default" if dataset_key == ANY_DATASET else dataset_key
            if not specs:
                raise ValueError(
                    f"per_user_partitions[{label}] must list at least one user"
                )

            total_budget = sum(spec.data_percent for spec in specs.values())
            if total_budget > budget_cap + 1e-9:
                raise ValueError(
                    f"per_user_partitions[{label}] total data_percent "
                    f"{total_budget:.4f}% exceeds cap {budget_cap:.4f}%"
                )

    def _resolve_user_seeds(self, user_seeds):
        # Optional explicit per-user overrides. Keys are normalised to str so
        # the same lookup works for GUID-keyed per_user mode and positional
        # int indices in global mode (callers stringify on lookup).
        # Anything not specified gets derived from the master seed at runtime
        # via SHA256.
        if user_seeds is None:
            return {}
        return {str(user_index): int(seed) for user_index, seed in user_seeds.items()}

    def get_finger_print(self, participants):
        # Sort participant fingerprints so config fingerprint is order-invariant.
        # `list.sort()` returns None, so use `sorted()` to actually capture the result.
        participants = sorted(
            participant.finger_print
            for participant in participants
        )

        data = {
            "dataset": self.dataset,
            "minimum_rounds": self.minimum_rounds,
            "min_buy_in": self.min_buy_in,
            "max_buy_in": self.max_buy_in,
            "standard_buy_in": self.standard_buy_in,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "punish_factor": self.punish_factor,
            "punish_factor_contrib": self.punish_factor_contrib,
            "first_round_fee": self.first_round_fee,
            "contribution_score_strategy": self.contribution_score_strategy,
            "loss_tolerance_pct": self.loss_tolerance_pct,
            "use_outlier_detection": self.use_outlier_detection,
            "freerider_noise_scale": self.freerider_noise_scale,
            "freerider_start_round": self.freerider_start_round,
            "malicious_start_round": self.malicious_start_round,
            "malicious_noise_scale": self.malicious_noise_scale,
            "force_merge_all": self.force_merge_all,
            "participants": participants,
            "seed": self.seed,
            "allow_overlap": self.allow_overlap,
            "replication_factor": self.replication_factor,
            "user_seeds": dict(sorted(self.user_seeds.items())),
            "partition_strategy": self.partition_strategy,
            "per_user_partitions": {
                dataset_key: [
                    spec.fingerprint_dict()
                    for _, spec in sorted(specs.items())
                ]
                for dataset_key, specs in sorted(self.per_user_partitions.items())
            },
        }

        blob = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()


    @staticmethod
    def _resolve_label_rules(label_rules):
        # Example:
        # {
        #   2: {"only_labels": [4, 9], "flip_map": {4: 9, 9: 4}},
        #   3: {"flip_map": {2: 5}}
        # }
        if label_rules is None:
            return {}

        resolved_rules = {}
        for user_index, rule in label_rules.items():
            only_labels = rule.get("only_labels")
            flip_map = rule.get("flip_map", {})

            normalized_rule = {
                "only_labels": [int(label) for label in only_labels] if only_labels is not None else None,
                "flip_map": {int(src): int(dst) for src, dst in flip_map.items()},
            }
            resolved_rules[int(user_index)] = normalized_rule

        return resolved_rules

    def to_dict(self):
        excluded = {"enabled_prints"}
        return {
            k: v for k, v in self.__dict__.items()
            if not callable(v) and not (k.startswith("_") or k.startswith("__")) and k not in excluded
        }