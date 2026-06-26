from dataclasses import dataclass
from enum import IntEnum


# Mirrors the Solidity `TaskType` enum in contracts/Types.sol. Values must
# match the Solidity ordinal positions (uint8). TaskType acts as the dataset
# key for TaskRep — one TaskRep per TaskType per user. Add new entries in
# lock-step with the Solidity enum.
class TaskType(IntEnum):
    template = 0
    Images = 1
    Language = 2
    Images_clothing = 3
    Images_objects = 4
    MNIST = 5
    CIFAR10 = 6
    FashionMNIST = 7
    IMDB = 8

    @classmethod
    def from_dataset_name(cls, name: str) -> "TaskType":
        if name is None:
            return cls.template
        normalized = name.replace("-", "").replace("_", "").replace(" ", "").lower()
        mapping = {
            "mnist": cls.MNIST,
            "cifar10": cls.CIFAR10,
            "fashionmnist": cls.FashionMNIST,
            "imdb": cls.IMDB,
        }
        return mapping.get(normalized, cls.template)


@dataclass
class TrainingSpecsJobListing:
    modelHash: bytes
    min_collateral: int
    max_collateral: int
    manager_address: str
    reward: int
    min_rounds: int
    punishfactor: int
    punishfactorContrib: int
    freeriderPenalty: int
    taskType: int
    q_weight: int = 0   # WAD-scaled (1e18); mirrors trainingSpecs.qWeight in Solidity
    tr_weight: int = 6  # taskRep multiplier; mirrors trainingSpecs.trWeight in Solidity
    gir_weight: int = 4 # GIR multiplier; mirrors trainingSpecs.girWeight in Solidity
    q_slot_limit_enabled: bool = False # mirrors trainingSpecs.qSlotLimitEnabled
    q_slot_limit: int = 0              # mirrors trainingSpecs.qSlotLimit
    q_hard_reset: bool = False         # mirrors trainingSpecs.qHardReset
    # Deploy-time TaskRep tunables; WAD-scaled (1e18) for the three fractions.
    # Defaults mirror the previous OpenFLChallenge.sol hardcoded constants.
    tr_alpha: int = int(2e17)
    tr_n_blend: int = int(2e17)
    tr_n_0: int = 2
    tr_lambda: int = 5
    tr_integrity_learning_rate: int = int(2e17)
    tr_gain_cap_multiplier: int = 2

    def to_solidity_job(self):
        return (
            self.modelHash,
            self.min_collateral,
            self.max_collateral,
            self.manager_address,
            self.reward,
            self.min_rounds,
            self.punishfactor,
            self.punishfactorContrib,
            self.freeriderPenalty,
            self.taskType,
        )

    def to_challenge(self, contribution_score_strategy, outlier_detection, joblisting_address, loss_tolerance_pct=0.1):
        return TrainingSpecsChallenge(
            modelHash=self.modelHash,
            min_collateral=self.min_collateral,
            max_collateral=self.max_collateral,
            manager_address=self.manager_address,
            reward=self.reward,
            min_rounds=self.min_rounds,
            punishfactor=self.punishfactor,
            punishfactorContrib=self.punishfactorContrib,
            freeriderPenalty=self.freeriderPenalty,
            taskType=self.taskType,
            q_weight=self.q_weight,
            tr_weight=self.tr_weight,
            gir_weight=self.gir_weight,
            q_slot_limit_enabled=self.q_slot_limit_enabled,
            q_slot_limit=self.q_slot_limit,
            q_hard_reset=self.q_hard_reset,
            tr_alpha=self.tr_alpha,
            tr_n_blend=self.tr_n_blend,
            tr_n_0=self.tr_n_0,
            tr_lambda=self.tr_lambda,
            tr_integrity_learning_rate=self.tr_integrity_learning_rate,
            tr_gain_cap_multiplier=self.tr_gain_cap_multiplier,
            contribution_score_strategy=contribution_score_strategy,
            joblisting_address=joblisting_address,
            outlier_detection=outlier_detection,
            loss_tolerance_pct=loss_tolerance_pct,
        )

@dataclass
class TrainingSpecsChallenge(TrainingSpecsJobListing):
    contribution_score_strategy: str = ""
    joblisting_address: str = "0x0000000000000000000000000000000000000000"
    outlier_detection: bool = False
    loss_tolerance_pct: float = 0.1

    def to_solidity_challenge(self):
        return (
            self.modelHash,
            self.min_collateral,
            self.max_collateral,
            self.manager_address,
            self.reward,
            self.min_rounds,
            self.punishfactor,
            self.punishfactorContrib,
            self.freeriderPenalty,
            self.taskType,
            self.joblisting_address,
            self.tr_alpha,
            self.tr_n_blend,
            self.tr_n_0,
            self.tr_lambda,
            self.tr_integrity_learning_rate,
            self.tr_gain_cap_multiplier,
        )

