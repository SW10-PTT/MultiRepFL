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
    q_weight: int = 0  # WAD-scaled (1e18); mirrors trainingSpecs.qWeight in Solidity

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

    def to_challenge(self, contribution_score_strategy, outlier_detection, joblisting_address, loss_tolerance_pct=0.05):
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
    loss_tolerance_pct: float = 0.05

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
        )

