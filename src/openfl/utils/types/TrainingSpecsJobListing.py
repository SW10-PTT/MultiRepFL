from dataclasses import dataclass

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
            contribution_score_strategy,
            joblisting_address,
            outlier_detection,
            loss_tolerance_pct,
        )

@dataclass
class TrainingSpecsChallenge(TrainingSpecsJobListing):
    contribution_score_strategy: str
    joblisting_address: str
    outlier_detection: bool
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

