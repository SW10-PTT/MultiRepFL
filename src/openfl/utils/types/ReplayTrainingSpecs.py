from dataclasses import dataclass


@dataclass
class ReplayTrainingSpecs:
    honest_participants: int
    bad_participants: int
    freeriding_participants: int
    dataset: str
    # TODO: data split