pragma solidity ^0.8.0;

enum TaskType {
    Images,
    Language,
    Images_clothing,
    Images_objects
}

struct TrainingSpecifications {
    bytes32 modelHash;
    uint min_collateral;
    uint max_collateral;
    address managerAddress;
    uint reward;
    uint8 min_rounds;
    uint8 punishfactor;
    uint8 punishfactorContrib;
    uint8 freeriderPenalty;
    TaskType taskType;
    address[] selectedParticipants;
}
