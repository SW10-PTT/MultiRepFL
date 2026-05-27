pragma solidity ^0.8.0;

enum TaskType {
    template,
    Images,
    Language,
    Images_clothing,
    Images_object,
    Images_numbers,
    Images_MNIST,
    Images_CIFAR10
}

struct TrainingSpecifications {
    uint min_collateral;
    uint max_collateral;
    address managerAddress;
    uint reward;
    uint8 min_rounds;
    uint8 punishfactor;
    uint8 punishfactorContrib;
    uint8 freeriderPenalty;
    TaskType taskType;
    address jobListingAddress;
}

struct ChallengeSpecifications {
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
    address jobListingAddress;
}
