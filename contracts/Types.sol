pragma solidity ^0.8.0;

// TaskType identifies the dataset/task this job is for. TaskRep is tracked
// per TaskType (one TaskRep per MNIST, another per CIFAR10, etc.). Keep
// `template` first so the default (uninitialised) value is the template marker.
enum TaskType {
    template,
    Images,
    Language,
    Images_clothing,
    Images_objects,
    MNIST,
    CIFAR10,
    FashionMNIST,
    IMDB
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
