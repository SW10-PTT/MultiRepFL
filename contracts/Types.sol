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

// Selects how reputation state is keyed on OpenFLManager.
//   PerTask   : default. TaskRep is stored per-TaskType (one slot per dataset)
//               and GIR is updated each task from cross-round vote tallies.
//   GlobalOnly: a single TaskRep slot per user is shared across all TaskTypes
//               and GIR is never written (logged as 0/unchanged).
// The mode is fixed at OpenFLManager deployment time.
enum ReputationMode {
    PerTask,
    GlobalOnly
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
    uint256 qWeight;   // WAD-scaled additive Q bonus: score = normalWeight + qWeight * q / WAD
    uint256 trWeight;  // taskRep multiplier in selection score (default 6)
    uint256 girWeight; // GIR multiplier in selection score (default 4)
}

// Per-user computed TaskRep outputs from one challenge. Stored on-chain by
// OpenFLChallenge.computeAndRecordTaskReps() so Python replay can pass them
// directly to OpenFLManager.applyPrecomputedTaskReps() without recalculation.
struct TaskRepRecord {
    address user;
    uint256 newTaskRep;
    uint256 newRunningCMean;
    uint256 newM2;
    uint256 newIntegrityRep; // only meaningful when applyGIR == true
    bool applyGIR;
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
