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
    bool qSlotLimitEnabled; // when true, cap how many slots may be won via the Q bonus
    uint256 qSlotLimit;     // max slots fillable using Q; the rest go by base score only
    bool qHardReset;        // when true, selected users' Q resets to 0; otherwise subtracts WAD
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
    // Transformed contribution score for this task: the output of
    // OpenFLChallenge._trTransformDelta(taskRepDelta, ...), WAD-scaled to
    // [0, 1e18]. This is the per-task contribution signal that feeds the
    // TaskRep EWMA. Recorded here purely so Python can read/print the exact
    // on-chain value; applyPrecomputedTaskReps ignores it.
    uint256 contribScore;
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
    // Deploy-time TaskRep tunables (WAD-scaled for the three fractions).
    uint256 trAlpha;
    uint256 trNBlend;
    uint256 trN0;
    uint256 trLambda;
    uint256 trIntegrityLearningRate;
    uint256 trGainCapMultiplier;
}
