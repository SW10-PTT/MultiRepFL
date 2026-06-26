// SPDX-License-Identifier: Apache-2.0
//  ___ _   _ ____       ____  _____ _
// |_ _| | | |  _ \     |  _ \|  ___| |
//  | || |_| | |_) |____| | | | |_  | |
//  | ||  _  |  __/_____| |_| |  _| | |___
// |___|_| |_|_|        |____/|_|   |_____|
// OpenFL is a Ethereum-based reputation system to facilitate federated learning.
// This contract is part of the OpenFL research paper by Anton Wahrstätter. The contracts do only
// represent Proof-of-Concepts and have not been developed to be used in productive
// environments. Do not use them, except for testing purpose.

pragma solidity ^0.8.0;

import "./Types.sol";
import "./Clones.sol";

interface IJobListing {
    function initialize(
        bytes32 _modelHash,
        uint _min_collateral,
        uint _max_collateral,
        uint _reward,
        uint8 _min_rounds,
        uint8 _punishfactor,
        uint8 _punishfactorContrib,
        uint8 _freeriderPenalty,
        address _managerAddress,
        TaskType _taskType
    ) external payable;
}

interface IOpenFLChallenge {
    struct TaskRep {
        address user;
        int256 taskRepDelta;
        uint globalReputationScore;
        uint8 roundsParticipated;
    }
    function getTaskRepDeltaAndGRS() external view returns (TaskRep[] memory);
    function taskType() external view returns (TaskType);
}

contract OpenFLManager {
    event JobListingValid(bool isValid);

    struct User {
        // GlobalTaskRep is the per-task (= per-dataset) TaskRep. TaskType acts
        // as the dataset key (e.g. MNIST, CIFAR10). Mutated by valid
        // JobListings via applyUserTaskRepDelta after a challenge round.
        mapping(TaskType => uint256) GlobalTaskRep;
        // Per-task running state for the TaskRepCalc formula. WAD-scaled (1e18).
        // RunningCMean = E_k (EWMA mean of raw per-task contribution score J_k);
        // M2 = F_k (EWMA squared-deviation accumulator, variance proxy s_k).
        mapping(TaskType => uint256) RunningCMean;
        mapping(TaskType => uint256) M2;
        uint256 GlobalIntegrityRep;
        uint128 TotalContribScore;
        mapping(TaskType => uint256) QValue;
        // Number of tasks completed per TaskType. Used by the JobListing's
        // confidence formula (k in k/(k+N_0)) so maturity grows correctly
        // across tasks. Incremented by incrementTaskCount after each task.
        mapping(TaskType => uint256) TaskCount;
        // ETH balance mirrored from the challenge contract's globalReputationScore.
        // Written by Python after each task (local: from challenge; replay: via delta).
        uint256 Balance;
    }

    struct TaskSpecificUser {
        address userAddress;
        uint256 taskRep;
        uint256 globalIntegrityRep;
        uint128 totalContribScore;
        uint256 qValue;
        uint256 balance;
        uint256 taskCount;
    }

    function getUser(
        address addr,
        TaskType taskType
    ) public view returns (TaskSpecificUser memory) {
        TaskType key = _repKey(taskType);
        User storage u = users[addr];
        return TaskSpecificUser({
            userAddress: addr,
            taskRep: u.GlobalTaskRep[key],
            globalIntegrityRep: u.GlobalIntegrityRep,
            totalContribScore: u.TotalContribScore,
            qValue: u.QValue[key],
            balance: u.Balance,
            taskCount: u.TaskCount[key]
        });
    }

    function getUsersBatch(
        address[] calldata addrs,
        TaskType taskType
    ) external view returns (TaskSpecificUser[] memory result) {
        result = new TaskSpecificUser[](addrs.length);
        for (uint256 i = 0; i < addrs.length; i++) {
            result[i] = getUser(addrs[i], taskType);
        }
    }

    // Number of real task types (TaskType enum values excluding template=0).
    // Update in lock-step with the TaskType enum in Types.sol.
    uint8 internal constant REAL_TASK_TYPE_COUNT = 8;

    // Returns the names of all TaskType enum members in ordinal order
    // (index 0 = template, index 1 = Images, …).  Python uses this to build
    // its TaskType IntEnum dynamically so the two definitions stay in sync.
    function getTaskTypeNames() external pure returns (string[] memory names) {
        names = new string[](REAL_TASK_TYPE_COUNT + 1);
        names[0] = "template";
        names[1] = "Images";
        names[2] = "Language";
        names[3] = "Images_clothing";
        names[4] = "Images_objects";
        names[5] = "MNIST";
        names[6] = "CIFAR10";
        names[7] = "FashionMNIST";
        names[8] = "IMDB";
    }

    // Returns one TaskSpecificUser per real TaskType (Images=1 … IMDB=8), in
    // enum order, for the given user.  Use for logging/graphing where all task
    // types are needed after every task, not just the one that just ran.
    function getUserAllTaskTypes(
        address addr
    ) external view returns (TaskSpecificUser[] memory result) {
        result = new TaskSpecificUser[](REAL_TASK_TYPE_COUNT);
        for (uint8 i = 0; i < REAL_TASK_TYPE_COUNT; i++) {
            result[i] = getUser(addr, TaskType(i + 1)); // skip template (0)
        }
    }

    function getGrsAndTrsBatch(
        address[] calldata userAddresses,
        TaskType taskType
    ) external view returns (TaskSpecificUser[] memory result) {
        result = new TaskSpecificUser[](userAddresses.length);
        for (uint256 i = 0; i < userAddresses.length; i++) {
            result[i] = getUser(userAddresses[i], taskType);
        }
    }

    // Set a user's mirrored ETH balance. Callable by publisher (replay path) or
    // a registered valid job (production path).
    function setUserBalance(address user, uint256 value) external {
        require(
            validJobs[msg.sender] || msg.sender == publisher,
            "OFLM: caller not valid job or publisher"
        );
        users[user].Balance = value;
    }

    mapping(address => User) public users;
    mapping(address => bool) public validJobs;

    address public implementation;
    bytes32 public jobListingCodeHash;
    bytes32 public challengeCodeHash;
    address public publisher;

    // Reputation mode is fixed at deploy time. PerTask preserves the original
    // per-dataset behaviour (one TaskRep slot per TaskType, GIR updated each
    // task). GlobalOnly collapses all TaskType-keyed reads/writes onto a
    // single sentinel slot per user (TaskType.template) and disables GIR
    // updates so the GIR value remains at its prior value (default 0).
    ReputationMode public immutable reputationMode;

    // Sentinel TaskType used by _repKey when reputationMode == GlobalOnly.
    // template is unused for real tasks (it marks the uninitialised default),
    // so reusing its slot to hold the global reputation bucket cannot collide
    // with per-task data.
    TaskType internal constant GLOBAL_KEY = TaskType.template;

    constructor(ReputationMode _reputationMode) {
        publisher = msg.sender;
        reputationMode = _reputationMode;
    }

    // Maps a caller-supplied TaskType to the storage slot used for TaskRep
    // (and TaskRepCalc running state). Per-task mode passes the TaskType
    // through unchanged; global-only mode redirects every key onto the
    // shared sentinel slot.
    function _repKey(TaskType taskType) internal view returns (TaskType) {
        return
            reputationMode == ReputationMode.GlobalOnly ? GLOBAL_KEY : taskType;
    }

    function getUserRep(
        address addr,
        TaskType taskType
    ) public view returns (uint, uint, uint) {
        TaskType _taskType = _repKey(taskType);
        return (
            users[addr].GlobalTaskRep[_taskType],
            users[addr].GlobalIntegrityRep,
            users[addr].QValue[_taskType]
        );
    }

    function getTaskCount(
        address addr,
        TaskType taskType
    ) public view returns (uint256) {
        return users[addr].TaskCount[_repKey(taskType)];
    }

    function setChallengeCodeHash(bytes32 _hash) external {
        if (msg.sender != publisher) {
            return;
        }

        if (challengeCodeHash != bytes32(0)) {
            return;
        }

        challengeCodeHash = _hash;
    }

    function setJobListingCodeHash(bytes32 _hash) external {
        if (msg.sender != publisher) {
            return;
        }

        if (jobListingCodeHash != bytes32(0)) {
            return;
        }

        jobListingCodeHash = _hash;
    }

    function validateJob(address job) public view returns (bool) {
        bytes32 codeHash;

        assembly {
            codeHash := extcodehash(job)
        }

        return codeHash == jobListingCodeHash;
    }

    function getChallengeCodeHash() public view returns (bytes32) {
        return challengeCodeHash;
    }

    function registerJob(address job) external {
        bool validJob = validateJob(job);

        validJobs[job] = validJob;

        emit JobListingValid(validJob);
    }

    // Pull reputation data from a completed challenge and update this manager's user records.
    // Callable by: the challenge contract itself (production), or the publisher (Python/replay).
    function updateReputationsFromChallenge(
        address challengeAddr,
        TaskType taskType
    ) external {
        require(
            msg.sender == publisher || msg.sender == challengeAddr,
            "Unauthorized"
        );

        IOpenFLChallenge challenge = IOpenFLChallenge(challengeAddr);
        IOpenFLChallenge.TaskRep[] memory taskReps = challenge
            .getTaskRepDeltaAndGRS();

        for (uint i = 0; i < taskReps.length; i++) {
            IOpenFLChallenge.TaskRep memory tr = taskReps[i];
            if (tr.user == address(0)) continue;

            User storage u = users[tr.user];

            // Apply task rep delta (clamp to zero)
            int256 newTaskRep = int256(u.GlobalTaskRep[taskType]) + tr.taskRepDelta;
            u.GlobalTaskRep[taskType] = newTaskRep > 0
                ? uint256(newTaskRep)
                : 0;

            // Update global integrity rep (pre-exit balance from challenge)
            u.GlobalIntegrityRep = tr.globalReputationScore;

            // Accumulate total rounds participated
            u.TotalContribScore += tr.roundsParticipated;

            // QValue is managed separately by updateQValuesAfterSelection; not touched here.
        }
    }

    // Update Q-values after a job listing's participant selection.
    // All registrants get Q incremented (patience bonus for waiting).
    // Q is WAD-scaled (1e18). The patience formula per round:
    //   increment = k / n  (k = selected count, n = all registrants count)
    //   not selected: q_new = q_old + increment
    //   selected:     q_new = max(0, q_old + increment - WAD)
    // Callable by: the job listing contract (production) or publisher (Python/replay).
    uint256 internal constant Q_WAD = 1e18;

    function updateQValuesAfterSelection(
        address[] calldata allRegistrants,
        address[] calldata selected,
        TaskType taskType,
        bool hardReset
    ) external {
        require(
            msg.sender == publisher || validJobs[msg.sender],
            "Unauthorized"
        );

        uint256 n = allRegistrants.length;
        if (n == 0) return;
        uint256 k = selected.length;
        uint256 increment = (k * Q_WAD) / n;

        // Route through _repKey so the slot written here matches the one read by
        // getUserRep/setUserQValue. In GlobalOnly every TaskType aliases onto the
        // shared sentinel bucket, giving a single user-bound Q (accumulates while
        // idle on any task, resets on selection for any task). In PerTask mode
        // _repKey(taskType) == taskType, so behaviour is unchanged.
        TaskType key = _repKey(taskType);

        // Mark selected addresses for O(k) lookup inside the O(n) loop.
        mapping(address => bool) storage isSelected = _tmpSelected;
        for (uint i = 0; i < k; i++) {
            isSelected[selected[i]] = true;
        }

        for (uint i = 0; i < n; i++) {
            address addr = allRegistrants[i];
            uint256 q = users[addr].QValue[key];
            uint256 newQ = q + increment;
            if (isSelected[addr]) {
                newQ = hardReset ? 0 : (newQ >= Q_WAD ? newQ - Q_WAD : 0);
            }
            users[addr].QValue[key] = newQ;
        }

        // Clean up the temporary selected-flag mapping.
        for (uint i = 0; i < k; i++) {
            delete isSelected[selected[i]];
        }
    }

    mapping(address => bool) private _tmpSelected;

    event UserTaskRepUpdated(
        address indexed user,
        TaskType indexed taskType,
        uint256 oldValue,
        uint256 newValue
    );

    // Replace a user's per-task (= per-dataset) TaskRep with a new value.
    // TaskRep is updated once per task on completion, so the JobListing
    // computes the full new value (typically as a weighted blend of the
    // previous TaskRep and the rep earned for this task) and calls this
    // setter to overwrite the stored value.
    //
    // Only callable by a registered (valid) JobListing — register via
    // registerJob().
    function setUserTaskRep(
        address user,
        TaskType taskType,
        uint256 newValue
    ) external {
        require(
            validJobs[msg.sender] || msg.sender == publisher,
            "OFLM: caller not valid job or publisher"
        );

        TaskType key = _repKey(taskType);
        uint256 current = users[user].GlobalTaskRep[key];
        users[user].GlobalTaskRep[key] = newValue;

        emit UserTaskRepUpdated(user, key, current, newValue);
    }

    event TaskRepCalcStateUpdated(
        address indexed user,
        TaskType indexed taskType,
        uint256 newRunningCMean,
        uint256 newM2
    );

    // Read the per-(user, taskType) TaskRepCalc running state used by the
    // JobListing's contribution-score formula. Both values are WAD-scaled.
    function getTaskRepCalcState(
        address addr,
        TaskType taskType
    ) public view returns (uint256 runningCMean, uint256 m2) {
        TaskType key = _repKey(taskType);
        return (users[addr].RunningCMean[key], users[addr].M2[key]);
    }

    // Persist updated TaskRepCalc running state. Same auth model as
    // setUserTaskRep — only callable by a registered (valid) JobListing.
    function setTaskRepCalcState(
        address user,
        TaskType taskType,
        uint256 newRunningCMean,
        uint256 newM2
    ) external {
        require(
            validJobs[msg.sender] || msg.sender == publisher,
            "OFLM: caller not valid job or publisher"
        );

        TaskType key = _repKey(taskType);
        users[user].RunningCMean[key] = newRunningCMean;
        users[user].M2[key] = newM2;

        emit TaskRepCalcStateUpdated(user, key, newRunningCMean, newM2);
    }

    // Increment the per-(user, taskType) task count by 1. Called by a
    // JobListing after computing the new TaskRep so k grows correctly
    // across tasks for the confidence formula.
    function incrementTaskCount(address user, TaskType taskType) external {
        require(
            validJobs[msg.sender] || msg.sender == publisher,
            "OFLM: caller not valid job or publisher"
        );
        TaskType key = _repKey(taskType);
        users[user].TaskCount[key] += 1;
    }

    // Seed all per-(user, taskType) rep state in one transaction.
    // Used by Python to seed a fresh manager with prior-session state before
    // the job listing is deployed, replacing 4-5 serial setter calls per user.
    function seedRepState(
        address user,
        TaskType taskType,
        uint256 tr,
        uint256 gir,
        uint256 c_mean,
        uint256 m2,
        uint256 k,
        uint256 q
    ) external {
        require(msg.sender == publisher, "OFLM: caller not publisher");
        TaskType key = _repKey(taskType);
        User storage u = users[user];
        u.GlobalTaskRep[key] = tr;
        u.RunningCMean[key] = c_mean;
        u.M2[key] = m2;
        u.TaskCount[key] = k;
        u.QValue[key] = q;
        u.GlobalIntegrityRep = gir;
    }

    event UserIntegrityRepUpdated(
        address indexed user,
        uint256 oldValue,
        uint256 newValue
    );

    // Replace a user's Global Integrity Reputation (GIR). Callable by a
    // registered (valid) JobListing (production) or the publisher EOA (replay /
    // initialisation from Python). GIR is WAD-scaled.
    function setUserIntegrityRep(address user, uint256 newValue) external {
        require(
            validJobs[msg.sender] || msg.sender == publisher,
            "OFLM: caller not valid job or publisher"
        );

        uint256 current = users[user].GlobalIntegrityRep;
        users[user].GlobalIntegrityRep = newValue;

        emit UserIntegrityRepUpdated(user, current, newValue);
    }

    // Apply pre-computed TaskRep outputs from a completed challenge.
    // Callable by a registered challenge (production) or the publisher (replay).
    // Does no calculation — blindly writes the values computed by the challenge.
    function applyPrecomputedTaskReps(
        TaskRepRecord[] calldata records,
        TaskType taskType
    ) external {
        require(
            msg.sender == publisher || _isValidChallenge(msg.sender),
            "OFLM: unauthorized"
        );

        TaskType key = _repKey(taskType);

        for (uint i = 0; i < records.length; i++) {
            TaskRepRecord calldata r = records[i];
            if (r.user == address(0)) continue;

            User storage u = users[r.user];

            uint256 oldTaskRep = u.GlobalTaskRep[key];
            u.GlobalTaskRep[key] = r.newTaskRep;
            emit UserTaskRepUpdated(r.user, key, oldTaskRep, r.newTaskRep);

            u.RunningCMean[key] = r.newRunningCMean;
            u.M2[key] = r.newM2;
            emit TaskRepCalcStateUpdated(r.user, key, r.newRunningCMean, r.newM2);

            u.TaskCount[key] += 1;

            if (r.applyGIR) {
                uint256 oldGIR = u.GlobalIntegrityRep;
                u.GlobalIntegrityRep = r.newIntegrityRep;
                emit UserIntegrityRepUpdated(r.user, oldGIR, r.newIntegrityRep);
            }
        }
    }

    function _isValidChallenge(address addr) internal view returns (bool) {
        if (challengeCodeHash == bytes32(0)) return false;
        bytes32 codeHash;
        assembly {
            codeHash := extcodehash(addr)
        }
        return codeHash == challengeCodeHash;
    }
}
