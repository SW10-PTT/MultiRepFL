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
        int256 delta;
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
        mapping(TaskType => uint128) QValue;
    }

    struct TaskSpecificUser {
        address userAddress;
        uint256 taskRep;
        uint256 globalIntegrityRep;
        uint128 totalContribScore;
        uint128 qValue;
    }

    function getGrsAndTrsBatch(
        address[] calldata userAddresses,
        TaskType taskType
    ) external view returns (TaskSpecificUser[] memory result) {
        result = new TaskSpecificUser[](userAddresses.length);

        for (uint256 i = 0; i < userAddresses.length; i++) {
            User storage u = users[userAddresses[i]];

            result[i] = TaskSpecificUser({
                userAddress: userAddresses[i],
                taskRep: u.GlobalTaskRep[taskType],
                globalIntegrityRep: u.GlobalIntegrityRep,
                totalContribScore: u.TotalContribScore,
                qValue: u.QValue[taskType]
            });
        }
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
        TaskType key = _repKey(taskType);
        return (
            users[addr].GlobalTaskRep[key],
            users[addr].GlobalIntegrityRep,
            users[addr].QValue[taskType]
        );
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
            int256 newTaskRep = int256(u.GlobalTaskRep[taskType]) + tr.delta;
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
    // Selected participants get Q reset to Q_BASE (selected = fresh start).
    // Callable by: the job listing contract (production) or publisher (Python/replay).
    uint128 constant Q_BASE = 1;
    uint128 constant Q_INCREMENT = 1;

    function updateQValuesAfterSelection(
        address[] calldata allRegistrants,
        address[] calldata selected,
        TaskType taskType
    ) external {
        require(
            msg.sender == publisher || validJobs[msg.sender],
            "Unauthorized"
        );

        // Everyone who registered gets a patience increment.
        for (uint i = 0; i < allRegistrants.length; i++) {
            users[allRegistrants[i]].QValue[taskType] += Q_INCREMENT;
        }

        // Selected participants reset to base (their wait is over).
        for (uint i = 0; i < selected.length; i++) {
            users[selected[i]].QValue[taskType] = Q_BASE;
        }
    }

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
        require(validJobs[msg.sender], "OFLM: caller not valid job");

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
        require(validJobs[msg.sender], "OFLM: caller not valid job");

        TaskType key = _repKey(taskType);
        users[user].RunningCMean[key] = newRunningCMean;
        users[user].M2[key] = newM2;

        emit TaskRepCalcStateUpdated(user, key, newRunningCMean, newM2);
    }

    // Increment a user's lifetime task-participation counter by 1. Called by
    // the JobListing at end-of-task for each participant so that
    // `NumberOfTasksJoined` reflects the round index used by TaskRepCalc.
    function incrementNumberOfTasksJoined(address user) external {
        require(validJobs[msg.sender], "OFLM: caller not valid job");
        users[user].NumberOfTasksJoined += 1;
    }

    event UserIntegrityRepUpdated(
        address indexed user,
        uint256 oldValue,
        uint256 newValue
    );

    // Replace a user's Global Integrity Reputation (GIR). Same auth model as
    // setUserTaskRep — only callable by a registered (valid) JobListing.
    // GIR is WAD-scaled in [0, WAD]; the JobListing computes the EWMA-blended
    // value off the per-task vote tallies at end-of-task and writes it here.
    function setUserIntegrityRep(address user, uint256 newValue) external {
        require(validJobs[msg.sender], "OFLM: caller not valid job");

        uint256 current = users[user].GlobalIntegrityRep;
        users[user].GlobalIntegrityRep = newValue;

        emit UserIntegrityRepUpdated(user, current, newValue);
    }
}
