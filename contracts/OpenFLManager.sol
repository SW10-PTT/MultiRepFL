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
        uint128 NumberOfTasksJoined;
    }

    mapping(address => User) public users;
    mapping(address => bool) public validJobs;

    address public implementation;
    bytes32 public jobListingCodeHash;
    bytes32 public challengeCodeHash;
    address public publisher;

    constructor() {
        publisher = msg.sender;
    }

    function getUserRep(
        address addr,
        TaskType taskType
    ) public view returns (uint, uint, uint) {
        return (
            users[addr].GlobalTaskRep[taskType],
            users[addr].GlobalIntegrityRep,
            users[addr].NumberOfTasksJoined
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

        uint256 current = users[user].GlobalTaskRep[taskType];
        users[user].GlobalTaskRep[taskType] = newValue;

        emit UserTaskRepUpdated(user, taskType, current, newValue);
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
        return (
            users[addr].RunningCMean[taskType],
            users[addr].M2[taskType]
        );
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

        users[user].RunningCMean[taskType] = newRunningCMean;
        users[user].M2[taskType] = newM2;

        emit TaskRepCalcStateUpdated(user, taskType, newRunningCMean, newM2);
    }

    // Increment a user's lifetime task-participation counter by 1. Called by
    // the JobListing at end-of-task for each participant so that
    // `NumberOfTasksJoined` reflects the round index used by TaskRepCalc.
    function incrementNumberOfTasksJoined(address user) external {
        require(validJobs[msg.sender], "OFLM: caller not valid job");
        users[user].NumberOfTasksJoined += 1;
    }
}
