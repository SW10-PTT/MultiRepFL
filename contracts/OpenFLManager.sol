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
        mapping(TaskType => uint256) GlobalTaskRep;
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

    constructor() {
        publisher = msg.sender;
    }

    function getUserRep(
        address addr,
        TaskType taskType
    ) public view returns (uint, uint, uint128) {
        return (
            users[addr].GlobalTaskRep[taskType],
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
}
