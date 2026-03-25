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
        mapping(TaskType => uint256) GlobalTaskRep;
        uint256 GlobalIntegrityRep;
        uint128 TotalContribScore;
        uint128 NumberOfTasksJoined;
    }

    mapping(address => User) public users;
    mapping(address => bool) public validJobs;

    address public implementation;
    bytes32 public jobListingCodeHash;
    constructor() {}

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

    //This is a constant in a final version
    function setJobListingCodeHash(bytes32 _hash) external {
        jobListingCodeHash = _hash;
    }

    function validateJob(address job) public view returns (bool) {
        bytes32 codeHash;

        assembly {
            codeHash := extcodehash(job)
        }

        return codeHash == jobListingCodeHash;
    }

    function registerJob(address job) external {
        bool validJob = validateJob(job);

        validJobs[job] = validJob;

        emit JobListingValid(validJob);
    }
}
