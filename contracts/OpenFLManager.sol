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
import "./JobListing.sol";
import "./Clones.sol";

contract OpenFLManager {
    using Clones for address;

    event JobCreated(address job);

    struct User {
        mapping(address => JobListing) listings;
        mapping(TaskType => uint256) GlobalTaskRep;
        uint256 GlobalIntegrityRep;
        uint128 TotalContribScore;
        uint128 NumberOfTasksJoined;
    }

    mapping(address => User) public users;

    address public implementation;
    constructor() {
        implementation = address(new JobListing());
    }

    function CreateNewJob(
        bytes32 _modelHash,
        uint _min_collateral,
        uint _max_collateral,
        uint _reward,
        uint8 _min_rounds,
        uint8 _punishfactor,
        uint8 _punishfactorContrib,
        uint8 _freeriderPenalty,
        TaskType _taskType
    ) public payable {
        require(msg.value >= _reward + _min_collateral, "NEV");

        address clone = implementation.clone();

        JobListing(clone).initialize{value: _reward}(
            _modelHash,
            _min_collateral,
            _max_collateral,
            _reward,
            _min_rounds,
            _punishfactor,
            _punishfactorContrib,
            _freeriderPenalty,
            address(this),
            _taskType
        );

        users[msg.sender].listings[clone] = JobListing(clone);

        emit JobCreated(clone);
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
}
