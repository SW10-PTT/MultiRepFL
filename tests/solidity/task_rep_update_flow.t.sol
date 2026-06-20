// SPDX-License-Identifier: MIT
pragma solidity ^0.8.9;

import "forge-std/Test.sol";
import "../../contracts/OpenFLChallenge.sol";
import "../../contracts/OpenFLManager.sol";
import "../../contracts/Types.sol";

// Harness that lets tests inject participant state and bypass the jobListing
// requirement. Uses TaskType.template in the constructor (skips jobListing
// check), then allows the task type and round to be overridden.
contract MockChallenge is OpenFLChallenge {
    constructor(address mgr) payable OpenFLChallenge(
        ChallengeSpecifications({
            modelHash: bytes32(0),
            min_collateral: 1e18,
            max_collateral: 1.8e18,
            managerAddress: mgr,
            reward: 1e18,
            min_rounds: 1,
            punishfactor: 3,
            punishfactorContrib: 3,
            freeriderPenalty: 0,
            taskType: TaskType.template,
            jobListingAddress: address(0),
            trAlpha: 2e17,
            trNBlend: 2e17,
            trN0: 2,
            trLambda: 5,
            trIntegrityLearningRate: 2e17,
            trGainCapMultiplier: 2
        })
    ) {}

    function addParticipant(
        address addr,
        int256 delta,
        uint256 posVotes,
        uint256 totVotes
    ) external {
        users[addr].taskRepDelta = delta;
        users[addr].isRegistered = true;
        participants.push(addr);
        nrOfActiveParticipants++;
        positiveVotesReceived[addr] = posVotes;
        totalVotesReceived[addr] = totVotes;
    }

    function setTaskTypeMNIST() external { taskType = TaskType.MNIST; }
    function setRound(uint8 r) external { round = r; }
}

contract TaskRepUpdateFlowTest is Test {
    uint256 constant WAD = 1e18;

    OpenFLManager manager;
    MockChallenge  challenge;
    address        user = address(0xBEEF);

    function setUp() public {
        manager = new OpenFLManager(ReputationMode.PerTask);
        challenge = new MockChallenge(address(manager));

        // Authorise challenge to call applyPrecomputedTaskReps on the manager.
        manager.setChallengeCodeHash(address(challenge).codehash);

        // Configure challenge for a real task.
        challenge.setTaskTypeMNIST();
        challenge.setRound(1);

        // Wire a participant with a positive delta (0.2e18 above stake).
        // positiveVotes=3, totalVotes=4 → GIR will be updated.
        challenge.addParticipant(user, int256(2e17), 3, 4);
    }

    // Two-step helper mirroring Python's _finalize_reputations:
    // challenge stores records, then publisher applies them to manager.
    function _computeAndApply(MockChallenge c, TaskType tt) internal {
        c.computeAndRecordTaskReps();
        TaskRepRecord[] memory records = c.getTaskRepRecords();
        manager.applyPrecomputedTaskReps(records, tt);
    }

    function testUpdateWritesNonZeroTaskRep() public {
        (uint256 tr0,,) = manager.getUserRep(user, TaskType.MNIST);
        uint256 k0 = manager.getTaskCount(user, TaskType.MNIST);
        assertEq(tr0, 0, "pre: TR should be 0");
        assertEq(k0,  0, "pre: task count should be 0");

        _computeAndApply(challenge, TaskType.MNIST);

        (uint256 tr1, uint256 gir1,) = manager.getUserRep(user, TaskType.MNIST);
        uint256 k1 = manager.getTaskCount(user, TaskType.MNIST);

        assertGt(tr1,  0, "TR must be non-zero after update");
        assertGt(gir1, 0, "GIR must be non-zero after update (positive votes)");
        assertEq(k1,   1, "task count must be 1 after first update");
    }

    function testIdempotency_secondCallReverts() public {
        challenge.computeAndRecordTaskReps();
        vm.expectRevert("OFC: already computed");
        challenge.computeAndRecordTaskReps();
    }

    function testGuard_revertsBeforeAnyRound() public {
        MockChallenge fresh = new MockChallenge(address(manager));
        manager.setChallengeCodeHash(address(fresh).codehash);
        fresh.setTaskTypeMNIST();
        vm.expectRevert("OFC: no rounds settled");
        fresh.computeAndRecordTaskReps();
    }

    function testSecondTaskIncrementsTaskCount() public {
        _computeAndApply(challenge, TaskType.MNIST);
        uint256 k1 = manager.getTaskCount(user, TaskType.MNIST);
        assertEq(k1, 1);

        MockChallenge c2 = new MockChallenge(address(manager));
        manager.setChallengeCodeHash(address(c2).codehash);
        c2.setTaskTypeMNIST();
        c2.setRound(1);
        c2.addParticipant(user, int256(2e17), 3, 4);
        _computeAndApply(c2, TaskType.MNIST);

        uint256 k2 = manager.getTaskCount(user, TaskType.MNIST);
        assertEq(k2, 2, "task count should be 2 after second challenge");
    }

    function testRecordsStoredOnChallenge() public {
        challenge.computeAndRecordTaskReps();
        TaskRepRecord[] memory records = challenge.getTaskRepRecords();
        assertEq(records.length, 1, "one record per participant");
        assertEq(records[0].user, user);
        assertGt(records[0].newTaskRep, 0);
        assertTrue(records[0].applyGIR);
    }

    function testCIFARAndMNISTCountsAreIndependent() public {
        _computeAndApply(challenge, TaskType.MNIST);
        assertEq(manager.getTaskCount(user, TaskType.MNIST),   1, "MNIST count after 1 task");
        assertEq(manager.getTaskCount(user, TaskType.CIFAR10), 0, "CIFAR10 count still 0");
    }
}
