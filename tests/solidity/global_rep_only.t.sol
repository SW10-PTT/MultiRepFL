// SPDX-License-Identifier: MIT
pragma solidity ^0.8.9;

import "forge-std/Test.sol";
import "../../contracts/OpenFLChallenge.sol";
import "../../contracts/OpenFLManager.sol";
import "../../contracts/Types.sol";

// Harness for GlobalOnly mode tests — exposes tApplyOne without the
// computeAndRecordTaskReps() idempotency guard.
contract GlobalOnlyHarness is OpenFLChallenge {
    constructor(address mgr) payable OpenFLChallenge(
        ChallengeSpecifications({
            modelHash: bytes32(0),
            min_collateral: 1e18,
            max_collateral: 1.8e18,
            managerAddress: mgr,
            reward: 10e18,
            min_rounds: 1,
            punishfactor: 3,
            punishfactorContrib: 3,
            freeriderPenalty: 0,
            taskType: TaskType.template,
            jobListingAddress: address(0)
        })
    ) {}

    function tApplyOne(
        address user,
        int256 delta,
        uint256 posVotes,
        uint256 totVotes,
        TaskType tt,
        uint256 reward,
        uint256 nrActive
    ) external {
        users[user].taskRepDelta = delta;
        positiveVotesReceived[user] = posVotes;
        totalVotesReceived[user] = totVotes;
        uint256 savedReward = totalReward;
        TaskType savedTT = taskType;
        totalReward = reward;
        taskType = tt;

        IOpenFLManager mgr = IOpenFLManager(managerAddress);
        bool applyGIR = mgr.reputationMode() == ReputationMode.PerTask;
        TaskRepRecord memory rec = _computeOneRecord(mgr, user, applyGIR, nrActive);

        totalReward = savedReward;
        taskType = savedTT;
        delete users[user].taskRepDelta;
        delete positiveVotesReceived[user];
        delete totalVotesReceived[user];

        TaskRepRecord[] memory records = new TaskRepRecord[](1);
        records[0] = rec;
        mgr.applyPrecomputedTaskReps(records, tt);
    }
}

contract GlobalRepOnlyTest is Test {
    OpenFLManager manager;
    GlobalOnlyHarness h;

    address constant USER = address(0xBEEF);
    uint256 constant WAD = 1e18;

    function setUp() public {
        manager = new OpenFLManager(ReputationMode.GlobalOnly);
        h = new GlobalOnlyHarness(address(manager));
        manager.setChallengeCodeHash(address(h).codehash);
    }

    function testMode_reportsGlobalOnly() public {
        assertEq(uint(manager.reputationMode()), uint(ReputationMode.GlobalOnly));
    }

    // GlobalOnly must not update the user's GIR even when votes are present.
    function testGIR_remainsZero_evenWithPerfectVotes() public {
        h.tApplyOne(USER, 0, 5, 5, TaskType.MNIST, 10e18, 5);
        (, uint256 storedGIR, ) = manager.getUserRep(USER, TaskType.MNIST);
        assertEq(storedGIR, 0, "GIR must be untouched in GlobalOnly");
    }

    // The TaskRep slot must be shared across TaskTypes.
    function testTaskRep_singleBucketAcrossTaskTypes() public {
        h.tApplyOne(USER, int256(2e18), 0, 0, TaskType.MNIST, 10e18, 5);
        (uint256 kMnist, , uint256 nrMnist) = manager.getUserRep(USER, TaskType.MNIST);
        (uint256 kCifar, , uint256 nrCifar) = manager.getUserRep(USER, TaskType.CIFAR10);

        assertEq(kMnist, kCifar, "TaskRep must alias across TaskTypes");
        assertEq(nrMnist, nrCifar, "task counter is per-user, not per-task");
        assertGt(kMnist, 0, "TaskRep should be non-zero after a positive task");
    }

    // Two tasks under different TaskTypes should compound into the same sentinel slot.
    function testTaskRep_secondTaskUsesFirstTasksPriorAcrossTaskTypes() public {
        h.tApplyOne(USER, int256(2e18), 0, 0, TaskType.MNIST, 10e18, 5);
        (uint256 afterMnist, , ) = manager.getUserRep(USER, TaskType.MNIST);

        h.tApplyOne(USER, int256(2e18), 0, 0, TaskType.CIFAR10, 10e18, 5);
        (uint256 afterCifar, , ) = manager.getUserRep(USER, TaskType.CIFAR10);

        assertGe(afterCifar, afterMnist, "TaskRep must compound, not reset");
    }
}
