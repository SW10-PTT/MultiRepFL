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

    // ---- Q-value: single user-bound bucket in GlobalOnly --------------------

    address constant USER2 = address(0xCAFE);

    function _pair(address a, address b) internal pure returns (address[] memory) {
        address[] memory arr = new address[](2);
        arr[0] = a;
        arr[1] = b;
        return arr;
    }

    function _one(address a) internal pure returns (address[] memory) {
        address[] memory arr = new address[](1);
        arr[0] = a;
        return arr;
    }

    // Idle user's Q must accumulate AND be the same value regardless of which
    // TaskType it is read under (one shared sentinel slot). With n=2, k=1 the
    // per-round increment is 0.5 WAD.
    function testQValue_accumulatesAndAliasesAcrossTaskTypes() public {
        // Round 1 on MNIST: select USER2, so USER is idle. (hardReset irrelevant for idle users.)
        manager.updateQValuesAfterSelection(_pair(USER, USER2), _one(USER2), TaskType.MNIST, false);
        // Round 2 on CIFAR10: select USER2 again, USER still idle.
        manager.updateQValuesAfterSelection(_pair(USER, USER2), _one(USER2), TaskType.CIFAR10, false);

        (, , uint256 qMnist) = manager.getUserRep(USER, TaskType.MNIST);
        (, , uint256 qCifar) = manager.getUserRep(USER, TaskType.CIFAR10);

        assertEq(qMnist, qCifar, "Q must read the same under any TaskType in GlobalOnly");
        assertEq(qMnist, 1e18, "idle Q must accumulate across datasets (0.5 + 0.5 WAD)");
    }

    // Soft reset (hardReset=false): selection on ANY TaskType subtracts one WAD
    // from the single user-bound Q.
    function testQValue_softResetsOnSelectionAnyTaskType() public {
        // Build USER's Q up to 1.0 WAD while idle (two idle rounds).
        manager.updateQValuesAfterSelection(_pair(USER, USER2), _one(USER2), TaskType.MNIST, false);
        manager.updateQValuesAfterSelection(_pair(USER, USER2), _one(USER2), TaskType.MNIST, false);
        (, , uint256 qBefore) = manager.getUserRep(USER, TaskType.MNIST);
        assertEq(qBefore, 1e18, "precondition: Q built to 1.0 WAD");

        // Select USER on a DIFFERENT TaskType (CIFAR10): newQ = 1.0 + 0.5 - 1.0 = 0.5 WAD.
        manager.updateQValuesAfterSelection(_pair(USER, USER2), _one(USER), TaskType.CIFAR10, false);
        (, , uint256 qAfter) = manager.getUserRep(USER, TaskType.MNIST);

        assertLt(qAfter, qBefore, "selection on any task must reduce the shared Q");
        assertEq(qAfter, 0.5e18, "soft reset subtracts exactly Q_WAD from the post-increment value");
    }

    // Hard reset (hardReset=true): selection on ANY TaskType zeroes the single
    // user-bound Q outright — must apply to the shared GlobalOnly slot.
    function testQValue_hardResetZeroesOnSelectionAnyTaskType() public {
        // Build USER's Q up to 1.0 WAD while idle.
        manager.updateQValuesAfterSelection(_pair(USER, USER2), _one(USER2), TaskType.MNIST, true);
        manager.updateQValuesAfterSelection(_pair(USER, USER2), _one(USER2), TaskType.MNIST, true);
        (, , uint256 qBefore) = manager.getUserRep(USER, TaskType.MNIST);
        assertEq(qBefore, 1e18, "precondition: Q built to 1.0 WAD");

        // Select USER on a DIFFERENT TaskType with hard reset: Q must drop to 0
        // (not 0.5), proving the hard reset hits the shared sentinel slot.
        manager.updateQValuesAfterSelection(_pair(USER, USER2), _one(USER), TaskType.CIFAR10, true);
        (, , uint256 qMnist) = manager.getUserRep(USER, TaskType.MNIST);
        (, , uint256 qCifar) = manager.getUserRep(USER, TaskType.CIFAR10);

        assertEq(qMnist, 0, "hard reset must zero the shared Q");
        assertEq(qCifar, 0, "hard reset is visible under any TaskType (single bucket)");
    }
}

// PerTask mode must keep Q strictly per-(user, task) — the fix must not leak
// the GlobalOnly aliasing into the default mode.
contract PerTaskQValueTest is Test {
    OpenFLManager manager;
    address constant USER = address(0xBEEF);
    address constant USER2 = address(0xCAFE);

    function setUp() public {
        manager = new OpenFLManager(ReputationMode.PerTask);
    }

    function _pair(address a, address b) internal pure returns (address[] memory) {
        address[] memory arr = new address[](2);
        arr[0] = a;
        arr[1] = b;
        return arr;
    }

    function _one(address a) internal pure returns (address[] memory) {
        address[] memory arr = new address[](1);
        arr[0] = a;
        return arr;
    }

    function testQValue_isolatedPerTaskType() public {
        // USER idle on MNIST only.
        manager.updateQValuesAfterSelection(_pair(USER, USER2), _one(USER2), TaskType.MNIST, false);

        (, , uint256 qMnist) = manager.getUserRep(USER, TaskType.MNIST);
        (, , uint256 qCifar) = manager.getUserRep(USER, TaskType.CIFAR10);

        assertEq(qMnist, 0.5e18, "MNIST Q accrues");
        assertEq(qCifar, 0, "CIFAR Q must stay isolated in PerTask mode");
    }

    // Hard reset in PerTask must zero only the selected task's Q slot.
    function testQValue_hardResetIsolatedPerTaskType() public {
        // Build MNIST Q to 1.0 while idle, leave CIFAR untouched.
        manager.updateQValuesAfterSelection(_pair(USER, USER2), _one(USER2), TaskType.MNIST, true);
        manager.updateQValuesAfterSelection(_pair(USER, USER2), _one(USER2), TaskType.MNIST, true);
        // Hard-reset USER on MNIST (select USER). MNIST Q -> 0; CIFAR Q stays 0.
        manager.updateQValuesAfterSelection(_pair(USER, USER2), _one(USER), TaskType.MNIST, true);

        (, , uint256 qMnist) = manager.getUserRep(USER, TaskType.MNIST);
        (, , uint256 qCifar) = manager.getUserRep(USER, TaskType.CIFAR10);

        assertEq(qMnist, 0, "MNIST Q hard-reset to 0");
        assertEq(qCifar, 0, "CIFAR slot never touched (per-task isolation holds)");
    }
}
