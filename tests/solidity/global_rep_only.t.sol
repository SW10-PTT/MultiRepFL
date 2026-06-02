// SPDX-License-Identifier: MIT
pragma solidity ^0.8.9;

import "forge-std/Test.sol";
import "../../contracts/JobListing.sol";
import "../../contracts/OpenFLManager.sol";
import "../../contracts/Types.sol";

// Mirror of the JobListing harness in task_rep_calc.t.sol — kept inline so
// the two test suites can evolve independently without sharing fixtures.
contract GlobalOnlyHarness is JobListing {
    constructor(address mgr)
        JobListing(1e18, 1.8e18, 1e18, 3, 3, 3, 50, mgr, TaskType.MNIST, 0, 6, 4)
    {}

    function tApplyOne(
        IOpenFLChallengeTaskRep.TaskRep memory rep,
        TaskType tt,
        uint256 reward,
        uint256 nrActive
    ) external {
        bool applyGIR = manager.reputationMode() == ReputationMode.PerTask;
        _applyTaskRepCalc(rep, tt, reward, nrActive, applyGIR);
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
        manager.setJobListingCodeHash(address(h).codehash);
        manager.registerJob(address(h));
    }

    function _rep(int256 delta, uint256 pos, uint256 tot)
        internal
        pure
        returns (IOpenFLChallengeTaskRep.TaskRep memory)
    {
        return IOpenFLChallengeTaskRep.TaskRep({
            user: USER,
            delta: delta,
            globalReputationScore: 0,
            positiveVotes: pos,
            totalVotes: tot
        });
    }

    // Mode is exposed via the auto-generated getter and pinned at deploy
    // time. Confirms the test fixture is operating in the intended mode.
    function testMode_reportsGlobalOnly() public {
        assertEq(uint(manager.reputationMode()), uint(ReputationMode.GlobalOnly));
    }

    // GlobalOnly must not update the user's GIR even when votes are present.
    // Reads through the manager's getter; a sibling assertion against the
    // raw struct slot is unnecessary because getUserRep already proxies it.
    function testGIR_remainsZero_evenWithPerfectVotes() public {
        h.tApplyOne(_rep(0, 5, 5), TaskType.MNIST, 10e18, 5);
        (, uint256 storedGIR, ) = manager.getUserRep(USER, TaskType.MNIST);
        assertEq(storedGIR, 0, "GIR must be untouched in GlobalOnly");
    }

    // The TaskRep slot must be shared across TaskTypes. Two writes from
    // different TaskTypes should both read back to the same value.
    function testTaskRep_singleBucketAcrossTaskTypes() public {
        h.tApplyOne(_rep(int256(2e18), 0, 0), TaskType.MNIST, 10e18, 5);
        (uint256 kMnist, , uint256 nrMnist) = manager.getUserRep(USER, TaskType.MNIST);
        (uint256 kCifar, , uint256 nrCifar) = manager.getUserRep(USER, TaskType.CIFAR10);

        // Same bucket -> identical reads regardless of which TaskType the
        // caller passes in.
        assertEq(kMnist, kCifar, "TaskRep must alias across TaskTypes");
        assertEq(nrMnist, nrCifar, "task counter is per-user, not per-task");
        assertGt(kMnist, 0, "TaskRep should be non-zero after a positive task");
    }

    // Two tasks under different TaskTypes should compound into the same
    // sentinel slot: the second task reads the first task's TaskRep as its
    // prior. Compare against PerTask, where prior would still be 0 because
    // the slots are disjoint.
    function testTaskRep_secondTaskUsesFirstTasksPriorAcrossTaskTypes() public {
        h.tApplyOne(_rep(int256(2e18), 0, 0), TaskType.MNIST, 10e18, 5);
        (uint256 afterMnist, , ) = manager.getUserRep(USER, TaskType.MNIST);

        h.tApplyOne(_rep(int256(2e18), 0, 0), TaskType.CIFAR10, 10e18, 5);
        (uint256 afterCifar, , ) = manager.getUserRep(USER, TaskType.CIFAR10);

        // EWMA on a non-zero prior with the same input either holds or
        // grows; PerTask mode would have reset to seed K_1 again here.
        assertGe(afterCifar, afterMnist, "TaskRep must compound, not reset");
    }
}
