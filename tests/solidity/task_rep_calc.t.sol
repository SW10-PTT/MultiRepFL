// SPDX-License-Identifier: MIT
pragma solidity ^0.8.9;

import "forge-std/Test.sol";
import "../../contracts/OpenFLChallenge.sol";
import "../../contracts/OpenFLManager.sol";
import "../../contracts/Types.sol";

// Test harness — exposes OpenFLChallenge's internal TR helpers for direct unit testing.
contract OpenFLChallengeHarness is OpenFLChallenge {
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
            jobListingAddress: address(0),
            trAlpha: 2e17,
            trNBlend: 2e17,
            trN0: 2,
            trLambda: 5,
            trIntegrityLearningRate: 2e17,
            trGainCapMultiplier: 2
        })
    ) {}

    function tTransformDelta(
        int256 delta,
        uint256 stake,
        uint256 reward,
        uint256 nrActive
    ) external view returns (uint256) {
        return _trTransformDelta(delta, stake, reward, nrActive);
    }

    function tUpdateRunningStats(
        uint256 ContributionScore,
        uint256 priorRunningCMean,
        uint256 priorM2,
        uint256 k
    ) external view returns (uint256, uint256) {
        return _trUpdateRunningStats(ContributionScore, priorRunningCMean, priorM2, k);
    }

    function tComputeConfidence(uint256 k, uint256 s_k)
        external
        view
        returns (uint256)
    {
        return _trComputeConfidence(k, s_k);
    }

    function tUpdateContribScore(
        uint256 PriorTaskRep,
        uint256 Confidence,
        uint256 ContributionScore
    ) external view returns (uint256) {
        return _trUpdateContribScore(PriorTaskRep, Confidence, ContributionScore);
    }

    function tUpdateIntegrityRep(
        uint256 priorIntegrityRep,
        uint256 positiveVotes,
        uint256 totalVotes
    ) external view returns (uint256) {
        return _trUpdateIntegrityRep(priorIntegrityRep, positiveVotes, totalVotes);
    }

    // Integration helper: directly applies one participant's TR update without
    // the computeAndRecordTaskReps() idempotency guard.
    // Temporarily sets per-challenge state so the inherited helpers can run,
    // then restores it and applies the result to the manager.
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

contract TaskRepCalcTest is Test {
    OpenFLManager manager;
    OpenFLChallengeHarness h;

    uint256 constant WAD = 1e18;
    uint256 constant ALPHA = 2e17;
    uint256 constant N_BLEND = 2e17;

    function setUp() public {
        manager = new OpenFLManager(ReputationMode.PerTask);
        h = new OpenFLChallengeHarness(address(manager));
        manager.setChallengeCodeHash(address(h).codehash);
    }

    // ============================================================
    // _transformDelta
    // ============================================================

    function testTransform_zeroDelta_mapsToStakeOverRange() public {
        // stake=1, reward=10, nrActive=5 -> maxGain = 1*10/5 = 2
        // range = 3, shifted = 1 -> ContributionScore = 1/3 * WAD ≈ 0.333e18
        assertEq(h.tTransformDelta(0, 1e18, 10e18, 5), WAD / 3);
    }

    function testTransform_atMaxGain_returnsWAD() public {
        assertEq(h.tTransformDelta(int256(4e18), 1e18, 10e18, 5), WAD);
    }

    function testTransform_aboveMaxGain_clipsToWAD() public {
        assertEq(h.tTransformDelta(int256(100e18), 1e18, 10e18, 5), WAD);
    }

    function testTransform_minusStake_returnsZero() public {
        assertEq(h.tTransformDelta(-int256(1e18), 1e18, 10e18, 5), 0);
    }

    function testTransform_belowMinusStake_clipsToZero() public {
        assertEq(h.tTransformDelta(-int256(50e18), 1e18, 10e18, 5), 0);
    }

    function testTransform_nrActiveZero_capCollapsesToStakeOnly() public {
        // maxGain=0, range=stake; delta=0 -> shifted=range -> ContributionScore=WAD
        assertEq(h.tTransformDelta(0, 1e18, 10e18, 0), WAD);
    }

    function testTransform_at1xAverage_clipsToWAD() public {
        // delta = reward/nrActive = 2; with 1x cap maxGain=2, range=3, shifted=3 -> clips to WAD
        assertEq(h.tTransformDelta(int256(2e18), 1e18, 10e18, 5), WAD);
    }

    // ============================================================
    // _updateRunningStats
    // ============================================================

    function testRunning_firstTask_seedsE_FzeroOnFreshUser() public {
        (uint256 newRunningCMean, uint256 newM2) = h.tUpdateRunningStats(5e17, 0, 0, 1);
        assertEq(newRunningCMean, 5e17);
        assertEq(newM2, 0);
    }

    function testRunning_secondTask_EWMAblendMean() public {
        // priorRunningCMean=0.5, ContributionScore=1: newRunningCMean = 0.8*0.5 + 0.2*1 = 0.6
        (uint256 newRunningCMean, ) = h.tUpdateRunningStats(1e18, 5e17, 0, 2);
        assertEq(newRunningCMean, 6e17);
    }

    function testRunning_secondTask_variancePositive() public {
        // priorRunningCMean=0.5, ContributionScore=1 -> newRunningCMean=0.6, absDelta=0.5, absDelta2=0.4
        // newM2 = (ALPHA * 0.5 * 0.4)/WAD^2 = (0.2 * 0.5 * 0.4) = 0.04 in WAD
        (, uint256 newM2) = h.tUpdateRunningStats(1e18, 5e17, 0, 2);
        assertEq(newM2, 4e16);
    }

    function testRunning_stableJEqualspriorRunningCMean_FdecaysOnly() public {
        // ContributionScore == priorRunningCMean -> Delta=0 -> variance contribution=0; existing F decays.
        (uint256 newRunningCMean, uint256 newM2) =
            h.tUpdateRunningStats(5e17, 5e17, 1e17, 2);
        assertEq(newRunningCMean, 5e17);
        assertEq(newM2, 8e16); // 0.8 * 0.1
    }

    function testRunning_signSafe_negativeDeltaProducesSameVariance() public {
        // |C*D| identical whether ContributionScore above or below priorRunningCMean (Welford symmetry).
        (, uint256 fAbove) = h.tUpdateRunningStats(1e18, 5e17, 0, 2);
        (, uint256 fBelow) = h.tUpdateRunningStats(0, 5e17, 0, 2);
        assertEq(fAbove, fBelow);
    }

    // ============================================================
    // _computeConfidence
    // ============================================================

    function testConfidence_kZero_returnsZero() public {
        assertEq(h.tComputeConfidence(0, 0), 0);
    }

    function testConfidence_k1_sZero_oneOverThree() public {
        assertEq(h.tComputeConfidence(1, 0), WAD / 3);
    }

    function testConfidence_largeK_approachesOne() public {
        assertEq(h.tComputeConfidence(1000, 0), (1000 * WAD) / 1002);
    }

    function testConfidence_highVariance_drivesDown() public {
        uint256 maturity = (10 * WAD) / 12;
        uint256 stability = (WAD * WAD) / (WAD + 20 * WAD);
        assertEq(h.tComputeConfidence(10, WAD), (maturity * stability) / WAD);
    }

    function testConfidence_monotoneInK_atFixedS() public {
        uint256 prev = 0;
        for (uint256 k = 1; k <= 50; k++) {
            uint256 Confidence = h.tComputeConfidence(k, 0);
            assertGe(Confidence, prev);
            prev = Confidence;
        }
    }

    function testConfidence_monotoneNonIncreasingInS() public {
        uint256 prev = type(uint256).max;
        for (uint256 i = 0; i <= 10; i++) {
            uint256 Confidence = h.tComputeConfidence(10, i * 1e17);
            assertLe(Confidence, prev);
            prev = Confidence;
        }
    }

    // ============================================================
    // _updateContribScore
    // ============================================================

    function testContribScore_freshUser_firstTask() public {
        uint256 Confidence = WAD / 3;
        uint256 ContributionScore = 5e17;
        uint256 weighted = (Confidence * ContributionScore) / WAD;
        uint256 expected = (N_BLEND * weighted) / WAD;
        assertEq(h.tUpdateContribScore(0, Confidence, ContributionScore), expected);
    }

    function testContribScore_zeroConfidence_decaysPrior20pct() public {
        assertEq(h.tUpdateContribScore(5e17, 0, 1e18), 4e17);
    }

    function testContribScore_perfectInputs_holdsAtOne() public {
        assertEq(h.tUpdateContribScore(1e18, 1e18, 1e18), 1e18);
    }

    // ============================================================
    // _updateIntegrityRep (Global Integrity Reputation)
    // ============================================================

    function testGIR_zeroTotalVotes_decaysPrior() public {
        // V = 0 when no votes received -> GIR = (1-LR)*prior
        // LR = 0.2 -> GIR = 0.8 * prior
        assertEq(h.tUpdateIntegrityRep(5e17, 0, 0), 4e17);
    }

    function testGIR_allPositive_sendsTowardsOne() public {
        // V = (5/5)^2 = 1 -> GIR = 0.8*0.5 + 0.2*1 = 0.6
        assertEq(h.tUpdateIntegrityRep(5e17, 5, 5), 6e17);
    }

    function testGIR_zeroPositive_decaysPriorOnly() public {
        // V = (0/5)^2 = 0 -> GIR = 0.8*0.5 + 0.2*0 = 0.4
        assertEq(h.tUpdateIntegrityRep(5e17, 0, 5), 4e17);
    }

    function testGIR_halfPositive_squared() public {
        // V = (1/2)^2 = 0.25 -> GIR = 0.8*0 + 0.2*0.25 = 0.05
        assertEq(h.tUpdateIntegrityRep(0, 1, 2), 5e16);
    }

    function testGIR_perfectPriorAllPositive_holdsAtOne() public {
        assertEq(h.tUpdateIntegrityRep(1e18, 7, 7), 1e18);
    }

    function testGIR_convergesToOne_underAllPositive() public {
        uint256 prior = 0;
        for (uint256 i = 0; i < 100; i++) {
            prior = h.tUpdateIntegrityRep(prior, 5, 5);
        }
        assertGt(prior, 99e16);
    }

    function testGIR_decaysToZero_underNoSignal() public {
        // No votes received every task -> V always 0 -> prior decays geometrically.
        uint256 prior = 1e18;
        for (uint256 i = 0; i < 100; i++) {
            prior = h.tUpdateIntegrityRep(prior, 0, 0);
        }
        assertLt(prior, 1e16);
    }

    function testGIR_stayswithinWAD_fuzzPattern() public {
        uint256 prior = 0;
        for (uint256 i = 1; i <= 60; i++) {
            uint256 total = (i % 7) + 1;
            uint256 positive = (i * 13) % (total + 1);
            prior = h.tUpdateIntegrityRep(prior, positive, total);
            assertLe(prior, 1e18);
        }
    }

    // ============================================================
    // Integration — tApplyOne persists GIR on the manager
    // ============================================================

    function testIntegration_appliesGIR_perfectVotes() public {
        TaskType tt = TaskType.MNIST;
        address user = address(0xCAFE);

        // 5/5 positive votes -> V = 1 -> GIR = 0.2 (with prior = 0)
        h.tApplyOne(user, 0, 5, 5, tt, 10e18, 5);

        (, uint256 storedGIR, ) = manager.getUserRep(user, tt);
        assertEq(storedGIR, 2e17);
    }

    function testIntegration_appliesGIR_noVotes_initialState() public {
        TaskType tt = TaskType.MNIST;
        address user = address(0xC0DE);

        // Prior GIR = 0, totalVotes = 0 -> V = 0 -> new GIR = 0.
        h.tApplyOne(user, 0, 0, 0, tt, 10e18, 5);

        (, uint256 storedGIR, ) = manager.getUserRep(user, tt);
        assertEq(storedGIR, 0);
    }

    // ============================================================
    // Invariants over many rounds
    // ============================================================

    function testInvariant_outputsStayWithinWAD() public {
        uint256 priorRunningCMean = 0;
        uint256 priorM2 = 0;
        uint256 PriorTaskRep = 0;
        for (uint256 k = 1; k <= 60; k++) {
            uint256 ContributionScore = ((k * 137) % 11) * (WAD / 10);
            if (ContributionScore > WAD) ContributionScore = WAD;
            (uint256 newRunningCMean, uint256 newM2) =
                h.tUpdateRunningStats(ContributionScore, priorRunningCMean, priorM2, k);
            uint256 Confidence = h.tComputeConfidence(k, newM2);
            uint256 newK = h.tUpdateContribScore(PriorTaskRep, Confidence, ContributionScore);
            assertLe(Confidence, WAD);
            assertLe(newRunningCMean, WAD);
            assertLe(newK, WAD);
            priorRunningCMean = newRunningCMean;
            priorM2 = newM2;
            PriorTaskRep = newK;
        }
    }

    function testInvariant_perfectScoreConverges() public {
        uint256 priorRunningCMean = 0;
        uint256 priorM2 = 0;
        uint256 PriorTaskRep = 0;
        for (uint256 k = 1; k <= 100; k++) {
            (uint256 newRunningCMean, uint256 newM2) =
                h.tUpdateRunningStats(WAD, priorRunningCMean, priorM2, k);
            uint256 Confidence = h.tComputeConfidence(k, newM2);
            uint256 newK = h.tUpdateContribScore(PriorTaskRep, Confidence, WAD);
            priorRunningCMean = newRunningCMean;
            priorM2 = newM2;
            PriorTaskRep = newK;
        }
        assertGt(PriorTaskRep, 95e16);
    }

    function testInvariant_zeroScoreStaysZero() public {
        uint256 priorRunningCMean = 0;
        uint256 priorM2 = 0;
        uint256 PriorTaskRep = 0;
        for (uint256 k = 1; k <= 100; k++) {
            (uint256 newRunningCMean, uint256 newM2) =
                h.tUpdateRunningStats(0, priorRunningCMean, priorM2, k);
            uint256 Confidence = h.tComputeConfidence(k, newM2);
            uint256 newK = h.tUpdateContribScore(PriorTaskRep, Confidence, 0);
            priorRunningCMean = newRunningCMean;
            priorM2 = newM2;
            PriorTaskRep = newK;
        }
        assertEq(PriorTaskRep, 0);
    }

    // ============================================================
    // Integration — tApplyOne persists state on the manager
    // ============================================================

    function testIntegration_firstTask_seedsManagerState() public {
        TaskType tt = TaskType.MNIST;
        address user = address(0xBEEF);

        // delta=0, stake=1e18, reward=10e18, nrActive=5 -> ContributionScore = WAD/3
        h.tApplyOne(user, 0, 0, 0, tt, 10e18, 5);

        (uint256 storedK, , ) = manager.getUserRep(user, tt);
        uint256 nrTasks = manager.getTaskCount(user, tt);
        (uint256 storedE, uint256 storedF) = manager.getTaskRepCalcState(user, tt);

        assertEq(nrTasks, 1, "task counter incremented");
        assertEq(storedE, WAD / 3, "E_1 = ContributionScore");
        assertEq(storedF, 0, "F_1 = 0");

        uint256 Confidence = WAD / 3;
        uint256 weighted = (Confidence * (WAD / 3)) / WAD;
        uint256 expectedK = (N_BLEND * weighted) / WAD;
        assertEq(storedK, expectedK);
    }

    function testIntegration_twoTasks_carriesState() public {
        TaskType tt = TaskType.MNIST;
        address user = address(0xBEEF);

        // Task 1: delta=2e18 -> ContributionScore = WAD (clips at maxGain)
        h.tApplyOne(user, int256(2e18), 0, 0, tt, 10e18, 5);

        (uint256 k1, , ) = manager.getUserRep(user, tt);
        uint256 nr1 = manager.getTaskCount(user, tt);
        (uint256 e1, uint256 f1) = manager.getTaskRepCalcState(user, tt);
        assertEq(nr1, 1);
        assertEq(e1, WAD);
        assertEq(f1, 0);

        // Task 2: same delta -> same ContributionScore = WAD.
        h.tApplyOne(user, int256(2e18), 0, 0, tt, 10e18, 5);

        (uint256 k2, , ) = manager.getUserRep(user, tt);
        uint256 nr2 = manager.getTaskCount(user, tt);
        (uint256 e2, uint256 f2) = manager.getTaskRepCalcState(user, tt);
        assertEq(nr2, 2);
        assertEq(e2, WAD);
        assertEq(f2, 0);

        uint256 Confidence = WAD / 2;
        uint256 weighted = (Confidence * WAD) / WAD;
        uint256 expectedK = ((WAD - N_BLEND) * k1 + N_BLEND * weighted) / WAD;
        assertEq(k2, expectedK);
    }

    function testIntegration_kickedUser_zeroJ_softDecay() public {
        TaskType tt = TaskType.MNIST;
        address user = address(0xBEEF);

        for (uint i = 0; i < 5; i++) {
            h.tApplyOne(user, int256(4e18), 0, 0, tt, 10e18, 5);
        }
        (uint256 PriorTaskRep, , ) = manager.getUserRep(user, tt);
        assertGt(PriorTaskRep, 0);

        // Simulate kick: delta = -stake, ContributionScore clips to 0.
        h.tApplyOne(user, -int256(1e18), 0, 0, tt, 10e18, 5);

        (uint256 newK, , ) = manager.getUserRep(user, tt);
        assertEq(newK, (PriorTaskRep * (WAD - N_BLEND)) / WAD);
    }

    // ============================================================
    // TaskCount — getTaskCount / incrementTaskCount
    // ============================================================

    function testTaskCount_startsAtZero() public {
        assertEq(manager.getTaskCount(address(0xABCD), TaskType.MNIST), 0);
        assertEq(manager.getTaskCount(address(0xABCD), TaskType.CIFAR10), 0);
    }

    function testTaskCount_incrementsPerApply() public {
        address user = address(0x1234);
        TaskType tt = TaskType.MNIST;

        assertEq(manager.getTaskCount(user, tt), 0);
        h.tApplyOne(user, 0, 0, 0, tt, 10e18, 5);
        assertEq(manager.getTaskCount(user, tt), 1, "after first apply");
        h.tApplyOne(user, 0, 0, 0, tt, 10e18, 5);
        assertEq(manager.getTaskCount(user, tt), 2, "after second apply");
        h.tApplyOne(user, 0, 0, 0, tt, 10e18, 5);
        assertEq(manager.getTaskCount(user, tt), 3, "after third apply");
    }

    function testTaskCount_independentPerTaskType() public {
        address user = address(0x5678);

        h.tApplyOne(user, 0, 0, 0, TaskType.MNIST,   10e18, 5);
        h.tApplyOne(user, 0, 0, 0, TaskType.MNIST,   10e18, 5);
        h.tApplyOne(user, 0, 0, 0, TaskType.CIFAR10, 10e18, 5);

        assertEq(manager.getTaskCount(user, TaskType.MNIST),   2, "MNIST count");
        assertEq(manager.getTaskCount(user, TaskType.CIFAR10), 1, "CIFAR10 count");
    }

    function testTaskCount_drivesConfidenceGrowth() public {
        address user = address(0x9ABC);
        TaskType tt = TaskType.CIFAR10;

        uint256 prevK = 0;
        for (uint i = 0; i < 10; i++) {
            h.tApplyOne(user, int256(2e18), 0, 0, tt, 10e18, 5);
            (uint256 storedK, , ) = manager.getUserRep(user, tt);
            assertGe(storedK, prevK, "task rep must not decrease under constant positive delta");
            prevK = storedK;
        }
        assertEq(manager.getTaskCount(user, tt), 10, "count after 10 tasks");

        uint256 k1Rep;
        {
            OpenFLManager m2 = new OpenFLManager(ReputationMode.PerTask);
            OpenFLChallengeHarness h2 = new OpenFLChallengeHarness(address(m2));
            m2.setChallengeCodeHash(address(h2).codehash);
            h2.tApplyOne(user, int256(2e18), 0, 0, tt, 10e18, 5);
            (k1Rep, , ) = m2.getUserRep(user, tt);
        }
        (uint256 finalK, , ) = manager.getUserRep(user, tt);
        assertGt(finalK, k1Rep, "10-task TR must exceed 1-task TR");
    }

    function testTaskCount_globalOnlyMode_sharedSlot() public {
        OpenFLManager globalMgr = new OpenFLManager(ReputationMode.GlobalOnly);
        OpenFLChallengeHarness gh = new OpenFLChallengeHarness(address(globalMgr));
        globalMgr.setChallengeCodeHash(address(gh).codehash);

        address user = address(0xDEAD);

        gh.tApplyOne(user, 0, 0, 0, TaskType.MNIST,   10e18, 5);
        gh.tApplyOne(user, 0, 0, 0, TaskType.CIFAR10, 10e18, 5);

        assertEq(globalMgr.getTaskCount(user, TaskType.MNIST),   2, "shared count via MNIST");
        assertEq(globalMgr.getTaskCount(user, TaskType.CIFAR10), 2, "shared count via CIFAR10");
    }
}
