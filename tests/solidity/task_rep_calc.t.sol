// SPDX-License-Identifier: MIT
pragma solidity ^0.8.9;

import "forge-std/Test.sol";
import "../../contracts/JobListing.sol";
import "../../contracts/OpenFLManager.sol";
import "../../contracts/Types.sol";

// Test harness — exposes JobListing's internal helpers + integration body for
// direct unit testing.
contract JobListingHarness is JobListing {
    constructor(address mgr)
        JobListing(1e18, 1.8e18, 1e18, 3, 3, 3, 50, mgr, TaskType.MNIST)
    {}

    function tTransformDelta(
        int256 delta,
        uint256 stake,
        uint256 reward,
        uint256 nrActive
    ) external pure returns (uint256) {
        return _transformDelta(delta, stake, reward, nrActive);
    }

    function tUpdateRunningStats(
        uint256 ContributionScore,
        uint256 priorRunningCMean,
        uint256 priorM2,
        uint256 k
    ) external pure returns (uint256, uint256) {
        return _updateRunningStats(ContributionScore, priorRunningCMean, priorM2, k);
    }

    function tComputeConfidence(uint256 k, uint256 s_k)
        external
        pure
        returns (uint256)
    {
        return _computeConfidence(k, s_k);
    }

    function tUpdateContribScore(
        uint256 PriorTaskRep,
        uint256 Confidence,
        uint256 ContributionScore
    ) external pure returns (uint256) {
        return _updateContribScore(PriorTaskRep, Confidence, ContributionScore);
    }

    function tUpdateIntegrityRep(
        uint256 priorIntegrityRep,
        uint256 positiveVotes,
        uint256 totalVotes
    ) external pure returns (uint256) {
        return _updateIntegrityRep(priorIntegrityRep, positiveVotes, totalVotes);
    }

    function tApplyOne(
        IOpenFLChallengeTaskRep.TaskRep memory rep,
        TaskType tt,
        uint256 reward,
        uint256 nrActive
    ) external {
        _applyTaskRepCalc(rep, tt, reward, nrActive);
    }
}

// Override variant — proves _computeConfidence is pluggable via virtual.
contract JobListingFlatConfidence is JobListingHarness {
    constructor(address mgr) JobListingHarness(mgr) {}

    function _computeConfidence(uint256, uint256)
        internal
        pure
        override
        returns (uint256)
    {
        return 5e17;
    }
}

contract TaskRepCalcTest is Test {
    OpenFLManager manager;
    JobListingHarness h;

    uint256 constant WAD = 1e18;
    uint256 constant ALPHA = 2e17;
    uint256 constant N_BLEND = 2e17;

    function setUp() public {
        manager = new OpenFLManager();
        h = new JobListingHarness(address(manager));
    }

    // ============================================================
    // _transformDelta
    // ============================================================

    function testTransform_zeroDelta_mapsToStakeOverRange() public {
        // stake=1, reward=10, nrActive=5 -> maxGain = 2*10/5 = 4
        // range = 5, shifted = 1 -> ContributionScore = 1/5 * WAD = 0.2e18
        assertEq(h.tTransformDelta(0, 1e18, 10e18, 5), 2e17);
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

    function testTransform_at1xAverage_isMidUpperBand() public {
        // delta = reward/nrActive = 2; with 2x cap range=5, shifted=3 -> 0.6
        assertEq(h.tTransformDelta(int256(2e18), 1e18, 10e18, 5), 6e17);
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

    function testConfidence_k1_sZero_oneOverSix() public {
        assertEq(h.tComputeConfidence(1, 0), WAD / 6);
    }

    function testConfidence_largeK_approachesOne() public {
        assertEq(h.tComputeConfidence(1000, 0), (1000 * WAD) / 1005);
    }

    function testConfidence_highVariance_drivesDown() public {
        uint256 maturity = (10 * WAD) / 15;
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
        uint256 Confidence = WAD / 6;
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
    // Integration — _applyTaskRepCalc persists GIR on the manager
    // ============================================================

    function testIntegration_appliesGIR_perfectVotes() public {
        _registerAsValidJob(h);

        TaskType tt = TaskType.MNIST;
        address user = address(0xCAFE);

        // 5/5 positive votes -> V = 1 -> GIR = 0.2 (with prior = 0)
        IOpenFLChallengeTaskRep.TaskRep memory rep = IOpenFLChallengeTaskRep
            .TaskRep({
                user: user,
                delta: 0,
                globalReputationScore: 0,
                positiveVotes: 5,
                totalVotes: 5
            });
        h.tApplyOne(rep, tt, 10e18, 5);

        (, uint256 storedGIR, ) = manager.getUserRep(user, tt);
        assertEq(storedGIR, 2e17);
    }

    function testIntegration_appliesGIR_noVotes_initialState() public {
        _registerAsValidJob(h);

        TaskType tt = TaskType.MNIST;
        address user = address(0xC0DE);

        // Prior GIR = 0, totalVotes = 0 -> V = 0 -> new GIR = 0 (0.8*0 + 0.2*0).
        IOpenFLChallengeTaskRep.TaskRep memory rep = IOpenFLChallengeTaskRep
            .TaskRep({
                user: user,
                delta: 0,
                globalReputationScore: 0,
                positiveVotes: 0,
                totalVotes: 0
            });
        h.tApplyOne(rep, tt, 10e18, 5);

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
    // Pluggable confidence
    // ============================================================

    function testPluggable_overrideTakesEffect() public {
        JobListingFlatConfidence flat =
            new JobListingFlatConfidence(address(manager));
        assertEq(flat.tComputeConfidence(7, 0), 5e17);
        assertEq(flat.tComputeConfidence(0, 999), 5e17);
    }

    // ============================================================
    // Integration — _applyTaskRepCalc persists state on the manager
    // ============================================================

    function _registerAsValidJob(JobListingHarness target) internal {
        manager.setJobListingCodeHash(address(target).codehash);
        manager.registerJob(address(target));
    }

    function testIntegration_firstTask_seedsManagerState() public {
        _registerAsValidJob(h);

        TaskType tt = TaskType.MNIST;
        address user = address(0xBEEF);

        // delta=0, stake=1e18, reward=10e18, nrActive=5 -> ContributionScore = 0.2e18
        IOpenFLChallengeTaskRep.TaskRep memory rep = IOpenFLChallengeTaskRep
            .TaskRep({
                user: user,
                delta: 0,
                globalReputationScore: 0,
                positiveVotes: 0,
                totalVotes: 0
            });
        h.tApplyOne(rep, tt, 10e18, 5);

        (uint256 storedK, , uint256 nrTasks) = manager.getUserRep(user, tt);
        (uint256 storedE, uint256 storedF) =
            manager.getTaskRepCalcState(user, tt);

        // k = 1 (first task), seeded E = ContributionScore, F = 0
        assertEq(nrTasks, 1, "task counter incremented");
        assertEq(storedE, 2e17, "E_1 = ContributionScore");
        assertEq(storedF, 0, "F_1 = 0");

        // Confidence_1 = WAD/6, weighted = ContributionScore/6,
        // K_1 = N_BLEND * ContributionScore/6 / WAD
        uint256 Confidence = WAD / 6;
        uint256 weighted = (Confidence * 2e17) / WAD;
        uint256 expectedK = (N_BLEND * weighted) / WAD;
        assertEq(storedK, expectedK);
    }

    function testIntegration_twoTasks_carriesState() public {
        _registerAsValidJob(h);

        TaskType tt = TaskType.MNIST;
        address user = address(0xBEEF);

        // Task 1: delta=2e18 -> ContributionScore = 0.6e18
        IOpenFLChallengeTaskRep.TaskRep memory rep1 = IOpenFLChallengeTaskRep
            .TaskRep({
                user: user,
                delta: int256(2e18),
                globalReputationScore: 0,
                positiveVotes: 0,
                totalVotes: 0
            });
        h.tApplyOne(rep1, tt, 10e18, 5);

        (uint256 k1, , uint256 nr1) = manager.getUserRep(user, tt);
        (uint256 e1, uint256 f1) = manager.getTaskRepCalcState(user, tt);
        assertEq(nr1, 1);
        assertEq(e1, 6e17);
        assertEq(f1, 0);

        // Task 2: same delta -> same ContributionScore = 0.6e18.
        // priorRunningCMean=0.6, ContributionScore=0.6 -> newRunningCMean = 0.6 (no movement); Delta=0 -> newM2=0
        h.tApplyOne(rep1, tt, 10e18, 5);

        (uint256 k2, , uint256 nr2) = manager.getUserRep(user, tt);
        (uint256 e2, uint256 f2) = manager.getTaskRepCalcState(user, tt);
        assertEq(nr2, 2);
        assertEq(e2, 6e17);
        assertEq(f2, 0);

        // k=2, s=0 -> Confidence = (2/7) * 1 * WAD = 2*WAD/7
        // weighted = Confidence * 0.6e18 / WAD = (2/7) * 0.6e18 = 12e17/70 ≈ 1.71e17
        // K2 = 0.8*k1 + 0.2*weighted
        uint256 Confidence = (2 * WAD) / 7;
        uint256 weighted = (Confidence * 6e17) / WAD;
        uint256 expectedK = ((WAD - N_BLEND) * k1 + N_BLEND * weighted) / WAD;
        assertEq(k2, expectedK);
    }

    function testIntegration_kickedUser_zeroJ_softDecay() public {
        _registerAsValidJob(h);

        TaskType tt = TaskType.MNIST;
        address user = address(0xBEEF);

        // Seed K with a strong prior via several positive tasks.
        IOpenFLChallengeTaskRep.TaskRep memory good = IOpenFLChallengeTaskRep
            .TaskRep({
                user: user,
                delta: int256(4e18),
                globalReputationScore: 0,
                positiveVotes: 0,
                totalVotes: 0
            });
        for (uint i = 0; i < 5; i++) {
            h.tApplyOne(good, tt, 10e18, 5);
        }
        (uint256 PriorTaskRep, , ) = manager.getUserRep(user, tt);
        assertGt(PriorTaskRep, 0);

        // Simulate kick: delta = -stake, ContributionScore clips to 0.
        IOpenFLChallengeTaskRep.TaskRep memory kicked = IOpenFLChallengeTaskRep
            .TaskRep({
                user: user,
                delta: -int256(1e18),
                globalReputationScore: 0,
                positiveVotes: 0,
                totalVotes: 0
            });
        h.tApplyOne(kicked, tt, 10e18, 5);

        (uint256 newK, , ) = manager.getUserRep(user, tt);
        // K_new = (1-N_BLEND)*PriorTaskRep + N_BLEND * Confidence * 0 = 0.8 * PriorTaskRep
        assertEq(newK, (PriorTaskRep * (WAD - N_BLEND)) / WAD);
    }
}
