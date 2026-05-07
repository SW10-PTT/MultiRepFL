from types import SimpleNamespace
from unittest.mock import MagicMock
import torch
import torch.nn as nn
import pytest
import numpy as np

# All scoring helpers come from the challenge contract. These tests
# validate how the contract normalizes/weights participant updates across the
# three scoring strategies exposed in production (dotproduct, naive, accuracy).
from openfl.contracts.FLChallenge import (
    FLChallenge,
    calc_contribution_score_naive,
    calc_contribution_scores_dotproduct,
)


def _build_loss_aggregator(prev_losses, user_losses, loss_tolerance_pct=0.05):
    """Lightweight aggregator stub for loss-based scoring strategies.

    Bypasses FLChallenge.__init__ (which requires a real web3 connection)
    by constructing a MagicMock with just the attributes the scoring methods
    touch. Mirrors the production access path: `self.contract.functions.*`
    and `self.loss_tolerance_pct`.
    """
    aggregator = MagicMock()
    aggregator.pytorch_model = MagicMock()
    aggregator.pytorch_model.round = 1
    aggregator.loss_tolerance_pct = loss_tolerance_pct

    users = []
    for i, losses in enumerate(user_losses):
        user = MagicMock()
        user.address = f"0xLossUser{i}"
        # display_label feeds f-string format specs (e.g. ":<12") in
        # round_scoring log lines; bare MagicMock can't be __format__'d.
        user.display_label.return_value = f"loss_user{i}"
        user._losses = losses
        users.append(user)

    aggregator.contract.functions.getAllPreviousAccuraciesAndLosses \
        .return_value.call.return_value = ([], prev_losses)

    losses_by_addr = {u.address: u._losses for u in users}

    def mock_get_losses(address):
        m = MagicMock()
        m.call.return_value = (None, losses_by_addr[address])
        return m

    aggregator.contract.functions.getAllLossesAbout.side_effect = mock_get_losses
    return aggregator, users


class TinyModel(nn.Module):
    # Minimal model used to simulate local updates. The parameters are small
    # tensors so we can reason about vector math in the tests without large
    # fixtures or real training runs.
    def __init__(self, base_weight: float, noise: float = 0.0):
        super().__init__()
        values = torch.tensor([base_weight + noise, base_weight + noise], dtype=torch.float32)
        self.params = nn.Parameter(values)

    def parameters(self):
        return [self.params]


def build_challenge(strategy: str, *, use_outlier_detection: bool = False, contract=None):
    """Construct an FLChallenge with the scoring strategy under test.

    Bypass __init__ via __new__ — production __init__ deploys the on-chain
    challenge through globals.w3, which isn't available under unit tests.
    Wire only the fields the scoring routines read: contract, pytorch_model,
    contribution_score_strategy, use_outlier_detection, loss_tolerance_pct,
    and the strategy lookup table.
    """
    contract_mock = contract if contract is not None else MagicMock()
    if not hasattr(contract_mock, "address"):
        contract_mock.address = "0xModelAddress"
    if not hasattr(contract_mock, "abi"):
        contract_mock.abi = []

    pytorch_model = MagicMock()
    pytorch_model.participants = []
    pytorch_model.round = 1

    challenge = FLChallenge.__new__(FLChallenge)
    challenge.contract = contract_mock
    challenge.contractAddress = contract_mock.address
    challenge.model = contract_mock  # legacy alias for tests written pre-rename
    challenge.pytorch_model = pytorch_model
    challenge._contribution_score_strategy = strategy
    challenge.contribution_score_strategy = strategy
    challenge.use_outlier_detection = use_outlier_detection
    challenge.loss_tolerance_pct = 0.05
    challenge.experiment_config = SimpleNamespace(
        contribution_score_strategy=strategy,
        use_outlier_detection=use_outlier_detection,
    )
    challenge._contribution_score_calculators = {
        "dotproduct": challenge._calculate_scores_dotproduct,
        "naive": challenge._calculate_scores_naive,
        "accuracy_loss": challenge._calculate_scores_accuracy_loss,
        "accuracy_only": challenge._calculate_scores_accuracy_only,
        "loss_only": challenge._calculate_scores_loss_only,
        "loss_tolerance_aware": challenge._calculate_scores_loss_tolerance_aware,
        "loss_tolerance_snap": challenge._calculate_scores_loss_tolerance_snap,
    }
    challenge._log_contribution_scores = MagicMock()
    challenge._log_warning = MagicMock()
    return challenge


def make_participant(idx: int, previous_model: nn.Module, merged_model: nn.Module):
    """Create a minimal participant stub used by scoring methods."""
    label = f"part{idx}"
    return SimpleNamespace(
        id=idx,
        address=f"0xparticipant{idx}",
        privateKey=f"priv{idx}",
        previousModel=previous_model,
        model=merged_model,
        # display_label is consumed by f-string format specs in scoring logs.
        display_label=lambda label=label: label,
    )


def make_accuracy_contract(prev_accs, prev_losses, user_metrics):
    """Fake accuracy contract that returns historical metrics for normalization."""
    functions = SimpleNamespace()

    functions.getAllPreviousAccuraciesAndLosses = lambda: SimpleNamespace(
        call=lambda: (prev_accs, prev_losses)
    )

    def about(addr):
        accs, losses = user_metrics[addr]
        return SimpleNamespace(call=lambda: (None, accs, losses))

    functions.getAllAccuraciesLossesAbout = about

    abi = [
        {"name": "getAllPreviousAccuraciesAndLosses", "type": "function"},
        {"name": "getAllAccuraciesLossesAbout", "type": "function"},
        {"name": "submitContributionScore", "type": "function"}
    ]

    return SimpleNamespace(functions=functions, abi=abi, address="0xModelAddress")


class TestDotProductScoring:
    def test_low_noise_freerider_has_small_penalty(self):
        '''
        Slightly noisy freerider should score below honest peers while keeping ordering deterministic.
        In this setup, participant[1] drifts 0.01 below the merged model, so
        the dot product between their update and the global update shrinks.
        '''
        merged = TinyModel(1.0)
        honest = make_participant(0, TinyModel(1.0), merged)
        freerider = make_participant(1, TinyModel(0.99), merged)
        backup = make_participant(2, TinyModel(1.05), merged)

        challenge = build_challenge("dotproduct", use_outlier_detection=False)
        scores = challenge._calculate_scores_dotproduct([honest, freerider, backup])

        local_updates = torch.stack([
            torch.tensor([1.0, 1.0]),
            torch.tensor([0.99, 0.99]),
            torch.tensor([1.05, 1.05]),
        ])
        expected = calc_contribution_scores_dotproduct(
            local_updates, torch.tensor([1.0, 1.0])
        )

        assert scores == expected
        assert scores[1] < scores[0]

    def test_medium_noise_shifts_rankings(self):
        '''
        Moderate noise should demote the freerider below both honest contributors.
        Noise magnitude of 0.1 makes participant[1] diverge further from the
        merged model, so their alignment (dot product) drops compared to the others.
        '''
        merged = TinyModel(1.0)
        honest = make_participant(0, TinyModel(1.0), merged)
        freerider = make_participant(1, TinyModel(0.9), merged)
        backup = make_participant(2, TinyModel(1.1), merged)

        challenge = build_challenge("dotproduct", use_outlier_detection=False)
        scores = challenge._calculate_scores_dotproduct([honest, freerider, backup])

        assert scores[2] > scores[0] > scores[1]

    def test_high_noise_filtered_by_mad(self):
        '''
        Outlier filtering should reduce extreme penalties while keeping honest scores positive.
        Here the freerider update is nearly opposite the merged model.
        The median absolute deviation (MAD) filter used in the dot-product path
        should clamp the negative impact so we don't over-penalize.
        '''
        merged = TinyModel(1.0)
        honest = make_participant(0, TinyModel(1.0), merged)
        freerider = make_participant(1, TinyModel(-1.0), merged)
        backup = make_participant(2, TinyModel(1.0), merged)

        challenge_no_filter = build_challenge("dotproduct", use_outlier_detection=False)
        challenge_filter = build_challenge("dotproduct", use_outlier_detection=True)

        raw_scores = challenge_no_filter._calculate_scores_dotproduct([honest, freerider, backup])
        filtered_scores = challenge_filter._calculate_scores_dotproduct([honest, freerider, backup])

        assert abs(filtered_scores[1]) <= abs(raw_scores[1])
        assert filtered_scores[0] > 0
        assert filtered_scores[2] > 0

    def test_zero_global_update_distributes_evenly(self):
        '''
        When the merged model is zero, all local updates should contribute equally.
        With a zero global baseline the dot-product denominator is constant, so
        relative ordering only depends on local update magnitudes which are
        identical here.
        '''
        merged = TinyModel(0.0)
        participants = [
            make_participant(i, TinyModel(val), merged)
            for i, val in enumerate([0.0, 0.5, 1.0])
        ]

        challenge = build_challenge("dotproduct", use_outlier_detection=False)
        scores = challenge._calculate_scores_dotproduct(participants)

        assert scores[0] == scores[1] == scores[2]

    def test_negative_alignment_produces_negative_score(self):
        '''
        Anti-aligned updates should receive negative scores relative to honest contributors.
        participant[1] pushes in the opposite direction of the merged model,
        so its dot product is negative while honest participants remain positive.
        '''
        merged = TinyModel(1.0)
        honest = make_participant(0, TinyModel(1.0), merged)
        anti_aligned = make_participant(1, TinyModel(-1.0), merged)
        freerider = make_participant(2, TinyModel(0.9), merged)

        challenge = build_challenge("dotproduct", use_outlier_detection=False)
        scores = challenge._calculate_scores_dotproduct([honest, anti_aligned, freerider])

        assert scores[1] < 0
        assert scores[0] > scores[2] > scores[1]

    def test_scores_stable_when_participants_reordered(self):
        '''
        The dot-product calculation should be deterministic regardless of participant ordering.
        This guards against position-dependent behavior when stacking updates
        and ensures addresses map to the same scores even if the list is shuffled.
        '''
        merged = TinyModel(1.0)
        participants = [
            make_participant(0, TinyModel(1.0), merged),
            make_participant(1, TinyModel(0.95), merged),
            make_participant(2, TinyModel(1.05), merged),
        ]

        challenge = build_challenge("dotproduct", use_outlier_detection=False)
        baseline = challenge._calculate_scores_dotproduct(participants)

        reversed_participants = list(reversed(participants))
        reversed_scores = challenge._calculate_scores_dotproduct(reversed_participants)

        baseline_by_addr = {p.address: score for p, score in zip(participants, baseline)}
        reversed_by_addr = {p.address: score for p, score in zip(reversed_participants, reversed_scores)}

        assert baseline_by_addr == reversed_by_addr


class TestNaiveScoring:
    def test_low_noise_participants_share_equally(self):
        '''
        All contributors split rewards evenly regardless of minor noise.
        The naive strategy ignores update alignment entirely, so any small
        perturbations should still yield the same pro-rata payout.
        '''
        merged = TinyModel(1.0)
        participants = [
            make_participant(0, TinyModel(1.0), merged),
            make_participant(1, TinyModel(1.0, noise=0.01), merged),
        ]

        challenge = build_challenge("naive")
        scores = challenge._calculate_scores_naive(participants)

        expected = [calc_contribution_score_naive(len(participants))] * len(participants)
        assert scores == expected

    def test_medium_noise_does_not_change_share(self):
        '''
        Moderate deviations still yield equal naive scores for all participants.
        With three contributors, calc_contribution_score_naive should divide
        the 1e18 reward pool by 3 even though the input weights differ.
        '''
        merged = TinyModel(1.0)
        participants = [
            make_participant(0, TinyModel(0.8), merged),
            make_participant(1, TinyModel(1.0, noise=0.1), merged),
            make_participant(2, TinyModel(1.2), merged),
        ]

        challenge = build_challenge("naive")
        scores = challenge._calculate_scores_naive(participants)

        assert len(set(scores)) == 1

    def test_high_noise_freerider_still_equal(self):
        '''
        Even large noise should not alter uniform naive contributions.
        participant[1] deviates by +1.0 but naive scoring ensures all four
        receive identical integer rewards.
        '''
        merged = TinyModel(1.0)
        participants = [
            make_participant(0, TinyModel(1.0), merged),
            make_participant(1, TinyModel(1.0, noise=1.0), merged),
            make_participant(2, TinyModel(1.0), merged),
            make_participant(3, TinyModel(1.0), merged),
        ]

        challenge = build_challenge("naive")
        scores = challenge._calculate_scores_naive(participants)

        assert all(score == scores[0] for score in scores)

    def test_single_participant_gets_full_share(self):
        '''
        Single contributor receives the full reward pool.
        The naive helper returns 1e18 for one participant, mirroring contract
        behavior of distributing the entire reward when there is no
        competition.
        '''
        merged = TinyModel(1.0)
        participants = [make_participant(0, TinyModel(1.5), merged)]

        challenge = build_challenge("naive")
        scores = challenge._calculate_scores_naive(participants)

        assert scores == [int(1e18)]

    def test_large_group_equal_distribution(self):
        '''
        Naive scoring should divide rewards uniformly across many participants.
        This guards against rounding issues when distributing to larger
        cohorts; every participant should still get the same integer amount.
        '''
        merged = TinyModel(1.0)
        participants = [
            make_participant(i, TinyModel(1.0 + 0.01 * i), merged)
            for i in range(10)
        ]

        challenge = build_challenge("naive")
        scores = challenge._calculate_scores_naive(participants)

        expected_score = calc_contribution_score_naive(len(participants))
        assert scores == [expected_score] * len(participants)

    def test_reward_pool_preserved_after_distribution(self):
        '''
        Naive scoring should conserve the reward pool aside from integer rounding.
        Summing the per-participant payout should equal the helper's output
        multiplied by the participant count, which mirrors on-chain behavior.
        '''
        merged = TinyModel(1.0)
        participants = [
            make_participant(i, TinyModel(0.9 + 0.02 * i), merged)
            for i in range(6)
        ]

        challenge = build_challenge("naive")
        scores = challenge._calculate_scores_naive(participants)

        per_user = calc_contribution_score_naive(len(participants))
        assert sum(scores) == per_user * len(participants)


class TestAccuracyScoring:
    # def test_low_noise_freerider_scores_lower(self):
    #     '''
    #     Small accuracy/loss differences should rank the freerider last despite similar baselines.
    #     Each participant reports two accuracy/loss entries. The freerider has
    #     slightly worse metrics, so after normalizing against previous
    #     experiment averages they should get the smallest share of the pool.
    #     '''
    #     users = [
    #         make_participant(0, TinyModel(1.0), TinyModel(1.0)),
    #         make_participant(1, TinyModel(1.0), TinyModel(1.0)),
    #         make_participant(2, TinyModel(1.0), TinyModel(1.0)),
    #     ]
    #     prev_accs = [0.7, 0.7, 0.7]
    #     prev_losses = [0.1, 0.1, 0.1]
    #     metrics = {
    #         users[0].address: ([0.9, 0.92], [0.2, 0.22]),
    #         users[1].address: ([0.85, 0.86], [0.25, 0.26]),
    #         users[2].address: ([0.8, 0.81], [0.3, 0.31]),
    #     }
    #     contract = make_accuracy_contract(prev_accs, prev_losses, metrics)
    #     challenge = build_challenge("accuracy", contract=contract)

    #     scores = challenge._calculate_scores_accuracy(users)

    #     avg_prev_acc = np.mean(prev_accs)
    #     avg_prev_loss = np.mean(prev_losses)
    #     avg_accuracies = [np.mean(v[0]) for v in metrics.values()]
    #     avg_losses = [np.mean(v[1]) for v in metrics.values()]
    #     norm_acc = calc_contribution_scores_accuracy(avg_accuracies, avg_prev_acc)
    #     norm_loss = calc_contribution_scores_accuracy(avg_losses, avg_prev_loss)
    #     inverted_losses = [1 - x for x in norm_loss]
    #     total = sum(norm_acc) + sum(inverted_losses)
    #     expected = [int(((a + l) / total) * 1e18) for a, l in zip(norm_acc, inverted_losses)]

    #     assert sum(scores) == pytest.approx(1e18, rel=0, abs=5)
    #     assert scores[0] > scores[1] > scores[2]

    def test_medium_noise_freerider_penalized_by_accuracy(self):
        '''
        Slightly worse accuracy should push the noisier participant below honest peers.
        participant[1] records lower accuracies and higher losses than the
        others; normalization should reduce their final payout even though
        noise is only 0.1.
        '''
        users = [
            make_participant(0, TinyModel(1.0), TinyModel(1.0)),
            make_participant(1, TinyModel(1.0, noise=0.1), TinyModel(1.0)),
            make_participant(2, TinyModel(1.0), TinyModel(1.0)),
        ]
        prev_accs = [0.6, 0.6, 0.6]
        prev_losses = [0.1, 0.1, 0.1]
        metrics = {
            users[0].address: ([0.85, 0.86], [0.26, 0.25]),
            users[1].address: ([0.55, 0.56], [0.55, 0.56]),
            users[2].address: ([0.83, 0.84], [0.28, 0.27]),
        }
        contract = make_accuracy_contract(prev_accs, prev_losses, metrics)
        challenge = build_challenge("accuracy", contract=contract)

        scores = challenge._calculate_scores_accuracy_loss(users)

        assert scores[1] < min(scores[0], scores[2])

    def test_high_noise_freerider_losses_dominate(self):
        '''
        Poor accuracy and high loss should heavily penalize the noisy participant.
        Accuracy/loss pairs for participant[1] are intentionally extreme to
        ensure loss normalization drives their score to the minimum.
        '''
        users = [
            make_participant(0, TinyModel(1.0), TinyModel(1.0)),
            make_participant(1, TinyModel(1.0, noise=1.0), TinyModel(1.0)),
            make_participant(2, TinyModel(1.0), TinyModel(1.0)),
        ]
        prev_accs = [0.6, 0.6, 0.6]
        prev_losses = [0.1, 0.1, 0.1]
        metrics = {
            users[0].address: ([0.9], [0.2]),
            users[1].address: ([0.3], [0.8]),
            users[2].address: ([0.78], [0.35]),
        }
        contract = make_accuracy_contract(prev_accs, prev_losses, metrics)
        challenge = build_challenge("accuracy", contract=contract)

        scores = challenge._calculate_scores_accuracy_loss(users)

        assert scores[1] == min(scores)
        assert scores[0] > scores[2] > scores[1]

    def test_handles_zero_sum_differences(self):
        '''
        When all participants tie on accuracy and loss, they should each receive equal scores.
        All metrics exactly match previous experiment averages, so normalized
        accuracy/loss values should be identical for every user.
        '''
        users = [
            make_participant(0, TinyModel(1.0), TinyModel(1.0)),
            make_participant(1, TinyModel(1.0, noise=0.1), TinyModel(1.0)),
            make_participant(2, TinyModel(1.0), TinyModel(1.0)),
        ]
        prev_accs = [0.8, 0.8, 0.8]
        prev_losses = [0.2, 0.2, 0.2]
        metrics = {
            users[0].address: ([0.8], [0.2]),
            users[1].address: ([0.8], [0.2]),
            users[2].address: ([0.8], [0.2]),
        }
        contract = make_accuracy_contract(prev_accs, prev_losses, metrics)
        challenge = build_challenge("accuracy", contract=contract)

        scores = challenge._calculate_scores_accuracy_loss(users)

        assert scores[0] == scores[1] == scores[2]

    # def test_accuracy_scores_return_integers_and_sum_to_pool(self):
    #     '''
    #     Accuracy-based scoring should emit integer values that collectively use the reward pool.
    #     Guards against regression where floating point division or rounding
    #     errors leak value from the 1e18 total supply allocated for contributions.
    #     '''
    #     users = [
    #         make_participant(0, TinyModel(1.0), TinyModel(1.0)),
    #         make_participant(1, TinyModel(1.0), TinyModel(1.0)),
    #         make_participant(2, TinyModel(1.0), TinyModel(1.0)),
    #     ]
    #     prev_accs = [0.4, 0.5, 0.6]
    #     prev_losses = [0.5, 0.5, 0.5]
    #     metrics = {
    #         users[0].address: ([0.7, 0.71], [0.3, 0.31]),
    #         users[1].address: ([0.68, 0.69], [0.32, 0.33]),
    #         users[2].address: ([0.65, 0.66], [0.34, 0.35]),
    #     }
    #     contract = make_accuracy_contract(prev_accs, prev_losses, metrics)
    #     challenge = build_challenge("accuracy", contract=contract)

    #     scores = challenge._calculate_scores_accuracy(users)

    #     assert all(isinstance(score, int) for score in scores)
    #     assert sum(scores) == pytest.approx(1e18, rel=0, abs=5)


class TestLossToleranceScoring:
    """End-to-end behavior for the two tolerance-aware loss strategies.

    Goal: confirm that honest participants whose updates are *slightly* worse
    than the global model are not punished, while participants whose updates
    are significantly worse still receive the expected negative contribution.
    """

    @pytest.fixture
    def baseline_prev_losses(self):
        # Simulate a converged global model with stable historical loss ~0.50.
        return [0.50, 0.50, 0.50]

    def test_loss_only_punishes_honest_minor_drift(self, baseline_prev_losses):
        '''
        Sanity check: with the legacy loss_only strategy, even tiny worsenings
        produce negative contribution scores. This is the regression that
        motivates the tolerance-aware strategies.
        '''
        # Each honest participant lands ~1-3% above the global loss after
        # local training (well within natural variance).
        user_losses = [[0.50], [0.51], [0.515]]
        aggregator, users = _build_loss_aggregator(baseline_prev_losses, user_losses, loss_tolerance_pct=0.0)

        scores = FLChallenge._calculate_scores_loss_only(aggregator, users, mad_threshold=1.1)
        normalized = [s / 1e18 for s in scores]

        # The two slightly-worse participants get penalized despite being honest.
        assert normalized[1] < 0
        assert normalized[2] < 0

    def test_aware_keeps_minor_drift_positive(self, baseline_prev_losses):
        '''
        loss_tolerance_aware: honest participants with ≤ 5% worse loss should
        still receive *positive* contribution scores because the reward
        threshold is shifted by ε.
        '''
        # With L_global ≈ 0.50 and pct=0.05, ε = 0.025 → shifted baseline = 0.525.
        # All three losses are at or below the shifted baseline.
        user_losses = [[0.50], [0.51], [0.515]]
        aggregator, users = _build_loss_aggregator(baseline_prev_losses, user_losses, loss_tolerance_pct=0.05)

        scores = FLChallenge._calculate_scores_loss_tolerance_aware(aggregator, users, mad_threshold=1.1)
        normalized = [s / 1e18 for s in scores]

        # All honest contributors stay positive; ranking by improvement preserved.
        assert all(s > 0 for s in normalized), f"expected all positive, got {normalized}"
        assert normalized[0] > normalized[1] > normalized[2]
        assert sum(normalized) == pytest.approx(1.0, abs=1e-9)

    def test_aware_still_punishes_clear_worsening(self, baseline_prev_losses):
        '''
        loss_tolerance_aware: a participant whose loss is well beyond ε must
        still get a negative score so the system retains its Byzantine
        defenses against bad/free-rider behavior.
        '''
        # 0.80 is 60% worse than baseline 0.50 — far beyond the 5% tolerance.
        user_losses = [[0.50], [0.51], [0.80]]
        aggregator, users = _build_loss_aggregator(baseline_prev_losses, user_losses, loss_tolerance_pct=0.05)

        scores = FLChallenge._calculate_scores_loss_tolerance_aware(aggregator, users, mad_threshold=1.1)
        normalized = [s / 1e18 for s in scores]

        assert normalized[2] < 0
        assert normalized[0] > 0
        assert normalized[1] > 0

    def test_snap_zeroes_small_worsenings(self, baseline_prev_losses):
        '''
        loss_tolerance_snap: a participant in the tolerance band should be
        treated as neutral (snap to baseline → zero diff) rather than
        penalized.
        '''
        # Idx 1 is in the tolerance band (diff = 0.02, ε = 0.025 → snap).
        # Idx 2 is well beyond (diff = 0.10) → real penalty.
        user_losses = [[0.40], [0.52], [0.60]]
        aggregator, users = _build_loss_aggregator(baseline_prev_losses, user_losses, loss_tolerance_pct=0.05)

        scores = FLChallenge._calculate_scores_loss_tolerance_snap(aggregator, users, mad_threshold=1.1)
        normalized = [s / 1e18 for s in scores]

        # Best contributor positive, snap user neutral (≈ 0), worst contributor negative.
        assert normalized[0] > 0
        assert normalized[1] == pytest.approx(0.0, abs=1e-9)
        assert normalized[2] < 0

    def test_snap_preserves_pure_improvements(self, baseline_prev_losses):
        '''
        loss_tolerance_snap: pure improvements bypass the snap entirely and
        retain their relative ranking.
        '''
        user_losses = [[0.40], [0.45], [0.475]]
        aggregator, users = _build_loss_aggregator(baseline_prev_losses, user_losses, loss_tolerance_pct=0.05)

        scores = FLChallenge._calculate_scores_loss_tolerance_snap(aggregator, users, mad_threshold=1.1)
        normalized = [s / 1e18 for s in scores]

        assert normalized[0] > normalized[1] > normalized[2] > 0
        assert sum(normalized) == pytest.approx(1.0, abs=1e-9)

    def test_aware_threshold_scales_with_baseline(self):
        '''
        ε = pct * L_global must scale with the dataset's loss magnitude:
        the same absolute drift is forgiven on a high-loss task and
        penalized on a low-loss task with identical pct.
        '''
        # Same absolute drift of 0.04, but on different L_global magnitudes.
        # For high-loss task (L_global = 1.0), ε = 0.05 → drift inside band → positive.
        # For low-loss task (L_global = 0.10), ε = 0.005 → drift outside band → negative.
        high_baseline = [1.0, 1.0, 1.0]
        low_baseline = [0.10, 0.10, 0.10]
        # Drifted user is at idx 1
        high_losses = [[1.0], [1.04], [1.0]]
        low_losses = [[0.10], [0.14], [0.10]]

        agg_hi, users_hi = _build_loss_aggregator(high_baseline, high_losses, loss_tolerance_pct=0.05)
        agg_lo, users_lo = _build_loss_aggregator(low_baseline, low_losses, loss_tolerance_pct=0.05)

        scores_hi = FLChallenge._calculate_scores_loss_tolerance_aware(agg_hi, users_hi, mad_threshold=1.1)
        scores_lo = FLChallenge._calculate_scores_loss_tolerance_aware(agg_lo, users_lo, mad_threshold=1.1)

        norm_hi = [s / 1e18 for s in scores_hi]
        norm_lo = [s / 1e18 for s in scores_lo]

        # High-loss task tolerates the drift → still positive.
        assert norm_hi[1] > 0
        # Low-loss task does not tolerate it → negative.
        assert norm_lo[1] < 0
