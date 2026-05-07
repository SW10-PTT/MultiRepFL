"""
Tests for the _roundrep aggregate-append fix in quick_feedback_round and the
per-user-failure logging in loss-based contribution score methods.

Background:
- quick_feedback_round historically appended only per-vote rep deltas to
  user._roundrep. Three downstream readers (logging at lines ~1356/1370 and
  the contributors filter at ~1452) use _roundrep[-1] expecting the
  aggregated round reputation. With per-vote semantics, a single trailing
  -1 vote could flip inclusion regardless of the user's total round rep.
- The fix appends the on-chain aggregated roundReputation at the end of
  quick_feedback_round so _roundrep[-1] reflects the post-voting total.
- The except branches in the loss-based score methods now include the user
  label and the underlying ValueError so per-user failures are visible.

We bypass FLChallenge.__init__ (which deploys contracts) and stamp the
attributes the methods under test require. This isolates the behavior we
care about and avoids depending on the (currently broken) fl_challenge
fixture in tests/unit/conftest.py.
"""
from unittest.mock import MagicMock, patch

import pytest

from openfl.contracts.FLChallenge import FLChallenge
from openfl.utils.types.EvaluationData import EvaluationData


def _make_challenge():
    """FLChallenge with __init__ skipped + minimum attributes set."""
    challenge = FLChallenge.__new__(FLChallenge)
    challenge.contract = MagicMock()
    challenge.contractAddress = "0xContract"
    challenge.pytorch_model = MagicMock()
    challenge.pytorch_model.disqualified = []
    challenge.contribution_score_strategy = "naive"
    challenge.gas_feedback = []
    challenge.txHashes = []
    challenge._logger = None
    challenge.loss_tolerance_pct = 0.2
    challenge.use_outlier_detection = False
    challenge.get_global_reputation_of_user = MagicMock(return_value=1)
    challenge._log_receipt = MagicMock()
    return challenge


def _make_voter(idx):
    u = MagicMock()
    addr = "0x" + f"{idx + 1:040x}"
    u.id = addr
    u.address = addr
    u.attitude = "honest"
    u.disqualified = False
    u.roundRep = 0
    u._roundrep = []
    u.privateKey = f"pk{idx}"
    return u


def _build_eval_data(participants, vote_matrix):
    ed = EvaluationData.new(participants)
    n = len(participants)
    for i in range(n):
        for j in range(n):
            ed.feedback_matrix[participants[i].id, participants[j].id] = vote_matrix[i][j]
            ed.accuracy_matrix[participants[i].id, participants[j].id] = 0
            ed.loss_matrix[participants[i].id, participants[j].id] = 0
        ed.prev_accuracies[participants[i].id] = 0
        ed.prev_losses[participants[i].id] = 0
    return ed


def _wire_for_feedback(challenge, participants, on_chain_round_rep):
    challenge.pytorch_model.participants = participants
    challenge.pytorch_model.get_participant = lambda addr: next(
        (p for p in participants if p.id == addr), None
    )
    challenge.get_round_reputation_of_user = MagicMock(
        side_effect=lambda addr: on_chain_round_rep[addr]
    )
    submit_fn = MagicMock()
    submit_fn.transact.return_value = b"\x00" * 32
    challenge.contract.functions.submitFeedbackBytes.return_value = submit_fn
    # build_tx is inherited from FLManager; stub it so no real tx is built.
    challenge.build_tx = MagicMock(return_value={"gas": 100000, "gasPrice": 1, "nonce": 1})


class TestRoundRepAggregateAppend:
    """quick_feedback_round must append the on-chain aggregate so
    _roundrep[-1] equals get_round_reputation_of_user for each participant."""

    def _run_feedback(self, challenge, ed):
        fake_w3 = MagicMock()
        fake_w3.eth.wait_for_transaction_receipt.return_value = {
            "gasUsed": 21000, "transactionHash": b"\x00" * 32,
        }
        with patch("openfl.contracts.FLChallenge.globals") as g, \
             patch("openfl.contracts.FLChallenge.printer.print_bar"), \
             patch("openfl.api.ConnectionHelper.ConnectionHelper.build_tx",
                   return_value={"gas": 1, "gasPrice": 1, "nonce": 1, "from": "0x" + "0" * 40}):
            g.fork = True
            g.w3 = fake_w3
            challenge.quick_feedback_round(ed)

    def test_roundrep_last_entry_matches_on_chain_aggregate(self):
        challenge = _make_challenge()
        users = [_make_voter(i) for i in range(3)]
        on_chain = {users[0].address: 100, users[1].address: -50, users[2].address: 0}
        _wire_for_feedback(challenge, users, on_chain)

        votes = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
        ed = _build_eval_data(users, votes)
        self._run_feedback(challenge, ed)

        for u in users:
            assert u._roundrep, f"{u.id} got no _roundrep entry"
            assert u._roundrep[-1] == on_chain[u.address], (
                f"{u.id}: _roundrep[-1]={u._roundrep[-1]} != "
                f"on-chain aggregate {on_chain[u.address]}"
            )

    def test_filter_uses_aggregate_not_last_per_vote_delta(self):
        """U1 receives a final -1 vote but has positive aggregate.
        After the fix the contributors filter (`_roundrep[-1] >= 0`) keeps U1."""
        challenge = _make_challenge()
        users = [_make_voter(i) for i in range(3)]
        on_chain = {users[0].address: 5, users[1].address: 3, users[2].address: -2}
        _wire_for_feedback(challenge, users, on_chain)

        # U2's last vote on U1 is -1 — pre-fix this would exclude U1.
        votes = [
            [0,  1, -1],
            [1,  0,  1],
            [1, -1,  0],
        ]
        ed = _build_eval_data(users, votes)
        self._run_feedback(challenge, ed)

        contributors = [u for u in users if u._roundrep[-1] >= 0]
        ids = {u.id for u in contributors}
        assert ids == {users[0].id, users[1].id}, (
            f"Expected U0 & U1 included, U2 excluded; got {ids}"
        )

    def test_disqualified_users_also_get_aggregate_appended(self):
        challenge = _make_challenge()
        active = [_make_voter(0), _make_voter(1)]
        disq = _make_voter(2)
        challenge.pytorch_model.disqualified = [disq]
        on_chain = {active[0].address: 7, active[1].address: 4, disq.address: -10}
        _wire_for_feedback(challenge, active, on_chain)

        votes = [[0, 1, 0], [1, 0, 0]]
        ed = EvaluationData.new(active)
        for i in range(2):
            for j in range(2):
                ed.feedback_matrix[active[i].id, active[j].id] = votes[i][j]
                ed.accuracy_matrix[active[i].id, active[j].id] = 0
                ed.loss_matrix[active[i].id, active[j].id] = 0
            ed.prev_accuracies[active[i].id] = 0
            ed.prev_losses[active[i].id] = 0

        self._run_feedback(challenge, ed)

        assert disq._roundrep[-1] == -10
        for u in active:
            assert u._roundrep[-1] == on_chain[u.address]


class TestPerUserFailureLogging:
    """Loss-based score methods must surface per-user ValueError with the
    user's display label rather than a generic 'An error occurred'."""

    def _make_user(self, label, address):
        u = MagicMock()
        u.address = address
        u.display_label.return_value = label
        return u

    def _wire_losses(self, challenge, prev_losses, per_user_losses):
        challenge.contract.functions.getAllPreviousAccuraciesAndLosses.return_value.call.return_value = (
            [0] * len(prev_losses), prev_losses,
        )

        def _per_user(addr):
            mock = MagicMock()
            mock.call.return_value = ([], per_user_losses[addr])
            return mock

        challenge.contract.functions.getAllLossesAbout.side_effect = _per_user

    def _patched_remove_outliers_mad(self, fail_for):
        """Real remove_outliers_mad except it raises ValueError when called
        with the loss vector belonging to `fail_for` (matched by content)."""
        from openfl.contracts.FLChallenge import remove_outliers_mad as _real

        def _fn(arr, *args, **kwargs):
            label = kwargs.get("label")
            if label == "current" and list(arr) == fail_for:
                raise ValueError("simulated MAD failure for this user")
            return _real(arr, *args, **kwargs)
        return _fn

    def _capture(self):
        captured = []
        def fake_log(tag, *args, **kwargs):
            captured.append(" ".join(str(a) for a in args))
        return captured, fake_log

    def test_loss_tolerance_aware_logs_user_label_on_value_error(self):
        challenge = _make_challenge()
        users = [
            self._make_user("User Good", "0xGood"),
            self._make_user("User Bad",  "0xBad"),
        ]
        self._wire_losses(
            challenge,
            prev_losses=[100, 100, 100],
            per_user_losses={"0xGood": [100, 110, 105], "0xBad": [100, 110]},
        )

        captured, fake_log = self._capture()
        bad_losses = [100, 110]
        with patch("openfl.contracts.FLChallenge.log", fake_log), \
             patch("openfl.contracts.FLChallenge.remove_outliers_mad",
                   side_effect=self._patched_remove_outliers_mad(bad_losses)):
            try:
                challenge._calculate_scores_loss_tolerance_aware(users)
            except Exception:
                pass

        skipped = [line for line in captured if "SKIPPED" in line]
        assert any("User Bad" in line for line in skipped), (
            f"Expected SKIPPED line tagged with 'User Bad', got: {skipped}"
        )
        assert any("User Good" in line and "loss=" in line for line in captured)

    def test_loss_only_logs_user_label_on_value_error(self):
        challenge = _make_challenge()
        users = [
            self._make_user("Alpha", "0xAlpha"),
            self._make_user("Beta",  "0xBeta"),
        ]
        self._wire_losses(
            challenge,
            prev_losses=[100, 100, 100],
            per_user_losses={"0xAlpha": [100, 110], "0xBeta": [120, 130]},
        )

        captured, fake_log = self._capture()
        bad_losses = [120, 130]
        with patch("openfl.contracts.FLChallenge.log", fake_log), \
             patch("openfl.contracts.FLChallenge.remove_outliers_mad",
                   side_effect=self._patched_remove_outliers_mad(bad_losses)):
            try:
                challenge._calculate_scores_loss_only(users)
            except Exception:
                pass

        assert any("Beta" in line and "SKIPPED" in line for line in captured), (
            f"Expected SKIPPED line tagged with 'Beta', got: {captured}"
        )

    def test_loss_tolerance_snap_logs_user_label_on_value_error(self):
        challenge = _make_challenge()
        users = [
            self._make_user("OK",   "0xOK"),
            self._make_user("Fail", "0xFail"),
        ]
        self._wire_losses(
            challenge,
            prev_losses=[100, 100, 100],
            per_user_losses={"0xOK": [100, 105], "0xFail": [115, 120]},
        )

        captured, fake_log = self._capture()
        bad_losses = [115, 120]
        with patch("openfl.contracts.FLChallenge.log", fake_log), \
             patch("openfl.contracts.FLChallenge.remove_outliers_mad",
                   side_effect=self._patched_remove_outliers_mad(bad_losses)):
            try:
                challenge._calculate_scores_loss_tolerance_snap(users)
            except Exception:
                pass

        assert any("Fail" in line and "SKIPPED" in line for line in captured), (
            f"Expected SKIPPED line tagged with 'Fail', got: {captured}"
        )
