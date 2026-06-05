"""
TR ceiling analysis — diagnose why TR stays in 0-0.7 range.

Writes detailed output to experiment/logs/tr_analysis.log.
Run with: python experiment/tr_analysis.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openfl.utils.printer import log, set_log_file, set_enabled_tags

LOG_FILE = Path(__file__).resolve().parent / "logs" / "tr_analysis.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
set_log_file(str(LOG_FILE))
set_enabled_tags({"tr_analysis"})

TAG = "tr_analysis"

# ── mirror multirep.py constants ──────────────────────────────────────────────
_WAD              = 10 ** 18
_ALPHA            = int(2e17)
_N_BLEND          = int(2e17)
_N_0              = 5
_LAMBDA           = 5
_STAKE_WAD        = int(1e18)

# Experiment defaults (from experiment_configuration.py + preset)
_REWARD_DEFAULT   = int(8e18)
_NR_ACTIVE        = 8


def _transform_delta(delta: int, stake: int, reward: int, nr_active: int,
                     gain_cap_mult: int) -> int:
    max_gain = (gain_cap_mult * reward) // nr_active if nr_active > 0 else 0
    range_   = stake + max_gain
    if range_ == 0:
        return 0
    shifted = delta + stake
    if shifted <= 0:
        return 0
    if shifted >= range_:
        return _WAD
    return (shifted * _WAD) // range_


def _update_running_stats(contrib: int, prior_mean: int, prior_m2: int, k: int):
    if k <= 1:
        new_mean = contrib
    else:
        new_mean = ((_WAD - _ALPHA) * prior_mean + _ALPHA * contrib) // _WAD
    abs_d1 = abs(contrib - prior_mean)
    abs_d2 = abs(contrib - new_mean)
    new_m2 = ((_WAD - _ALPHA) * prior_m2) // _WAD + (_ALPHA * abs_d1 * abs_d2) // (_WAD * _WAD)
    return new_mean, new_m2


def _compute_confidence(k: int, s_k: int) -> int:
    if k == 0:
        return 0
    maturity  = (k * _WAD) // (k + _N_0)
    stability = (_WAD * _WAD) // (_WAD + _LAMBDA * s_k)
    return (maturity * stability) // _WAD


def _update_contrib_score(prior_tr: int, confidence: int, contrib: int) -> int:
    weighted = (confidence * contrib) // _WAD
    return ((_WAD - _N_BLEND) * prior_tr + _N_BLEND * weighted) // _WAD


def simulate(label: str, n_tasks: int, delta_eth: float, gain_cap_mult: int,
             reward: int = _REWARD_DEFAULT, nr_active: int = _NR_ACTIVE):
    """Simulate TR over n_tasks with constant delta."""
    delta   = int(delta_eth * _WAD)
    max_gain = (gain_cap_mult * reward) // nr_active
    range_   = _STAKE_WAD + max_gain

    log(TAG, "")
    log(TAG, f"=== {label} ===")
    log(TAG, f"  params: delta={delta_eth:+.2f} ETH  gain_cap_mult={gain_cap_mult}x"
             f"  reward={reward/_WAD:.1f} ETH  nr_active={nr_active}"
             f"  n_tasks={n_tasks}")
    log(TAG, f"  computed: max_gain={max_gain/_WAD:.3f} ETH  range={range_/_WAD:.3f} ETH")

    cs_single = _transform_delta(delta, _STAKE_WAD, reward, nr_active, gain_cap_mult)
    log(TAG, f"  contrib_score (constant): {cs_single/_WAD:.4f}  ← ceiling for this delta")

    header = (f"  {'k':>3}  {'contrib':>8}  {'maturity':>9}  {'s_k':>8}  "
              f"{'stability':>10}  {'confidence':>11}  {'TR':>8}")
    log(TAG, header)
    log(TAG, "  " + "─" * (len(header) - 2))

    tr = 0
    mean = 0
    m2   = 0
    for k in range(1, n_tasks + 1):
        cs           = _transform_delta(delta, _STAKE_WAD, reward, nr_active, gain_cap_mult)
        mean, m2     = _update_running_stats(cs, mean, m2, k)
        confidence   = _compute_confidence(k, m2)
        tr           = _update_contrib_score(tr, confidence, cs)

        maturity  = (k * _WAD) // (k + _N_0)
        stability = (_WAD * _WAD) // (_WAD + _LAMBDA * m2)

        log(TAG, f"  {k:>3}  {cs/_WAD:>8.4f}  {maturity/_WAD:>9.4f}  {m2/_WAD:>8.4f}  "
                 f"{stability/_WAD:>10.4f}  {confidence/_WAD:>11.4f}  {tr/_WAD:>8.4f}")

    log(TAG, f"  → final TR after {n_tasks} tasks: {tr/_WAD:.4f}")
    return tr


def show_delta_sweep():
    """What contrib_score does each delta level produce?"""
    log(TAG, "")
    log(TAG, "=== contrib_score vs delta (current params: gain_cap_mult=2, reward=8 ETH, nr_active=8) ===")
    log(TAG, f"  {'delta (ETH)':>12}  {'shifted (ETH)':>14}  {'range (ETH)':>12}  {'contrib_score':>14}")
    log(TAG, "  " + "─" * 58)
    max_gain = (2 * _REWARD_DEFAULT) // _NR_ACTIVE  # = 2 ETH
    range_   = _STAKE_WAD + max_gain                 # = 3 ETH
    for d_eth in [-1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5]:
        delta   = int(d_eth * _WAD)
        shifted = delta + _STAKE_WAD
        cs      = _transform_delta(delta, _STAKE_WAD, _REWARD_DEFAULT, _NR_ACTIVE, 2)
        log(TAG, f"  {d_eth:>12.2f}  {shifted/_WAD:>14.3f}  {range_/_WAD:>12.3f}  {cs/_WAD:>14.4f}")
    log(TAG, f"  NOTE: delta > +1 ETH is needed to push contrib_score above 0.667 with gain_cap_mult=2")


def main():
    log(TAG, "=" * 72)
    log(TAG, "TR CEILING ANALYSIS")
    log(TAG, "=" * 72)

    # ── Part 1: explain the contrib_score cap ────────────────────────────────
    log(TAG, "")
    log(TAG, "PART 1 — contrib_score cap from _transform_delta")
    show_delta_sweep()

    # ── Part 2: simulate TR for the actual experiment setup ──────────────────
    log(TAG, "")
    log(TAG, "PART 2 — TR simulation (constant delta, zero variance)")

    # Actual experiment: 5 tasks/type, 8/20 participants selected → avg ~2 tasks per user
    simulate("Current params | delta=+1 ETH | 2 tasks (avg in 5-task experiment)",
             n_tasks=2, delta_eth=1.0, gain_cap_mult=2)

    simulate("Current params | delta=+1 ETH | 5 tasks",
             n_tasks=5, delta_eth=1.0, gain_cap_mult=2)

    simulate("Current params | delta=+1 ETH | 10 tasks",
             n_tasks=10, delta_eth=1.0, gain_cap_mult=2)

    simulate("Current params | delta=+1 ETH | 20 tasks",
             n_tasks=20, delta_eth=1.0, gain_cap_mult=2)

    # ── Part 3: what if gain_cap_mult=1 (delta range matches actual) ─────────
    log(TAG, "")
    log(TAG, "PART 3 — Fix: gain_cap_mult=1 (max_gain = reward/nr_active = 1 ETH = actual max delta)")

    simulate("gain_cap_mult=1 | delta=+1 ETH | 5 tasks",
             n_tasks=5, delta_eth=1.0, gain_cap_mult=1)

    simulate("gain_cap_mult=1 | delta=+1 ETH | 10 tasks",
             n_tasks=10, delta_eth=1.0, gain_cap_mult=1)

    simulate("gain_cap_mult=1 | delta=+1 ETH | 20 tasks",
             n_tasks=20, delta_eth=1.0, gain_cap_mult=1)

    # ── Part 4: what does the system need to reach TR=0.8? ───────────────────
    log(TAG, "")
    log(TAG, "PART 4 — TR=0.8 requires:")
    log(TAG, "  confidence * contrib_score = 0.8 at steady state")
    log(TAG, "  With contrib_score=0.667 (gain_cap_mult=2, delta=+1):  confidence >= 1.20 → impossible")
    log(TAG, "  With contrib_score=1.000 (gain_cap_mult=1, delta=+1):  confidence >= 0.80")
    log(TAG, "    → maturity=0.8 needs k=20 (k/(k+5)=0.8) → 20 tasks per type per user")
    log(TAG, "    → EWMA converges in ~20 tasks (N_BLEND=0.2, 0.8^20 ≈ 0.012)")
    log(TAG, "")
    log(TAG, "  Summary of blockers:")
    log(TAG, "  [1] gain_cap_mult=2 caps contrib_score at 0.667 for actual delta (+1 ETH)")
    log(TAG, "      Fix: set gain_cap_mult=1 so max delta matches actual max reward share")
    log(TAG, "  [2] Low task count per type (avg ~2 with 5-task experiment, 8/20 selected)")
    log(TAG, "      Fix: more tasks OR reduce N_0 (e.g. N_0=2 gives 0.8 maturity at k=8)")
    log(TAG, "  [3] Slow EWMA (N_BLEND=0.2) needs ~20 tasks to converge")
    log(TAG, "      Fix: increase N_BLEND (e.g. 0.3-0.4) for faster buildup")

    # ── Part 5: show what N_0 change does ────────────────────────────────────
    log(TAG, "")
    log(TAG, "PART 5 — Effect of N_0 on maturity (current N_0=5):")
    log(TAG, f"  {'k':>4}  N0=5: {'maturity':>8}  N0=2: {'maturity':>8}  N0=1: {'maturity':>8}")
    for k in [1, 2, 3, 5, 8, 10, 20]:
        m5 = (k * _WAD) // (k + 5)
        m2 = (k * _WAD) // (k + 2)
        m1 = (k * _WAD) // (k + 1)
        log(TAG, f"  {k:>4}  N0=5: {m5/_WAD:>8.4f}  N0=2: {m2/_WAD:>8.4f}  N0=1: {m1/_WAD:>8.4f}")

    log(TAG, "")
    log(TAG, "=" * 72)
    log(TAG, f"Output written to: {LOG_FILE}")


if __name__ == "__main__":
    main()
