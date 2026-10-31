"""Out-of-sample tests of the position model and the control model."""
import numpy as np

from .config import FRAME_DT
from .control import control_rates, pitch_control, time_to_intercept
from .ghosts import masked_errors, sample_positions

COVERAGE_LEVELS = (0.5, 0.8, 0.95)
TAU_BINS = ((0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, np.inf))
LEGACY_DRIFT = 1.446  # s; the original simulator's mean displacement per unit velocity.
LEGACY_SD = (3.5, 1.25)  # m; the original simulator's longitudinal and lateral SDs.
EPS = 1e-6
N_FOLDS = 5
N_BOOTSTRAP = 2000
GHOST_MARGIN = 1.0  # s; a ghost counts if it could arrive this soon after the ball.


def _gaussian_nll(errors, sd_a, sd_b):
    return np.mean(
        0.5 * ((errors[:, 0] / sd_a) ** 2 + (errors[:, 1] / sd_b) ** 2)
        + np.log(2 * np.pi * sd_a * sd_b)
    )


def position_validation(match, sample, curve, fixed_sigma):
    """Coverage and log score on masked gaps. Fit and test samples must not overlap."""
    errors, tau = masked_errors(match, sample)
    sigma = np.maximum(curve(tau), 1e-3)
    squared = np.sum((errors / sigma[:, None]) ** 2, axis=1)
    # For a 2D isotropic Gaussian, the radius containing mass q satisfies r^2 = -2 log(1 - q).
    thresholds = {q: -2 * np.log(1 - q) for q in COVERAGE_LEVELS}
    bins = []
    for low, high in TAU_BINS:
        inside = (tau >= low) & (tau < high)
        if inside.any():
            bins.append(
                {
                    "tau_low": low,
                    "tau_high": None if np.isinf(high) else high,
                    "n": int(inside.sum()),
                    "sigma": float(np.sqrt(np.mean(errors[inside] ** 2))),
                    "coverage": {
                        str(q): float((squared[inside] < t).mean()) for q, t in thresholds.items()
                    },
                }
            )

    j, a, m = sample["player"], sample["a"], sample["m"]
    velocity = np.column_stack(
        [
            (match.x[a, j] - match.x[np.maximum(a - 5, 0), j]) / (5 * FRAME_DT),
            (match.y[a, j] - match.y[np.maximum(a - 5, 0), j]) / (5 * FRAME_DT),
        ]
    )
    truth = np.column_stack([match.x[m, j], match.y[m, j]])
    start = np.column_stack([match.x[a, j], match.y[a, j]])
    velocity = np.nan_to_num(velocity)
    residual = truth - start - velocity * LEGACY_DRIFT
    # The original SDs are along and across the direction of travel, so rotate into that frame.
    heading = velocity / (np.hypot(velocity[:, 0], velocity[:, 1])[:, None] + 1e-9)
    legacy = np.column_stack(
        [
            residual[:, 0] * heading[:, 0] + residual[:, 1] * heading[:, 1],
            residual[:, 1] * heading[:, 0] - residual[:, 0] * heading[:, 1],
        ]
    )
    keep = ~np.isnan(legacy).any(axis=1)
    return {
        "n": int(len(tau)),
        "bins": bins,
        "overall_coverage": {str(q): float((squared < t).mean()) for q, t in thresholds.items()},
        "nll": {
            "bridge": float(np.mean(np.sum(errors**2, axis=1) / (2 * sigma**2) + np.log(2 * np.pi * sigma**2))),
            "chord_fixed_sigma": float(_gaussian_nll(errors, fixed_sigma, fixed_sigma)),
            "legacy_forward_simulation": float(_gaussian_nll(legacy[keep], *LEGACY_SD)),
        },
    }


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_logistic(features, outcome, iterations=50):
    """Newton-Raphson logistic regression with a small ridge for stability."""
    design = np.column_stack([np.ones(len(features)), features])
    beta = np.zeros(design.shape[1])
    for _ in range(iterations):
        p = np.clip(1 / (1 + np.exp(-design @ beta)), EPS, 1 - EPS)
        gradient = design.T @ (outcome - p) - 1e-6 * beta
        hessian = (design * (p * (1 - p))[:, None]).T @ design + 1e-6 * np.eye(len(beta))
        step = np.linalg.solve(hessian, gradient)
        beta += step
        if np.abs(step).max() < 1e-8:
            break
    return beta


def predict_logistic(beta, features):
    return 1 / (1 + np.exp(-np.column_stack([np.ones(len(features)), features]) @ beta))


def log_loss(outcome, p):
    p = np.clip(p, EPS, 1 - EPS)
    return -(outcome * np.log(p) + (1 - outcome) * np.log(1 - p))


def grouped_folds(groups, rng):
    unique = rng.permutation(np.unique(groups))
    assignment = dict(zip(unique, np.arange(len(unique)) % N_FOLDS))
    return np.array([assignment[g] for g in groups])


def cross_fitted(features, outcome, folds):
    out = np.zeros(len(outcome))
    for fold in range(N_FOLDS):
        test = folds == fold
        out[test] = predict_logistic(fit_logistic(features[~test], outcome[~test]), features[test])
    return out


def score_passes(match, passes, vx, vy, model, gaps, params, n_draws, rng):
    """PPCF at the target's tracked position at release, plug-in and over ghost draws."""
    rows = []
    for row in passes.itertuples():
        frame = row.frame
        players = np.flatnonzero(match.present[frame])
        attacking = match.team_ids[players] == row.team_id
        is_gk = match.is_gk[players]
        positions = np.column_stack([match.x[frame, players], match.y[frame, players]])
        velocities = np.column_stack([vx[frame, players], vy[frame, players]])
        ball = np.array([match.x[frame, row.passer], match.y[frame, row.passer]])
        target = np.array([[match.x[frame, row.target], match.y[frame, row.target]]])

        plugin, _ = pitch_control(positions, velocities, target, ball, attacking, is_gk, params)
        draws = sample_positions(match, frame, players, model, n_draws, rng, *gaps)
        speeds = np.broadcast_to(velocities, draws.shape)
        sampled, _ = pitch_control(draws, speeds, target, ball, attacking, is_gk, params)

        distance = float(np.linalg.norm(target[0] - ball))
        tau = time_to_intercept(positions, velocities, target, params)[:, 0]
        hidden = ~match.detected[frame, players]
        reachable = hidden & (tau <= distance / params.ball_speed + GHOST_MARGIN)
        rows.append(
            (float(plugin[0]), float(sampled.mean()), float(sampled[:, 0].std()), distance, bool(reachable.any()))
        )
    return np.array(rows)


def _bootstrap_difference(loss_a, loss_b, groups, rng):
    """Mean loss_a - loss_b with a 95% CI, resampling whole possessions."""
    unique, inverse = np.unique(groups, return_inverse=True)
    totals = np.bincount(inverse, loss_a - loss_b)
    counts = np.bincount(inverse)
    resampled = rng.integers(0, len(unique), size=(N_BOOTSTRAP, len(unique)))
    values = totals[resampled].sum(axis=1) / counts[resampled].sum(axis=1)
    low, high = np.percentile(values, [2.5, 97.5])
    return {"mean": float((loss_a - loss_b).mean()), "ci95": [float(low), float(high)]}


def pass_validation(passes, scores, rng):
    """Compare forecasts of completion. Calibrated forecasts are cross-fitted by possession."""
    outcome = passes["completed"].to_numpy(dtype=float)
    groups = passes["group"].to_numpy()
    plugin, mean, sd, distance, ghost_relevant = scores.T
    ghost_relevant = ghost_relevant.astype(bool)
    folds = grouped_folds(groups, rng)

    train_rate = np.zeros(len(outcome))
    for fold in range(N_FOLDS):
        train_rate[folds == fold] = outcome[folds != fold].mean()
    forecasts = {
        "base_rate": train_rate,
        "distance_logistic": cross_fitted(distance[:, None], outcome, folds),
        "plugin_raw": plugin,
        "expected_raw": mean,
        "plugin_calibrated": cross_fitted(_logit(plugin)[:, None], outcome, folds),
        "expected_calibrated": cross_fitted(_logit(mean)[:, None], outcome, folds),
    }

    def table(mask):
        return {
            name: {
                "log_loss": float(log_loss(outcome[mask], p[mask]).mean()),
                "brier": float(((outcome[mask] - p[mask]) ** 2).mean()),
            }
            for name, p in forecasts.items()
        }

    def contrast(mask, a, b):
        return _bootstrap_difference(
            log_loss(outcome[mask], forecasts[a][mask]),
            log_loss(outcome[mask], forecasts[b][mask]),
            groups[mask],
            rng,
        )

    everyone = np.ones(len(outcome), dtype=bool)
    xpass = passes["xpass"].to_numpy(dtype=float)
    has_xpass = ~np.isnan(xpass)
    bins = np.clip((mean * 10).astype(int), 0, 9)
    return {
        "n_passes": int(len(outcome)),
        "n_possession_groups": int(len(np.unique(groups))),
        "completion_rate": float(outcome.mean()),
        "n_ghost_relevant": int(ghost_relevant.sum()),
        "all_passes": table(everyone),
        "ghost_relevant_passes": table(ghost_relevant),
        "primary_contrast_plugin_minus_expected": contrast(
            ghost_relevant, "plugin_calibrated", "expected_calibrated"
        ),
        "all_passes_plugin_minus_expected": contrast(everyone, "plugin_calibrated", "expected_calibrated"),
        "skillcorner_xpass_benchmark": {
            "n": int(has_xpass.sum()),
            "xpass_log_loss": float(log_loss(outcome[has_xpass], xpass[has_xpass]).mean()),
            "expected_calibrated_log_loss": float(
                log_loss(outcome[has_xpass], forecasts["expected_calibrated"][has_xpass]).mean()
            ),
        },
        "mean_draw_sd_ghost_relevant": float(sd[ghost_relevant].mean()) if ghost_relevant.any() else None,
        "calibration": [
            {
                "forecast": float(mean[bins == k].mean()),
                "observed": float(outcome[bins == k].mean()),
                "n": int((bins == k).sum()),
            }
            for k in range(10)
            if (bins == k).any()
        ],
    }
