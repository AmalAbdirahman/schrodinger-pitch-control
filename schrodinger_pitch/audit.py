"""How much of the match does the broadcast see, by player speed and pitch zone?"""
import numpy as np
from scipy.ndimage import gaussian_filter

from .config import MAX_PLAYER_SPEED, SPRINT_SPEED

SPEED_BANDS = ((0.0, 2.0), (2.0, 4.0), (4.0, SPRINT_SPEED), (SPRINT_SPEED, MAX_PLAYER_SPEED + 1e-6))
WIDE_THRESHOLDS = (20.0, 24.0, 28.0)
HEATMAP_BINS = (52, 34)
HEATMAP_SMOOTHING = 1.5
MIN_BIN_EVENTS = 5


def observability_audit(match, vx, vy):
    """P(undetected | ...), always beside the base rate.

    Ghost speeds come from SkillCorner's interpolated tracks, so these rates are estimates.
    Straightened paths hide fast running, while jumps at tracking switches can invent it, so
    the direction of the bias is not known.
    """
    speed = np.hypot(vx, vy)
    valid = ~np.isnan(speed)
    hidden = valid & ~match.detected
    sprint = valid & (speed >= SPRINT_SPEED)
    fast = valid & (speed >= 4.0)

    def rate(mask):
        total = int(mask.sum())
        return {"n": total, "p_undetected": float(hidden[mask].mean()) if total else None}

    half_length, half_width = match.pitch_length / 2, match.pitch_width / 2
    extent = [[-half_length, half_length], [-half_width, half_width]]
    fast_all, x_edges, y_edges = np.histogram2d(
        match.x[fast], match.y[fast], bins=HEATMAP_BINS, range=extent
    )
    fast_hidden, _, _ = np.histogram2d(
        match.x[fast & hidden], match.y[fast & hidden], bins=HEATMAP_BINS, range=extent
    )
    # Smooth numerator and denominator separately, so empty bins do not read as zero risk.
    smooth_all = gaussian_filter(fast_all, HEATMAP_SMOOTHING)
    smooth_hidden = gaussian_filter(fast_hidden, HEATMAP_SMOOTHING)
    heatmap = np.full(smooth_all.shape, np.nan)
    enough = smooth_all >= MIN_BIN_EVENTS
    heatmap[enough] = smooth_hidden[enough] / smooth_all[enough]

    summary = {
        "base_rate": rate(valid),
        "at_least_4_mps": rate(fast),
        "sprint": rate(sprint),
        "bands": [
            {"low": low, "high": high, **rate(valid & (speed >= low) & (speed < high))}
            for low, high in SPEED_BANDS
        ],
        "wide": [
            {
                "abs_y_over": limit,
                "at_least_4_mps": rate(fast & (np.abs(match.y) > limit)),
                "all_speeds": rate(valid & (np.abs(match.y) > limit)),
            }
            for limit in WIDE_THRESHOLDS
        ],
    }
    return summary, heatmap, (x_edges[0], x_edges[-1], y_edges[0], y_edges[-1])
