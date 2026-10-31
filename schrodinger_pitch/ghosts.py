"""Position uncertainty for players the broadcast cannot see ("ghosts").

SkillCorner's off-camera tracks end within about 0.2 m of the re-detection point at any gap
length, so they are conditioned on the future. The right object is therefore a bridge pinned at
both detections: the SkillCorner position is the mean, and the spread is largest mid-gap.

The spread is a function of the bridge time tau = s u / (s + u), with s the time since the last
detection and u the time to the next one. For a Brownian bridge the variance is proportional to
tau. Player motion is not Brownian, so sigma(tau) is measured from data instead: gaps are masked
inside detected tracks, where the truth is known.
"""
import json
from dataclasses import dataclass

import numpy as np
from matplotlib.path import Path

from .config import FRAME_DT

MIN_GAP = 0.3  # s
MAX_GAP = 60.0  # s
N_CURVE_BINS = 24


@dataclass(frozen=True)
class SigmaCurve:
    """Per-axis position SD (m) as a piecewise-linear function of bridge time (s)."""

    taus: tuple
    sigmas: tuple

    def __call__(self, tau):
        return np.interp(tau, self.taus, self.sigmas)

    def as_dict(self):
        return {"taus": self.taus, "sigmas": self.sigmas}


@dataclass(frozen=True)
class GhostModel:
    """Goalkeepers are mostly off camera yet barely move, so they get their own curve."""

    outfield: SigmaCurve
    goalkeeper: SigmaCurve
    camera_ratio: float

    def sigma(self, tau, is_gk):
        return self.goalkeeper(tau) if is_gk else self.outfield(tau)

    def save(self, path):
        path.write_text(
            json.dumps(
                {
                    "outfield": self.outfield.as_dict(),
                    "goalkeeper": self.goalkeeper.as_dict(),
                    "camera_ratio": self.camera_ratio,
                },
                indent=2,
            )
        )

    @classmethod
    def load(cls, path):
        stored = json.loads(path.read_text())
        curves = [
            SigmaCurve(tuple(stored[key]["taus"]), tuple(stored[key]["sigmas"]))
            for key in ("outfield", "goalkeeper")
        ]
        return cls(*curves, stored["camera_ratio"])


def bridge_time(since, until):
    """s u / (s + u), tending to the finite argument if one side is unbounded."""
    since, until = np.asarray(since, dtype=float), np.asarray(until, dtype=float)
    a, b = np.minimum(since, until), np.maximum(since, until)
    ratio = np.divide(a, b, out=np.zeros_like(a), where=np.isfinite(b) & (b > 0))
    return a / (1.0 + ratio)


def detection_gaps(match):
    """Seconds since the last and until the next detection, within the period.

    Both are 0 on detected frames and inf when no such detection exists.
    """
    n, p = match.x.shape
    since, until = np.full((n, p), np.inf), np.full((n, p), np.inf)
    for period in np.unique(match.period[match.period > 0]):
        rows = np.flatnonzero(match.period == period)
        detected = match.detected[rows]
        last = np.where(detected, rows[:, None], -1)
        last = np.maximum.accumulate(last, axis=0).astype(float)
        last[last < 0] = -np.inf
        nxt = np.where(detected, rows[:, None], match.n_frames * 2)
        nxt = np.minimum.accumulate(nxt[::-1], axis=0)[::-1].astype(float)
        nxt[nxt >= match.n_frames * 2] = np.inf
        since[rows] = (rows[:, None] - last) * FRAME_DT
        until[rows] = (nxt - rows[:, None]) * FRAME_DT
    return since, until


def _detected_runs(match, periods, players):
    runs = []
    for j in players:
        for period in periods:
            mask = match.detected[:, j] & (match.period == period)
            edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
            starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
            runs.extend((j, a, b) for a, b in zip(starts, stops) if b - a >= 8)
    return np.array(runs, dtype=int).reshape(-1, 3)


def real_gap_lengths(match):
    """Lengths in seconds of real bounded ghost episodes."""
    since, until = detection_gaps(match)
    ghost = match.present & ~match.detected & np.isfinite(since) & np.isfinite(until)
    # A gap of length T contributes one frame per 0.1 s, so sample from its first frame.
    first = ghost & (np.abs(since - FRAME_DT) < FRAME_DT / 2)
    return (since[first] + until[first]).ravel()


def sample_masked_gaps(match, periods, n, rng, players=None):
    """Hide a synthetic gap inside detected tracks and pick one interior frame of each.

    Gap lengths follow the real gap distribution, but gaps longer than any detected run are
    dropped, so very long gaps are under-represented. Returns a dict of arrays: player, the
    frames a (last seen), m (evaluated) and b (seen again).
    """
    lengths = real_gap_lengths(match)
    lengths = lengths[(lengths >= MIN_GAP) & (lengths <= MAX_GAP)]
    players = range(match.x.shape[1]) if players is None else players
    runs = _detected_runs(match, periods, players)
    run_frames = runs[:, 2] - runs[:, 1]
    order = np.argsort(run_frames)
    runs, run_frames = runs[order], run_frames[order]
    # Number of valid start frames in each run is run_frames - gap; choose run by that weight.
    chosen = {key: [] for key in ("player", "a", "m", "b")}
    gaps = np.round(rng.choice(lengths, size=n) / FRAME_DT).astype(int)
    for gap in np.unique(gaps):
        count = int((gaps == gap).sum())
        first = int(np.searchsorted(run_frames, gap + 1))
        if first == len(runs):
            continue
        weights = (run_frames[first:] - gap).astype(float)
        pick = rng.choice(len(weights), size=count, p=weights / weights.sum()) + first
        a = runs[pick, 1] + rng.integers(0, run_frames[pick] - gap)
        m = a + rng.integers(1, gap, size=count) if gap > 1 else a
        chosen["player"].append(runs[pick, 0])
        chosen["a"].append(a)
        chosen["m"].append(m)
        chosen["b"].append(a + gap)
    return {key: np.concatenate(values) for key, values in chosen.items()}


def masked_errors(match, sample):
    """Errors of the chord (the bridge mean) at the hidden frames, and their bridge times."""
    j, a, m, b = sample["player"], sample["a"], sample["m"], sample["b"]
    weight = (m - a) / (b - a)
    chord_x = match.x[a, j] + weight * (match.x[b, j] - match.x[a, j])
    chord_y = match.y[a, j] + weight * (match.y[b, j] - match.y[a, j])
    errors = np.column_stack([match.x[m, j] - chord_x, match.y[m, j] - chord_y])
    return errors, bridge_time((m - a) * FRAME_DT, (b - m) * FRAME_DT)


def fit_sigma_curve(errors, tau):
    """Bin by bridge time, take the per-axis RMS, and enforce monotonicity."""
    edges = np.unique(np.quantile(tau, np.linspace(0, 1, N_CURVE_BINS + 1)))
    bins = np.clip(np.searchsorted(edges, tau, side="right") - 1, 0, len(edges) - 2)
    taus, sigmas = [0.0], [0.0]
    for k in range(len(edges) - 1):
        inside = bins == k
        if inside.any():
            taus.append(float(tau[inside].mean()))
            sigmas.append(float(np.sqrt(np.mean(errors[inside] ** 2))))
    return SigmaCurve(tuple(taus), tuple(np.maximum.accumulate(sigmas).tolist()))


def camera_likelihood_ratio(match, stride=10):
    """P(undetected | inside the camera view) / P(undetected | outside it).

    Being undetected makes the inside of the view unlikely, not impossible.
    """
    counts = np.zeros((2, 2))  # [inside][undetected]
    for frame in range(0, match.n_frames, stride):
        corners, present = match.camera[frame], match.present[frame]
        if np.isnan(corners).any() or not present.any():
            continue
        points = np.column_stack([match.x[frame, present], match.y[frame, present]])
        inside = Path(corners).contains_points(points)
        hidden = ~match.detected[frame, present]
        for i, is_inside in enumerate((False, True)):
            counts[i, 1] += (hidden & (inside == is_inside)).sum()
            counts[i, 0] += (~hidden & (inside == is_inside)).sum()
    rates = counts[:, 1] / counts.sum(axis=1)
    return float(rates[1] / rates[0])


def sample_positions(match, frame, players, model, n_draws, rng, since, until):
    """Joint draws of the players' positions, shape (draws, players, 2).

    Detected players stay fixed. Each ghost is drawn independently around the SkillCorner
    position and reweighted by the camera likelihood ratio.
    """
    positions = np.column_stack([match.x[frame, players], match.y[frame, players]])
    draws = np.repeat(positions[None], n_draws, axis=0)
    ghosts = np.flatnonzero(~match.detected[frame, players])
    corners = match.camera[frame]
    camera = None if np.isnan(corners).any() else Path(corners)
    for g in ghosts:
        j = players[g]
        scale = model.sigma(bridge_time(since[frame, j], until[frame, j]), match.is_gk[j])
        candidates = positions[g] + rng.normal(0.0, scale, size=(n_draws * 8, 2))
        if camera is None:
            draws[:, g] = candidates[:n_draws]
            continue
        weights = np.where(camera.contains_points(candidates), model.camera_ratio, 1.0)
        pick = rng.choice(len(candidates), size=n_draws, p=weights / weights.sum())
        draws[:, g] = candidates[pick]
    return draws
