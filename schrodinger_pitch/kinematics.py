"""Velocity estimation from tracking positions."""
import numpy as np
from scipy.signal import savgol_filter

from .config import FRAME_DT, MAX_PLAYER_SPEED, VELOCITY_POLYORDER, VELOCITY_WINDOW


def _segments(mask):
    """(start, stop) pairs of the True runs in a boolean vector."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def _derivative(values):
    if len(values) >= VELOCITY_WINDOW:
        return savgol_filter(
            values, VELOCITY_WINDOW, VELOCITY_POLYORDER, deriv=1, delta=FRAME_DT, mode="interp"
        )
    if len(values) >= 2:
        return np.gradient(values, FRAME_DT)
    return np.zeros_like(values)


def estimate_velocities(x, y):
    """Savitzky-Golay velocities for [frame, player] arrays, per contiguous track segment.

    Segments never span frames without data, and speeds are capped at MAX_PLAYER_SPEED.
    Frames with no position stay NaN.
    """
    vx, vy = np.full_like(x, np.nan), np.full_like(y, np.nan)
    for j in range(x.shape[1]):
        for start, stop in _segments(~np.isnan(x[:, j])):
            vx[start:stop, j] = _derivative(x[start:stop, j])
            vy[start:stop, j] = _derivative(y[start:stop, j])
    speed = np.hypot(vx, vy)
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = np.where(speed > MAX_PLAYER_SPEED, MAX_PLAYER_SPEED / speed, 1.0)
    return vx * scale, vy * scale
