"""E[PPCF] over ghost positions, compared with the plug-in surface."""
from dataclasses import dataclass

import numpy as np

from .control import pitch_control, pitch_grid
from .ghosts import sample_positions

BATCH_SIZE = 1  # Larger batches measured slower per draw (0.10 s against 0.056 s).
MATERIAL_CHANGE = 0.05


@dataclass
class FrameSurface:
    frame: int
    players: np.ndarray
    attacking: np.ndarray
    ball: np.ndarray
    shape: tuple
    cell_area: float
    plugin: np.ndarray
    mean: np.ndarray
    sd: np.ndarray


def frame_is_usable(match, frame):
    return (
        match.present[frame].sum() == 22
        and not np.isnan(match.ball[frame]).any()
        and match.possession_team[frame] != 0
    )


def usable_frames(match, count):
    """Evenly spaced frames with all players, a known ball and known possession."""
    frames = np.array([f for f in range(match.n_frames) if frame_is_usable(match, f)])
    return frames[np.linspace(0, len(frames) - 1, min(count, len(frames))).astype(int)]


def frame_surface(match, vx, vy, frame, model, gaps, params, n_draws, rng, cell=2.0):
    """Plug-in control, E[control] and SD[control] for the team in possession."""
    if not frame_is_usable(match, frame):
        raise ValueError(f"Frame {frame} lacks players, ball or possession.")
    players = np.flatnonzero(match.present[frame])
    attacking = match.team_ids[players] == match.possession_team[frame]
    is_gk = match.is_gk[players]
    ball = match.ball[frame]
    velocities = np.column_stack([vx[frame, players], vy[frame, players]])
    positions = np.column_stack([match.x[frame, players], match.y[frame, players]])
    targets, shape, cell_area = pitch_grid(match.pitch_length, match.pitch_width, cell)

    plugin, _ = pitch_control(positions, velocities, targets, ball, attacking, is_gk, params)
    draws = sample_positions(match, frame, players, model, n_draws, rng, *gaps)
    samples = []
    for start in range(0, n_draws, BATCH_SIZE):
        batch = draws[start : start + BATCH_SIZE]
        speeds = np.broadcast_to(velocities, batch.shape)
        samples.append(pitch_control(batch, speeds, targets, ball, attacking, is_gk, params)[0])
    samples = np.concatenate(samples)
    return FrameSurface(
        frame=int(frame),
        players=players,
        attacking=attacking,
        ball=ball,
        shape=shape,
        cell_area=cell_area,
        plugin=plugin,
        mean=samples.mean(axis=0),
        sd=samples.std(axis=0),
    )


def surface_metrics(surface):
    """Signed and absolute change in m^2, so values do not depend on the grid."""
    delta = surface.mean - surface.plugin
    return {
        "frame": surface.frame,
        "net_m2": float(delta.sum() * surface.cell_area),
        "absolute_m2": float(np.abs(delta).sum() * surface.cell_area),
        "max_increase": float(delta.max()),
        "max_decrease": float(delta.min()),
        "share_changed": float((np.abs(delta) > MATERIAL_CHANGE).mean()),
        "max_sd": float(surface.sd.max()),
        "mean_attacking_control": float(surface.plugin.mean()),
    }
