"""Potential pitch control in the style of Spearman (2018).

For player j, with arrival probability f_j and control rate lambda_j,

    dPPCF_j/dt = (1 - sum_k PPCF_k) * f_j(t) * lambda_j,    t >= T_ball.

Writing h_j = lambda_j f_j and L = sum_k h_k, the uncontrolled probability R = 1 - sum_k PPCF_k
obeys dR/dt = -L R. Holding each h_j constant over a step gives an exact update:

    R' = R exp(-L dt),    PPCF_j' = PPCF_j + (R - R') h_j / L.

Unlike forward Euler this cannot leave [0, 1] at any step size, and sum_k PPCF_k = 1 - R holds
to rounding error at every step.
"""
import numpy as np
from scipy.special import expit


def time_to_intercept(positions, velocities, targets, params):
    """Expected arrival times, shape (..., players, targets).

    The player keeps their velocity for the reaction time, then runs straight at max speed.
    """
    reaction_point = positions + velocities * params.reaction_time
    offset = targets[..., None, :, :] - reaction_point[..., :, None, :]
    return params.reaction_time + np.linalg.norm(offset, axis=-1) / params.max_speed


def control_rates(attacking, is_gk, params):
    rates = np.where(attacking, params.lambda_att, params.lambda_att * params.kappa_def)
    return np.where(~attacking & is_gk, rates * params.gk_multiplier, rates)


def control_from_arrival_times(tau, ball_time, rates, params):
    """Integrate the control ODE for tau of shape (players, cells).

    Returns (ppcf of shape (players, cells), residual of shape (cells,)). The horizon is set
    per cell by the residual tolerance, so a late-arriving lone player still converges.
    """
    ppcf = np.zeros_like(tau)
    residual = np.ones(tau.shape[1])
    # Nothing happens until someone is within reach, so skip that time. The hazard skipped is
    # at most lambda * exp(-steepness * margin) per second, below 1e-6 for the default margin.
    time = np.maximum(ball_time, tau.min(axis=0) - params.skip_margin)
    steepness = np.pi / np.sqrt(3.0) / params.arrival_sigma
    active = np.arange(tau.shape[1])
    for _ in range(int(np.ceil(params.max_time / params.dt))):
        midpoint = time[active] + params.dt / 2
        hazard = rates[:, None] * expit(steepness * (midpoint - tau[:, active]))
        total = hazard.sum(axis=0)
        captured = -np.expm1(-total * params.dt) * residual[active]
        share = np.divide(hazard, total, out=np.zeros_like(hazard), where=total > 0)
        ppcf[:, active] += captured * share
        residual[active] -= captured
        time[active] += params.dt
        active = active[residual[active] > params.residual_tol]
        if active.size == 0:
            break
    return ppcf, residual


def pitch_control(positions, velocities, targets, ball, attacking, is_gk, params):
    """Attacking and defending control at each target.

    positions and velocities are (players, 2) or (draws, players, 2). The result drops or keeps
    the draws axis to match: (targets,) or (draws, targets).
    """
    single = positions.ndim == 2
    if single:
        positions, velocities = positions[None], velocities[None]
    draws, players = positions.shape[:2]
    tau = time_to_intercept(positions, velocities, targets, params)
    tau = tau.transpose(1, 0, 2).reshape(players, -1)
    ball_time = np.linalg.norm(targets - ball, axis=-1) / params.ball_speed
    ppcf, _ = control_from_arrival_times(
        tau, np.tile(ball_time, draws), control_rates(attacking, is_gk, params), params
    )
    attack = ppcf[attacking].sum(axis=0).reshape(draws, -1)
    defend = ppcf[~attacking].sum(axis=0).reshape(draws, -1)
    return (attack[0], defend[0]) if single else (attack, defend)


def pitch_grid(length, width, cell=2.0):
    """Cell-centred grid over the actual pitch, so cell_area * n_cells is the pitch area."""
    nx, ny = int(round(length / cell)), int(round(width / cell))
    xs = (np.arange(nx) + 0.5) * length / nx - length / 2
    ys = (np.arange(ny) + 0.5) * width / ny - width / 2
    grid_x, grid_y = np.meshgrid(xs, ys)
    targets = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    return targets, grid_x.shape, length * width / (nx * ny)
