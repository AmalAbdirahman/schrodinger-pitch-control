"""Model parameters and project-wide constants."""
import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR_ENV = "SCHRODINGER_DATA_DIR"

DEFAULT_MATCH_ID = 1886347
DEFAULT_FRAME = 5059
DEFAULT_SEED = 20251228

FRAME_DT = 0.1  # SkillCorner tracking is 10 fps; load_match verifies contiguous frames.
VELOCITY_WINDOW = 7  # Savitzky-Golay window in frames (0.7 s).
VELOCITY_POLYORDER = 2
MAX_PLAYER_SPEED = 12.0  # m/s; above this a speed is treated as a tracking glitch.
SPRINT_SPEED = 7.0  # m/s (about 25 km/h).


def default_data_dir():
    return Path(os.environ.get(DATA_DIR_ENV, PROJECT_ROOT / "data"))


@dataclass(frozen=True)
class ModelParams:
    """Spearman-style parameters, using the defaults of L. Shaw's public implementation."""

    reaction_time: float = 0.7  # s
    max_speed: float = 5.0  # m/s
    arrival_sigma: float = 0.45  # s; standard deviation of the logistic arrival time.
    lambda_att: float = 4.3  # 1/s
    kappa_def: float = 1.0  # defender rate multiplier.
    gk_multiplier: float = 3.0  # applied to a defending goalkeeper only.
    ball_speed: float = 15.0  # m/s; constant, ball flight is not modelled further.
    dt: float = 0.04  # s
    residual_tol: float = 1e-3  # stop once P(ball uncontrolled) falls below this.
    max_time: float = 60.0  # s of integration; exceeds any cross-pitch run.
    skip_margin: float = 3.5  # s before the first arrival at which integration starts.
