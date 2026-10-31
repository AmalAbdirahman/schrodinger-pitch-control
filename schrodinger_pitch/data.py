"""Read-only access to SkillCorner open data. Nothing is ever written to the data directory."""
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import default_data_dir

LFS_HEADER = b"version https://git-lfs"
CAMERA_CORNERS = ("top_left", "bottom_left", "bottom_right", "top_right")


class DataNotAvailable(FileNotFoundError):
    """The tracking file is missing or is an unresolved Git LFS pointer."""


@dataclass
class Match:
    """Tracking arrays are indexed [frame, player]; row i is frame i."""

    match_id: int
    pitch_length: float
    pitch_width: float
    home_team_id: int
    away_team_id: int
    home_sides: list  # home team's attacking side, one entry per period.
    player_ids: np.ndarray
    team_ids: np.ndarray
    is_gk: np.ndarray
    period: np.ndarray  # 0 where the frame has no period.
    x: np.ndarray
    y: np.ndarray
    detected: np.ndarray
    ball: np.ndarray  # [frame, (x, y)], NaN if unknown.
    ball_detected: np.ndarray
    possession_team: np.ndarray  # team id, 0 if unknown.
    camera: np.ndarray  # [frame, corner, (x, y)], NaN if unknown.

    @property
    def n_frames(self):
        return self.x.shape[0]

    @property
    def present(self):
        return ~np.isnan(self.x)

    def attack_sign(self, team_id, period):
        """+1 if the team attacks towards +x in this period, otherwise -1."""
        side = self.home_sides[int(period) - 1]
        home_sign = 1 if side == "left_to_right" else -1
        return home_sign if team_id == self.home_team_id else -home_sign


def _open_tracking(match_dir, match_id):
    jsonl = match_dir / f"{match_id}_tracking_extrapolated.jsonl"
    archive = match_dir / f"{match_id}_tracking_extrapolated.zip"
    if jsonl.exists():
        with open(jsonl, "rb") as handle:
            if handle.read(len(LFS_HEADER)) != LFS_HEADER:
                return open(jsonl, "r", encoding="utf-8")
    if archive.exists():
        with zipfile.ZipFile(archive) as bundle:
            raw = bundle.read(bundle.namelist()[0])
        return io.StringIO(raw.decode("utf-8"))
    raise DataNotAvailable(
        f"No tracking data for match {match_id} in {match_dir}. The file is missing or is a "
        "Git LFS pointer; run `git lfs pull` in the SkillCorner opendata repository and pass "
        "its data folder with --data-dir."
    )


def load_match(match_id, data_dir=None):
    match_dir = Path(data_dir or default_data_dir()) / "matches" / str(match_id)
    with open(match_dir / f"{match_id}_match.json", encoding="utf-8") as handle:
        meta = json.load(handle)
    with _open_tracking(match_dir, match_id) as handle:
        rows = [json.loads(line) for line in handle]

    frames = np.array([row["frame"] for row in rows])
    if not np.array_equal(frames, np.arange(len(frames))):
        raise ValueError("Expected contiguous frames starting at 0.")

    # Everyone appearing in the match metadata, in a stable order.
    players = sorted(meta["players"], key=lambda player: player["id"])
    index = {player["id"]: j for j, player in enumerate(players)}
    n, p = len(rows), len(players)
    x, y = np.full((n, p), np.nan), np.full((n, p), np.nan)
    detected = np.zeros((n, p), dtype=bool)
    ball = np.full((n, 2), np.nan)
    ball_detected = np.zeros(n, dtype=bool)
    camera = np.full((n, 4, 2), np.nan)
    period = np.zeros(n, dtype=int)
    possession = np.zeros(n, dtype=int)
    group_team = {"home team": meta["home_team"]["id"], "away team": meta["away_team"]["id"]}

    for i, row in enumerate(rows):
        period[i] = row["period"] or 0
        possession[i] = group_team.get(row["possession"]["group"], 0)
        ball_row = row["ball_data"]
        if ball_row["x"] is not None:
            ball[i] = ball_row["x"], ball_row["y"]
            ball_detected[i] = bool(ball_row["is_detected"])
        corners = row["image_corners_projection"]
        if corners["x_top_left"] is not None:
            camera[i] = [[corners[f"x_{c}"], corners[f"y_{c}"]] for c in CAMERA_CORNERS]
        for player in row["player_data"]:
            j = index[player["player_id"]]
            x[i, j], y[i, j] = player["x"], player["y"]
            detected[i, j] = player["is_detected"]

    return Match(
        match_id=int(match_id),
        pitch_length=float(meta["pitch_length"]),
        pitch_width=float(meta["pitch_width"]),
        home_team_id=meta["home_team"]["id"],
        away_team_id=meta["away_team"]["id"],
        home_sides=list(meta["home_team_side"]),
        player_ids=np.array([player["id"] for player in players]),
        team_ids=np.array([player["team_id"] for player in players]),
        is_gk=np.array([player["player_role"]["acronym"] == "GK" for player in players]),
        period=period,
        x=x,
        y=y,
        detected=detected,
        ball=ball,
        ball_detected=ball_detected,
        possession_team=possession,
        camera=camera,
    )


def load_passes(match, data_dir=None):
    """Attempted passes with their outcome, positioned from tracking at the release frame.

    Event-file coordinates are not used: they disagree with tracking by a flip of both axes
    on many rows. `high_pass` and `pass_distance` are not used either: they are blank for
    every unsuccessful pass, so filtering on them would leak the outcome.
    """
    match_dir = Path(data_dir or default_data_dir()) / "matches" / str(match.match_id)
    events = pd.read_csv(match_dir / f"{match.match_id}_dynamic_events.csv", low_memory=False)
    passes = events[
        (events["event_type"] == "player_possession")
        & (events["end_type"] == "pass")
        & events["pass_outcome"].isin(["successful", "unsuccessful"])
        & events["player_targeted_id"].notna()
    ]
    index = {player_id: j for j, player_id in enumerate(match.player_ids)}
    records = []
    for row in passes.itertuples():
        frame = int(row.frame_end)
        passer, target = index.get(row.player_id), index.get(int(row.player_targeted_id))
        if passer is None or target is None or frame >= match.n_frames:
            continue
        if np.isnan(match.x[frame, passer]) or np.isnan(match.x[frame, target]):
            continue
        records.append(
            {
                "frame": frame,
                "team_id": row.team_id,
                "passer": passer,
                "target": target,
                "completed": int(row.pass_outcome == "successful"),
                "group": row.phase_index,
                "xpass": row.player_targeted_xpass_completion,
            }
        )
    return pd.DataFrame(records)
