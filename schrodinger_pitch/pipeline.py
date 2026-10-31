"""End-to-end analysis. All numbers in the report are computed here, from a fixed seed."""
import json
from pathlib import Path

import numpy as np

from . import figures
from .audit import observability_audit
from .config import PROJECT_ROOT, ModelParams
from .data import load_match, load_passes
from .ghosts import (
    GhostModel,
    camera_likelihood_ratio,
    detection_gaps,
    fit_sigma_curve,
    masked_errors,
    sample_masked_gaps,
)
from .kinematics import estimate_velocities
from .uncertainty import frame_surface, surface_metrics, usable_frames
from .validation import pass_validation, position_validation, score_passes

N_MASKED_GAPS = 60000
PROGRESS_EVERY = 20


def run(match_id, frame, seed, n_draws, n_frames, data_dir=None, output_dir=None, log=print):
    output = Path(output_dir or PROJECT_ROOT / "results")
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    params = ModelParams()

    log(f"Loading match {match_id}")
    match = load_match(match_id, data_dir)
    vx, vy = estimate_velocities(match.x, match.y)
    gaps = detection_gaps(match)

    log("Observability audit")
    audit, heatmap, extent = observability_audit(match, vx, vy)
    figures.observability_figure(match, audit, heatmap, extent, output / "observability.png")

    log("Calibrating position uncertainty on period 1, testing on period 2")
    outfield = np.flatnonzero(~match.is_gk)
    goalkeepers = np.flatnonzero(match.is_gk)
    fit_errors, fit_tau = masked_errors(match, sample_masked_gaps(match, [1], N_MASKED_GAPS, rng, outfield))
    gk_errors, gk_tau = masked_errors(match, sample_masked_gaps(match, [1], N_MASKED_GAPS, rng, goalkeepers))
    model = GhostModel(
        fit_sigma_curve(fit_errors, fit_tau),
        fit_sigma_curve(gk_errors, gk_tau),
        camera_likelihood_ratio(match),
    )
    model.save(output / "ghost_model.json")
    positions = position_validation(
        match,
        sample_masked_gaps(match, [2], N_MASKED_GAPS, rng, outfield),
        model.outfield,
        float(np.sqrt(np.mean(fit_errors**2))),
    )
    keeper_positions = position_validation(
        match,
        sample_masked_gaps(match, [2], N_MASKED_GAPS, rng, goalkeepers),
        model.goalkeeper,
        float(np.sqrt(np.mean(gk_errors**2))),
    )
    figures.position_figure(model, positions, output / "position_calibration.png")

    log(f"Frame {frame}")
    surface = frame_surface(match, vx, vy, frame, model, gaps, params, max(n_draws, 200), rng)
    figures.frame_figure(match, surface, output / f"frame_{frame}.png")

    log(f"Sampling {n_frames} frames")
    sampled = []
    for i, f in enumerate(usable_frames(match, n_frames), start=1):
        sampled.append(
            surface_metrics(frame_surface(match, vx, vy, f, model, gaps, params, n_draws, rng))
        )
        if i % PROGRESS_EVERY == 0:
            log(f"  {i}/{n_frames} frames")

    log("Scoring passes")
    passes = load_passes(match, data_dir)
    scores = score_passes(match, passes, vx, vy, model, gaps, params, n_draws, rng)
    pass_results = pass_validation(passes, scores, rng)
    figures.calibration_figure(pass_results, output / "pass_calibration.png")

    metrics = {
        "match_id": match.match_id,
        "seed": seed,
        "pitch": [match.pitch_length, match.pitch_width],
        "observability": audit,
        "camera_likelihood_ratio": model.camera_ratio,
        "position_validation": positions,
        "goalkeeper_position_validation": keeper_positions,
        "focus_frame": surface_metrics(surface),
        "sampled_frames": _summarise(sampled),
        "pass_validation": pass_results,
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (output / "REPORT.md").write_text(_report(metrics, frame), encoding="utf-8")
    log(f"Results written to {output}")
    return metrics


def _summarise(rows):
    out = {"n_frames": len(rows)}
    for key in ("net_m2", "absolute_m2", "max_increase", "max_decrease", "share_changed", "max_sd"):
        values = np.array([row[key] for row in rows])
        out[key] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90)),
        }
    return out


def _report(m, frame):
    audit, pos, focus, many, passes = (
        m["observability"],
        m["position_validation"],
        m["focus_frame"],
        m["sampled_frames"],
        m["pass_validation"],
    )
    keeper = m["goalkeeper_position_validation"]
    primary = passes["primary_contrast_plugin_minus_expected"]
    overall = passes["all_passes_plugin_minus_expected"]
    low, high = primary["ci95"]
    if low > 0:
        verdict = "Expected control beats plug-in on these passes."
    elif high < 0:
        verdict = "Plug-in beats expected control on these passes."
    else:
        verdict = "Inconclusive: the interval includes zero, so no difference is established."

    lines = [
        f"# Results: match {m['match_id']}",
        "",
        f"Generated by `python -m schrodinger_pitch` with seed {m['seed']}. Single match only.",
        "",
        "## Observability",
        "",
        f"- Undetected, all player-frames: {audit['base_rate']['p_undetected']:.1%}",
        f"- Undetected at 4 m/s or more: {audit['at_least_4_mps']['p_undetected']:.1%}",
        f"- Undetected at sprint speed: {audit['sprint']['p_undetected']:.1%}",
        "",
        "Off-camera speeds come from interpolated tracks, so these rates are estimates of unknown bias.",
        "",
        "| Wide zone | Undetected at 4 m/s or more | Undetected, all speeds |",
        "|---|---|---|",
    ]
    for row in audit["wide"]:
        lines.append(
            f"| \\|y\\| > {row['abs_y_over']:g} m | {row['at_least_4_mps']['p_undetected']:.1%} "
            f"| {row['all_speeds']['p_undetected']:.1%} |"
        )
    lines += [
        "",
        "## Position model (fit on period 1, tested on period 2)",
        "",
        f"Outfield, held-out masked gaps: {pos['n']:,}. Overall coverage at nominal 50/80/95%: "
        + " / ".join(f"{pos['overall_coverage'][q]:.1%}" for q in ("0.5", "0.8", "0.95")),
        "",
        f"Goalkeepers, held-out masked gaps: {keeper['n']:,}. Overall coverage: "
        + " / ".join(f"{keeper['overall_coverage'][q]:.1%}" for q in ("0.5", "0.8", "0.95")),
        "",
        "Outfield by bridge time:",
        "",
        "| Bridge time (s) | n | RMS per axis (m) | 50% | 80% | 95% |",
        "|---|---|---|---|---|---|",
    ]
    for row in pos["bins"]:
        label = f"{row['tau_low']:g}–{row['tau_high']:g}" if row["tau_high"] else f"{row['tau_low']:g}+"
        cover = row["coverage"]
        lines.append(
            f"| {label} | {row['n']:,} | {row['sigma']:.2f} "
            f"| {cover['0.5']:.1%} | {cover['0.8']:.1%} | {cover['0.95']:.1%} |"
        )
    lines += [
        "",
        "| Forecast | Mean negative log-likelihood (lower is better) |",
        "|---|---|",
        f"| Bridge with sigma(tau) | {pos['nll']['bridge']:.3f} |",
        f"| Chord with one fixed sigma | {pos['nll']['chord_fixed_sigma']:.3f} |",
        f"| Original forward simulation | {pos['nll']['legacy_forward_simulation']:.3f} |",
        "",
        "## Effect on control",
        "",
        f"Frame {frame}: largest changes {focus['max_increase']:+.2f} and {focus['max_decrease']:+.2f}; "
        f"net {focus['net_m2']:+.0f} m²; absolute {focus['absolute_m2']:.0f} m²; "
        f"{focus['share_changed']:.1%} of cells move by more than 0.05; maximum SD {focus['max_sd']:.2f}.",
        "",
        f"Across {many['n_frames']} evenly spaced frames: median absolute change "
        f"{many['absolute_m2']['median']:.0f} m² (90th percentile {many['absolute_m2']['p90']:.0f} m²), "
        f"median net change {many['net_m2']['median']:+.0f} m², median share of cells moved "
        f"{many['share_changed']['median']:.1%}.",
        "",
        "## Pass completion",
        "",
        f"{passes['n_passes']} passes in {passes['n_possession_groups']} phases, "
        f"{passes['completion_rate']:.1%} completed. Ghost-relevant passes: {passes['n_ghost_relevant']}.",
        "",
        "| Forecast | Log loss, all | Brier, all | Log loss, ghost-relevant |",
        "|---|---|---|---|",
    ]
    for name, row in passes["all_passes"].items():
        ghost = passes["ghost_relevant_passes"][name]
        lines.append(f"| {name} | {row['log_loss']:.4f} | {row['brier']:.4f} | {ghost['log_loss']:.4f} |")
    xpass = passes["skillcorner_xpass_benchmark"]
    lines += [
        "",
        f"Primary contrast, plug-in minus expected log loss on ghost-relevant passes: "
        f"{primary['mean']:+.4f} (95% CI {low:+.4f} to {high:+.4f}, resampling phases). {verdict}",
        "",
        f"All passes: {overall['mean']:+.4f} (95% CI {overall['ci95'][0]:+.4f} to {overall['ci95'][1]:+.4f}).",
        "",
        f"Benchmark on {xpass['n']} passes with SkillCorner xPass: xPass log loss "
        f"{xpass['xpass_log_loss']:.4f}, calibrated expected control {xpass['expected_calibrated_log_loss']:.4f}.",
    ]
    return "\n".join(lines) + "\n"
