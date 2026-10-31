"""Static figures. Every function writes one PNG and draws no window."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from mplsoccer import Pitch

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
SEQUENTIAL = LinearSegmentedColormap.from_list("control", ["#f3f6fb", "#9cc3ee", BLUE, "#123f7a"])
DIVERGING = LinearSegmentedColormap.from_list("delta", [ORANGE, "#efeeea", BLUE])

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "axes.labelcolor": INK_SECONDARY,
        "axes.edgecolor": GRID,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "font.size": 11,
    }
)


def _pitch(match, ax):
    pitch = Pitch(
        pitch_type="skillcorner",
        pitch_length=match.pitch_length,
        pitch_width=match.pitch_width,
        pitch_color=SURFACE,
        line_color=INK_SECONDARY,
        linewidth=0.8,
        line_zorder=3,
    )
    pitch.draw(ax=ax)
    return (-match.pitch_length / 2, match.pitch_length / 2, -match.pitch_width / 2, match.pitch_width / 2)


def _players(match, surface, ax):
    x, y = match.x[surface.frame, surface.players], match.y[surface.frame, surface.players]
    seen = match.detected[surface.frame, surface.players]
    for team, colour in ((surface.attacking, BLUE), (~surface.attacking, ORANGE)):
        ax.scatter(x[team & seen], y[team & seen], s=34, c=colour, edgecolors=SURFACE, linewidths=1, zorder=4)
        ax.scatter(x[team & ~seen], y[team & ~seen], s=34, facecolors=SURFACE, edgecolors=colour, linewidths=1.6, zorder=4)
    ax.scatter(*surface.ball, s=26, c=INK, marker="D", edgecolors=SURFACE, linewidths=1, zorder=5)


def _colorbar(fig, ax, image, label):
    bar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    bar.set_label(label, color=INK_SECONDARY)
    bar.outline.set_visible(False)


def observability_figure(match, summary, heatmap, extent, path):
    fig, (left, right) = plt.subplots(1, 2, figsize=(14, 5.4), width_ratios=(1, 1.7))
    labels = [
        f"{band['low']:g}–{band['high']:g}" if band["high"] < 12 else f"{band['low']:g}+"
        for band in summary["bands"]
    ]
    values = [100 * band["p_undetected"] for band in summary["bands"]]
    bars = left.bar(labels, values, color=BLUE, width=0.56, zorder=3)
    left.bar_label(bars, fmt="%.0f%%", padding=3, color=INK)
    base = 100 * summary["base_rate"]["p_undetected"]
    left.axhline(base, color=INK_SECONDARY, lw=1.5, ls="--", zorder=2)
    left.text(3.45, base + 1.2, f"all frames {base:.0f}%", color=INK_SECONDARY, ha="right")
    left.set(xlabel="Player speed (m/s)", ylabel="Share of frames undetected (%)", ylim=(0, 100))
    left.grid(axis="x", visible=False)
    left.set_title("Undetected share by player speed", loc="left", fontweight="bold")

    bounds = _pitch(match, right)
    image = right.imshow(
        heatmap.T, extent=extent, origin="lower", cmap=SEQUENTIAL, vmin=0, vmax=0.5, aspect="equal", zorder=1
    )
    right.set(xlim=bounds[:2], ylim=bounds[2:])
    _colorbar(fig, right, image, "P(undetected | speed ≥ 4 m/s)")
    right.set_title("Where running players go unseen", loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def frame_figure(match, surface, path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.4))
    shape = surface.shape
    panels = (
        (surface.plugin, "Plug-in control", SEQUENTIAL, 0, 1, "P(attacking control)"),
        (surface.mean, "Expected control over ghost positions", SEQUENTIAL, 0, 1, "P(attacking control)"),
        (surface.mean - surface.plugin, "Expected minus plug-in", DIVERGING, -0.5, 0.5, "Change in probability"),
        (surface.sd, "Spread across ghost positions", SEQUENTIAL, 0, 0.5, "SD of control"),
    )
    for ax, (values, title, cmap, low, high, label) in zip(axes.ravel(), panels):
        bounds = _pitch(match, ax)
        image = ax.imshow(
            values.reshape(shape), extent=bounds, origin="lower", cmap=cmap, vmin=low, vmax=high, zorder=1
        )
        ax.set(xlim=bounds[:2], ylim=bounds[2:])
        _players(match, surface, ax)
        _colorbar(fig, ax, image, label)
        ax.set_title(title, loc="left", fontweight="bold")
    handles = [
        Line2D([], [], marker="o", ls="", color=BLUE, label="In possession"),
        Line2D([], [], marker="o", ls="", color=ORANGE, label="Out of possession"),
        Line2D([], [], marker="o", ls="", markerfacecolor=SURFACE, color=INK_SECONDARY, label="Undetected (hollow)"),
        Line2D([], [], marker="D", ls="", color=INK, label="Ball"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    fig.suptitle(f"Match {match.match_id}, frame {surface.frame}", x=0.02, ha="left", fontweight="bold", fontsize=15)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def position_figure(model, validation, path):
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4.8))
    left.plot(model.outfield.taus, model.outfield.sigmas, color=BLUE, lw=2, label="Outfield")
    left.plot(model.goalkeeper.taus, model.goalkeeper.sigmas, color=ORANGE, lw=2, label="Goalkeeper")
    left.legend(frameon=False)
    left.set(xlabel="Bridge time s·u/(s+u) (seconds)", ylabel="Per-axis position SD (m)")
    left.set_title("Position uncertainty by time unseen", loc="left", fontweight="bold")

    bins = validation["bins"]
    labels = [f"{b['tau_low']:g}–{b['tau_high']:g}" if b["tau_high"] else f"{b['tau_low']:g}+" for b in bins]
    positions = np.arange(len(bins))
    for level, colour in (("0.5", BLUE), ("0.8", ORANGE), ("0.95", AQUA)):
        values = [b["coverage"][level] for b in bins]
        right.plot(positions, values, color=colour, lw=2, marker="o", ms=8, label=f"Nominal {float(level):.0%}")
        right.axhline(float(level), color=colour, lw=1, ls=":")
    right.set_xticks(positions, labels)
    right.set(xlabel="Bridge time (seconds)", ylabel="Observed coverage", ylim=(0, 1.02))
    right.legend(frameon=False, loc="lower left")
    right.set_title("Held-out coverage on masked gaps (outfield)", loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def calibration_figure(validation, path):
    fig, ax = plt.subplots(figsize=(6, 5.6))
    bins = validation["calibration"]
    forecast = [b["forecast"] for b in bins]
    observed = [b["observed"] for b in bins]
    ax.plot([0, 1], [0, 1], color=INK_SECONDARY, lw=1, ls="--")
    ax.plot(forecast, observed, color=BLUE, lw=2, zorder=3)
    ax.scatter(forecast, observed, s=[max(30, b["n"]) for b in bins], c=BLUE, edgecolors=SURFACE, linewidths=2, zorder=4, clip_on=False)
    ax.set(xlabel="Expected control at the target", ylabel="Observed completion rate", xlim=(0, 1), ylim=(0, 1.02))
    ax.set_title("Raw model vs pass outcomes (dot size = passes)", loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
