"""図の作成（matplotlib）。各関数は Figure を返す。"""

from __future__ import annotations

import math
from contextlib import contextmanager

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from .attribution import BRINSON_EFFECTS, FACTOR_CLASS_ORDER
from .io import month_end

# 検証済みカテゴリカルパレット（固定順。ランの並び順に割り当てる）
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MAX_SERIES = len(SERIES_COLORS)
POSITIVE = "#2a78d6"
NEGATIVE = "#e34948"
EFFECT_COLORS = dict(zip(BRINSON_EFFECTS, SERIES_COLORS[:3]))

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

STYLE = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": BASELINE,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "legend.labelcolor": INK_2,
    "lines.linewidth": 1.5,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
}


@contextmanager
def style():
    with plt.rc_context(STYLE):
        yield


def run_colors(labels: list[str]) -> dict[str, str]:
    """ラベル → 色。並び順で固定割り当て（フィルタしても色は変えない）。"""
    if len(labels) > MAX_SERIES:
        raise ValueError(
            f"1 レポートで比較できるのは最大 {MAX_SERIES} 本です（{len(labels)} 本指定）。レポートを分けてください。"
        )
    return dict(zip(labels, SERIES_COLORS))


def _pct(ax, axis="y", decimals=1):
    fmt = PercentFormatter(1.0, decimals=decimals)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def _zero_line(ax, axis="y"):
    (ax.axhline if axis == "y" else ax.axvline)(0, color=INK_2, linewidth=0.8, zorder=1)


def _legend_right(ax, title=None):
    ax.legend(title=title, loc="center left", bbox_to_anchor=(1.01, 0.5), title_fontsize=8)


def _bar_labels(ax, bars, values, fmt="{:.2%}", extra=None, fontsize=7):
    """棒の端（外側）に値ラベルを置く。"""
    ymin, ymax = ax.get_ylim()
    pad = (ymax - ymin) * 0.015
    for i, (bar, v) in enumerate(zip(bars, values)):
        if pd.isna(v):
            continue
        text = fmt.format(v) + (f"\n{extra[i]}" if extra is not None else "")
        x = bar.get_x() + bar.get_width() / 2
        if v >= 0:
            ax.text(x, bar.get_height() + pad, text, ha="center", va="bottom", fontsize=fontsize, color=INK)
        else:
            ax.text(
                x,
                min(bar.get_y(), bar.get_y() + bar.get_height()) - pad,
                text,
                ha="center",
                va="top",
                fontsize=fontsize,
                color=INK,
            )


def _expand_ylim(ax, frac=0.18):
    lo, hi = ax.get_ylim()
    span = hi - lo
    ax.set_ylim(lo - span * frac * (lo < 0), hi + span * frac)


def _facet_grid(n: int, ncols: int, panel_w: float, panel_h: float, sharey=True):
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_w * ncols, panel_h * nrows), sharey=sharey, squeeze=False)
    flat = axes.ravel()
    for ax in flat[n:]:
        ax.set_visible(False)
    return fig, flat[:n]


# ---------------------------------------------------------------- performance summary


def plot_lines(ax, df: pd.DataFrame, colors: dict[str, str], ylabel: str, decimals=0, percent=True):
    x = month_end(df.index)
    for label in df.columns:
        ax.plot(x, df[label].to_numpy(), color=colors[label], label=label)
    ax.set_ylabel(ylabel)
    if percent:
        _pct(ax, decimals=decimals)
    _zero_line(ax)
    _legend_right(ax, "type")


def fig_cumulative_excess(cum: pd.DataFrame, colors: dict[str, str], ax=None):
    with style():
        fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(8, 4.2))
        plot_lines(ax, cum, colors, "Cumulative excess (net of cost)")
        ax.set_title("Excess", loc="left")
        return fig


def fig_alpha(alpha: pd.DataFrame, colors: dict[str, str], ax=None):
    with style():
        fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(8, 4.2))
        plot_lines(ax, alpha, colors, "Expected alpha of portfolio", percent=False)
        ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
        ax.set_title("Alpha", loc="left")
        return fig


def fig_excess_ir(summary: pd.DataFrame, colors: dict[str, str], axes=None):
    """年率超過リターン（上）と IR（下）。超過リターンの降順、横軸共通。"""
    s = summary.sort_values("Excess Return", ascending=False)
    with style():
        if axes is None:
            fig, axes = plt.subplots(2, 1, figsize=(8, 5.2), sharex=True, height_ratios=[3, 2])
        fig = axes[0].figure
        ax1, ax2 = axes
        c = [colors[t] for t in s.index]
        x = np.arange(len(s))
        b1 = ax1.bar(x, s["Excess Return"], color=c, width=0.7)
        _pct(ax1)
        _zero_line(ax1)
        _expand_ylim(ax1)
        _bar_labels(ax1, b1, s["Excess Return"])
        ax1.set_ylabel("Excess p.a.")
        ax1.set_title("Excess return p.a. and information ratio", loc="left")
        b2 = ax2.bar(x, s["Information Ratio"], color=c, width=0.7)
        _zero_line(ax2)
        _expand_ylim(ax2)
        _bar_labels(ax2, b2, s["Information Ratio"], fmt="{:.2f}")
        ax2.set_ylabel("IR")
        ax2.set_xticks(x, s.index)
        ax2.grid(axis="x", visible=False)
        ax1.grid(axis="x", visible=False)
        return fig


def fig_annual_excess(annual: pd.DataFrame, colors: dict[str, str], ax=None):
    with style():
        fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(8, 4.2))
        n = len(annual.columns)
        width = 0.8 / n
        x = np.arange(len(annual))
        for i, label in enumerate(annual.columns):
            ax.bar(x + (i - (n - 1) / 2) * width, annual[label], width=width * 0.9, color=colors[label], label=label)
        ax.set_xticks(x, annual.index.astype(str))
        ax.set_xlabel("Year")
        ax.set_ylabel("Annual excess")
        ax.grid(axis="x", visible=False)
        _pct(ax)
        _zero_line(ax)
        _legend_right(ax, "type")
        ax.set_title("Excess by calendar year", loc="left")
        return fig


def _draw_table(ax, summary: pd.DataFrame):
    ax.axis("off")
    cells = [
        [f"{row[c]:.2f}" if c == "Information Ratio" else f"{row[c]:.2%}" for c in summary.columns]
        for _, row in summary.iterrows()
    ]
    tbl = ax.table(
        cellText=cells,
        rowLabels=list(summary.index),
        colLabels=list(summary.columns),
        cellLoc="right",
        rowLoc="left",
        colLoc="right",
        loc="upper center",
        bbox=[0.06, 0, 0.94, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.5)
        cell.visible_edges = "B"
        cell.get_text().set_color(INK if r > 0 else INK_2)
        if r == 0:
            cell.get_text().set_fontweight("bold")
        if c == -1:
            cell.get_text().set_fontweight("bold")
    return tbl


def fig_summary_table(summary: pd.DataFrame):
    with style():
        fig, ax = plt.subplots(figsize=(11, 0.35 * (len(summary) + 1) + 0.3))
        _draw_table(ax, summary)
        return fig


def fig_performance_page(summary, cum, alpha, annual, colors, title="Performance summary"):
    """スライド 1 枚相当（表 + 4 つの図）。"""
    with style():
        fig = plt.figure(figsize=(16, 11))
        gs = fig.add_gridspec(
            3,
            2,
            height_ratios=[0.06 + 0.04 * len(summary), 1, 1],
            hspace=0.35,
            wspace=0.28,
            left=0.05,
            right=0.93,
            top=0.93,
            bottom=0.05,
        )
        fig.suptitle(title, x=0.05, ha="left", fontsize=15, fontweight="bold", color=INK)
        _draw_table(fig.add_subplot(gs[0, :]), summary)
        fig_cumulative_excess(cum, colors, ax=fig.add_subplot(gs[1, 0]))
        fig_alpha(alpha, colors, ax=fig.add_subplot(gs[1, 1]))
        sub = gs[2, 0].subgridspec(2, 1, height_ratios=[3, 2], hspace=0.08)
        a1 = fig.add_subplot(sub[0])
        a2 = fig.add_subplot(sub[1], sharex=a1)
        plt.setp(a1.get_xticklabels(), visible=False)
        fig_excess_ir(summary, colors, axes=(a1, a2))
        fig_annual_excess(annual, colors, ax=fig.add_subplot(gs[2, 1]))
        return fig


# ---------------------------------------------------------------- Barra


def fig_barra_classes(annualized: pd.DataFrame):
    """ラン別に Country/Industry/Style/Cur/specific の年率寄与（index=ラベル）。"""
    labels = list(annualized.index)
    with style():
        fig, axes = _facet_grid(len(labels), ncols=len(labels) if len(labels) <= 8 else 4, panel_w=2.4, panel_h=4.2)
        for ax, label in zip(axes, labels):
            v = annualized.loc[label, FACTOR_CLASS_ORDER]
            ax.bar(FACTOR_CLASS_ORDER, v, color=[POSITIVE if x >= 0 else NEGATIVE for x in v], width=0.7)
            ax.set_title(label, loc="center", fontsize=10)
            ax.tick_params(axis="x", rotation=90)
            ax.grid(axis="x", visible=False)
            _zero_line(ax)
        _expand_ylim(axes[0])
        for ax, label in zip(axes, labels):
            _bar_labels(ax, ax.patches, annualized.loc[label, FACTOR_CLASS_ORDER], fontsize=6.5)
        _pct(axes[0], decimals=2)
        axes[0].set_ylabel("Annualized contribution")
        fig.suptitle("Barra factor attribution (annualized)", x=0.02, ha="left", fontweight="bold")
        fig.tight_layout()
        return fig


def fig_cumulative_class(cum: pd.DataFrame, colors: dict[str, str], cls: str):
    with style():
        fig, ax = plt.subplots(figsize=(10, 5))
        plot_lines(ax, cum, colors, "Cumulative sum of contribution", decimals=1)
        ax.set_title(f"Cumulative {cls} contribution", loc="left")
        fig.tight_layout()
        return fig


def fig_style_top(summaries: dict[str, pd.DataFrame], n: int = 5):
    """ラン別に Style 寄与の下位 n・上位 n を表示（ラベル: 寄与 + OW/UW と ±）。"""
    labels = list(summaries)
    with style():
        fig, axes = _facet_grid(len(labels), ncols=2, panel_w=7.0, panel_h=2.9)
        lo = min(s["contribution"].min() for s in summaries.values())
        hi = max(s["contribution"].max() for s in summaries.values())
        span = hi - lo
        for ax, label in zip(axes, labels):
            s = summaries[label].sort_values("contribution")
            s = pd.concat([s.head(n), s.tail(n)]) if len(s) > 2 * n else s
            s = s[~s.index.duplicated()]
            v = s["contribution"]
            x = np.arange(len(s))
            bars = ax.bar(x, v, color=[POSITIVE if c >= 0 else NEGATIVE for c in v], width=0.7)
            ax.set_ylim(min(lo, 0) - span * 0.45, hi + span * 0.45)
            _bar_labels(ax, bars, v, extra=list(s["position"]), fontsize=6.5)
            ax.set_xticks(x, s.index, rotation=35, ha="right", fontsize=7)
            ax.grid(axis="x", visible=False)
            ax.set_title(label, loc="center", fontsize=10)
            _zero_line(ax)
            _pct(ax, decimals=2)
        fig.supylabel("Annualized contribution", fontsize=9, color=INK_2)
        fig.suptitle(
            "Risk index contribution: top and bottom (OW/UW = mean active exposure, +/- = mean factor return)",
            x=0.02,
            ha="left",
            fontweight="bold",
        )
        fig.tight_layout()
        return fig


# ---------------------------------------------------------------- Brinson


def fig_brinson(totals: dict[str, pd.DataFrame], order: list[str]):
    """ラン別・国別のブリンソン分解（期間合計）。棒 = 3 効果、点 = 合計。"""
    labels = list(totals)
    with style():
        fig, axes = _facet_grid(len(labels), ncols=2, panel_w=8.0, panel_h=2.6)
        x = np.arange(len(order))
        w = 0.27
        for ax, label in zip(axes, labels):
            t = totals[label].reindex(order).fillna(0.0)
            for i, eff in enumerate(BRINSON_EFFECTS):
                ax.bar(x + (i - 1) * w, t[eff], width=w * 0.9, color=EFFECT_COLORS[eff], label=eff, alpha=0.85)
            ax.scatter(x, t["total"], s=22, color=INK, edgecolor=SURFACE, linewidth=1.2, zorder=3, label="total")
            ax.set_xticks(x, order, rotation=90, fontsize=7)
            ax.grid(axis="x", visible=False)
            ax.set_title(label, loc="center", fontsize=10)
            _zero_line(ax)
            _pct(ax, decimals=0)
        handles, names = axes[0].get_legend_handles_labels()
        fig.legend(handles, names, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0))
        fig.supylabel("Brinson contribution (sum over period)", fontsize=9, color=INK_2)
        fig.suptitle("Brinson decomposition by country", x=0.02, ha="left", fontweight="bold")
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        return fig
