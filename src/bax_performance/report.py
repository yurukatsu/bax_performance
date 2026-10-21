"""レポート生成：集計表（CSV）・図（PNG）・まとめ PDF を出力する。"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from . import attribution as attr
from . import metrics, plots
from .config import ReportConfig
from .io import BaxOutput

log = logging.getLogger(__name__)


@dataclass
class ReportData:
    """図の元になる集計結果。"""

    outputs: list[BaxOutput]
    summary: pd.DataFrame
    cumulative_excess: pd.DataFrame
    annual_excess: pd.DataFrame
    alpha: pd.DataFrame
    brinson: pd.DataFrame  # long: label, month, country, effects, total
    factor: dict[str, pd.DataFrame] = field(default_factory=dict)  # label → 月次ファクター別寄与
    factor_class: dict[str, pd.DataFrame] = field(default_factory=dict)  # label → 月次分類別寄与
    style: dict[str, pd.DataFrame] = field(default_factory=dict)  # label → Style サマリー

    @property
    def labels(self) -> list[str]:
        return [o.label for o in self.outputs]

    @property
    def has_factor(self) -> bool:
        return bool(self.factor)

    def brinson_totals(self) -> dict[str, pd.DataFrame]:
        cols = [*attr.BRINSON_EFFECTS, "total"]
        g = self.brinson.groupby(["label", "country"])[cols].sum()
        return {label: g.loc[label] for label in self.labels}

    def brinson_country_order(self) -> list[str]:
        return list(self.brinson.groupby("country")["total"].sum().sort_values().index)

    def barra_annualized(self) -> pd.DataFrame:
        return pd.DataFrame({k: v.mean() * 12 for k, v in self.factor_class.items()}).T

    def cumulative_class(self, cls: str) -> pd.DataFrame:
        return pd.DataFrame({k: v[cls].cumsum() for k, v in self.factor_class.items()})


def compute(config: ReportConfig) -> ReportData:
    outputs = [BaxOutput.from_path(r.path, r.label) for r in config.runs]
    plots.run_colors([o.label for o in outputs])  # 本数チェック

    series = {o.label: metrics.monthly_series(o, config.start, config.end) for o in outputs}
    brinson = []
    for o in outputs:
        log.info("Brinson: %s", o.label)
        brinson.append(attr.brinson(o, list(series[o.label].index)).assign(label=o.label))

    data = ReportData(
        outputs=outputs,
        summary=metrics.summary_table(series),
        cumulative_excess=metrics.cumulative_excess(series),
        annual_excess=metrics.annual_excess(series),
        alpha=metrics.expected_alpha(outputs, config.start, config.end),
        brinson=pd.concat(brinson, ignore_index=True),
    )

    if config.factor_returns is not None:
        fr = attr.load_factor_returns(config.factor_returns, config.factor_returns_unit)
        for o in outputs:
            log.info("Factor attribution: %s", o.label)
            contrib = attr.factor_attribution(o, fr, list(series[o.label].index))
            data.factor[o.label] = contrib
            data.factor_class[o.label] = attr.by_class(contrib)
            data.style[o.label] = attr.style_summary(o, contrib, fr)
    return data


def write_tables(data: ReportData, out_dir: Path) -> None:
    d = out_dir / "tables"
    d.mkdir(parents=True, exist_ok=True)
    data.summary.to_csv(d / "summary.csv")
    data.cumulative_excess.to_csv(d / "cumulative_excess.csv")
    data.annual_excess.to_csv(d / "annual_excess.csv")
    data.alpha.to_csv(d / "expected_alpha.csv")
    data.brinson.to_csv(d / "brinson_monthly.csv", index=False)
    pd.concat(data.brinson_totals(), names=["label"]).to_csv(d / "brinson_total.csv")
    if data.has_factor:
        pd.concat(data.factor, names=["label"]).to_csv(d / "factor_contribution_monthly.csv")
        pd.concat(data.factor_class, names=["label"]).to_csv(d / "factor_class_monthly.csv")
        data.barra_annualized().to_csv(d / "factor_class_annualized.csv")
        pd.concat(data.style, names=["label"]).to_csv(d / "style_summary.csv")


def figures(data: ReportData, config: ReportConfig) -> dict[str, plt.Figure]:
    """ファイル名（拡張子なし）→ Figure。並びは PDF のページ順。"""
    colors = plots.run_colors(data.labels)
    figs: dict[str, plt.Figure] = {
        "00_performance_page": plots.fig_performance_page(
            data.summary,
            data.cumulative_excess,
            data.alpha,
            data.annual_excess,
            colors,
            title=f"{config.title}: performance summary",
        ),
        "01_summary_table": plots.fig_summary_table(data.summary),
        "02_cumulative_excess": plots.fig_cumulative_excess(data.cumulative_excess, colors),
        "03_expected_alpha": plots.fig_alpha(data.alpha, colors),
        "04_excess_ir": plots.fig_excess_ir(data.summary, colors),
        "05_annual_excess": plots.fig_annual_excess(data.annual_excess, colors),
    }
    if data.has_factor:
        figs["06_barra_classes"] = plots.fig_barra_classes(data.barra_annualized())
        figs["07_cumulative_style"] = plots.fig_cumulative_class(data.cumulative_class("Style"), colors, "Style")
        figs["08_cumulative_specific"] = plots.fig_cumulative_class(
            data.cumulative_class("specific"), colors, "specific"
        )
        figs["09_risk_index_top"] = plots.fig_style_top(data.style, n=config.style_top_n)
    figs["10_brinson_country"] = plots.fig_brinson(data.brinson_totals(), data.brinson_country_order())
    return figs


PDF_PAGES = [
    "00_performance_page",
    "06_barra_classes",
    "07_cumulative_style",
    "08_cumulative_specific",
    "09_risk_index_top",
    "10_brinson_country",
]


def build_report(config: ReportConfig) -> ReportData:
    out_dir = config.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    data = compute(config)
    write_tables(data, out_dir)

    figs = figures(data, config)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    with PdfPages(out_dir / "report.pdf") as pdf:
        for name in PDF_PAGES:
            if name in figs:
                pdf.savefig(figs[name])
    for name, fig in figs.items():
        fig.savefig(fig_dir / f"{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    if not data.has_factor:
        log.warning(
            "ファクターリターンが未指定のため Barra 系の図（要因分解・Style/Specific 累積・リスクインデックス寄与）は省略しました"
        )
    return data
