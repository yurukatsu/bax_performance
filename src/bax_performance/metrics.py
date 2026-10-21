"""パフォーマンス指標（サマリー表・累積超過・年次超過）。

定義（月次・小数ベース）:
- Total Return     : ファンドリターンの年率幾何平均
- Total Risk       : ファンドリターンの標準偏差 × √12
- Excess Return    : 超過リターン（取引コスト控除後）の算術平均 × 12
- Tracking Error   : 超過リターンの標準偏差 × √12
- Information Ratio: Excess Return / Tracking Error
- Max Drawdown     : ファンド NAV（期初 = 1）の最大下落率
- Turnover         : 月次回転率（片道）の平均 × 12
- Cost             : 月次取引コストの平均 × 12
- Per_Trade        : Cost / (2 × Turnover)  … 片道 1 回あたりのコスト
累積超過・年次超過は月次超過リターンの単純和（算術）。
リターンが NA の月（初回の組成月）は集計から除く。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .io import BaxOutput

SUMMARY_COLUMNS = [
    "Total Return",
    "Total Risk",
    "Excess Return",
    "Tracking Error",
    "Information Ratio",
    "Max Drawdown",
    "Turnover",
    "Cost",
    "Per_Trade",
]


def select_months(months, start: int | None = None, end: int | None = None) -> list[int]:
    return [m for m in months if (start is None or m >= start) and (end is None or m <= end)]


def monthly_series(out: BaxOutput, start: int | None = None, end: int | None = None) -> pd.DataFrame:
    """評価対象月の月次系列（bm, fund, excess, cost, turnover）。"""
    p = out.prfm.dropna(subset=["fund_rtn"])
    p = p.loc[select_months(p.index, start, end)]
    return pd.DataFrame(
        {
            "bm": p["bm_rtn"],
            "fund": p["fund_rtn"],
            "excess": p["fund_ex_rtn"],
            "cost": p["tr_cost"],
            "turnover": p["rot"],
        }
    )


def max_drawdown(returns: pd.Series) -> float:
    nav = np.concatenate([[1.0], (1 + returns).cumprod().to_numpy()])
    return float((1 - nav / np.maximum.accumulate(nav)).max())


def summary_stats(m: pd.DataFrame) -> dict[str, float]:
    n = len(m)
    if n < 2:
        raise ValueError("指標の計算には 2 か月以上のデータが必要です")
    total = (1 + m["fund"]).prod() ** (12 / n) - 1
    excess = m["excess"].mean() * 12
    te = m["excess"].std() * np.sqrt(12)
    turnover = m["turnover"].mean() * 12
    cost = m["cost"].mean() * 12
    return {
        "Total Return": total,
        "Total Risk": m["fund"].std() * np.sqrt(12),
        "Excess Return": excess,
        "Tracking Error": te,
        "Information Ratio": excess / te if te > 0 else np.nan,
        "Max Drawdown": max_drawdown(m["fund"]),
        "Turnover": turnover,
        "Cost": cost,
        "Per_Trade": cost / (2 * turnover) if turnover > 0 else np.nan,
    }


def summary_table(series: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """ラベル → monthly_series の辞書からサマリー表を作る（行順は辞書の順）。"""
    table = pd.DataFrame({label: summary_stats(m) for label, m in series.items()}).T
    table.index.name = "Type"
    return table[SUMMARY_COLUMNS]


def cumulative_excess(series: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """月次超過リターンの累積和。index=month、columns=ラベル。"""
    return pd.DataFrame({label: m["excess"].cumsum() for label, m in series.items()})


def annual_excess(series: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """暦年ごとの超過リターン（月次の単純和）。index=year、columns=ラベル。"""
    out = {}
    for label, m in series.items():
        years = pd.Index(m.index) // 100
        out[label] = m["excess"].groupby(years).sum()
    df = pd.DataFrame(out)
    df.index.name = "year"
    return df


def expected_alpha(outputs: list[BaxOutput], start: int | None = None, end: int | None = None) -> pd.DataFrame:
    """st_objv の alpha（最適化ポートフォリオの期待アルファ）。index=month、columns=ラベル。"""
    df = pd.DataFrame({o.label: o.objv["alpha"] for o in outputs})
    return df.loc[select_months(df.index, start, end)]
