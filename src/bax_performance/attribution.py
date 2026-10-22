"""要因分解（ブリンソン型・Barra ファクター）。

ブリンソン（BHB、国別）:
  月 t の対象は me_t の銘柄。ウェイトは前月 O_{t-1}（保有・ユニバース銘柄）から取り、
  O に無い BM のみ銘柄は w_b = -prev_awgt で補う。ファンド・BM それぞれ合計 1 に正規化。
  allocation  = (w_p,c - w_b,c) (R_b,c - R_b)
  selection   = w_b,c (R_p,c - R_b,c)
  interaction = (w_p,c - w_b,c) (R_p,c - R_b,c)
  （ファンド非保有の国は R_p,c = R_b,c、BM 外の国は R_b,c = R_b とする）
  リターンは me の rtn（銘柄リターン、取引コスト控除前）。

Barra ファクター:
  寄与_k(t) = opt_active_k(t-1) × f_k(t)
  specific(t) = 取引コスト控除前の超過リターン（fund_ex_rtn + tr_cost） − Σ_k 寄与_k(t)
  ファクターリターン f は BAX 出力に含まれないため、外部ファイルで与える。
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from .io import BaxOutput, factor_class

BRINSON_EFFECTS = ["allocation", "selection", "interaction"]
FACTOR_CLASS_ORDER = ["Country", "Industry", "Style", "Cur", "specific"]


def country_from_code(code: str) -> str:
    """銘柄コード先頭 3 文字を国コードとする（例: BRAA261 → BRA）。"""
    return code[:3]


def _is_cash(code: str) -> bool:
    return code.endswith("CURR")


# ---------------------------------------------------------------- Brinson


def brinson_weights(out: BaxOutput, month: int, prev_month: int) -> pd.DataFrame:
    """月 month の銘柄別 w_p, w_b, r（正規化前）。"""
    me = out.active_returns(month)
    o = out.holdings(prev_month)
    df = me.join(o[["wgt_o", "wgt_b"]], how="left")
    df = df[[not _is_cash(c) for c in df.index]]
    in_o = df["wgt_o"].notna()
    w_p = df["wgt_o"].where(in_o, 0.0)
    w_b = df["wgt_b"].where(in_o, -df["prev_awgt"])
    return pd.DataFrame({"w_p": w_p, "w_b": w_b.clip(lower=0.0), "r": df["rtn"].fillna(0.0)})


def brinson_month(w: pd.DataFrame, country_of: Callable[[str], str] = country_from_code) -> pd.DataFrame:
    """1 か月分の国別ブリンソン分解。index=country、columns=allocation/selection/interaction。"""
    w = w.copy()
    w["w_p"] /= w["w_p"].sum()
    w["w_b"] /= w["w_b"].sum()
    w["country"] = [country_of(c) for c in w.index]
    g = w.assign(rp=w.w_p * w.r, rb=w.w_b * w.r).groupby("country")[["w_p", "w_b", "rp", "rb"]].sum()
    r_b = g["rb"].sum()
    r_bc = (g["rb"] / g["w_b"]).where(g["w_b"] > 0, r_b)
    r_pc = (g["rp"] / g["w_p"]).where(g["w_p"] > 0, r_bc)
    dw = g["w_p"] - g["w_b"]
    return pd.DataFrame(
        {
            "allocation": dw * (r_bc - r_b),
            "selection": g["w_b"] * (r_pc - r_bc),
            "interaction": dw * (r_pc - r_bc),
        }
    )


def brinson(
    out: BaxOutput,
    months: list[int],
    country_of: Callable[[str], str] = country_from_code,
) -> pd.DataFrame:
    """月次・国別のブリンソン分解（long 形式: month, country, 3 効果, total）。"""
    all_months = out.months
    frames = []
    for m in months:
        i = all_months.index(m)
        if i == 0:
            continue
        b = brinson_month(brinson_weights(out, m, all_months[i - 1]), country_of)
        frames.append(b.assign(month=m))
    df = pd.concat(frames).rename_axis("country").reset_index()
    df["total"] = df[BRINSON_EFFECTS].sum(axis=1)
    return df[["month", "country", *BRINSON_EFFECTS, "total"]]


# ---------------------------------------------------------------- Barra


# これ未満のアクティブエクスポージャーはリターン欠損を警告しない（寄与は無視できる大きさ）
MATERIAL_EXPOSURE = 1e-3

LONG_COLUMN_ALIASES = {"dateym": "month", "fac": "factor", "frtn": "return"}


def load_factor_returns(path: str | Path, unit: str = "decimal") -> pd.DataFrame:
    """ファクターリターンを読む。index=month(YYYYMM)、columns=ファクター名、値は小数。

    受け付ける CSV:
      - long 形式: 列 `month, factor, return`（`fctrtn_monthly.csv` の `dateym, fac, frtn` も可）
      - wide 形式: 列 `month` と各ファクター名（A ファイルの属性名、例 R13_MOMENTUM）
    ファクター名が空欄の行は除外して警告する（fctrtn_monthly.csv では世界ファクター C89_WORLD と思われる。
    アクティブエクスポージャーはほぼ 0 なので寄与への影響はない）。
    """
    raw = pd.read_csv(path, dtype={"fac": str, "factor": str}, keep_default_na=False, na_values=["", "NA", "NaN"])
    raw.columns = [c.strip() for c in raw.columns]
    raw = raw.rename(columns=LONG_COLUMN_ALIASES)
    if {"month", "factor", "return"} <= set(raw.columns):
        blank = raw["factor"].fillna("").str.strip() == ""
        if blank.any():
            warnings.warn(f"{path}: ファクター名が空欄の行を除外しました（{int(blank.sum())} 行）", stacklevel=2)
            raw = raw[~blank]
        df = raw.pivot_table(index="month", columns="factor", values="return", aggfunc="first")
    elif "month" in raw.columns:
        df = raw.set_index("month")
    else:
        raise ValueError(f"{path}: 'month' 列が必要です")
    df.index = df.index.astype(int)
    if unit == "percent":
        df = df / 100.0
    elif unit != "decimal":
        raise ValueError("unit は 'percent' か 'decimal' を指定してください")
    return df.sort_index()


def factor_attribution(out: BaxOutput, factor_returns: pd.DataFrame, months: list[int]) -> pd.DataFrame:
    """月次・ファクター別寄与。index=month、columns=ファクター名 + 'specific'。

    ファクターリターンが欠けている月は除外し警告する。
    """
    all_months = out.months
    pairs = [(m, all_months[all_months.index(m) - 1]) for m in months if all_months.index(m) > 0]
    missing_months = [m for m, _ in pairs if m not in factor_returns.index]
    if missing_months:
        warnings.warn(f"{out.label}: ファクターリターンが無い月を除外します: {missing_months}", stacklevel=2)
    pairs = [(m, p) for m, p in pairs if m in factor_returns.index]
    if not pairs:
        raise ValueError(f"{out.label}: ファクターリターンと重なる月がありません")

    exp = out.active_exposure_panel([p for _, p in pairs])
    exp.index = [m for m, _ in pairs]
    missing = sorted(set(exp.columns) - set(factor_returns.columns))
    nonzero_missing = [f for f in missing if exp[f].abs().max() > MATERIAL_EXPOSURE]
    if nonzero_missing:
        warnings.warn(
            f"{out.label}: ファクターリターンが無いファクター（寄与 0 扱い）: {nonzero_missing}", stacklevel=2
        )
    fr = factor_returns.reindex(index=exp.index, columns=exp.columns)
    gaps = fr.isna() & (exp.abs() > MATERIAL_EXPOSURE)
    gaps = gaps.drop(columns=nonzero_missing)
    if gaps.values.any():
        cols = list(gaps.columns[gaps.any()])
        warnings.warn(
            f"{out.label}: エクスポージャーがあるのにファクターリターンが欠けている月があります（寄与 0 扱い）: {cols}",
            stacklevel=2,
        )
    contrib = exp.fillna(0.0) * fr.fillna(0.0)

    gross_active = (out.prfm["fund_ex_rtn"] + out.prfm["tr_cost"]).reindex(exp.index)
    contrib["specific"] = gross_active - contrib.sum(axis=1)
    contrib.index.name = "month"
    return contrib


def by_class(contrib: pd.DataFrame) -> pd.DataFrame:
    """ファクター別寄与を分類（Country/Industry/Style/Cur/specific）ごとに合算。"""
    classes = {c: (factor_class(c) or c) for c in contrib.columns}
    grouped = contrib.T.groupby(classes).sum().T
    return grouped.reindex(columns=FACTOR_CLASS_ORDER, fill_value=0.0)


def style_summary(out: BaxOutput, contrib: pd.DataFrame, factor_returns: pd.DataFrame) -> pd.DataFrame:
    """リスクインデックス（Style）ごとの年率寄与・平均アクティブエクスポージャー・平均ファクターリターン。

    `position` は OW/UW（平均アクティブエクスポージャーの符号）と +/-（平均ファクターリターンの符号）。
    """
    styles = [c for c in contrib.columns if factor_class(c) == "Style"]
    all_months = out.months
    prev = [all_months[all_months.index(m) - 1] for m in contrib.index]
    exp = out.active_exposure_panel(prev).reindex(columns=styles)
    exp.index = contrib.index
    fr = factor_returns.reindex(index=contrib.index, columns=styles)
    df = pd.DataFrame(
        {
            "contribution": contrib[styles].mean() * 12,
            "mean_active_exposure": exp.mean(),
            "mean_factor_return": fr.mean(),
        }
    )
    df["position"] = [
        f"{'OW' if e >= 0 else 'UW'} {'+' if f >= 0 else '-'}"
        for e, f in zip(df["mean_active_exposure"], df["mean_factor_return"].fillna(0.0))
    ]
    df.index.name = "factor"
    return df
