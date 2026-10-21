"""BAX 出力ディレクトリの読み込み。

単位の規約（docs/bax_output.md の alphaeval.bax に合わせる）:
ウェイト・リターン・回転率・コスト・omega はファイル上 % だが、ここで小数に変換して返す。
アルファスコアと A ファイルのエクスポージャーは変換しない。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cached_property, lru_cache
from pathlib import Path

import pandas as pd

PCT = 100.0

# A ファイルの属性名の接頭辞 → ファクター分類
FACTOR_CLASSES = {"C": "Country", "I": "Industry", "R": "Style", "c": "Cur"}
_FACTOR_RE = re.compile(r"^([CIRc])\d+_")


def factor_class(attribute: str) -> str | None:
    """A ファイルの属性名からファクター分類を返す（ファクターでなければ None）。"""
    m = _FACTOR_RE.match(attribute)
    return FACTOR_CLASSES[m.group(1)] if m else None


def read_table(path: str | Path) -> pd.DataFrame:
    """空白区切り・先頭行が `#col ...` のヘッダ付きファイルを読む。"""
    df = pd.read_csv(path, sep=r"\s+", na_values=["NA"], comment=None)
    df.columns = [c.lstrip("#") for c in df.columns]
    return df


def read_params(path: str | Path) -> dict[str, str]:
    """`bc.txt` の `key = value` を辞書で返す。"""
    params: dict[str, str] = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        params[key.strip()] = value.strip()
    return params


def month_end(months) -> pd.DatetimeIndex:
    """YYYYMM（int）を月末日付に変換する。"""
    return pd.to_datetime(pd.Index(months).astype(str), format="%Y%m") + pd.offsets.MonthEnd(0)


@dataclass(frozen=True)
class BaxOutput:
    """1 回分の BAX 出力ディレクトリ（例: data/output/raw/v7）。"""

    path: Path
    label: str

    @classmethod
    def from_path(cls, path: str | Path, label: str | None = None) -> BaxOutput:
        p = Path(path)
        if not (p / "st_prfm.dat").exists():
            raise FileNotFoundError(f"st_prfm.dat が見つかりません: {p}")
        return cls(path=p, label=label or p.name)

    # ---- 集計ファイル ----

    @cached_property
    def prfm(self) -> pd.DataFrame:
        """st_prfm.dat。index=month(YYYYMM)、リターン・コスト・回転率は小数。"""
        df = read_table(self.path / "st_prfm.dat").set_index("month")
        cols = ["bm_rtn", "fund_rtn", "fund_ex_rtn", "tr_cost", "rot", "w_cash"]
        df[cols] = df[cols] / PCT
        return df

    @cached_property
    def objv(self) -> pd.DataFrame:
        """st_objv.dat。index=month、omega と取引コストは小数、alpha はそのまま。"""
        df = read_table(self.path / "st_objv.dat").rename(columns={"day": "month"}).set_index("month")
        df[["omega", "tr_cost", "h_tr_cost"]] = df[["omega", "tr_cost", "h_tr_cost"]] / PCT
        return df

    @cached_property
    def params(self) -> dict[str, str]:
        return read_params(self.path / "bc.txt")

    @property
    def months(self) -> list[int]:
        return list(self.prfm.index)

    # ---- 月次ファイル ----

    def holdings(self, month: int) -> pd.DataFrame:
        """O{month}.dat。index=code、ウェイト（wgt_o/wgt_b/wgt_i/上下限）は小数。"""
        return _read_holdings(self.path / f"O{month}.dat")

    def active_returns(self, month: int) -> pd.DataFrame:
        """me_{month}.dat。前月末アクティブウェイトと当月リターン（いずれも小数）。"""
        return _read_me(self.path / f"me_{month}.dat")

    def exposures(self, month: int) -> pd.DataFrame:
        """A{month}.dat。index=attribute、値は変換なし。"""
        return _read_exposures(self.path / f"A{month}.dat")

    def active_exposure_panel(self, months: list[int] | None = None) -> pd.DataFrame:
        """最適化後アクティブエクスポージャー（opt_active）。index=month、columns=ファクター。"""
        months = self.months if months is None else months
        rows = {}
        for m in months:
            a = self.exposures(m)
            rows[m] = a.loc[[i for i in a.index if factor_class(i)], "opt_active"]
        return pd.DataFrame(rows).T.sort_index()


@lru_cache(maxsize=4096)
def _read_holdings(path: Path) -> pd.DataFrame:
    df = read_table(path).set_index("code")
    wcols = [c for c in df.columns if c != "alpha"]
    df[wcols] = df[wcols] / PCT
    return df


@lru_cache(maxsize=4096)
def _read_me(path: Path) -> pd.DataFrame:
    df = read_table(path).set_index("code")
    return df / PCT


@lru_cache(maxsize=4096)
def _read_exposures(path: Path) -> pd.DataFrame:
    return read_table(path).set_index("attribute")
