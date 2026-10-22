"""レポート設定（TOML）。相対パスは設定ファイルの場所を基準に解決する。

例:
    title = "EMAI_Transformer (raw)"
    output_dir = "../reports/raw"
    start = 201702        # 省略可（YYYYMM）
    end = 202609          # 省略可
    exclude = [202204, "202001-202003"]   # 省略可。集計から除く月（YYYYMM か "YYYYMM-YYYYMM"）

    [factor_returns]      # 省略可。Barra 系の図に必要
    path = "../data/fctrtn_monthly.csv"
    unit = "decimal"      # "decimal"（既定）or "percent"

    [[runs]]
    label = "ai"
    path = "../data/output/raw/ai"
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

_MONTH_RE = re.compile(r"^\d{6}$")


def _to_month(value: int | str) -> int:
    text = str(value).strip()
    if not _MONTH_RE.match(text) or not 1 <= int(text[4:]) <= 12:
        raise ValueError(f"月は YYYYMM で指定してください: {value!r}")
    return int(text)


def _next_month(m: int) -> int:
    y, mm = divmod(m, 100)
    return (y + 1) * 100 + 1 if mm == 12 else m + 1


def parse_months(values: Iterable[int | str]) -> list[int]:
    """`[202204, "202201-202203"]` のような指定を月（YYYYMM）のリストに展開する。範囲は両端を含む。"""
    months: set[int] = set()
    for v in values:
        text = str(v).strip()
        if "-" in text:
            lo, hi = (_to_month(x) for x in text.split("-", 1))
            if lo > hi:
                raise ValueError(f"範囲の開始が終了より後です: {v!r}")
            m = lo
            while m <= hi:
                months.add(m)
                m = _next_month(m)
        else:
            months.add(_to_month(text))
    return sorted(months)


def month_spans(months: Iterable[int]) -> list[tuple[int, int]]:
    """月のリストを連続区間 [(開始, 終了), ...] にまとめる。"""
    spans: list[tuple[int, int]] = []
    for m in sorted(set(months)):
        if spans and _next_month(spans[-1][1]) == m:
            spans[-1] = (spans[-1][0], m)
        else:
            spans.append((m, m))
    return spans


def format_months(months: Iterable[int]) -> str:
    """月のリストを連続区間にまとめて表示用文字列にする（例: "2020-01–2020-03, 2022-04"）。"""
    spans = month_spans(months)

    def fmt(m: int) -> str:
        return f"{m // 100}-{m % 100:02d}"

    return ", ".join(fmt(a) if a == b else f"{fmt(a)}–{fmt(b)}" for a, b in spans)


@dataclass
class RunSpec:
    label: str
    path: Path


@dataclass
class ReportConfig:
    runs: list[RunSpec]
    output_dir: Path
    title: str = "BAX performance"
    start: int | None = None
    end: int | None = None
    exclude: list[int] = field(default_factory=list)
    factor_returns: Path | None = None
    factor_returns_unit: str = "decimal"
    style_top_n: int = 5

    @classmethod
    def from_toml(cls, path: str | Path) -> ReportConfig:
        path = Path(path)
        base = path.parent
        data = tomllib.loads(path.read_text())

        def resolve(p: str) -> Path:
            q = Path(p).expanduser()
            return q if q.is_absolute() else (base / q).resolve()

        runs = [RunSpec(label=r.get("label") or Path(r["path"]).name, path=resolve(r["path"])) for r in data["runs"]]
        fr = data.get("factor_returns") or {}
        return cls(
            runs=runs,
            output_dir=resolve(data.get("output_dir", "reports")),
            title=data.get("title", "BAX performance"),
            start=data.get("start"),
            end=data.get("end"),
            exclude=parse_months(data.get("exclude", [])),
            factor_returns=resolve(fr["path"]) if fr.get("path") else None,
            factor_returns_unit=fr.get("unit", "decimal"),
            style_top_n=data.get("style_top_n", 5),
        )
