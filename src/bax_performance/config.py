"""レポート設定（TOML）。相対パスは設定ファイルの場所を基準に解決する。

例:
    title = "EMAI_Transformer (raw)"
    output_dir = "../reports/raw"
    start = 201702        # 省略可（YYYYMM）
    end = 202609          # 省略可

    [factor_returns]      # 省略可。Barra 系の図に必要
    path = "../data/factor_returns.csv"
    unit = "percent"      # "percent" or "decimal"

    [[runs]]
    label = "ai"
    path = "../data/output/raw/ai"
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


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
    factor_returns: Path | None = None
    factor_returns_unit: str = "percent"
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
            factor_returns=resolve(fr["path"]) if fr.get("path") else None,
            factor_returns_unit=fr.get("unit", "percent"),
            style_top_n=data.get("style_top_n", 5),
        )
