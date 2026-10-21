"""BAX（ポートフォリオ最適化ツール）出力の集計と可視化。"""

from .config import ReportConfig, RunSpec
from .io import BaxOutput
from .report import ReportData, build_report, compute

__all__ = ["BaxOutput", "ReportConfig", "ReportData", "RunSpec", "build_report", "compute", "main"]


def main() -> None:
    from .cli import main as _main

    raise SystemExit(_main())
