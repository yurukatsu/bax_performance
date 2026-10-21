"""コマンドライン: `bax-performance report ...`"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import ReportConfig, RunSpec
from .report import build_report


def _parse_run(text: str) -> RunSpec:
    label, sep, path = text.partition("=")
    if not sep:
        return RunSpec(label=Path(text).name, path=Path(text))
    return RunSpec(label=label, path=Path(path))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bax-performance", description="BAX 出力のパフォーマンス集計と可視化")
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser(
        "report",
        help="集計表・図・PDF を出力する",
        description="設定ファイル（TOML）か --run でランを指定する。両方あれば --run 以外のオプションで設定を上書き。",
    )
    r.add_argument("config", nargs="?", type=Path, help="設定ファイル（TOML）")
    r.add_argument("--run", action="append", default=[], metavar="LABEL=PATH", help="比較するラン（複数可）")
    r.add_argument("-o", "--output-dir", type=Path, help="出力先ディレクトリ")
    r.add_argument("--title", help="レポートのタイトル")
    r.add_argument("--start", type=int, help="集計開始月（YYYYMM）")
    r.add_argument("--end", type=int, help="集計終了月（YYYYMM）")
    r.add_argument("--factor-returns", type=Path, help="ファクターリターン CSV（Barra 系の図に必要）")
    r.add_argument("--factor-returns-unit", choices=["percent", "decimal"], help="ファクターリターンの単位")
    r.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")

    if args.config:
        cfg = ReportConfig.from_toml(args.config)
        if args.run:
            cfg.runs = [_parse_run(s) for s in args.run]
    elif args.run:
        cfg = ReportConfig(runs=[_parse_run(s) for s in args.run], output_dir=Path("reports"))
    else:
        print("設定ファイルか --run を指定してください", file=sys.stderr)
        return 2

    for key in ("output_dir", "title", "start", "end", "factor_returns", "factor_returns_unit"):
        value = getattr(args, key)
        if value is not None:
            setattr(cfg, key, value)

    build_report(cfg)
    print(f"出力しました: {cfg.output_dir}")
    return 0
