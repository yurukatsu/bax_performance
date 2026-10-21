"""テスト用の小さな BAX 出力ディレクトリを生成する。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

MONTHS = [202001, 202002, 202003, 202004, 202005]
HELD = ["BRAA001", "BRAA002", "KORA001"]
BM_ONLY = ["KORA002"]
FACTORS = ["R01_BETA", "R13_MOMENTUM", "I05_BANKS", "C05_BRA", "c05_BRAC"]


def _write(path: Path, header: str, rows: list[str]) -> None:
    path.write_text("\n".join([header, *rows]) + "\n")


def make_bax_output(root: Path, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    codes = HELD + BM_ONLY
    wgt_o, wgt_b = {}, {}
    for m in MONTHS:
        o = rng.dirichlet(np.ones(len(HELD))) * 100
        b = rng.dirichlet(np.ones(len(codes))) * 100
        wgt_o[m] = dict(zip(HELD, o)) | {c: 0.0 for c in BM_ONLY}
        wgt_b[m] = dict(zip(codes, b))
        rows = ["USACURR 0.0 0.0 0.0 0.0 0.0 0.0"]
        rows += [f"{c} {wgt_o[m][c]:.6f} {wgt_b[m][c]:.6f} 0.0 0.5 0.0 7.0" for c in HELD]
        _write(root / f"O{m}.dat", "#code wgt_o wgt_b wgt_i alpha cs_wgt_l cs_wgt_u", rows)
        exp = rng.normal(0, 0.2, len(FACTORS))
        rows = ["sum_pos_w 100 100 100 50 100 0"] + [f"{f} 0.1 0.1 0.1 {e:.6f} 0.1 0" for f, e in zip(FACTORS, exp)]
        _write(root / f"A{m}.dat", "#attribute opt bm init opt_active opt_long opt_short", rows)

    prfm = [f"{MONTHS[0]} NA NA NA 0.5 NA 0 NA 50.0 -0.5 NA NA"]
    objv = []
    for i, m in enumerate(MONTHS):
        objv.append(f"{m} 0.4 {0.5 + 0.01 * i:.4f} 2.0 0.1 0.5 0 0 0 0 1.0")
        if i == 0:
            continue
        prev = MONTHS[i - 1]
        rtn = dict(zip(codes, rng.normal(1, 5, len(codes))))
        aw = {c: wgt_o[prev][c] - wgt_b[prev][c] for c in codes}
        _write(root / f"me_{m}.dat", "#code prev_awgt rtn", [f"{c} {aw[c]:.6f} {rtn[c]:.6f}" for c in codes])
        bm = sum(wgt_b[prev][c] * rtn[c] for c in codes) / 100
        cost = 0.1
        ex = sum(aw[c] * rtn[c] for c in codes) / 100 - cost
        prfm.append(f"{m} {bm:.6f} {bm + ex:.6f} {ex:.6f} {cost} 0 0 0 12.0 -0.1 NA NA")
    _write(
        root / "st_prfm.dat",
        "#month bm_rtn fund_rtn fund_ex_rtn tr_cost fixed_fee tr_fee br_cost rot w_cash altbm_rtn altbm2_rtn",
        prfm,
    )
    _write(root / "st_objv.dat", "#day opt_obj alpha omega tr_cost h_tr_cost br_cost altrisk alt2risk rc time", objv)
    (root / "bc.txt").write_text("# test\nnumeraire = USA\ncs_rot = 0.15\n")
    return root


def make_factor_returns(path: Path, seed: int = 1) -> Path:
    rng = np.random.default_rng(seed)
    rows = [{"month": m, "factor": f, "return": rng.normal(0, 1)} for m in MONTHS for f in FACTORS]
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture
def bax_dirs(tmp_path) -> list[Path]:
    return [make_bax_output(tmp_path / name, seed=i) for i, name in enumerate(["run_a", "run_b"])]


@pytest.fixture
def factor_csv(tmp_path) -> Path:
    return make_factor_returns(tmp_path / "factor_returns.csv")
