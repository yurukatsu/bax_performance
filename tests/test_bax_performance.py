from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bax_performance import attribution as attr
from bax_performance import metrics
from bax_performance.cli import main
from bax_performance.io import BaxOutput, factor_class


def test_units_are_converted_to_decimals(bax_dirs):
    out = BaxOutput.from_path(bax_dirs[0])
    assert out.prfm.loc[202001, "rot"] == pytest.approx(0.5)
    h = out.holdings(202001)
    assert h["wgt_o"].sum() == pytest.approx(1.0, abs=1e-5)
    assert h["alpha"].iloc[1] == pytest.approx(0.5)  # alpha はスコアのまま
    assert out.params["numeraire"] == "USA"


def test_factor_class():
    assert factor_class("R13_MOMENTUM") == "Style"
    assert factor_class("I05_BANKS") == "Industry"
    assert factor_class("C15_HKG") == "Country"
    assert factor_class("c14_HKGC") == "Cur"
    assert factor_class("sum_pos_w") is None
    assert factor_class("fbeta_wld") is None


def test_summary_stats_definitions():
    rng = np.random.default_rng(0)
    m = pd.DataFrame(
        {
            "fund": rng.normal(0.01, 0.04, 36),
            "excess": rng.normal(0.002, 0.008, 36),
            "cost": np.full(36, 0.001),
            "turnover": np.full(36, 0.12),
        }
    )
    s = metrics.summary_stats(m)
    assert s["Total Return"] == pytest.approx((1 + m.fund).prod() ** (12 / 36) - 1)
    assert s["Tracking Error"] == pytest.approx(m.excess.std() * np.sqrt(12))
    assert s["Information Ratio"] == pytest.approx(s["Excess Return"] / s["Tracking Error"])
    assert s["Turnover"] == pytest.approx(1.44)
    assert s["Cost"] == pytest.approx(0.012)
    assert s["Per_Trade"] == pytest.approx(0.012 / 2.88)


def test_max_drawdown():
    assert metrics.max_drawdown(pd.Series([0.1, -0.5, 0.2])) == pytest.approx(0.5)
    assert metrics.max_drawdown(pd.Series([-0.1, 0.05])) == pytest.approx(0.1)


def test_monthly_series_skips_initial_month_and_filters(bax_dirs):
    out = BaxOutput.from_path(bax_dirs[0])
    s = metrics.monthly_series(out)
    assert list(s.index) == [202002, 202003, 202004, 202005]
    assert list(metrics.monthly_series(out, start=202003, end=202004).index) == [202003, 202004]


def test_brinson_sums_to_active_return(bax_dirs):
    out = BaxOutput.from_path(bax_dirs[0])
    w = attr.brinson_weights(out, 202003, 202002)
    b = attr.brinson_month(w)
    wp, wb = w.w_p / w.w_p.sum(), w.w_b / w.w_b.sum()
    active = (wp * w.r).sum() - (wb * w.r).sum()
    assert b.to_numpy().sum() == pytest.approx(active)
    assert set(b.index) == {"BRA", "KOR"}


def test_brinson_weights_reconstruct_bm_only_stocks(bax_dirs):
    out = BaxOutput.from_path(bax_dirs[0])
    w = attr.brinson_weights(out, 202003, 202002)
    o = out.holdings(202002)
    assert w.loc["KORA002", "w_p"] == 0.0
    assert w.loc["KORA002", "w_b"] == pytest.approx(-out.active_returns(202003).loc["KORA002", "prev_awgt"])
    assert w.loc["BRAA001", "w_b"] == pytest.approx(o.loc["BRAA001", "wgt_b"])
    assert "USACURR" not in w.index


def test_load_factor_returns_long_and_wide(factor_csv, tmp_path):
    long = attr.load_factor_returns(factor_csv)
    raw = pd.read_csv(factor_csv).pivot(index="month", columns="factor", values="return").reset_index()
    raw.to_csv(tmp_path / "wide.csv", index=False)
    wide = attr.load_factor_returns(tmp_path / "wide.csv")
    pd.testing.assert_frame_equal(long, wide[long.columns], check_names=False)


def test_load_factor_returns_fctrtn_format(factor_csv, tmp_path):
    """fctrtn_monthly.csv 形式（dateym, fac, frtn、名前が空欄の行あり）を読める。"""
    raw = pd.read_csv(factor_csv).rename(columns={"month": "dateym", "factor": "fac", "return": "frtn"})
    blank = raw.head(3).assign(fac="")
    pd.concat([raw, blank]).to_csv(tmp_path / "fctrtn.csv", index=False)
    with pytest.warns(UserWarning, match="空欄"):
        got = attr.load_factor_returns(tmp_path / "fctrtn.csv")
    pd.testing.assert_frame_equal(got, attr.load_factor_returns(factor_csv), check_names=False)


def test_load_factor_returns_percent_unit(factor_csv):
    pct = attr.load_factor_returns(factor_csv, unit="percent")
    np.testing.assert_allclose(pct.to_numpy() * 100, attr.load_factor_returns(factor_csv).to_numpy())


def test_factor_attribution_adds_up(bax_dirs, factor_csv):
    out = BaxOutput.from_path(bax_dirs[0])
    fr = attr.load_factor_returns(factor_csv)
    months = [202002, 202003, 202004, 202005]
    c = attr.factor_attribution(out, fr, months)
    gross = out.prfm.loc[months, "fund_ex_rtn"] + out.prfm.loc[months, "tr_cost"]
    np.testing.assert_allclose(c.sum(axis=1), gross)
    # 寄与 = 前月アクティブエクスポージャー × 当月ファクターリターン
    expo = out.exposures(202002).loc["R13_MOMENTUM", "opt_active"]
    assert c.loc[202003, "R13_MOMENTUM"] == pytest.approx(expo * fr.loc[202003, "R13_MOMENTUM"])
    cls = attr.by_class(c)
    assert list(cls.columns) == attr.FACTOR_CLASS_ORDER
    np.testing.assert_allclose(cls.sum(axis=1), gross)


def test_style_summary_position_labels(bax_dirs, factor_csv):
    out = BaxOutput.from_path(bax_dirs[0])
    fr = attr.load_factor_returns(factor_csv)
    c = attr.factor_attribution(out, fr, [202002, 202003, 202004, 202005])
    s = attr.style_summary(out, c, fr)
    assert set(s.index) == {"R01_BETA", "R13_MOMENTUM"}
    for _, row in s.iterrows():
        assert row.position.startswith("OW" if row.mean_active_exposure >= 0 else "UW")
        assert row.position.endswith("+" if row.mean_factor_return >= 0 else "-")


def test_cli_end_to_end(bax_dirs, factor_csv, tmp_path):
    cfg = tmp_path / "cfg.toml"
    cfg.write_text(
        f"""
title = "test"
output_dir = "out"
[factor_returns]
path = "{factor_csv.name}"
[[runs]]
label = "A"
path = "{bax_dirs[0].name}"
[[runs]]
label = "B"
path = "{bax_dirs[1].name}"
"""
    )
    assert main(["report", str(cfg)]) == 0
    out = tmp_path / "out"
    assert (out / "report.pdf").stat().st_size > 0
    figs = {p.stem for p in (out / "figures").glob("*.png")}
    assert {"00_performance_page", "06_barra_classes", "09_risk_index_top", "10_brinson_country"} <= figs
    summary = pd.read_csv(out / "tables" / "summary.csv", index_col=0)
    assert list(summary.index) == ["A", "B"]


def test_cli_with_runs_only(bax_dirs, tmp_path):
    out = tmp_path / "quick"
    assert main(["report", "--run", f"X={bax_dirs[0]}", "--run", str(bax_dirs[1]), "-o", str(out)]) == 0
    figs = {p.stem for p in (out / "figures").glob("*.png")}
    assert "06_barra_classes" not in figs  # ファクターリターン無しでは Barra 系は出さない
    assert list(pd.read_csv(out / "tables" / "summary.csv", index_col=0).index) == ["X", "run_b"]


def test_too_many_runs_rejected():
    from bax_performance.plots import run_colors

    with pytest.raises(ValueError):
        run_colors([str(i) for i in range(9)])
