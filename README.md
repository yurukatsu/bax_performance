# bax-performance

BAX（ポートフォリオ最適化ツール）の出力を集計し、複数ランを比較する図・表・PDF を作るパッケージ。
`docs/sample.pdf` のスライドの図を再現する。

## セットアップ

```sh
uv sync
```

## 使い方

### 設定ファイルで実行

```sh
uv run bax-performance report configs/raw.toml          # 素点スコア版
uv run bax-performance report configs/discretized.toml  # 離散化版
```

設定ファイル（TOML）の例。相対パスは設定ファイルの場所が基準。

```toml
title = "EMAI_Transformer"
output_dir = "../reports/raw"
start = 201702        # 省略可（YYYYMM）。省略時は全期間
end = 202603          # 省略可

[factor_returns]      # 省略可。Barra 系の図に必要（後述）
path = "../data/fctrtn_monthly.csv"
unit = "decimal"      # "decimal"（既定）or "percent"

[[runs]]
label = "ai"
path = "../data/output/raw/ai"

[[runs]]
label = "v3"
path = "../data/output/raw/v3"
```

1 レポートで比較できるランは最大 16 本。色は 8 色を並び順で固定割り当てし、9〜16 本目は同じ色を線グラフでは破線、棒グラフでは斜線柄で区別する。

### コマンドラインだけで実行

```sh
uv run bax-performance report \
  --run ai=data/output/raw/ai --run v7=data/output/raw/v7 \
  -o reports/quick --start 201702 --end 202603
```

`--run` はラベル省略可（ディレクトリ名がラベルになる）。設定ファイルと併用すると CLI の値が優先。

### Python から

```python
from bax_performance import ReportConfig, compute
cfg = ReportConfig.from_toml("configs/raw.toml")
data = compute(cfg)          # 集計のみ（図は作らない）
data.summary                 # サマリー表（DataFrame）
data.brinson                 # 月次・国別ブリンソン分解
```

## 出力

```
reports/<name>/
  report.pdf          スライド相当のまとめ（1 ページ 1 テーマ）
  figures/*.png       個別の図（PowerPoint 貼り付け用）
  tables/*.csv        図の元データ
```

| 図 | ファイル | 必要なデータ |
| --- | --- | --- |
| パフォーマンスサマリー（表 + 下記 4 図） | `00_performance_page` | BAX 出力のみ |
| サマリー表 | `01_summary_table` | 〃 |
| 累積超過リターン | `02_cumulative_excess` | 〃 |
| 期待アルファ推移 | `03_expected_alpha` | 〃 |
| 年率超過リターンと IR | `04_excess_ir` | 〃 |
| 暦年別超過リターン | `05_annual_excess` | 〃 |
| Barra 要因分解（Country/Industry/Style/Cur/specific） | `06_barra_classes` | ファクターリターン |
| Style 寄与の累積和 | `07_cumulative_style` | ファクターリターン |
| Specific 寄与の累積和 | `08_cumulative_specific` | ファクターリターン |
| リスクインデックス寄与 上位・下位 | `09_risk_index_top` | ファクターリターン |
| ブリンソン型要因分解（国別） | `10_brinson_country` | BAX 出力のみ |

サンプルとの違い: 「超過リターン + IR」は 2 軸グラフをやめ、横軸を共通にした上下 2 段に分けた。

## 指標の定義

月次・小数ベース。リターンが NA の月（初回の組成月）は除外。

| 指標 | 定義 |
| --- | --- |
| Total Return | ファンドリターンの年率幾何平均 |
| Total Risk | ファンドリターンの標準偏差 × √12 |
| Excess Return | 超過リターン（取引コスト控除後）の平均 × 12 |
| Tracking Error | 超過リターンの標準偏差 × √12 |
| Information Ratio | Excess Return / Tracking Error |
| Max Drawdown | ファンド NAV の最大下落率 |
| Turnover | 月次回転率（片道）の平均 × 12 |
| Cost | 月次取引コストの平均 × 12 |
| Per_Trade | Cost / (2 × Turnover) |

- 累積超過・暦年別超過: 月次超過リターンの単純和
- 期待アルファ: `st_objv.dat` の `alpha`

### ブリンソン型要因分解（国別、BHB）

- 月 t の銘柄リターンは `me_t` の `rtn`、ウェイトは前月 `O_{t-1}` の `wgt_o` / `wgt_b`。`O` に載らない BM のみ銘柄は `w_b = -prev_awgt` で補う。ファンド・BM それぞれ合計 1 に正規化。
- 国は銘柄コード先頭 3 文字（`BRAA261` → `BRA`）。キャッシュ（`*CURR`）は除外。
- allocation = (w_p,c − w_b,c)(R_b,c − R_b)、selection = w_b,c(R_p,c − R_b,c)、interaction = (w_p,c − w_b,c)(R_p,c − R_b,c)
- 図は期間合計。取引コスト控除前。
- 注意: `me` の `rtn` は `st_prfm` の BM リターンと厳密には一致しない（現地通貨建てと思われる）。超過リターンとの整合は概ね取れている（相関 0.99）。

### Barra ファクター分解

- 寄与_k(t) = 前月末のアクティブエクスポージャー（`A_{t-1}` の `opt_active`）× 当月のファクターリターン
- specific = コスト控除前超過リターン（`fund_ex_rtn + tr_cost`）− ファクター寄与の合計
- 分類は属性名の接頭辞: `R` = Style、`I` = Industry、`C` = Country、`c` = Cur
- リスクインデックス図のラベル: OW/UW = 期間平均アクティブエクスポージャーの符号、+/- = 期間平均ファクターリターンの符号

**ファクターリターンは BAX 出力に含まれない**（`bc.txt` で `f_get_attr = 0`）ため、CSV で別途渡す。ファクター名は `A{date}.dat` の属性名（`R13_MOMENTUM`, `I05_BANKS`, `C15_HKG`, `c14_HKGC` など）に合わせる。

long 形式の例（値は架空。単位は小数）:

```csv
dateym,fac,frtn
201702,R13_MOMENTUM,0.0085
201702,I05_BANKS,-0.0120
```

- 列名は `dateym, fac, frtn` か `month, factor, return`。列 `month` と各ファクター名を列に持つ wide 形式も可
- 単位は既定で小数（`unit = "decimal"`）。% で書かれたファイルは設定で `unit = "percent"` を指定する
- ファクター名が空欄の行は警告を出して除外する
- エクスポージャーが 0.001 以上あるのにリターンが欠けているファクター・月は、寄与 0 として警告する
- 指定がなければ Barra 系の図は省略される

## 開発

```sh
uv run pytest
uv run ruff check src tests
```

テストは合成した小さな BAX 出力で動くため、実データは不要。
