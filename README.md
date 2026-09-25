# JR_train_snow

JR西日本の積雪予測ソリューションの学習・推論パイプラインです。
Notebook は残しつつ、実行ロジックを Python モジュールへ分離して、再利用しやすい構成に整理しています。

## 1. ディレクトリ構成

```text
JR_train_snow/
├── 00.config/
│   ├── config.yaml          # 学習設定 / 実験メモ / LightGBM params
│   └── path.yaml            # 入力データのパス定義
├── 10.Notebook/
│   ├── base_line.ipynb      # 既存の分析 Notebook
│   ├── データの前処理.ipynb
│   ├── 基礎分析.ipynb
│   └── submit.csv
├── 20.Data/
│   ├── train_*.pkl          # 学習データ
│   ├── test_*.pkl           # 推論用データ
│   ├── weather.csv          # 気象データ
│   ├── snowfall.csv         # 積雪計データ
│   └── ...
├── 30.src/
│   ├── jr_snow/
│   │   ├── config.py         # 設定ファイル読み込みと特徴量定義
│   │   ├── data.py           # データ読込
│   │   ├── features.py       # 前処理と特徴量強化
│   │   ├── cross_validation.py  # 時系列CV
│   │   ├── feature_importance.py  # 特徴量重要度保存
│   │   ├── model_registry.py # モデルの保存・読み込み
│   │   ├── modeling.py       # LightGBM 学習・推論
│   │   ├── evaluation.py     # WMAE 等の評価指標
│   │   ├── reporting.py      # 評価レポートCSV出力
│   │   ├── logging_utils.py  # log.info 用ロガー
│   │   └── __init__.py
│   └── utils/
│       ├── utils.py
│       └── validation.py
├── artifacts/
│   ├── lightgbm_model_*.pkl
│   ├── feature_importance.csv
│   └── validation_report.csv
├── logs/
│   └── pipeline.log
├── tests/
│   ├── test_run_pipeline.py
│   ├── test_dia_pass_interpolation.py
│   └── test_pipeline_modules.py
├── run.py
├── submit.csv
├── .venv/
├── pytest.ini (if present)
├── README.md
└── requirements.txt (if added later)
```

## 2. 実行フロー

```text
Raw data in 20.Data
    ↓
config.py / path.yaml
    ↓
load_train_test_data()
    ↓
features.py
  - 日付特徴量
  - 気温閾値フラグ
  - 既存特徴量の再整形
    ↓
train/valid split
    ↓
modeling.py
  - LightGBM training
  - inference
    ↓
feature_importance.py
model_registry.py
reporting.py
    ↓
logs/pipeline.log
artifacts/*
submit.csv
```

### 役割分担

- data.py
  - 学習データとテストデータの読み込み
- features.py
  - 前処理と特徴量生成
- cross_validation.py
  - 時系列 CV フォールド生成
- feature_importance.py
  - 特徴量重要度 CSV を出力
- model_registry.py
  - 学習済みモデルを版管理付きで保存・読み込み
- modeling.py
  - LightGBM の学習と推論
- evaluation.py
  - WMAE 等の計算
- reporting.py
  - 評価レポートを CSV に保存
- logging_utils.py
  - 実験メモや学習指標を log.info で出力

## 3. 入力ディレクトリの整理

### 00.config
- `config.yaml`
  - `EXPERIMENT.experiment_note`
  - `MODEL_PARAMS`
  - `FEATURE`
  - `WEATHER_FEATURE`
  - `SNOW_FEATURE`
- `path.yaml`
  - `INTERIUM_PATH.train_data`
  - `INTERIUM_PATH.test_data`

### 20.Data
- 事前に生成済みの学習・検証用データを置くディレクトリ
- 例:
  - `train_time_weather.pkl`
  - `test_time_weather.pkl`

## 4. 出力ディレクトリの整理

### artifacts/
- 学習済みモデルの保存先
- 特徴量重要度CSV
- validation report CSV

### logs/
- `pipeline.log`
- 学習件数、使用特徴量、LightGBMパラメータ、予測値統計量などを記録

### submit.csv
- 推論結果の提出用ファイル

## 5. 実行方法

### 1) 仮想環境を有効化

```bash
source .venv/bin/activate
```

### 2) 学習 + 推論を実行

```bash
python run.py --mode full
```

### 3) 学習のみ

```bash
python run.py --mode train
```

### 4) 推論のみ

```bash
python run.py --mode predict
```

### 5) 時系列CVの確認

```bash
python run.py --mode cv --cv-folds 3
```

### 6) 最終推論モデル戦略の切り替え

```bash
# ① 現状の single split を使う（既存の標準動作）
python run.py --mode full --final-model-strategy single_split

# ② 各 fold モデルの平均予測を使う
python run.py --mode full --final-model-strategy cv_average

# ③ 各 fold の WMAE の中央値に最も近いモデルを採用する
python run.py --mode full --final-model-strategy median_wmae

# ④ WMAE が最もよかった fold モデルを採用する
python run.py --mode full --final-model-strategy best_fold
```

- `single_split`: 既存の `split_date` ベースの train/valid 分割のモデルをそのまま使用
- `cv_average`: 各 fold の学習済みモデルを使い、テスト予測を平均する
- `median_wmae`: fold ごとの WMAE の中央値に最も近いモデルを最終推論に使う
- `best_fold`: WMAE が最も良かった fold モデルを最終推論に使う

### 7) 着雪量の2段階予測を有効化

```bash
# テストデータの「着雪量予測フラグ」が 0 の行は 0 に固定し、1 の行だけモデル予測を使う
python run.py --mode predict --two-stage-snow-prediction
```

- `着雪量予測フラグ == 0` の行: 予測値を `0.0` に固定
- `着雪量予測フラグ == 1` の行: モデルの予測値をそのまま利用
- フラグ列が存在しない場合: 警告を出し、通常の単一段階予測へフォールバック

この引数を使うことで、着雪あり/なしの判定を先に行い、着雪がないと判定されたレコードには着雪量を出さない2段階の処理を実行できます。

### 8) オプションの例

```bash
python run.py \
  --mode full \
  --final-model-strategy cv_average \
  --two-stage-snow-prediction \
  --num-boost-round 1000 \
  --early-stopping-rounds 100
```

## 6. CV の設計

本プロジェクトの時系列CV は、従来の「固定区間の分割」ではなく、
`train 期間を広げていく expanding-window 型` に変更しています。

- 各 fold の検証期間は将来の連続区間を使う
- 学習期間は fold が進むごとに広がる
- これにより、過去のデータで学習し、より新しい期間を検証する構造を再現する

### 7. ログに残す主な情報

- Train件数
- Validation件数
- Test件数
- 特徴量数
- Target名
- 使用特徴量
- LightGBMパラメータ
- 学習結果
- Prediction stats
- Target stats
- Experiment note

ログは `logs/pipeline.log` に保存されます。

## 8. Notebook との関係

Notebook は分析・検証のために残していますが、実行本体は `run.py` と `30.src/jr_snow` 配下へ分離しました。
これにより、

- 実験の再現性が上がる
- モジュールごとにテストがしやすい
- 学習・推論・評価の責務が明確になる
- 提出前の整理がしやすい

という利点があります。

## 9. 今後の拡張候補

- Cross Validation を本格的に回す実験スクリプト
- 学習済みモデルの比較表出力
- feature importance の可視化図
- バージョン別のモデル管理と回帰テスト
- 実験条件ごとの設定ファイル切り替え
