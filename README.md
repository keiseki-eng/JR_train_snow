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

### 6) オプションの例

```bash
python run.py \
  --mode full \
  --num-boost-round 1000 \
  --early-stopping-rounds 100
```

## 6. ログに残す主な情報

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

## 7. Notebook との関係

Notebook は分析・検証のために残していますが、実行本体は `run.py` と `30.src/jr_snow` 配下へ分離しました。
これにより、

- 実験の再現性が上がる
- モジュールごとにテストがしやすい
- 学習・推論・評価の責務が明確になる
- 提出前の整理がしやすい

という利点があります。

## 8. 今後の拡張候補

- Cross Validation を本格的に回す実験スクリプト
- 学習済みモデルの比較表出力
- feature importance の可視化図
- バージョン別のモデル管理と回帰テスト
- 実験条件ごとの設定ファイル切り替え
