# JR_train_snow

JR西日本の積雪予測ソリューションです。現状の構成は、
2段階で予測する設計を意識しながら、既存コードと整合する形で運用されています。

- 着雪確率予測モデル: 各レコードで着雪が発生する確率を予測
- 着雪量予測モデル: 着雪有無判定後に、着雪量を回帰で予測

## 1. 2段階のモデル構成

```text
入力データ
  ↓
着雪確率予測モデル（binary classification）
  ↓
着雪有無の判定（config の thresholds を利用）
  ↓
着雪量予測モデル（regression）
  ↓
着雪量の予測
```

このプロジェクトでは、既存の `run.py` と `binary_predict_run.py` を分けて管理し、
実験 Notebook と実行用 Python スクリプトを両立させています。

## 2. 主要ファイル

- `run.py` : 着雪量予測モデルの学習・推論・提出ファイル生成
- `binary_predict_run.py` : 着雪確率予測モデルを Notebook と同等の処理で実行
- `00.config/config.yaml` : モデル設定、thresholds、LightGBM param を管理
- `00.config/path.yaml` : 学習・テストデータのロード先を管理
- `30.src/jr_snow/` : 前処理、特徴量生成、CV、評価、保存処理
- `10.Notebook/binary_base_line.ipynb` : 着雪確率モデルの元実験 Notebook

## 3. 着雪確率予測モデル

### 目的
各レコードについて、「着雪が発生する確率」を予測します。

### 使用データ
- `train_data` / `test_data` は `00.config/path.yaml` の設定に従って読み込む
- 既存 Notebook のロジックに合わせて、`着雪有無フラグ` をターゲットとして学習する
- 時系列 split は `binary_split_date` で維持する

### 実行方法

```bash
source .venv/bin/activate
python binary_predict_run.py
```

### 処理内容
- データ読込
- 特徴量生成
- train/validation 分割
- LightGBM binary classification
- ROC-AUC 評価
- モデル保存
- 予測確率の出力

### 出力
- `artifacts/binary_model_*.pkl`
- `artifacts/binary_prediction_probabilities.csv`

## 4. 着雪量予測モデル

### 目的
着雪が発生すると判定されたレコードに対して、着雪量を回帰予測します。

### 重要な要件
今回の修正では、`thresholds` 以上のレコードのみを学習・validation の対象に使用します。

具体的には、`着雪有無フラグ` または `着雪確率` が `thresholds` 以上の行だけを残し、
その後に従来通りの時系列 split を適用します。

```python
# 概念
filtered_df = df[df["着雪有無フラグ"] >= thresholds]
```

- `thresholds` は `00.config/config.yaml` で管理する
- `test_data` は今回の要件で勝手に除外しない
- train/validation で同じフィルタ条件が適用される
- 時系列 split のロジックそのものは変更しない

### 実行方法

```bash
source .venv/bin/activate
python run.py --mode full
```

または、推論のみ:

```bash
python run.py --mode predict
```

### 出力
- `submit.csv`
- `artifacts/lightgbm_model_*.pkl`
- `artifacts/feature_importance.csv`
- `artifacts/validation_report.csv`
- `logs/pipeline.log`

## 5. 既存コードとの整合性

- 学習・validation の時系列分割はそのまま維持
- LightGBM のハイパーパラメータ、評価指標、特徴量生成の流れは維持
- 不要な大規模な仕様変更は行わず、既存モジュールを再利用する
- 追加した threshold フィルタは `run.py` の学習前処理で実施

## 6. 実行例

### 学習 + 推論（着雪量モデル）

```bash
python run.py --mode full
```

### 学習のみ

```bash
python run.py --mode train
```

### 推論のみ

```bash
python run.py --mode predict
```

### CV 確認

```bash
python run.py --mode cv --cv-folds 3
```

### 2段階予測の適用

```bash
python run.py --mode predict --two-stage-snow-prediction
```

### 確率予測モデル

```bash
python binary_predict_run.py
```

## 7. config の重要項目

### `00.config/config.yaml`

- `binary_split_date`: 着雪確率モデルの時系列 split 日
- `thresholds`: 着雪有無判定の閾値（0.005 など）
- `MODEL_PARAMS`: 着雪量回帰モデル用 LightGBM パラメータ
- `BINARY_MODEL_PARAMS`: 着雪確率分類モデル用 LightGBM パラメータ

### `00.config/path.yaml`

- `INTERIUM_PATH.train_data`
- `INTERIUM_PATH.test_data`

## 8. 補足

- Notebook は分析・検証用に残している
- 実行本体は Python ロジックとして管理し、再現性を高めている
- 新たな仕様を勝手に追加せず、既存のコードから読み取れる範囲で実装している

- 実験条件ごとの設定ファイル切り替え
