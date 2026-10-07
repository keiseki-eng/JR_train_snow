# JR_train_snow 処理フロー可視化

以下は、現在の src 配下の実装に沿って、データの流れと特徴量生成の位置を可視化した図です。

## 1. 全体像

```text
[train.csv / test.csv]
        │
        │ 00.config/path.yaml で読込先を管理
        ▼
load_train_test_data()
  - 30.src/jr_snow/data.py
        │
        ├─ train_df = pd.read_pickle(...)
        └─ test_df = pd.read_pickle(...)
        │
        ▼
[冬季フラグ処理 / 欠損補完 / 特徴量生成]
        │
        ├─ fill_missing_by_same_day_time_location_average()
        │   - 30.src/jr_snow/features.py
        │   - 同じ日・同じ時間帯の他地点平均で欠損値を埋める
        │
        ├─ build_engineered_feature_frame()
        │   - 30.src/jr_snow/features.py
        │   - 日付特徴量追加
        │   - 既存の地域・時間帯特徴量追加
        │   - 気温閾値特徴量追加
        │
        └─ prepare_model_inputs()
            - 学習/検証分割
            - カテゴリ変数変換
            - 数値整形
            - feature_list / categorical_cols を返却

        │
        ▼
  [binary model: 着雪確率予測]
        │
        ├─ binary_predict_run.py
        │   - df_train に対して 冬季フラグ=1 のみを学習対象にする
        │   - 予測確率を出力
        │   - thresholds で対象判定を行う
        │
        └─ 予測結果: 着雪確率

        │
        ▼
  [two-stage filter]
        │
        ├─ filter_snow_presence_records()
        │   - 30.src/run.py
        │   - 冬季フラグを先に絞る
        │   - 着雪確率 or 着雪有無フラグ >= thresholds
        │
        └─ 着雪量モデルの学習対象レコードを決定

        │
        ▼
  [snow amount regression model]
        │
        ├─ run.py
        │   - train_df から threshold 以上のレコードのみを学習
        │   - test_df では 冬季フラグ=0 を 0 として固定
        │   - 冬季フラグ=1 かつ threshold 以上のみ回帰予測
        │
        ├─ train_lightgbm_model()
        │   - 30.src/jr_snow/modeling.py
        │   - LightGBM 回帰学習
        │
        ├─ predict_submission()
        │   - 推論
        │
        └─ save_submission()
            - submit.csv を出力
```

## 2. 実際のコード対応

### 2-1. データ読込

実際の入口は以下です。

- [30.src/jr_snow/data.py](30.src/jr_snow/data.py)
- [00.config/path.yaml](00.config/path.yaml)
- [run.py](run.py)
- [binary_predict_run.py](binary_predict_run.py)

この段階では、pickle 形式（または適切なデータファイル）を読み込んで、
`train_df`, `test_df` という DataFrame に変換します。

### 2-2. 欠損値補完

特徴量作成前に、欠損値を埋める処理が入ります。

- [30.src/jr_snow/features.py](30.src/jr_snow/features.py)

`fill_missing_by_same_day_time_location_average()` は次を実施します。

- 日付が同じ
- 時間帯が同じ
- 地点が異なる
- その列の平均値で該当欠損値を補完

これは、ある地点の観測が欠損しているときに、同じ時間帯の他地点を使って近い値を埋める安全策です。

### 2-3. 特徴量生成

特徴量は以下の流れで生成されます。

- `add_date_features()`
  - 年・月・日・曜日の特徴量追加
- `add_document_weather_features()`
  - 地域・時間帯別降雪量・日射量・天気期待値を追加
- `add_temperature_threshold_features()`
  - 気温閾値特徴量を追加

最終的に `build_engineered_feature_frame()` で一つの特徴量 DataFrame にまとめます。

### 2-4. 学習/検証分割

- [30.src/jr_snow/features.py](30.src/jr_snow/features.py)

`prepare_model_inputs()` の内部で、

- `split_train_valid()`
- `X_train`, `X_valid`, `y_train`, `y_valid`

に分割されます。

ここで target は通常 `合計`、二値モデルは `着雪有無フラグ` を使います。

### 2-5. 学習と推論

- [30.src/jr_snow/modeling.py](30.src/jr_snow/modeling.py)
- [run.py](run.py)
- [binary_predict_run.py](binary_predict_run.py)

学習は `train_lightgbm_model()` や `lgb.train()` で実施され、
推論は `predict_submission()` により生成されます。

## 3. 特徴量作成の妥当性

### 3-1. 元データが適切に扱われているか

現在の流れでは、以下が守られています。

- 元データの読み込みは一箇所で管理
- 欠損値補完を特徴量生成前に実施
- 特徴量生成と学習データ分割が分離されている
- モデル入力の列選定が明示的
- `feature_list` と `categorical_cols` を明確に分けている

つまり、特徴量の生成とモデル入力の整形が「前処理に集中」しており、
学習時に元データが直接そのまま使われる形ではありません。

### 3-2. 注意点

ただし、データの前処理が肥大化しやすいため、今後は次の観点を確認すると安全です。

1. `df_train` と `df_test` の列名が完全一致しているか
2. 欠損補完対象列が本当に数値列であるか
3. `冬季フラグ` で絞ったデータが本当に学習/推論に使われているか
4. `thresholds` を適用した後に、学習データに異常値が残っていないか

## 4. 現在の実装でのまとめ

```text
元データ
  ↓
読み込み
  ↓
欠損補完（同じ日・同じ時間帯・他地点平均）
  ↓
特徴量生成
  ↓
train/valid split
  ↓
model input 整形
  ↓
LightGBM 学習 / 推論
```

この流れにより、特徴量作成の前に欠損値を埋め、
そのうえでモデルに渡す構成になっていて、理論的には比較的安全です。

---

必要なら次に、
「このままのフローを Mermaid 図で見せる」形へ整えて、README に組み込みやすい版も作成できます。
