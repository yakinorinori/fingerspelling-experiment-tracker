# fingerspelling-experiment-tracker

指文字認識研究の**実験記録Webアプリ**。実験ごとに「行（クラス・モデルなど）× 列（正認識/誤認識/未検出などの条件）」の表を記録し、認識率を自動計算、TSV/Markdownでコピーできる。

## 構成

- `app.py` — Flaskアプリ本体（SQLiteに実験データを保存）。初回起動時に研究で行うべき実験7件を自動登録
- `templates/index.html` — 実験一覧・編集UI
- `singlechar.py`, `templates/single.html` — 単体文字動画で認識する画面（実験5）
- `data/single_char_labels.csv` — 単体文字動画の正解（動画を見て入力したもの）
- `data/experiments.db` — 実験データ（SQLite）。本リポジトリには現時点の記録を同梱

## 主な機能

- **実験一覧（左サイドバー）**: カテゴリ別に表示、状態（予定/実施中/完了）をバッジ表示
- **条件（列）を自由に設定**: チップUIで列の追加・削除・名前変更が可能。各列に「✓ 認識率の分母に含める」を個別設定でき、例えば「未検出」を分母に含めるかどうかを実験ごとに選べる
- **認識率の自動計算**: 「正認識」列があれば行ごと・合計の認識率を自動算出
- **コピペ可能な表**: TSV（Excel/スプレッドシート用）、Markdown（GitHub/論文メモ用）の2形式でクリップボードにコピー
- **全項目を後から編集可能**: タイトル・カテゴリ・状態・クラス数・説明・条件列・結果セル・メモをいつでも再編集→保存

## 単体文字動画で認識（実験5）— `/single`

`Video/単語分割/単体文字` の動画を、静止状態モデル（SVM）・動作状態モデル（Transformer）・統一モデルで認識し、
正認識・誤認識・未検出を集計して実験5の表に反映する画面。実験記録の左下「🎬 単体文字動画で認識」から開く。

1. 動画を選ぶと、右手の骨格を取って3つのモデルで認識する（結果は保存され、もう一度開いても計算し直さない）
2. 右手が1フレームも取れない動画は **未検出**。自動再生しない（「もう一度再生」で見られる）
3. 動画を見て **正解の文字** をボタンか直接入力で入れる → `data/single_char_labels.csv` に保存され、次回も使われる
4. 認識結果は動画を1回見終わってから表示（正解の入力が結果に引きずられないように。設定で「すぐに表示」にもできる）
5. 正解と比べて自動で判定。おかしければ 正認識／誤認識／未検出／対象外 のボタンで直せる（「自動」で元に戻る）
6. 「📥 実験5の表に反映」で、モデルごとの サンプル数・正認識・誤認識・未検出 を実験5に書き込む
7. 「📄 動画ごとの結果をCSVで保存」で、動画ごとの正解・認識結果・判定を書き出す

- **対象外**：そのモデルが認識できない文字（静止SVMに動作指文字、動作モデルに静止指文字）。サンプル数に数えない
- **サンプル数** = 正認識 + 誤認識 + 未検出（正解が未入力の動画は数えない）
- 認識には fingerspelling-character のコード（右手の判定・特徴量・`recognize_video.py` と同じ計算）とモデルを使う
- キー操作：Space=もう一度再生、← →=前・次の動画、1〜3=静止・動作・統一の判定を自動に戻す

### 設定（画面左下の ⚙）

フォルダとモデルの場所を指定する（`data/single_char_settings.json` に保存。パスが正しいと ✓ が付く）。

| 項目 | 既定値 |
|---|---|
| 単体文字フォルダ | `../../Fingerspelling/Video/単語分割/単体文字` |
| fingerspelling-character | `../fingerspelling-character` |
| 静止・動作・統一モデル | `fingerspelling-character/results/models/` の `svm_H3+O.joblib`、`dynamic_transformer_H3+O+M.pt`、`unified_transformer_S+H3+O+M.pt` |

### 起動（fingerspelling-character と同じ環境で）

```bash
source ../../Fingerspelling/venv_hand/bin/activate   # mediapipe・torch・scikit-learn が入っている環境
pip install flask
python app.py        # http://127.0.0.1:5002/single
```

動画が再生されないときは Safari で開く（iPhone の HEVC 動画は Chrome で再生できないことがある）。

## 初期登録されている実験

1. 手型モデル精度（7クラスSVM）
2. 動作モデル比較（Transformer vs 1D-CNN）
3. 指文字単語分割（3クラス: 静止/動作/遷移）
4. 指文字単語分割（2クラス: 文字区間/遷移区間）
5. 指文字単体認識精度（静止/動作/統一モデル）
6. 骨格保存導入後の比較実験
7. 手話文章中の指文字単語判別

## 起動方法

```bash
pip install flask
python app.py
```

ブラウザで `http://127.0.0.1:5002` を開く。

## 必要ライブラリ
`flask`（単体文字動画の認識には、加えて fingerspelling-character と同じ `mediapipe`・`torch`・`scikit-learn`・`opencv-python`）
