# System Architecture

## 1．概要

UNe3dMe は，Gradio ベースの Web UI から，画像／動画データセットの作成，COLMAP 前処理，複数の 3D 再構築手法の実行，可視化，レンダリング評価までを一貫して扱う統合プラットフォームである．  
システムは主に以下の 4 モジュールで構成される．

- `main.py`：起動エントリポイント
- `demo.py`：UI 定義，イベント接続，セッション変数の保持
- `local_backend.py`：共通処理，前処理，ファイル操作，viewer 起動，評価
- `methods.py`：各手法の実行ラッパ，学習・推論・レンダリング呼び出し

`main.py` は起動時に一時作業ディレクトリを作成し，その配下に `datasets/`， `outputs/`　および  `logs/` を生成する．さらに，その一時ディレクトリを `local_backend.TMPDIR` および `methods.TMPDIR` に設定した後，`main_demo(tmpdir, datasets, outputs)` を呼び出して Web UI を起動する．

## 2．設計方針

本システムは，責務を以下のように分離している．

- **起動と環境初期化**
  - `main.py` が担当．
- **ユーザー操作の受付と画面更新**
  - `demo.py` が担当．
- **データ準備や共通ユーティリティ**
  - `local_backend.py` が担当．
- **個別手法の実行**
  - `methods.py` が担当．

この分離により，UI の変更と，実行ロジックや外部ツール呼び出しを切り離して保守できる構成となっている．

## 3．モジュール構成

### 3.1 `main.py`

`main.py` はシステムの起動点である．主な役割は以下である．

1. 一時作業ディレクトリの生成
2. `datasets/`， `outputs/` および `logs/` の作成
3. 共通作業ディレクトリパスの各モジュールへの共有
4. `demo.py` の `main_demo()` 呼び出し

このモジュールは，アプリケーション全体のライフサイクル開始時に必要な最小限の初期化のみを担い，業務ロジックは持たない．

### 3.2 `demo.py`

`demo.py` は UI 層であり，Gradio コンポーネントの定義，言語切替，イベント接続，状態管理を担当する．  
翻訳 JSON を読み込む `load_translations()` と，UI 表示ラベルを一括更新する `update_ui()` を持ち，多数の UI コンポーネントを言語別に更新する構成となっている． 

また，`main_demo()` では以下の `State` を保持する．

- 一時ディレクトリ
- datasets ディレクトリ
- outputs ディレクトリ
- UI 言語状態
- 現在セットされている画像データセット
- 現在セットされている COLMAP データセット
- viewer スクリプト種別

これにより，画面上の操作結果をシステム内で共有できる． 

さらに `demo.py` は，ボタンや入力変更イベントを `local_backend.py` および `methods.py` の関数へ接続する役割を持つ．たとえば，

- 画像から画像データセットを作成する操作は `local_backend.copy_images`
- 動画から画像を抽出する操作は `local_backend.extract_frames_with_filter`
- COLMAP 実行は `local_backend.run_colmap`
- 各手法の再構築は `methods.recon_*`
- エクスポートは `methods.export_*`
- 評価は `methods.render_eval_*`

へ接続されている．

### 3.3 `local_backend.py`

`local_backend.py` は，共通バックエンド処理を担うモジュールである．主な機能は以下のとおりである．

#### 画像／動画データセット作成
- `copy_images()`：指定画像群を `datasets/<name>/images` にコピーする． 
- `extract_frames_with_filter()`：動画から ffmpeg でフレームを抽出し，必要に応じて SSIM により類似フレームを削除する． 

#### COLMAP 前処理
- `run_colmap()`：入力画像の準備，特徴抽出，マッチング，マッピング，undistortion，multiscale 画像生成，Nerfstudio 形式変換をまとめて実行する． 

#### ファイル操作
- `zip_dataset()`：`images/` や `colmap/` を含むデータセットを ZIP 化する．

#### 可視化
- `viewer()`：`scripts/viewer/` 配下の viewer スクリプトを subprocess で起動し，ソケット接続確認により URL を返す．

#### 評価
- `evaluate_all_metrics()`：GT 画像と生成画像から，PSNR，SSIM，MS-SSIM，LPIPS，FSIM，VIF，BRISQUE，FID を計算し，JSON と結果を保存する． 

### 3.4 `methods.py`

`methods.py` は，各 3D 再構築手法や関連処理の**実行ラッパ層**である．  
このモジュールは，個別手法の UI 固有ロジックを持たず，`demo.py` から受け取った入力をもとにコマンドラインを組み立て，実行し，実行結果保存ディレクトリ・実行時間・ステータス・ログを返す．

共通の実行基盤として `run_subprocess_popen()` を持ち，標準出力と標準エラー出力を逐次取得しながらログファイルへ保存する．ログは既定で `TMPDIR/logs` に出力される． 

Nerfstudio 系では，さらに以下の共通関数で処理を抽象化している．

- `train_nerfstudio()`
- `train_nerfstudio_slurm()`
- `export_nerfstudio()`
- `export_nerfstudio_slurm()`
- `render_eval_nerfstudio()`

たとえば `recon_vnerf()` は，mode に応じて local 実行か slurm 実行かを切り替えたうえで，Nerfstudio 共通学習関数へ委譲する．`export_vnerf()` と `render_eval_vnerf()` も同様に，共通のエクスポート関数・評価関数へ委譲する． 

この設計により，各手法ラッパは
- 手法固有の引数
- 出力先規約
- local／slurm の切替
のみを記述すればよく，共通実行処理の重複を抑えられる．

## 4．処理フロー

システム全体の代表的な処理フローを以下に示す．

```{mermaid}
---
config: {"flowchart": {"curve": "linear"}}
---
flowchart TB

    A[main.py<br/>起動エントリポイント]
    B[demo.py<br/>UI 定義<br/>イベント接続<br/>State 管理]
    C[local_backend.py<br/>共通処理<br/>前処理<br/>ファイル操作<br/>viewer の起動]
    D[methods.py<br/>手法別3次元再構築処理<br/>評価処理<br/>実行ラッパ]
    I[datasets/<br/>入力データ]
    E[scripts/<br/>補助スクリプト<br/>viewer スクリプト]
    F[models/<br/>各手法実装<br/>submodules]
    G[外部ツール<br/>FFmpeg / COLMAP]
    J[outputs/<br/>再構築結果<br/>評価結果]
    
    A --> B

    B --> C
    B --> D

    C --> I
    C --> E
    C --> G

    D --> E
    E --> F
    D --> F
    E --> J

    F --> J
    G --> I

    linkStyle default stroke:#000000,stroke-width:5px;

```