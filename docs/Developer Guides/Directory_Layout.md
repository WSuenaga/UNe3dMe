# Directory Layout

## 1．概要

本ページでは，UNe3dMe の主要ディレクトリ構成について説明する．  
開発時に「どのファイルがどこに生成されるのか」，「どのモジュールがどのディレクトリを扱うのか」を把握するための参照ページである．

## 2．ルートディレクトリ構成

リポジトリ直下の主な構成は以下のとおりである．

```text
UNe3dMe/
├── main.py
├── demo.py
├── local_backend.py
├── methods.py
├── models/
├── scripts/
├── translations/
├── docs/
└── ...
```

### 2.1 `main.py`
- アプリケーションの起動エントリポイント．
- 一時作業ディレクトリを作成し，`datasets/`，`outputs/`，`logs/` を初期化する．

### 2.2 `demo.py`
- Gradio UI の定義を行う．
- State 管理，言語切替，イベント接続を担当する．

### 2.3 `local_backend.py`
- データセット作成，COLMAP 前処理，viewer 起動，評価処理などの共通機能を担当する．

### 2.4 `methods.py`
- 各手法の学習・推論・エクスポート・レンダリング評価を呼び出す実行ラッパ．

### 2.5 `scripts/`
- 補助スクリプト群を格納するディレクトリ．
- 再構築処理の補助スクリプトや，viewer 起動スクリプトを含む．

### 2.6 `models/`
- 各再構築手法の実装本体を配置するディレクトリ．
- Git submodule により管理されることがある．

### 2.7 `translations/`
- UI 文言の翻訳ファイルを格納するディレクトリ．

### 2.8 `docs/`
- Sphinx ドキュメント一式を格納するディレクトリ．

## 3．実行時に作成される作業ディレクトリ

アプリケーション起動時に，一時作業ディレクトリ配下へ以下のディレクトリが作成される．

```text
TMPDIR/
├── datasets/
├── outputs/
└── logs/
```

### 3.1 `datasets/`
- 入力画像・動画から作成されたデータセットを保存する．
- COLMAP 前処理結果もここに含まれる場合がある．

### 3.2 `outputs/`
- 学習結果，推論結果，エクスポート結果，レンダリング結果，評価結果を保存する．

### 3.3 `logs/`
- subprocess 実行ログや補助処理ログを保存する．

## 4．データセット構造

典型的なデータセット構造の例を以下に示す．

```text
datasets/
└── <dataset_name>/
    ├── images/
    ├── colmap/
    |  ├── images/    
    |  ├── images_2/
    |  ├── images_4/
    |  ├── images_8/
    |  ...
    |  ├── sparse/
    |  └──transforms.json
    └── ...
```

### 4.1 `images/`
- 画像データセットを格納するディレクトリ．
- 画像のみからなる．

### 4.2 `colmap/`
- COLMAP による前処理結果を格納するディレクトリ．
- `images/`，`sparse/`，`transforms.json`などを含む．

## 5．出力ディレクトリ構造

典型的な出力構造の例を以下に示す．

```text
outputs/
└── <method_name>/
    └── <dataset_name>/
        └── ...
```

### 5.1 `<method>/`
- 手法ごとの出力ディレクトリ．

### 5.2 `<dataset_name>/`
- 入力データセット単位の出力ディレクトリ．

## 6．主要モジュールとディレクトリの対応

各モジュールと，主に扱うディレクトリの対応を以下に示す．

- `main.py`
  - `TMPDIR/`
  - `datasets/`
  - `outputs/`
  - `logs/`

- `demo.py`
  - UI 上でこれらのパスを State として保持する．

- `local_backend.py`
  - `datasets/`
  - `logs/`
  - `scripts/`
  - 一部 `outputs/`

- `methods.py`
  - `outputs/`
  - `logs/`
  - `models/`
  - `scripts/`

## 7．データの流れ

代表的なデータの流れは以下のとおりである．

1. 入力画像または動画から `datasets/<dataset_name>/` を作成する．
2. 必要に応じて COLMAP 前処理を実行し，対応する構造を同データセット配下へ保存する．
3. 再構築手法を実行し，`outputs/` 配下へ結果を保存する．また，実行ログを`logs/` 配下へ保存する．
4. 出力ディレクトリ内の 3D モデルを viewer が参照する．
5. render / evaluation が `outputs/` 配下の結果を参照して追加出力を生成する．