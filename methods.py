import os
import cv2
import time
import shutil
import subprocess
import platform

from local_backend import get_imagelist, evaluate_all_metrics

# subprocessのshellフラグの設定
SHELL_FLAG = platform.system() == "Windows"

# 保存先一時ディレクトリ
TMPDIR = ""


# =========================
# 共通関数
# =========================

def msg(lang, jp_msg, en_msg):
    """
    言語コードに応じて日本語または英語のメッセージを返す．

    Args:
        lang (str): 言語コード．`"jp"` のとき日本語，それ以外は英語を返す．
        jp_msg (str): 日本語メッセージ．
        en_msg (str): 英語メッセージ．

    Returns:
        str: 選択された言語のメッセージ．
    """
    return jp_msg if lang == "jp" else en_msg


def run_subprocess_popen(lang, cmd, workdir, log_dir=None):
    """
    サブプロセスを起動し，標準出力と標準エラー出力を逐次取得しながら，
    ログファイルへ保存する．

    Args:
        lang (str）: ログやステータス表示に使う言語コード．
            `"jp"` のとき日本語，それ以外は英語を用いる．
            デフォルトは `"jp"` ．
        cmd (list[str] | tuple[str, ...]): 実行するコマンド．
        workdir (str): コマンドを実行する作業ディレクトリ．
        log_dir (str | None, optional): ログ保存先ディレクトリ．
            `None` の場合は `TMPDIR/logs` を使用する．
            デフォルトは `None` ．

    Returns:
        tuple[str, str, str]: 以下を返す．
            - 実行時間．`HH:MM:SS` 形式．
            - 実行ステータス文字列．
            - ログ全文．
    """
    global SHELL_FLAG
    global TMPDIR

    start_time = time.time()
    
    cmd_str = " ".join(map(str, cmd))
    print(msg(lang, "実行中", "Running") + f": {cmd_str}")

    # ログ保存ディレクトリを決定する
    if log_dir is None:
        log_dir = os.path.join(TMPDIR, "logs")

    # 実行開始時刻をもとにログファイル名を決定する
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(start_time))
    log_path = os.path.join(log_dir, f"{timestamp}.log")

    # ログ内容を保持する
    log_lines = []

    # ログヘッダを作成する
    header = (
        f"[{msg(lang, 'コマンド', 'COMMAND')}]\n"
        f"{cmd_str}\n"
        f"{'-' * 60}\n"
    )

    # コマンドの実行
    try:
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"

        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=workdir,
            shell=SHELL_FLAG,
            bufsize=1,
        )

        with open(log_path, "w", encoding="utf-8") as log_file:
            # ログ先頭に実行コマンドを書き込む
            log_file.write(header)
            log_file.flush()
            log_lines.append(header)

            # 標準出力を逐次取得し，コンソール表示とログ保存を行う
            if process.stdout is not None:
                for line in process.stdout:
                    print(line, end="")
                    log_file.write(line)
                    log_file.flush()
                    log_lines.append(line)

        returncode = process.wait()
    except Exception as e:
        error_log = msg(
            lang,
            f"実行に失敗しました: {e}",
            f"Execution failed: {e}"
        )
        status = msg(lang, "❌ 失敗（例外）", "❌ Failed (Exception)")
        return "", status, error_log

    end_time = time.time()

    # 実行時間を計算
    run_seconds = int(end_time - start_time)
    h, rem = divmod(run_seconds, 3600)
    m, s = divmod(rem, 60)
    run_time = f"{h:02d}:{m:02d}:{s:02d}"

    # 終了ステータスを決定
    if returncode == 0:
        status = msg(lang, "✅ 成功", "✅ Success")
    else:
        status = msg(lang, "❌ 失敗", "❌ Failed")

    # ログ末尾に結果を追記
    footer = (
        f"\n{'-' * 60}\n"
        f"[{msg(lang, 'ステータス', 'STATUS')}] {status}\n"
        f"[{msg(lang, '実行時間', 'ELAPSED TIME')}] {run_time}\n"
    )

    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(footer)

    log_lines.append(footer)
    full_log = "".join(log_lines)

    return run_time, status, full_log


# =========================
# Nerfstudio 共通関数
# =========================    

def train_nerfstudio(lang, dataset, outputs_dir, method_name, train_args=None):
    """
    Nerfstudio を用いてモデルを学習する．

    Args:
        lang (str): ログ出力に使用する言語コード．
            `"jp"` のとき日本語，それ以外は英語を用いる．
            デフォルトは `"jp"` ．
        dataset (str): 学習に使用するデータセットディレクトリのパス．
        outputs_dir (str): 学習結果の出力先ルートディレクトリ．
        method_name (str): 使用する Nerfstudio の手法名．
        train_args (list[str] | None, optional): `ns-train` に追加で渡す引数．
            デフォルトは `None` ．

    Returns:
        tuple[str, str, str, str]: 以下を返す．
            - 出力ディレクトリのパス．
            - 実行時間．
            - 実行ステータス．
            - 実行ログ全文．
    """
    # データセットのパス
    dataset = os.path.abspath(dataset)

    # 出力ディレクトリの作成
    name = os.path.basename(os.path.dirname(dataset))
    outdir = os.path.join(outputs_dir, method_name, name)
    os.makedirs(outdir, exist_ok=True)

    # 実行コマンド
    cmd = [
        "conda", "run", "--no-capture-output",
        "-n", "nerfstudio",
        "ns-train", method_name,
        "--output-dir", outdir,
        "--experiment-name", "results",
        "--timestamp", "results",
        "--vis", "viewer",
        "--viewer.quit-on-train-completion", "True",
    ]
    # 追加引数
    if train_args:
        cmd.extend(train_args)

    cmd.extend([
        "nerfstudio-data",
        "--data", dataset,
        "--downscale-factor", "1",
    ])

    # 学習の実行
    runtime, status, log = run_subprocess_popen(lang=lang, cmd=cmd, workdir=".", )

    return outdir, runtime, status, log


def train_nerfstudio_slurm(lang, dataset, outputs_dir, method_name, num_iters, port):
    """
    Slurm 経由で Nerfstudio の学習ジョブを投入する．

    Args:
        lang (str): ログ出力に使用する言語コード．
            `"jp"` のとき日本語，それ以外は英語を用いる．
            デフォルトは `"jp"` ．
        dataset (str): 学習に使用するデータセットディレクトリのパス．
        outputs_dir (str): 学習結果の出力先ルートディレクトリ．
        method_name (str): 使用する Nerfstudio の手法名．
        num_iters (int): 学習反復回数．
        port (int): Viewer などで使用するポート番号．

    Returns:
        tuple[str, str, str, str]: 以下を返す．
            - 出力ディレクトリのパス．
            - 実行時間．
            - 実行ステータス．
            - 実行ログ全文．
    """
    dataset = os.path.abspath(dataset)

    name = os.path.basename(os.path.dirname(dataset))
    outdir = os.path.join(outputs_dir, method_name, name)
    os.makedirs(outdir, exist_ok=True)

    sbatch_script = os.path.join("scripts", "recon_nerfstudio.sh")

    cmd = [
        "sbatch",
        f"--job-name={method_name}",
        sbatch_script,
        method_name,
        outdir,
        str(num_iters),
        str(port),
        dataset,
    ]

    runtime, status, log = run_subprocess_popen(
        lang=lang,
        cmd=cmd,
        workdir=".",
    )

    return outdir, runtime, status, log


def export_nerfstudio(lang, outdir, method_name, filetype, export_args=None):
    """
    学習済みの Nerfstudio モデルをエクスポートする．

    Args:
        lang (str): ログ出力に使用する言語コード．
            `"jp"` のとき日本語，それ以外は英語を用いる．
        outdir (str): 学習結果が保存されている出力ディレクトリ．
        method_name (str): 使用した Nerfstudio の手法名．
        filetype (str): `ns-export` に指定するエクスポート形式．
        export_args (list[str] | None, optional): `ns-export` に追加で渡す引数．

    Returns:
        tuple[str, str, str, str, str]: 以下を返す．
            - 出力ディレクトリのパス．
            - 実行時間．
            - 実行ステータス．
            - 実行ログ全文．
            - エクスポートされたモデルファイルの想定パス．
    """
    config_path = os.path.join(
        outdir, "results", method_name, "results", "config.yml"
    )

    cmd = [
        "conda", "run", "--no-capture-output",
        "-n", "nerfstudio",
        "ns-export", filetype,
        "--load-config", config_path,
        "--output-dir", outdir,
    ]

    if export_args:
        cmd.extend(export_args)

    run_time, status, log = run_subprocess_popen(
        lang,
        cmd,
        ".",
    )

    outmodel = os.path.join(outdir, "point_cloud.ply")

    return outdir, run_time, status, log, outmodel


def export_nerfstudio_slurm(lang, outdir, method_name, filetype, export_args=None):
    """
    学習済みの Nerfstudio モデルを Slurm 環境でエクスポートする．

    Args:
        lang (str): ログ出力に使用する言語コード．
            `"jp"` のとき日本語，それ以外は英語を用いる．
        outdir (str): 学習結果が保存されている出力ディレクトリ．
        method_name (str): 使用した Nerfstudio の手法名．
        filetype (str): `ns-export` に指定するエクスポート形式．
        export_args (list[str] | None, optional): `ns-export` に追加で渡す引数．

    Returns:
        tuple[str, str, str, str]: 以下を返す．
            - 出力ディレクトリのパス．
            - 実行時間．
            - 実行ステータス．
            - 実行ログ全文．
    """
    config_path = os.path.join(
        outdir, "results", method_name, "results", "config.yml"
    )

    cmd = [
        "conda", "run", "-n", "nerfstudio",
        "ns-export", filetype,
        "--load-config", config_path,
        "--output-dir", outdir,
    ]

    if export_args:
        cmd.extend(export_args)

    run_time, status, log = run_subprocess_popen(
        lang,
        cmd,
        ".",
    )

    return outdir, run_time, status, log


def normalize_render_output(base_dir, name, exts=(".png", ".jpg", ".jpeg")):
    """
    レンダリング出力を必ず `test/<name>/` 配下に統一する．

    想定する出力形式は以下の 2 通りである．
    
    1. 複数枚出力:
       `test/<name>/*.jpg`
    2. 単一ファイル出力:
       `test/<name>.jpg` など

    単一ファイルの場合は，`test/<name>/<name>.jpg` となるように
    ディレクトリ配下へ移動して形式をそろえる．

    Args:
        base_dir (str): `test` ディレクトリのパス．
        name (str): レンダリング出力名．
        exts (tuple[str, ...], optional): 対象とする画像拡張子．

    Returns:
        tuple[list[str], str | None]: 以下を返す．
            - 画像ファイルパスのリスト．
            - 正規化後のディレクトリパス．見つからない場合は `None` ．
    """
    dir_path = os.path.join(base_dir, name)

    # すでにディレクトリ形式で出力されている場合．
    if os.path.isdir(dir_path):
        files = sorted(
            os.path.join(dir_path, f)
            for f in os.listdir(dir_path)
            if f.lower().endswith(exts)
        )
        if files:
            return files, dir_path

    # 単一ファイル形式で出力されている場合．
    for ext in exts:
        flat_file = os.path.join(base_dir, f"{name}{ext}")
        if os.path.isfile(flat_file):
            os.makedirs(dir_path, exist_ok=True)

            dst_file = os.path.join(dir_path, os.path.basename(flat_file))
            if os.path.exists(dst_file):
                os.remove(dst_file)

            shutil.move(flat_file, dst_file)
            return [dst_file], dir_path

    # 出力が見つからない場合．
    return [], None


def render_eval_nerfstudio(lang, outdir, method_name, gt_name, pred_name):
    """
    Nerfstudio のレンダリング結果を用いて評価を行う．

    学習済みの `config.yml` を用いて GT と予測画像をレンダリングし，
    評価指標を計算したうえで，表示用ギャラリーを作成する．

    Args:
        lang (str): ログ出力に使用する言語コード．
            `"jp"` のとき日本語，それ以外は英語を用いる．
        outdir (str): 学習結果が保存されている出力ディレクトリ．
        method_name (str): 使用した Nerfstudio の手法名．
        gt_name (str): GT 側のレンダリング出力名．
        pred_name (str): 予測側のレンダリング出力名．

    Returns:
        tuple[str, str, str, str, list, list]: 以下を返す．
            - 出力ディレクトリのパス．
            - 実行時間．
            - 実行ステータス．
            - 実行ログ全文．
            - 評価指標の要約リスト．
            - ギャラリー表示用の画像パスリスト．
    """
    config_path = os.path.join(
        outdir, "results", method_name, "results", "config.yml"
    )
    test_dir = os.path.join(outdir, "test")

    gt_cmd = [
        "conda", "run", "--no-capture-output",
        "-n", "nerfstudio",
        "ns-render", "dataset",
        "--load-config", config_path,
        "--rendered-output-names", gt_name,
        "--output-path", outdir,
    ]
    pred_cmd = [
        "conda", "run", "--no-capture-output",
        "-n", "nerfstudio",
        "ns-render", "dataset",
        "--load-config", config_path,
        "--rendered-output-names", pred_name,
        "--output-path", outdir,
    ]

    gt_run_time, gt_status, gt_log = run_subprocess_popen(lang, gt_cmd, ".")
    pred_run_time, pred_status, pred_log = run_subprocess_popen(lang, pred_cmd, ".")

    render_log = (
        f"[GT Render]\n{gt_log}\n\n"
        f"[Pred Render]\n{pred_log}"
    )

    if "❌" in gt_status:
        return outdir, gt_run_time, gt_status, render_log, [], []
    if "❌" in pred_status:
        return outdir, pred_run_time, pred_status, render_log, [], []

    gt_files, gt_dir = normalize_render_output(test_dir, gt_name)
    pred_files, pred_dir = normalize_render_output(test_dir, pred_name)

    if not gt_files:
        full_log = render_log + "\n\n" + msg(
            lang,
            f"GT 画像が見つかりません: {gt_name}",
            f"GT images were not found: {gt_name}",
        )
        return outdir, "", msg(lang, "❌ 失敗", "❌ Failed"), full_log, [], []

    if not pred_files:
        full_log = render_log + "\n\n" + msg(
            lang,
            f"予測画像が見つかりません: {pred_name}",
            f"Predicted images were not found: {pred_name}",
        )
        return outdir, "", msg(lang, "❌ 失敗", "❌ Failed"), full_log, [], []

    if len(gt_files) != len(pred_files):
        count_log = msg(
            lang,
            "GT と予測の画像枚数が一致しません．\n"
            f"{gt_name}: {len(gt_files)} 枚\n"
            f"{pred_name}: {len(pred_files)} 枚",
            "The number of GT and predicted images does not match.\n"
            f"{gt_name}: {len(gt_files)} images\n"
            f"{pred_name}: {len(pred_files)} images",
        )
        full_log = render_log + "\n\n" + count_log
        return outdir, "", msg(lang, "❌ 失敗", "❌ Failed"), full_log, [], []

    run_time, status, full_log, summary_list = evaluate_all_metrics(
        lang,
        method_name,
        gt_dir,
        pred_dir,
        outdir,
    )

    full_log = render_log + "\n\n" + full_log

    gallery = []
    for gt_file, pred_file in zip(gt_files, pred_files):
        gallery.append(gt_file)
        gallery.append(pred_file)

    return outdir, run_time, status, full_log, summary_list, gallery


def render_eval_nerfstudio_slurm(lang, outdir, method_name, gt_name, pred_name):
    """
    Nerfstudio のレンダリング結果を用いて評価を行う．

    この関数は Slurm 用の呼び出し口として用いるが，処理内容は通常版と同様に，
    GT と予測画像のレンダリング，評価指標の計算，ギャラリー作成を行う．

    Args:
        lang (str): ログ出力に使用する言語コード．
            `"jp"` のとき日本語，それ以外は英語を用いる．
        outdir (str): 学習結果が保存されている出力ディレクトリ．
        method_name (str): 使用した Nerfstudio の手法名．
        gt_name (str): GT 側のレンダリング出力名．
        pred_name (str): 予測側のレンダリング出力名．

    Returns:
        tuple[str, str, str, str, list, list]: 以下を返す．
            - 出力ディレクトリのパス．
            - 実行時間．
            - 実行ステータス．
            - 実行ログ全文．
            - 評価指標の要約リスト．
            - ギャラリー表示用の画像パスリスト．
    """
    config_path = os.path.join(
        outdir, "results", method_name, "results", "config.yml"
    )
    test_dir = os.path.join(outdir, "test")

    gt_cmd = [
        "conda", "run", "-n", "nerfstudio",
        "ns-render", "dataset",
        "--load-config", config_path,
        "--rendered-output-names", gt_name,
        "--output-path", outdir,
    ]
    pred_cmd = [
        "conda", "run", "-n", "nerfstudio",
        "ns-render", "dataset",
        "--load-config", config_path,
        "--rendered-output-names", pred_name,
        "--output-path", outdir,
    ]

    gt_run_time, gt_status, gt_log = run_subprocess_popen(lang, gt_cmd, ".")
    pred_run_time, pred_status, pred_log = run_subprocess_popen(lang, pred_cmd, ".")

    render_log = (
        f"[GT Render]\n{gt_log}\n\n"
        f"[Pred Render]\n{pred_log}"
    )

    if "❌" in gt_status:
        return outdir, gt_run_time, gt_status, render_log, [], []
    if "❌" in pred_status:
        return outdir, pred_run_time, pred_status, render_log, [], []

    gt_files, gt_dir = normalize_render_output(test_dir, gt_name)
    pred_files, pred_dir = normalize_render_output(test_dir, pred_name)

    if not gt_files:
        full_log = render_log + "\n\n" + msg(
            lang,
            f"GT 画像が見つかりません: {gt_name}",
            f"GT images were not found: {gt_name}",
        )
        return outdir, "", msg(lang, "❌ 失敗", "❌ Failed"), full_log, [], []

    if not pred_files:
        full_log = render_log + "\n\n" + msg(
            lang,
            f"予測画像が見つかりません: {pred_name}",
            f"Predicted images were not found: {pred_name}",
        )
        return outdir, "", msg(lang, "❌ 失敗", "❌ Failed"), full_log, [], []

    if len(gt_files) != len(pred_files):
        count_log = msg(
            lang,
            "GT と予測の画像枚数が一致しません．\n"
            f"{gt_name}: {len(gt_files)} 枚\n"
            f"{pred_name}: {len(pred_files)} 枚",
            "The number of GT and predicted images does not match.\n"
            f"{gt_name}: {len(gt_files)} images\n"
            f"{pred_name}: {len(pred_files)} images",
        )
        full_log = render_log + "\n\n" + count_log
        return outdir, "", msg(lang, "❌ 失敗", "❌ Failed"), full_log, [], []

    run_time, status, full_log, summary_list = evaluate_all_metrics(
        method_name,
        gt_dir,
        pred_dir,
        outdir,
    )

    full_log = render_log + "\n\n" + full_log

    gallery = []
    for gt_file, pred_file in zip(gt_files, pred_files):
        gallery.append(gt_file)
        gallery.append(pred_file)

    return outdir, run_time, status, full_log, summary_list, gallery


# =========================
# Vanilla-NeRF
# =========================    

def recon_vnerf(lang, mode, dataset, outdir, num_iters):
    """
    Vanilla NeRF による再構築を実行する．

    Args:
        lang (str): ログ出力に使用する言語コード．
            `"jp"` のとき日本語，それ以外は英語を用いる．
        mode (str): 実行モード．`"local"` または `"slurm"` を指定する．
        dataset (str): 入力データセットのパス．
        outdir (str): 出力先ディレクトリのパス．
        num_iters (int): 学習反復回数．

    Returns:
        tuple: 学習関数の戻り値．
    """
    port = 7007

    if mode == "local":
        train_args = [
            "--max-num-iterations", str(num_iters),
            "--viewer.websocket-port-default", str(port),
        ]
        return train_nerfstudio(lang, dataset, outdir, "vanilla-nerf", train_args, )
    elif mode == "slurm":
        return train_nerfstudio_slurm(lang, dataset, outdir, "vanilla-nerf", num_iters, port, )


def export_vnerf(lang, mode, outdir):
    """
    Vanilla NeRF の学習結果を点群としてエクスポートする．

    Args:
        lang (str): ログ出力に使用する言語コード．
            `"jp"` のとき日本語，それ以外は英語を用いる．
        mode (str): 実行モード．`"local"` または `"slurm"` を指定する．
        outdir (str): 学習結果が保存された出力ディレクトリのパス．

    Returns:
        tuple: エクスポート関数の戻り値．
    """
    export_args = [
        "--normal-method", "open3d",
        "--rgb-output-name", "rgb_fine",
        "--depth-output-name", "depth_fine",
    ]

    if mode == "local":
        return export_nerfstudio(lang, outdir, "vanilla-nerf", "pointcloud", export_args, )
    elif mode == "slurm":
        return export_nerfstudio_slurm(lang, outdir, "vanilla-nerf", "pointcloud", export_args, )


def render_eval_vnerf(lang, mode, outdir):
    """
    Vanilla NeRF のレンダリング結果を用いて評価を行う．

    Args:
        lang (str): ログ出力に使用する言語コード．
            `"jp"` のとき日本語，それ以外は英語を用いる．
        mode (str): 実行モード．`"local"` または `"slurm"` を指定する．
        outdir (str): 学習結果が保存された出力ディレクトリのパス．

    Returns:
        tuple: 評価関数の戻り値．
    """
    gt_name = "gt-rgb"
    pred_name = "rgb_fine"

    if mode == "local":
        return render_eval_nerfstudio(
            lang,
            outdir,
            "vanilla-nerf",
            gt_name,
            pred_name,
        )

    if mode == "slurm":
        return render_eval_nerfstudio_slurm(
            lang,
            outdir,
            "vanilla-nerf",
            gt_name,
            pred_name,
        )

    raise ValueError(
        msg(
            lang,
            f"無効な mode です: {mode}",
            f"Invalid mode: {mode}",
        )
    )


# =========================
# Nerfacto
# =========================    

def recon_nerfacto(lang, mode, dataset, outdir, num_iters):
    """
    Nerfacto による再構築を実行する．

    Args:
        lang (str): ログ出力に使用する言語コード．
            `"jp"` のとき日本語，それ以外は英語を用いる．
        mode (str): 実行モード．`"local"` または `"slurm"` を指定する．
        dataset (str): 入力データセットのパス．
        outdir (str): 出力先ディレクトリのパス．
        num_iters (int): 学習反復回数．

    Returns:
        tuple: 学習関数の戻り値．
    """
    port = 7008

    if mode == "local":
        train_args = [
            "--max-num-iterations", str(num_iters),
            "--viewer.websocket-port-default", str(port),
        ]
        return train_nerfstudio(lang, dataset, outdir, "nerfacto-huge", train_args)
    elif mode == "slurm":
        return train_nerfstudio_slurm(lang, dataset, outdir, "nerfacto-huge", num_iters, port)


def export_nerfacto(lang, mode, outdir):
    """
    Nerfacto の学習結果を点群としてエクスポートする．

    Args:
        lang (str): ログ出力に使用する言語コード．
            `"jp"` のとき日本語，それ以外は英語を用いる．
        mode (str): 実行モード．`"local"` または `"slurm"` を指定する．
        outdir (str): 学習結果が保存された出力ディレクトリのパス．

    Returns:
        tuple: エクスポート関数の戻り値．
    """
    export_args = [
        "--normal-method", "open3d",
        "--rgb-output-name", "rgb",
        "--depth-output-name", "depth",
    ]

    if mode == "local":
        return export_nerfstudio(lang, outdir, "nerfacto", "pointcloud", export_args)
    elif mode == "slurm":
        return export_nerfstudio_slurm(lang, outdir, "nerfacto", "pointcloud", export_args)


def render_eval_nerfacto(lang, mode, outdir):
    """
    Nerfacto のレンダリング結果を用いて評価を行う．

    Args:
        lang (str): ログ出力に使用する言語コード．
            `"jp"` のとき日本語，それ以外は英語を用いる．
        mode (str): 実行モード．`"local"` または `"slurm"` を指定する．
        outdir (str): 学習結果が保存された出力ディレクトリのパス．

    Returns:
        tuple: 評価関数の戻り値．
    """
    if mode == "local":
        return render_eval_nerfstudio(lang, outdir, "nerfacto", "gt-rgb", "rgb")
    elif mode == "slurm":
        return render_eval_nerfstudio_slurm(lang, outdir, "nerfacto", "gt-rgb", "rgb")


# =========================
# mip-NeRF
# =========================    

def recon_mipnerf(lang, mode, dataset, outdir, num_iters):
    """
    Mip-NeRF による再構築を実行する．

    Args:
        lang (str): ログ出力に使用する言語コード．
            `"jp"` のとき日本語，それ以外は英語を用いる．
        mode (str): 実行モード．`"local"` または `"slurm"` を指定する．
        dataset (str): 入力データセットのパス．
        outdir (str): 出力先ディレクトリのパス．
        num_iters (int): 学習反復回数．

    Returns:
        tuple: 学習関数の戻り値．
    """
    port = 7009

    if mode == "local":
        train_args = [
            "--max-num-iterations", str(num_iters),
            "--viewer.websocket-port-default", str(port),
        ]
        return train_nerfstudio(lang, dataset, outdir, "mipnerf", train_args)
    elif mode == "slurm":
        return train_nerfstudio_slurm(lang, dataset, outdir, "mipnerf", num_iters, port)


def export_mipnerf(lang, mode, outdir):
    """
    Mip-NeRF の学習結果を点群としてエクスポートする．

    Args:
        lang (str): ログ出力に使用する言語コード．
            `"jp"` のとき日本語，それ以外は英語を用いる．
        mode (str): 実行モード．`"local"` または `"slurm"` を指定する．
        outdir (str): 学習結果が保存された出力ディレクトリのパス．

    Returns:
        tuple: エクスポート関数の戻り値．
    """
    export_args = [
        "--normal-method", "open3d",
        "--rgb-output-name", "rgb_fine",
        "--depth-output-name", "depth_fine",
    ]

    if mode == "local":
        return export_nerfstudio(lang, outdir, "mipnerf", "pointcloud", export_args)
    elif mode == "slurm":
        return export_nerfstudio_slurm(lang, outdir, "mipnerf", "pointcloud", export_args)


def render_eval_mipnerf(lang, mode, outdir):
    """
    Mip-NeRF のレンダリング結果を用いて評価を行う．

    Args:
        lang (str): ログ出力に使用する言語コード．
            `"jp"` のとき日本語，それ以外は英語を用いる．
        mode (str): 実行モード．`"local"` または `"slurm"` を指定する．
        outdir (str): 学習結果が保存された出力ディレクトリのパス．

    Returns:
        tuple: 評価関数の戻り値．
    """
    if mode == "local":
        return render_eval_nerfstudio(lang, outdir, "mipnerf", "gt-rgb", "rgb_fine")
    elif mode == "slurm":
        return render_eval_nerfstudio_slurm(lang, outdir, "mipnerf", "gt-rgb", "rgb_fine")
    

# =========================
# SeaThru-NeRF
# =========================    

def recon_stnerf(lang, mode, dataset, outdir, num_iters):
    """
    SeaThru-NeRF による再構築を実行する．

    Args:
        lang (str): ログ出力に使用する言語コード．
            `"jp"` のとき日本語，それ以外は英語を用いる．
        mode (str): 実行モード．`"local"` または `"slurm"` を指定する．
        dataset (str): 入力データセットのパス．
        outdir (str): 出力先ディレクトリのパス．
        num_iters (int): 学習反復回数．

    Returns:
        tuple: 学習関数の戻り値．
    """
    port = 7010

    if mode == "local":
        train_args = [
            "--max-num-iterations", str(num_iters),
            "--viewer.websocket-port-default", str(port),
        ]
        return train_nerfstudio(lang, dataset, outdir, "seathru-nerf", train_args)
    elif mode == "slurm":
        return train_nerfstudio_slurm(lang, dataset, outdir, "seathru-nerf", num_iters, port)


def export_stnerf(lang, mode, outdir):
    """
    SeaThru-NeRF の学習結果を点群としてエクスポートする．

    Args:
        lang (str): ログ出力に使用する言語コード．
            `"jp"` のとき日本語，それ以外は英語を用いる．
        mode (str): 実行モード．`"local"` または `"slurm"` を指定する．
        outdir (str): 学習結果が保存された出力ディレクトリのパス．

    Returns:
        tuple: エクスポート関数の戻り値．
    """
    export_args = [
        "--normal-method", "open3d",
        "--rgb-output-name", "rgb",
        "--depth-output-name", "depth",
    ]

    if mode == "local":
        return export_nerfstudio(lang, outdir, "seathru-nerf", "pointcloud", export_args)
    elif mode == "slurm":
        return export_nerfstudio_slurm(lang, outdir, "seathru-nerf", "pointcloud", export_args)


def render_eval_stnerf(lang, mode, outdir):
    """
    SeaThru-NeRF のレンダリング結果を用いて評価を行う．

    Args:
        lang (str): ログ出力に使用する言語コード．
            `"jp"` のとき日本語，それ以外は英語を用いる．
        mode (str): 実行モード．`"local"` または `"slurm"` を指定する．
        outdir (str): 学習結果が保存された出力ディレクトリのパス．

    Returns:
        tuple: 評価関数の戻り値．
    """
    if mode == "local":
        return render_eval_nerfstudio(lang, outdir, "seathru-nerf", "gt-rgb", "rgb")
    elif mode == "slurm":
        return render_eval_nerfstudio_slurm(lang, outdir, "seathru-nerf", "gt-rgb", "rgb")


# =========================
# Vanilla-GS
# =========================   

def recon_vgs(lang, mode, dataset, outputs_dir, save_iter):
    """
    Vanilla-GS による再構築を実行する．

    Args:
        lang (str): ログ出力に使用する言語コード．
            `"jp"` のとき日本語，それ以外は英語を用いる．
        mode (str): 実行モード．`"local"` または `"slurm"` を指定する．
        dataset (str): 入力データセットのパス．
        outputs_dir (str): 出力先ルートディレクトリのパス．
        save_iter (int): 保存対象の反復回数．

    Returns:
        tuple[str, str, str, str, str]: 以下を返す．
            - 出力ディレクトリのパス．
            - 実行時間．
            - 実行ステータス．
            - 実行ログ全文．
            - 学習結果ディレクトリのパス．
    """
    # データセットのパス
    dataset = os.path.abspath(dataset)

    # 出力ディレクトリの作成
    name = os.path.basename(os.path.dirname(dataset))
    outdir = os.path.join(outputs_dir, "3dgs", name)
    os.makedirs(outdir, exist_ok=True)

    if mode == "local":
        recon_script = "train.py"

        # 実行コマンド
        cmd = [
            "conda", "run", "--no-capture-output",
            "-n", "gaussian_splatting", "python", recon_script,
            "--source_path", dataset,
            "--model_path", outdir,
            "--iterations", str(save_iter),
            "--save_iterations", str(save_iter),
            "--eval",
        ]

        # 作業ディレクトリ
        workdir = os.path.join("models", "gaussian-splatting")

    elif mode == "slurm":
        sbatch_script = os.path.join("scripts", "recon_vanillags.sh")
        recon_script = os.path.join("models", "gaussian-splatting", "train.py")
        cmd = ["sbatch", sbatch_script, recon_script, dataset, outdir, str(save_iter)]
        workdir = "."

    # 学習の実行
    runtime, status, log = run_subprocess_popen(lang, cmd, workdir)

    # 3D モデル
    outmodel = os.path.join(outdir, "point_cloud", f"iteration_{save_iter}", "point_cloud.ply")

    return outdir, runtime, status, log, outmodel


def render_eval_3dgs(lang, model_dir, skip_train, skip_test, iteration):
    """
    Vanilla-GS のレンダリングと評価を実行する．

    Args:
        lang (str): ログ出力に使用する言語コード．
            `"jp"` のとき日本語，それ以外は英語を用いる．
        model_dir (str): Gaussian Splatting の学習結果ディレクトリのパス．
        skip_train (bool): 学習用ビューのレンダリングを省略するかどうか．
        skip_test (bool): テスト用ビューのレンダリングを省略するかどうか．
        iteration (int | None): レンダリング対象の反復回数．

    Returns:
        tuple[str, str, str, str, list, list]: 以下を返す．
            - モデルディレクトリのパス．
            - 実行時間．
            - 実行ステータス．
            - 実行ログ全文．
            - 評価指標の要約リスト．
            - ギャラリー表示用の画像パスリスト．
    """
    # 作業ディレクトリ
    workdir = os.path.join("models", "gaussian-splatting")

    render_script = "render.py"

    render_cmd = [
        "conda", "run", "--no-capture-output",
        "-n", "gaussian_splatting", "python", render_script,
        "--model_path", model_dir,
    ]
    if skip_train:
        render_cmd.append("--skip_train")
    if skip_test:
        render_cmd.append("--skip_test")
    if iteration is not None:
        render_cmd.extend(["--iteration", str(iteration)])

    # レンダリングの実行
    runtime_r, status_r, log_r = run_subprocess_popen(lang, render_cmd, workdir)

    success_status = msg(lang, "✅ 成功", "✅ Success")
    failed_status = msg(lang, "❌ 失敗", "❌ Failed")

    if status_r != success_status:
        full_log = msg(
            lang,
            "レンダリングに失敗しました．\n\n" + log_r,
            "Rendering failed.\n\n" + log_r,
        )
        return model_dir, runtime_r, failed_status, full_log, [], []

    test_dir = os.path.join(model_dir, "test", f"ours_{iteration}")
    gt_dir = os.path.join(test_dir, "gt")
    pred_dir = os.path.join(test_dir, "renders")

    # 評価の実行
    run_time, status, full_log, summary_list = evaluate_all_metrics(
        lang,
        "gaussian-splatting",
        gt_dir,
        pred_dir,
        model_dir,
    )

    gt_files = sorted(
        os.path.join(gt_dir, f)
        for f in os.listdir(gt_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    pred_files = sorted(
        os.path.join(pred_dir, f)
        for f in os.listdir(pred_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )

    gallery = []
    for gt_file, pred_file in zip(gt_files, pred_files):
        gallery.append(gt_file)
        gallery.append(pred_file)

    return model_dir, run_time, status, full_log, summary_list, gallery


# =========================
# Mip-Splatting
# =========================

def recon_mipsplatting(lang, mode, dataset, outputs_dir, save_iter):
    """
    Mip-Splatting による再構築を実行する．

    Args:
        lang (str): ログ出力に使用する言語コード．
            `"jp"` のとき日本語，それ以外は英語を用いる．
        mode (str): 実行モード．`"local"` または `"slurm"` を指定する．
        dataset (str): 入力データセットのパス．
        outputs_dir (str): 出力先ルートディレクトリのパス．
        save_iter (int): 保存対象の反復回数．

    Returns:
        tuple[str, str, str, str, str]: 以下を返す．
            - 出力ディレクトリのパス．
            - 実行時間．
            - 実行ステータス．
            - 実行ログ全文．
            - 学習結果ディレクトリのパス．
    """
    # データセットのパス
    dataset = os.path.abspath(dataset)

    # 出力ディレクトリの作成
    name = os.path.basename(os.path.dirname(dataset))
    outdir = os.path.join(outputs_dir, "mip-splatting", name)
    os.makedirs(outdir, exist_ok=True)

    if mode == "local":
        recon_script = "train.py"

        cmd = [
            "conda", "run", "--no-capture-output",
            "-n", "mip-splatting", "python", recon_script,
            "-s", dataset,
            "-m", outdir,
            "--iterations", str(save_iter),
            "--test_iterations", str(save_iter),
            "--save_iterations", str(save_iter),
            "--eval",
        ]

        workdir = os.path.join("models", "mip-splatting")

    elif mode == "slurm":
        sbatch_script = os.path.join("scripts", "recon_mipsplatting.sh")
        recon_script = os.path.join("models", "mip-splatting", "train.py")
        cmd = ["sbatch", sbatch_script, recon_script, dataset, outdir, str(save_iter)]
        workdir = "."

    # 学習の実行
    runtime, status, log = run_subprocess_popen(lang, cmd, workdir)

    # 3D モデル
    outmodel = os.path.join(outdir, "point_cloud", f"iteration_{save_iter}", "point_cloud.ply")

    return outdir, runtime, status, log, outmodel


def render_eval_mips(lang, model_dir, skip_train, skip_test, iteration):
    """
    Mip-Splatting のレンダリングと評価を実行する．

    Args:
        lang (str): ログ出力に使用する言語コード．
            `"jp"` のとき日本語，それ以外は英語を用いる．
        model_dir (str): Mip-Splatting の学習結果ディレクトリのパス．
        skip_train (bool): 学習用ビューのレンダリングを省略するかどうか．
        skip_test (bool): テスト用ビューのレンダリングを省略するかどうか．
        iteration (int | None): レンダリング対象の反復回数．

    Returns:
        tuple[str, str, str, str, list, list]: 以下を返す．
            - モデルディレクトリのパス．
            - 実行時間．
            - 実行ステータス．
            - 実行ログ全文．
            - 評価指標の要約リスト．
            - ギャラリー表示用の画像パスリスト．
    """
    workdir = os.path.join("models", "mip-splatting")

    render_script = "render.py"
    render_cmd = [
        "conda", "run", "--no-capture-output",
        "-n", "mip-splatting", "python", render_script,
        "-m", model_dir,
    ]

    if skip_train:
        render_cmd.append("--skip_train")
    if skip_test:
        render_cmd.append("--skip_test")
    if iteration is not None:
        render_cmd.extend(["--iteration", str(iteration)])

    runtime_r, status_r, log_r = run_subprocess_popen(lang, render_cmd, workdir)

    success_status = msg(lang, "✅ 成功", "✅ Success")
    failed_status = msg(lang, "❌ 失敗", "❌ Failed")

    if status_r != success_status:
        full_log = msg(
            lang,
            "レンダリングに失敗しました．\n\n" + log_r,
            "Rendering failed.\n\n" + log_r,
        )
        return model_dir, runtime_r, failed_status, full_log, [], []

    test_dir = os.path.join(model_dir, "test", f"ours_{iteration}")
    gt_dir = os.path.join(test_dir, "gt_-1")
    pred_dir = os.path.join(test_dir, "test_preds_-1")

    run_time, status, full_log, summary_list = evaluate_all_metrics(
        lang,
        "mip-splatting",
        gt_dir,
        pred_dir,
        model_dir,
    )

    gt_files = sorted(
        os.path.join(gt_dir, f)
        for f in os.listdir(gt_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    pred_files = sorted(
        os.path.join(pred_dir, f)
        for f in os.listdir(pred_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )

    gallery = []
    for gt_file, pred_file in zip(gt_files, pred_files):
        gallery.append(gt_file)
        gallery.append(pred_file)

    return model_dir, run_time, status, full_log, summary_list, gallery


# =========================
# Splatfacto
# =========================

def recon_sfacto(lang, mode, dataset, outdir, num_iters):
    """
    Splatfacto による再構築を実行する．

    Args:
        lang (str): ログ出力に使用する言語コード．
            `"jp"` のとき日本語，それ以外は英語を用いる．
        mode (str): 実行モード．`"local"` または `"slurm"` を指定する．
        dataset (str): 入力データセットのパス．
        outdir (str): 出力先ディレクトリのパス．
        num_iters (int): 学習反復回数．

    Returns:
        tuple: 学習関数の戻り値．
    """
    port = 7011

    if mode == "local":
        train_args = [
            "--max-num-iterations", str(num_iters),
            "--viewer.websocket-port-default", str(port),
        ]
        return train_nerfstudio(lang, dataset, outdir, "splatfacto-big", train_args)
    elif mode == "slurm":
        return train_nerfstudio_slurm(lang, dataset, outdir, "splatfacto-big", num_iters, port)


def export_sfacto(lang, mode, outdir):
    """
    Splatfacto の学習結果をエクスポートする．

    Args:
        lang (str): ログ出力に使用する言語コード．
            `"jp"` のとき日本語，それ以外は英語を用いる．
        mode (str): 実行モード．`"local"` または `"slurm"` を指定する．
        outdir (str): 学習結果が保存された出力ディレクトリのパス．

    Returns:
        tuple: エクスポート関数の戻り値．
    """
    export_args = []

    if mode == "local":
        return export_nerfstudio(lang, outdir, "splatfacto", "gaussian-splat", export_args)
    elif mode == "slurm":
        return export_nerfstudio_slurm(lang, outdir, "splatfacto", "gaussian-splat", export_args)


def render_eval_sfacto(lang, mode, outdir):
    """
    Splatfacto のレンダリング結果を用いて評価を行う．

    Args:
        lang (str): ログ出力に使用する言語コード．
            `"jp"` のとき日本語，それ以外は英語を用いる．
        mode (str): 実行モード．`"local"` または `"slurm"` を指定する．
        outdir (str): 学習結果が保存された出力ディレクトリのパス．

    Returns:
        tuple: 評価関数の戻り値．
    """
    if mode == "local":
        return render_eval_nerfstudio(lang, outdir, "splatfacto", "gt-rgb", "rgb")
    elif mode == "slurm":
        return render_eval_nerfstudio_slurm(lang, outdir, "splatfacto", "gt-rgb", "rgb")


# =========================
# 4D-Gaussians
# =========================

def recon_4dgaussians(lang, mode, dataset, outputs_dir, save_iter):
    """
    4D-Gaussians による再構築を実行する．

    Args:
        lang (str): ログ出力に使用する言語コード．
            `"jp"` のとき日本語，それ以外は英語を用いる．
        mode (str): 実行モード．`"local"` または `"slurm"` を指定する．
        dataset (str): 入力データセットのパス．
        outputs_dir (str): 出力先ルートディレクトリのパス．
        save_iter (int): 保存対象の反復回数．

    Returns:
        tuple[str, str, str, str, str]: 以下を返す．
            - 出力ディレクトリのパス．
            - 実行時間．
            - 実行ステータス．
            - 実行ログ全文．
            - 学習結果ディレクトリのパス．
    """
    # データセットの作成
    dataset = os.path.abspath(dataset)

    # 出力ディレクトリの作成
    name = os.path.basename(os.path.dirname(dataset))
    outdir = os.path.join(outputs_dir, "4D-Gaussians", name)
    os.makedirs(outdir, exist_ok=True)

    if mode == "local":
        recon_script = "train.py"

        cmd = [
            "conda", "run", "--no-capture-output",
            "-n", "Gaussians4D", "python", recon_script,
            "--source_path", dataset,
            "--model_path", outdir,
            "--iterations", str(save_iter),
            "--test_iterations", str(save_iter),
            "--save_iterations", str(save_iter),
            "--eval",
        ]

        workdir = os.path.join("models", "4DGaussians")
    elif mode == "slurm":
        sbatch_script = os.path.join("scripts", "recon_4dgaussians.sh")
        recon_script = os.path.join("models", "4DGaussians", "train.py")
        cmd = ["sbatch", sbatch_script, recon_script, dataset, outdir, str(save_iter)]
        workdir = "."

    # 学習の実行
    runtime, status, log = run_subprocess_popen(lang, cmd, workdir)

    # 3D モデル
    outmodel = os.path.join(outdir, "point_cloud", f"iteration_{save_iter}", "point_cloud.ply")

    return outdir, runtime, status, log, outmodel


def render_eval_4dgs(lang, model_dir, skip_train, skip_test, iteration):
    """
    4D-Gaussians のレンダリングと評価を実行する．

    Args:
        lang (str): ログ出力に使用する言語コード．
            `"jp"` のとき日本語，それ以外は英語を用いる．
        model_dir (str): 4D-Gaussians の学習結果ディレクトリのパス．
        skip_train (bool): 学習用ビューのレンダリングを省略するかどうか．
        skip_test (bool): テスト用ビューのレンダリングを省略するかどうか．
        iteration (int | None): レンダリング対象の反復回数．

    Returns:
        tuple[str, str, str, str, list, list]: 以下を返す．
            - モデルディレクトリのパス．
            - 実行時間．
            - 実行ステータス．
            - 実行ログ全文．
            - 評価指標の要約リスト．
            - ギャラリー表示用の画像パスリスト．
    """
    workdir = os.path.join("models", "4DGaussians")

    render_script = "render.py"

    render_cmd = [
        "conda", "run", "--no-capture-output",
        "-n", "Gaussians4D", "python", render_script,
        "-m", model_dir,
    ]

    if skip_train:
        render_cmd.append("--skip_train")
    if skip_test:
        render_cmd.append("--skip_test")
    if iteration is not None:
        render_cmd.extend(["--iteration", str(iteration)])

    # レンダリングの実行
    runtime_r, status_r, log_r = run_subprocess_popen(lang, render_cmd, workdir)

    success_status = msg(lang, "✅ 成功", "✅ Success")
    failed_status = msg(lang, "❌ 失敗", "❌ Failed")

    if status_r != success_status:
        full_log = msg(
            lang,
            "レンダリングに失敗しました．\n\n" + log_r,
            "Rendering failed.\n\n" + log_r,
        )
        return model_dir, runtime_r, failed_status, full_log, [], []

    test_dir = os.path.join(model_dir, "test", f"ours_{iteration}")
    gt_dir = os.path.join(test_dir, "gt")
    pred_dir = os.path.join(test_dir, "renders")

    # 評価の実行
    run_time, status, full_log, summary_list = evaluate_all_metrics(
        lang,
        "4d-gaussians",
        gt_dir,
        pred_dir,
        model_dir,
    )

    gt_files = sorted(
        os.path.join(gt_dir, f)
        for f in os.listdir(gt_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    pred_files = sorted(
        os.path.join(pred_dir, f)
        for f in os.listdir(pred_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )

    gallery = []
    for gt_file, pred_file in zip(gt_files, pred_files):
        gallery.append(gt_file)
        gallery.append(pred_file)

    return model_dir, run_time, status, full_log, summary_list, gallery


# =========================
# DUSt3R
# =========================

def recon_dust3r(lang, mode, dataset, outputs_dir, schedule, niter, min_conf_thr, as_pointcloud, mask_sky,
                 clean_depth, transparent_cams, cam_size, scenegraph_type, winsize, refid):
    """
    DUSt3R による再構築を実行する．

    Args:
        lang (str): ログ出力に使用する言語コード．
            `"jp"` のとき日本語，それ以外は英語を用いる．
        mode (str): 実行モード．`"local"` または `"slurm"` を指定する．
        dataset (str): 入力データセットのパス．
        outputs_dir (str): 出力先ルートディレクトリのパス．
        schedule (str): 最適化スケジュール．
        niter (int): 反復回数．
        min_conf_thr (float): 最小信頼度しきい値．
        as_pointcloud (bool): 点群として出力するかどうか．
        mask_sky (bool): 空領域をマスクするかどうか．
        clean_depth (bool): 深度をクリーンアップするかどうか．
        transparent_cams (bool): カメラを半透明表示にするかどうか．
        cam_size (float): カメラ表示サイズ．
        scenegraph_type (str): シーングラフの種類．
        winsize (int): ウィンドウサイズ．
        refid (int): 参照画像 ID．

    Returns:
        tuple[str, str, str, str, str, str]: 以下を返す．
            - 出力ディレクトリのパス．
            - 実行時間．
            - 実行ステータス．
            - 実行ログ全文．
            - 再構築結果のパス．
            - レンダリング画像ディレクトリのパス．
    """
    # データセットのパス
    dataset = os.path.abspath(dataset)

    # 出力ディレクトリの作成
    name = os.path.basename(os.path.dirname(dataset))
    outdir = os.path.join(outputs_dir, "dust3r", name)
    os.makedirs(outdir, exist_ok=True)

    model_name = "DUSt3R_ViTLarge_BaseDecoder_512_dpt" # 使用モデル
    image_size = 512
    device = "cuda"

    if mode == "local":
        recon_script = os.path.join("scripts", "recon", "recon_dust3r.py")
        cmd = [
            "conda", "run", "--no-capture-output",
            "-n", "dust3r", "python", recon_script,
            "--lang", lang,
            "--model_name", model_name,
            "--device", device,
            "--outdir", outdir,
            "--image_size", str(image_size),
            "--filelist", dataset,
            "--schedule", schedule,
            "--niter", str(niter),
            "--min_conf_thr", str(min_conf_thr),
            "--cam_size", str(cam_size),
            "--scenegraph_type", scenegraph_type,
            "--winsize", str(winsize),
            "--refid", str(refid),
        ]
        if as_pointcloud:
            cmd.append("--as_pointcloud")
        if mask_sky:
            cmd.append("--mask_sky")
        if clean_depth:
            cmd.append("--clean_depth")
        if transparent_cams:
            cmd.append("--transparent_cams")
        workdir = "."
    elif mode == "slurm":
        sbatch_script = os.path.join("scripts", "recon_dust3r.sh")
        recon_script = os.path.join("scripts", "recon_dust3r.py")
        cmd = ["sbatch", sbatch_script, recon_script, model_name, device, outdir, str(image_size), dataset]
        workdir = "."

    # 再構築の実行
    runtime, status, log = run_subprocess_popen(lang, cmd, workdir)

    # 3Dモデル
    model_path = os.path.join(outdir, "scene.glb")

    # レンダリング画像
    outimgs = os.path.join(outdir, "render")

    return outdir, runtime, status, log, model_path, outimgs


# =========================
# MASt3R
# =========================

def recon_mast3r(lang, mode, dataset, outputs_dir):
    """
    MASt3R による再構築を実行する．

    Args:
        lang (str): ログ出力に使用する言語コード．
            `"jp"` のとき日本語，それ以外は英語を用いる．
        mode (str): 実行モード．`"local"` または `"slurm"` を指定する．
        dataset (str): 入力データセットのパス．
        outputs_dir (str): 出力先ルートディレクトリのパス．

    Returns:
        tuple[str, str, str, str, str]: 以下を返す．
            - 出力ディレクトリのパス．
            - 実行時間．
            - 実行ステータス．
            - 実行ログ全文．
            - 再構築結果のパス．
    """
    dataset = os.path.abspath(dataset)

    # 出力ディレクトリの作成
    name = os.path.basename(os.path.dirname(dataset))
    outdir = os.path.join(outputs_dir, "mast3r", name)
    os.makedirs(outdir, exist_ok=True)

    # 使用モデル
    model_name = "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"

    # データセットのパス
    filelist = get_imagelist(dataset)

    if mode == "local":
        recon_script = os.path.join("scripts", "recon", "recon_mast3r.py")
        cmd = [
            "conda", "run", "--no-capture-output",
            "-n", "mast3r", "python", recon_script,
            "--lang", lang,
            "--filelist", str(filelist),
            "--outdir", outdir,
            "--model_name", model_name,
            "--as_pointcloud",
        ]
        workdir = "."
    elif mode == "slurm":
        sbatch_script = os.path.join("scripts", "recon_mast3r.sh")
        recon_script = os.path.join("scripts", "recon_mast3r.py")
        cmd = ["sbatch", sbatch_script, recon_script, str(filelist), outdir, model_name]
        workdir = "."

    # 再構築の実行
    runtime, status, log = run_subprocess_popen(lang, cmd, workdir)

    # 3Dモデル
    model_path = os.path.join(outdir, "scene.glb")

    return outdir, runtime, status, log, model_path


# =========================
# MonST3R
# =========================

def recon_monst3r(lang, mode, dataset, outputs_dir):
    """
    MonST3R による再構築を実行する．

    Args:
        lang (str): ログ出力に使用する言語コード．
            `"jp"` のとき日本語，それ以外は英語を用いる．
        mode (str): 実行モード．`"local"` または `"slurm"` を指定する．
        dataset (str): 入力データセットのパス．
        outputs_dir (str): 出力先ルートディレクトリのパス．

    Returns:
        tuple[str, str, str, str, str]: 以下を返す．
            - 出力ディレクトリのパス．
            - 実行時間．
            - 実行ステータス．
            - 実行ログ全文．
            - 再構築結果のパス．
    """
    # データセットのパス
    dataset = os.path.abspath(dataset)
    # 出力ディレクトリの作成
    name = os.path.basename(os.path.dirname(dataset))
    outdir = os.path.join(outputs_dir, "monst3r", name)
    os.makedirs(outdir, exist_ok=True)

    if mode == "local":
        recon_script = "demo.py"

        cmd = [
            "conda", "run", "--no-capture-output",
            "-n", "monst3r", "python", recon_script,
            "--input_dir", dataset,
            "--output_dir", outdir,
            "--seq_name", name,
        ]

        workdir = os.path.join("models", "monst3r")
    elif mode == "slurm":
        sbatch_script = os.path.join("scripts", "recon_mast3r.sh")
        recon_script = os.path.join("models", "monst3r", "demo.py")
        cmd = ["sbatch", sbatch_script, recon_script, dataset, outdir, name]
        workdir = "."

    # 再構築の実行
    runtime, status, log = run_subprocess_popen(lang, cmd, workdir)

    # 3Dモデル
    model_path = os.path.join(outdir, name, "scene.glb")

    return outdir, runtime, status, log, model_path


# =========================
# Easi3R
# =========================

def recon_easi3r(lang, mode, dataset, outputs_dir):
    """
    Easi3R による再構築を実行する．

    Args:
        lang (str): ログ出力に使用する言語コード．
            `"jp"` のとき日本語，それ以外は英語を用いる．
        mode (str): 実行モード．`"local"` または `"slurm"` を指定する．
        dataset (str): 入力データセットのパス．
        outputs_dir (str): 出力先ルートディレクトリのパス．

    Returns:
        tuple[str, str, str, str, str]: 以下を返す．
            - 出力ディレクトリのパス．
            - 実行時間．
            - 実行ステータス．
            - 実行ログ全文．
            - 再構築結果のパス．
    """
    dataset = os.path.abspath(dataset)

    name = os.path.basename(os.path.dirname(dataset))
    outdir = os.path.join(outputs_dir, "easi3r", name)
    os.makedirs(outdir, exist_ok=True)

    if mode == "local":
        recon_script = "demo.py"
        cmd = [
            "conda", "run", "--no-capture-output",
            "-n", "easi3r", "python", recon_script,
            "--input_dir", dataset,
            "--output_dir", outdir,
            "--seq_name", name,
        ]
        workdir = os.path.join("models", "Easi3R")
    elif mode == "slurm":
        sbatch_script = os.path.join("scripts", "recon_easi3r.sh")
        recon_script = os.path.join("models", "Easi3R", "demo.py")
        cmd = ["sbatch", sbatch_script, recon_script, dataset, outdir, name]
        workdir = "."

    runtime, status, log = run_subprocess_popen(lang, cmd, workdir)

    model_path = os.path.join(outdir, name, "scene.glb")

    return outdir, runtime, status, log, model_path


# =========================
# MUSt3R
# =========================

def recon_must3r(lang, mode, dataset, outputs_dir):
    """
    MUSt3R による再構築を実行する．

    Args:
        lang (str): ログ出力に使用する言語コード．
            `"jp"` のとき日本語，それ以外は英語を用いる．
        mode (str): 実行モード．`"local"` または `"slurm"` を指定する．
        dataset (str): 入力データセットのパス．
        outputs_dir (str): 出力先ルートディレクトリのパス．

    Returns:
        tuple[str, str, str, str, str]: 以下を返す．
            - 出力ディレクトリのパス．
            - 実行時間．
            - 実行ステータス．
            - 実行ログ全文．
            - 再構築結果のパス．
    """
    # データセットのパス
    dataset = os.path.abspath(dataset)

    # 出力ディレクトリの作成
    name = os.path.basename(os.path.dirname(dataset))
    outdir = os.path.join(outputs_dir, "must3r", name)
    os.makedirs(outdir, exist_ok=True)

    if mode == "local":
        recon_script = "get_reconstruction.py"

        cmd = [
            "micromamba", "run", 
            "-n", "must3r", "python", recon_script,
            "--image_dir", dataset,
            "--output", outdir,
            "--weights", "ckpt/MUSt3R_512.pth",
            "--retrieval", "ckpt/MUSt3R_512_retrieval_trainingfree.pth",
            "--image_size", "512",
            "--file_type", "glb",
        ]

        workdir = os.path.join("models", "must3r")
    elif mode == "slurm":
        sbatch_script = os.path.join("scripts", "recon_must3r.sh")
        recon_script = os.path.join("models", "must3r", "get_reconstruction.py")
        cmd = ["sbatch", sbatch_script, recon_script, dataset, outdir]
        workdir = "."

    # 再構築の実行
    runtime, status, log = run_subprocess_popen(lang, cmd, workdir)

    # 3Dモデル
    model_path = os.path.join(outdir, "scene_1.05.glb")

    return outdir, runtime, status, log, model_path


# =========================
# Fast3R
# =========================

def recon_fast3r(lang, mode, dataset, outputs_dir):
    """
    Fast3R による再構築を実行する．

    Args:
        lang (str): ログ出力に使用する言語コード．
            `"jp"` のとき日本語，それ以外は英語を用いる．
        mode (str): 実行モード．`"local"` または `"slurm"` を指定する．
        dataset (str): 入力データセットのパス．
        outputs_dir (str): 出力先ルートディレクトリのパス．

    Returns:
        tuple[str, str, str, str, str]: 以下を返す．
            - 出力ディレクトリのパス．
            - 実行時間．
            - 実行ステータス．
            - 実行ログ全文．
            - 再構築結果のパス．
    """
    # データセットのパス
    dataset = os.path.abspath(dataset)

    # 出力ディレクトリの作成
    name = os.path.basename(os.path.dirname(dataset))
    outdir = os.path.join(outputs_dir, "fast3r", name)
    os.makedirs(outdir, exist_ok=True)

    if mode == "local":
        recon_script = os.path.join("scripts", "recon", "recon_fast3r.py")

        cmd = [
            "conda", "run", "--no-capture-output",
            "-n", "fast3r", "python", recon_script,
            "--lang", lang,
            "--inpdir", dataset,
            "--outdir", outdir,
        ]

        workdir = "."
    elif mode == "slurm":
        sbatch_script = os.path.join("scripts", "recon_fast3r.sh")
        recon_script = os.path.join("scripts", "recon_fast3r.py")
        cmd = ["sbatch", sbatch_script, recon_script, dataset, outdir]
        workdir = "."

    # 再構築の実行
    runtime, status, log = run_subprocess_popen(lang, cmd, workdir)

    # 3Dモデル
    model_path = os.path.join(outdir, "scene.glb")

    return outdir, runtime, status, log, model_path


# =========================
# Splatt3R
# =========================

def recon_splatt3r(lang, mode, dataset, outputs_dir):
    """
    Splatt3R による再構築を実行する．

    Args:
        lang (str): ログ出力に使用する言語コード．
            `"jp"` のとき日本語，それ以外は英語を用いる．
        mode (str): 実行モード．`"local"` または `"slurm"` を指定する．
        dataset (str): 入力画像のパス．
        outputs_dir (str): 出力先ルートディレクトリのパス．

    Returns:
        tuple[str, str, str, str, str]: 以下を返す．
            - 出力ディレクトリのパス．
            - 実行時間．
            - 実行ステータス．
            - 実行ログ全文．
            - 再構築結果のパス．
    """
    # データセットのパス
    dataset = os.path.abspath(dataset)

    # 出力ディレクトリの作成
    name = os.path.splitext(os.path.basename(dataset))[0]
    outdir = os.path.join(outputs_dir, "splatt3r", name)
    os.makedirs(outdir, exist_ok=True)

    if mode == "local":
        recon_script = os.path.join("scripts", "recon", "recon_splatt3r.py")

        cmd = [
            "conda", "run", "--no-capture-output",
            "-n", "splatt3r", "python", recon_script,
            "--lang", lang,
            "--image1", dataset,
            "--outdir", outdir,
        ]

        workdir = "."
    elif mode == "slurm":
        sbatch_script = os.path.join("scripts", "recon_splatt3r.sh")
        recon_script = os.path.join("scripts", "recon_splatt3r.py")
        cmd = ["sbatch", sbatch_script, recon_script, dataset, outdir]
        workdir = "."

    # 再構築の実行
    runtime, status, log = run_subprocess_popen(lang, cmd, workdir)

    # 3Dモデル
    outmodel = os.path.join(outdir, "gaussians.ply")

    return outdir, runtime, status, log, outmodel


# =========================
# CUT3R
# =========================

def recon_cut3r(lang, mode, dataset, outputs_dir):
    """
    CUT3R による再構築を実行する．

    Args:
        lang (str): ログ出力に使用する言語コード．
            `"jp"` のとき日本語，それ以外は英語を用いる．
        mode (str): 実行モード．`"local"` または `"slurm"` を指定する．
        dataset (str): 入力データセットのパス．
        outputs_dir (str): 出力先ルートディレクトリのパス．

    Returns:
        tuple[str, str, str, str, str]: 以下を返す．
            - 出力ディレクトリのパス．
            - 実行時間．
            - 実行ステータス．
            - 実行ログ全文．
            - 再構築結果のパス．
    """
    # データセットのパス
    dataset = os.path.abspath(dataset)

    # 出力ディレクトリの作成
    name = os.path.basename(os.path.dirname(dataset))
    outdir = os.path.join(outputs_dir, "cutt3r", name)
    os.makedirs(outdir, exist_ok=True)

    # 学習済みモデル
    model_ckpt = os.path.join("models", "CUT3R", "src", "cut3r_512_dpt_4_64.pth")

    if mode == "local":
        recon_script = os.path.join("scripts", "recon", "recon_cut3r.py")
        cmd = [
            "conda", "run", "--no-capture-output",
            "-n", "cut3r", "python", recon_script,
            "--lang", lang,
            "--inpdir", dataset,
            "--outdir", outdir,
            "--model_path", model_ckpt,
            "--image_size", "512",
            "--vis_threshold", "1.5",
            "--device", "cuda",
        ]
        workdir = "."
    elif mode == "slurm":
        sbatch_script = os.path.join("scripts", "recon_cut3r.sh")
        recon_script = os.path.join("scripts", "recon_cut3r.py")
        cmd = ["sbatch", sbatch_script, recon_script, dataset, outdir, model_ckpt]
        workdir = "."

    # 再構築の実行
    runtime, status, log = run_subprocess_popen(lang, cmd, workdir)

    # 3Dモデル
    outmodel = os.path.join(outdir, "scene.glb")

    return outdir, runtime, status, log, outmodel


# =========================
# WinT3R
# =========================

def recon_wint3r(lang, mode, dataset, outputs_dir):
    """
    WinT3R による再構築を実行する．

    Args:
        lang (str): ログ出力に使用する言語コード．
            `"jp"` のとき日本語，それ以外は英語を用いる．
        mode (str): 実行モード．`"local"` または `"slurm"` を指定する．
        dataset (str): 入力データセットのパス．
        outputs_dir (str): 出力先ルートディレクトリのパス．

    Returns:
        tuple[str, str, str, str, str]: 以下を返す．
            - 出力ディレクトリのパス．
            - 実行時間．
            - 実行ステータス．
            - 実行ログ全文．
            - 再構築結果のパス．
    """
    # データセットのパス
    dataset = os.path.abspath(dataset)

    # 出力ディレクトリの作成
    name = os.path.basename(os.path.dirname(dataset))
    outdir = os.path.join(outputs_dir, "wint3r", name)
    os.makedirs(outdir, exist_ok=True)

    # 学習済みモデル
    ckpt_path = os.path.join("checkpoints", "pytorch_model.bin")

    if mode == "local":
        recon_script = "recon.py"
        cmd = [
            "conda", "run", "--no-capture-output",
            "-n", "wint3r", "python", recon_script,
            "--data_path", dataset,
            "--save_dir", outdir,
            "--inference_mode", "offline",
            "--ckpt", ckpt_path,
        ]
        workdir = os.path.join("models", "WinT3R")
    elif mode == "slurm":
        sbatch_script = os.path.join("scripts", "recon_wint3r.sh")
        recon_script = os.path.join("models", "WinT3R", "recon.py")
        cmd = ["sbatch", sbatch_script, recon_script, dataset, outdir, ckpt_path]
        workdir = "."

    # 再構築の実行
    runtime, status, log = run_subprocess_popen(lang, cmd, workdir)

    # 3Dモデル
    outmodel = os.path.join(outdir, "recon.ply")

    return outdir, runtime, status, log, outmodel


# =========================
# VGGT
# =========================

def recon_vggt(lang, mode, dataset, outputs_dir):
    """
    VGGT による再構築を実行する．

    Args:
        lang (str): ログ出力に使用する言語コード．
            `"jp"` のとき日本語，それ以外は英語を用いる．
        mode (str): 実行モード．`"local"` または `"slurm"` を指定する．
        dataset (str): 入力データセットまたは画像パス．
        outputs_dir (str): 出力先ルートディレクトリのパス．

    Returns:
        tuple[str, str, str, str, str]: 以下を返す．
            - 出力ディレクトリのパス．
            - 実行時間．
            - 実行ステータス．
            - 実行ログ全文．
            - 再構築結果のパス．
    """
    # 入力ディレクトリ
    dataset = os.path.dirname(os.path.abspath(dataset))

    # 出力ディレクトリの作成
    name = os.path.basename(dataset)
    outdir = os.path.join(outputs_dir, "vggt", name)
    os.makedirs(outdir, exist_ok=True)

    if mode == "local":
        # 再構築スクリプトパス
        recon_script = os.path.join("scripts", "recon", "recon_vggt.py")

        # 実行コマンド
        cmd = [
            "conda", "run", "--no-capture-output",
            "-n", "vggt", "python", recon_script,
            "--lang", lang,
            "--image-dir", dataset,
            "--out-dir", outdir,
            "--conf-thres", "3.0",
            "--frame-filter", "All",
            "--prediction-mode", "Pointmap Regression",
            "--mode", "crop",
            "--device", "cuda",
            "--show-cam",
        ]

        # 実行ディレクトリ
        workdir = "."
    elif mode == "slurm":
        # sbatchスクリプト
        sbatch_script = os.path.join("scripts", "recon_vggt.sh")

        # 再構築スクリプトパス
        recon_script = os.path.join("scripts", "recon_vggt.py")

        # 実行コマンド
        cmd = ["sbatch", sbatch_script, recon_script, dataset, outdir]

        # 実行ディレクトリ
        workdir = "."

    # 推論実行
    runtime, status, log = run_subprocess_popen(lang, cmd, workdir)

    # 再構築結果のパス
    outmodel = os.path.join(outdir, "scene.glb")

    return outdir, runtime, status, log, outmodel


# =========================
# VGGSfM
# =========================

def recon_vggsfm(lang, mode, dataset):
    """
    VGGSfM による再構築を実行する．

    Args:
        lang (str): ログ出力に使用する言語コード．
            `"jp"` のとき日本語，それ以外は英語を用いる．
        mode (str): 実行モード．`"local"` または `"slurm"` を指定する．
        dataset (str): 入力データセットまたは画像パス．

    Returns:
        tuple[str, str, str, str]: 以下を返す．
            - 出力ディレクトリのパス．
            - 実行時間．
            - 実行ステータス．
            - 実行ログ全文．
    """
    # 入力ディレクトリ
    dataset = os.path.dirname(os.path.abspath(dataset))

    # 出力先のパス
    outdir = os.path.join(dataset, "sparse")

    if mode == "local":
        # 再構築スクリプトパス
        recon_script = "demo.py"

        # 実行コマンド
        cmd = [
            "conda", "run", "--no-capture-output",
            "-n", "vggsfm_tmp", "python", recon_script,
            f"SCENE_DIR={dataset}",
        ]

        # 実行ディレクトリ
        workdir = os.path.join("models", "vggsfm")
    elif mode == "slurm":
        # sbatchスクリプト
        sbatch_script = os.path.join("scripts", "recon_vggsfm.sh")

        # 再構築スクリプトパス
        recon_script = os.path.join("models", "vggsfm", "demo.py")

        # 実行コマンド
        cmd = ["sbatch", sbatch_script, recon_script, dataset, outdir]

        # 実行ディレクトリ
        workdir = "."

    # 推論実行
    runtime, status, log = run_subprocess_popen(lang, cmd, workdir)

    return outdir, runtime, status, log


def export_vggsfm(lang, dataset, outputs_dir):  # 軽量なのでlocalのみ
    """
    VGGSfM の再構築結果を PLY 形式へエクスポートする．

    Args:
        lang (str): ログ出力に使用する言語コード．
            `"jp"` のとき日本語，それ以外は英語を用いる．
        dataset (str): 入力データセットまたは画像パス．
        outputs_dir (str): 出力先ルートディレクトリのパス．

    Returns:
        tuple[str, str, str, str, str]: 以下を返す．
            - 出力ディレクトリのパス．
            - 実行時間．
            - 実行ステータス．
            - 実行ログ全文．
            - 出力モデルファイルのパス．
    """
    # 出力ディレクトリの作成
    dataset = os.path.abspath(dataset)
    name = os.path.basename(os.path.dirname(dataset))
    outdir = os.path.join(outputs_dir, "vggsfm", name)
    os.makedirs(outdir, exist_ok=True)

    # データセットのパス
    sparse_dir = os.path.join(dataset, "sparse")

    # 出力ファイル
    outmodel = os.path.join(outdir, "scene.ply")

    # 実行コマンド
    cmd = [
        "colmap",
        "model_converter",
        "--input_path", sparse_dir,
        "--output_path", outmodel,
        "--output_type", "ply",
    ]

    # 実行ディレクトリ
    workdir = "."

    # 推論実行
    runtime, status, log = run_subprocess_popen(lang, cmd, workdir)

    return outdir, runtime, status, log, outmodel


# =========================
# VGGT-SLAM
# =========================

def recon_vggtslam(lang, mode, dataset, outputs_dir):
    """
    VGGT-SLAM による再構築を実行する．

    Args:
        lang (str): ログ出力に使用する言語コード．
            `"jp"` のとき日本語，それ以外は英語を用いる．
        mode (str): 実行モード．`"local"` または `"slurm"` を指定する．
        dataset (str): 入力データセットまたは画像パス．
        outputs_dir (str): 出力先ルートディレクトリのパス．

    Returns:
        tuple[str, str, str, str, None]: 以下を返す．
            - 出力ディレクトリのパス．
            - 実行時間．
            - 実行ステータス．
            - 実行ログ全文．
            - 再構築結果のパス．現状は `None` ．
    """
    # 入力ディレクトリ
    dataset = os.path.dirname(os.path.abspath(dataset))

    # 出力ディレクトリの作成
    name = os.path.basename(dataset)
    outdir = os.path.join(outputs_dir, "vggt-slam", name)
    os.makedirs(outdir, exist_ok=True)

    if mode == "local":
        # データセットのパス
        image_dir = os.path.join(dataset, "images")

        # 再構築スクリプトパス
        recon_script = "main.py"

        # 実行コマンド
        cmd = [
            "conda", "run", "--no-capture-output",
            "-n", "vggt-slam", "python", recon_script,
            "--image_folder", image_dir,
            "--vis_map",
        ]

        # 実行ディレクトリ
        workdir = os.path.join("models", "VGGT-SLAM")
    elif mode == "slurm":
        # sbatchスクリプト
        sbatch_script = os.path.join("scripts", "recon_vggtslam.sh")

        # 再構築スクリプトパス
        recon_script = os.path.join("models", "VGGT-SLAM", "main.py")

        # 実行コマンド
        cmd = ["sbatch", sbatch_script, recon_script, dataset]

        # 実行ディレクトリ
        workdir = "."

    # 推論実行
    runtime, status, log = run_subprocess_popen(lang, cmd, workdir)

    # 再構築結果のパス
    outmodel = None

    return outdir, runtime, status, log, outmodel

# =========================
# StreamVGGT
# =========================

def recon_stmvggt(lang, mode, dataset, outputs_dir):
    """
    StreamVGGT による再構築を実行する．

    Args:
        lang (str): ログ出力に使用する言語コード．
            `"jp"` のとき日本語，それ以外は英語を用いる．
        mode (str): 実行モード．`"local"` または `"slurm"` を指定する．
        dataset (str): 入力データセットまたは画像パス．
        outputs_dir (str): 出力先ルートディレクトリのパス．

    Returns:
        tuple[str, str, str, str, str]: 以下を返す．
            - 出力ディレクトリのパス．
            - 実行時間．
            - 実行ステータス．
            - 実行ログ全文．
            - 再構築結果のパス．
    """
    # 入力ディレクトリ
    dataset = os.path.dirname(os.path.abspath(dataset))

    # 出力ディレクトリの作成
    name = os.path.basename(dataset)
    outdir = os.path.join(outputs_dir, "stmvggt", name)
    os.makedirs(outdir, exist_ok=True)

    if mode == "local":
        # 再構築スクリプトパス
        recon_script = os.path.join("scripts", "recon", "recon_streamvggt.py")

        # 実行コマンド
        cmd = [
            "conda", "run", "--no-capture-output",
            "-n", "streamvggt", "python", recon_script,
            "--lang", lang,
            "--input_dir", dataset,
            "--output_dir", outdir,
            "--show_cam",
        ]

        # 実行ディレクトリ
        workdir = "."
    elif mode == "slurm":
        # sbatchスクリプト
        sbatch_script = os.path.join("scripts", "recon_streamvggt.sh")

        # 再構築スクリプトパス
        recon_script = os.path.join("scripts", "recon_streamvggt.py")

        # 実行コマンド
        cmd = ["sbatch", sbatch_script, recon_script, dataset, outdir]

        # 実行ディレクトリ
        workdir = "."

    # 推論実行
    runtime, status, log = run_subprocess_popen(lang, cmd, workdir)

    # 再構築結果のパス
    outmodel = os.path.join(outdir, "scene.glb")

    return outdir, runtime, status, log, outmodel


# =========================
# FastVGGT
# =========================

def recon_fastvggt(lang, mode, dataset, outputs_dir):
    """
    FastVGGT による再構築を実行する．

    Args:
        lang (str): ログ出力に使用する言語コード．
            `"jp"` のとき日本語，それ以外は英語を用いる．
        mode (str): 実行モード．`"local"` または `"slurm"` を指定する．
        dataset (str): 入力データセットまたは画像パス．
        outputs_dir (str): 出力先ルートディレクトリのパス．

    Returns:
        tuple[str, str, str, str, str]: 以下を返す．
            - 出力ディレクトリのパス．
            - 実行時間．
            - 実行ステータス．
            - 実行ログ全文．
            - 再構築結果のパス．
    """
    # 入力ディレクトリ
    dataset = os.path.dirname(os.path.abspath(dataset))

    # 出力ディレクトリの作成
    name = os.path.basename(dataset)
    outdir = os.path.join(outputs_dir, "fastvggt", name)
    os.makedirs(outdir, exist_ok=True)

    # データセット
    image_dir = os.path.join(dataset, "images")

    # check pointのパス
    ckpt_path = os.path.join("ckpt", "model_tracker_fixed_e20.pt")

    if mode == "local":
        # 再構築スクリプトパス
        recon_script = os.path.join("eval", "eval_custom.py")

        # 実行コマンド
        cmd = [
            "conda", "run", "--no-capture-output",
            "-n", "fastvggt", "python", recon_script,
            "--data_path", image_dir,
            "--output_path", outdir,
            "--ckpt_path", ckpt_path,
            "--plot",
        ]

        # 実行ディレクトリ
        workdir = os.path.join("models", "FastVGGT")
    elif mode == "slurm":
        # sbatchスクリプト
        sbatch_script = os.path.join("scripts", "recon_fastvggt.sh")

        # 再構築スクリプトパス
        recon_script = os.path.join("models", "FastVGGT", "eval", "eval_custom.py")

        # 実行コマンド
        cmd = ["sbatch", sbatch_script, recon_script, image_dir, outdir, ckpt_path]

        # 実行ディレクトリ
        workdir = "."

    # 推論実行
    runtime, status, log = run_subprocess_popen(lang, cmd, workdir)

    # 再構築結果のパス
    outmodel = os.path.join(outdir, "custom_dataset", "reconstructed_points.ply")

    return outdir, runtime, status, log, outmodel


# =========================
# Pi3
# =========================

def recon_pi3(lang, mode, dataset, outputs_dir):
    """
    Pi3 による再構築を実行する．

    Args:
        lang (str): ログ出力に使用する言語コード．
            `"jp"` のとき日本語，それ以外は英語を用いる．
        mode (str): 実行モード．`"local"` または `"slurm"` を指定する．
        dataset (str): 入力データセットまたは画像パス．
        outputs_dir (str): 出力先ルートディレクトリのパス．

    Returns:
        tuple[str, str, str, str, str]: 以下を返す．
            - 出力ディレクトリのパス．
            - 実行時間．
            - 実行ステータス．
            - 実行ログ全文．
            - 再構築結果のパス．
    """
    # 入力ディレクトリ
    dataset = os.path.dirname(os.path.abspath(dataset))

    # 出力ディレクトリの作成
    name = os.path.basename(dataset)
    outdir = os.path.join(outputs_dir, "pi3", name)
    os.makedirs(outdir, exist_ok=True)

    # データセットパス
    image_dir = os.path.join(dataset, "images")

    # 再構築結果のパス
    outmodel = os.path.join(outdir, "recon.ply")

    if mode == "local":
        # 再構築スクリプトパス
        recon_script = "example.py"

        # 実行コマンド
        cmd = [
            "conda", "run", "--no-capture-output",
            "-n", "Pi3", "python", recon_script,
            "--data_path", image_dir,
            "--save_path", outmodel,
        ]

        # 実行ディレクトリ
        workdir = os.path.join("models", "Pi3")
    elif mode == "slurm":
        # sbatchスクリプト
        sbatch_script = os.path.join("scripts", "recon_pi3.sh")

        # 再構築スクリプトパス
        recon_script = os.path.join("models", "Pi3", "example.py")

        # 実行コマンド
        cmd = ["sbatch", sbatch_script, recon_script, image_dir, outdir]

        # 実行ディレクトリ
        workdir = "."

    # 推論実行
    runtime, status, log = run_subprocess_popen(lang, cmd, workdir)

    return outdir, runtime, status, log, outmodel


# =========================
# MoGe
# =========================

def recon_moge2(lang, mode, dataset, outputs_dir, img_type):
    """
    MoGe による再構築を実行する．

    Args:
        lang (str): ログ出力に使用する言語コード．
            `"jp"` のとき日本語，それ以外は英語を用いる．
        mode (str): 実行モード．`"local"` または `"slurm"` を指定する．
        dataset (str): 入力画像のパス．
        outputs_dir (str): 出力先ルートディレクトリのパス．
        img_type (str): 画像種別．`"標準画像"`，`"Standard Image"`，
            `"パノラマ画像"`，`"Panorama Image"` を想定する．

    Returns:
        tuple[str, str, str, str, str]: 以下を返す．
            - 出力ディレクトリのパス．
            - 実行時間．
            - 実行ステータス．
            - 実行ログ全文．
            - 再構築結果のパス．
    """
    # 入力画像のパス
    dataset = os.path.abspath(dataset)

    # 出力ディレクトリの作成
    name = os.path.splitext(os.path.basename(dataset))[0]
    outdir = os.path.join(outputs_dir, "moge")
    os.makedirs(outdir, exist_ok=True)

    # 再構築スクリプトパス
    if img_type == "標準画像" or img_type == "Standard Image":
        recon_script = os.path.join("models", "MoGe", "moge", "scripts", "infer.py")
    elif img_type == "パノラマ画像" or img_type == "Panorama Image":
        recon_script = os.path.join("models", "MoGe", "moge", "scripts", "infer_panorama.py")

    if mode == "local":
        # 実行コマンド
        cmd = [
            "conda", "run", "--no-capture-output", 
            "-n", "MoGe", "python", recon_script,
            "-i", dataset,
            "-o", outdir,
            "--maps",
            "--glb",
            "--ply",
        ]

        # 実行ディレクトリ
        workdir = "."
    elif mode == "slurm":
        # sbatchスクリプト
        sbatch_script = os.path.join("scripts", "recon_moge.sh")

        # 実行コマンド
        cmd = ["sbatch", sbatch_script, recon_script, dataset, outdir]

        # 実行ディレクトリ
        workdir = "."

    # 推論実行
    runtime, status, log = run_subprocess_popen(lang, cmd, workdir)

    # 再構築結果のパス
    outmodel = os.path.join(outdir, name, "mesh.glb")

    return outdir, runtime, status, log, outmodel


# =========================
# UniK3D
# =========================

def recon_unik3d(lang, mode, dataset, outputs_dir):
    """
    UniK3D による再構築を実行する．

    Args:
        lang (str): ログ出力に使用する言語コード．
            `"jp"` のとき日本語，それ以外は英語を用いる．
        mode (str): 実行モード．`"local"` または `"slurm"` を指定する．
        dataset (str): 入力画像のパス．
        outputs_dir (str): 出力先ルートディレクトリのパス．

    Returns:
        tuple[str, str, str, str, str]: 以下を返す．
            - 出力ディレクトリのパス．
            - 実行時間．
            - 実行ステータス．
            - 実行ログ全文．
            - 再構築結果のパス．
    """
    # 入力画像のパス
    dataset = os.path.abspath(dataset)

    # 出力ディレクトリの作成
    name = os.path.splitext(os.path.basename(dataset))[0]
    outdir = os.path.join(outputs_dir, "unik3d")
    os.makedirs(outdir, exist_ok=True)

    if mode == "local":
        # 再構築スクリプトパス
        recon_script = os.path.join("scripts", "infer.py")

        # 実行コマンド
        cmd = [
            "conda", "run", "--no-capture-output",
            "-n", "UniK3D", "python", recon_script,
            "--input", dataset,
            "--output", outdir,
            "--config-file", "configs/eval/vitl.json",
            "--save",
            "--save-ply",
        ]

        # 実行ディレクトリ
        workdir = os.path.join("models", "UniK3D")
    elif mode == "slurm":
        # sbatchスクリプト
        sbatch_script = os.path.join("scripts", "recon_unik3d.sh")

        # 再構築スクリプトパス
        recon_script = os.path.join("models", "UniK3D", "scripts", "infer.py")

        # 実行コマンド
        cmd = ["sbatch", sbatch_script, recon_script, dataset, outdir]

        # 実行ディレクトリ
        workdir = "."

    # 推論実行
    runtime, status, log = run_subprocess_popen(lang, cmd, workdir)

    # 再構築結果のパス
    outmodel = os.path.join(outdir, f"{name}.ply")

    return outdir, runtime, status, log, outmodel


# =========================
# Depth-Anything-V2
# =========================

def run_image_da2(lang, mode, dataset, outputs_dir, encoder):
    """
    Depth-Anything-V2 で画像の深度推論を実行する．

    Args:
        lang (str): ログ出力に使用する言語コード．
            `"jp"` のとき日本語，それ以外は英語を用いる．
        mode (str): 実行モード．`"local"` または `"slurm"` を指定する．
        dataset (str): 入力画像のパス．
        outputs_dir (str): 出力先ルートディレクトリのパス．
        encoder (str): 使用するエンコーダ名．

    Returns:
        tuple[str, str, str, str, str]: 以下を返す．
            - 出力ディレクトリのパス．
            - 実行時間．
            - 実行ステータス．
            - 実行ログ全文．
            - 出力ディレクトリのパス．
    """
    # 入力画像のパス
    dataset = os.path.abspath(dataset)

    # 出力ディレクトリの作成
    name = os.path.splitext(os.path.basename(dataset))[0]
    outdir = os.path.join(outputs_dir, "Depth-Anything-V2", name)
    os.makedirs(outdir, exist_ok=True)

    if mode == "local":
        # 推論スクリプトパス
        infer_script = "run.py"

        # 実行コマンド
        cmd = [
            "conda", "run", "--no-capture-output",
            "-n", "DA2", "python", infer_script,
            "--img-path", dataset,
            "--outdir", outdir,
            "--encoder", encoder,
        ]

        # 実行ディレクトリ
        workdir = os.path.join("models", "Depth-Anything-V2")
    elif mode == "slurm":
        # sbatchスクリプト
        sbatch_script = os.path.join("scripts", "recon_da2.sh")

        # 推論スクリプトパス
        infer_script = os.path.join("models", "Depth-Anything-V2", "run.py")

        # 実行コマンド
        cmd = ["sbatch", sbatch_script, infer_script, dataset, outdir]

        # 実行ディレクトリ
        workdir = "."

    # 推論実行
    runtime, status, log = run_subprocess_popen(lang, cmd, workdir)

    return outdir, runtime, status, log, outdir


def run_video_da2(lang, mode, dataset, outputs_dir, encoder):
    """
    Depth-Anything-V2 で動画の深度推論を実行する．

    Args:
        lang (str): ログ出力に使用する言語コード．
            `"jp"` のとき日本語，それ以外は英語を用いる．
        mode (str): 実行モード．`"local"` または `"slurm"` を指定する．
        dataset (str): 入力動画のパス．
        outputs_dir (str): 出力先ルートディレクトリのパス．
        encoder (str): 使用するエンコーダ名．

    Returns:
        tuple[str, str, str, str, str]: 以下を返す．
            - 出力ディレクトリのパス．
            - 実行時間．
            - 実行ステータス．
            - 実行ログ全文．
            - 出力動画のパス．
    """
    # 入力動画のパス
    dataset = os.path.abspath(dataset)

    # 出力ディレクトリの作成
    name = os.path.splitext(os.path.basename(dataset))[0]
    outdir = os.path.join(outputs_dir, "Depth-Anything-V2", name)
    os.makedirs(outdir, exist_ok=True)

    if mode == "local":
        # 推論スクリプトパス
        infer_script = "run_video.py"

        # 実行コマンド
        cmd = [
            "conda", "run", "--no-capture-output",
            "-n", "DA2", "python", infer_script,
            "--video-path", dataset,
            "--outdir", outdir,
            "--encoder", encoder,
        ]

        # 実行ディレクトリ
        workdir = os.path.join("models", "Depth-Anything-V2")
    elif mode == "slurm":
        # sbatchスクリプト
        sbatch_script = os.path.join("scripts", "recon_da2.sh")

        # 推論スクリプトパス
        infer_script = os.path.join("models", "Depth-Anything-V2", "run.py")

        # 実行コマンド
        cmd = ["sbatch", sbatch_script, infer_script, dataset, outdir]

        # 実行ディレクトリ
        workdir = "."

    # 推論実行
    runtime, status, log = run_subprocess_popen(lang, cmd, workdir)

    # 出力動画
    outvideo = os.path.join(outdir, f"{name}.mp4")

    return outdir, runtime, status, log, outvideo


# =========================
# Depth-Anything-3
# =========================

def recon_da3(lang, mode, dataset, outputs_dir):
    """
    Depth-Anything-3 による推論を実行する．

    Args:
        lang (str): ログ出力に使用する言語コード．
            `"jp"` のとき日本語，それ以外は英語を用いる．
        mode (str): 実行モード．`"local"` または `"slurm"` を指定する．
        dataset (str): 入力データセットまたは画像パス．
        outputs_dir (str): 出力先ルートディレクトリのパス．

    Returns:
        tuple[str, str, str, str, str, str, str, str]: 以下を返す．
            - 出力ディレクトリのパス．
            - 実行時間．
            - 実行ステータス．
            - 実行ログ全文．
            - 再構築結果のパス．
            - 出力画像ディレクトリのパス．
            - 出力動画のパス．
            - 出力 GS 動画のパス．
    """
    # 出力ディレクトリの作成
    dataset = os.path.dirname(os.path.abspath(dataset))
    name = os.path.basename(dataset)
    outdir = os.path.join(outputs_dir, "Depth-Anything-3", name)
    os.makedirs(outdir, exist_ok=True)

    # 内部関数（depth画像 → mp4生成）
    def images_to_video(image_dir, output_path, fps=5):
        """
        画像列から mp4 動画を生成する．

        Args:
            image_dir (str): 入力画像ディレクトリのパス．
            output_path (str): 出力動画ファイルのパス．
            fps (int, optional): 動画のフレームレート．
        """
        if not os.path.exists(image_dir):
            return

        images = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
        if not images:
            return

        first = cv2.imread(os.path.join(image_dir, images[0]))
        if first is None:
            return

        h, w = first.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        for img in images:
            frame = cv2.imread(os.path.join(image_dir, img))
            if frame is None:
                continue
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h))
            out.write(frame)

        out.release()

    if mode == "local":
        # 推論スクリプトパス
        infer_script = os.path.join("scripts", "recon", "recon_da3.py")

        # 実行コマンド
        cmd = [
            "conda", "run", "--no-capture-output",
            "-n", "DA3", "python", infer_script,
            "--lang", lang,
            "--input_dir", dataset,
            "--output_dir", outdir,
            "--infer_gs",
        ]

        # 実行ディレクトリ
        workdir = "."
    elif mode == "slurm":
        # sbatchスクリプト
        sbatch_script = os.path.join("scripts", "recon_da3.sh")

        # 推論スクリプトパス
        infer_script = os.path.join("scripts", "recon_da3.py")

        # 実行コマンド
        cmd = ["sbatch", sbatch_script, infer_script, dataset, outdir]

        # 実行ディレクトリ
        workdir = "."

    # 推論実行
    runtime, status, log = run_subprocess_popen(lang, cmd, workdir)

    # 再構築結果のパス
    outmodel = os.path.join(outdir, "scene.glb")

    # 出力画像ディレクトリ
    outimages = os.path.join(outdir, "depth_vis")

    # depth画像 → mp4生成
    outvideo = os.path.join(outdir, "scene.mp4")
    images_to_video(outimages, outvideo)

    # 出力gs動画ディレクトリ
    outgsvideo = os.path.join(outdir, "gs_video", "gs.mp4")

    return outdir, runtime, status, log, outmodel, outimages, outvideo, outgsvideo