from __future__ import annotations

import argparse
import os
import signal
import threading
import time
from pathlib import Path

import numpy as np
import trimesh
import viser
from viser import transforms as tf


I18N = {
    "ja": {
        "lang_label": "言語 / Language",
        "lang_ja": "日本語",
        "lang_en": "English",
        "help_md": """
### 操作方法

- **マウス左ドラッグ**：シーンの回転．回転の中心はワールド座標系の原点です．  
- **マウス右ドラッグ**：シーンを上下左右に平行移動．  
- **マウスホイール**：ズームイン・ズームアウト．  
- **各種ボタン**：ワールド座標系に沿ったカメラ移動，および yaw / pitch / roll 回転．  
        """,
        "reload_fix_ud": "上下反転を補正して再読み込み",
        "reload_fix_lr": "左右反転を補正して再読み込み",
        "camera_folder": "カメラ設定",
        "server_folder": "サーバー設定",
        "move_speed": "移動量",
        "yaw_deg": "yaw角度",
        "pitch_deg": "pitch角度",
        "roll_deg": "roll角度",
        "x_axis": "X軸移動",
        "y_axis": "Y軸移動",
        "z_axis": "Z軸移動",
        "yaw": "yaw回転",
        "pitch": "pitch回転",
        "roll": "roll回転",
        "yaw_left": "yaw 左",
        "yaw_right": "yaw 右",
        "pitch_up": "pitch 上",
        "pitch_down": "pitch 下",
        "roll_left": "roll 左",
        "roll_right": "roll 右",
        "confirm_shutdown": "サーバーを停止する",
        "shutdown_button": "サーバー停止",
    },
    "en": {
        "lang_label": "言語 / Language",
        "lang_ja": "日本語",
        "lang_en": "English",
        "help_md": """
### Controls

- **Left mouse drag**: Rotate the scene around the world-coordinate origin.  
- **Right mouse drag**: Pan the scene horizontally and vertically.  
- **Mouse wheel**: Zoom in / out.  
- **Buttons**: Move the camera along world axes and apply yaw / pitch / roll rotations.  
        """,
        "reload_fix_ud": "Reload with upside-down fix",
        "reload_fix_lr": "Reload with left-right fix",
        "camera_folder": "Camera Settings",
        "server_folder": "Server Settings",
        "move_speed": "Move Speed",
        "yaw_deg": "Yaw Angle",
        "pitch_deg": "Pitch Angle",
        "roll_deg": "Roll Angle",
        "x_axis": "Move on X",
        "y_axis": "Move on Y",
        "z_axis": "Move on Z",
        "yaw": "Yaw",
        "pitch": "Pitch",
        "roll": "Roll",
        "yaw_left": "Yaw Left",
        "yaw_right": "Yaw Right",
        "pitch_up": "Pitch Up",
        "pitch_down": "Pitch Down",
        "roll_left": "Roll Left",
        "roll_right": "Roll Right",
        "confirm_shutdown": "Confirm server shutdown",
        "shutdown_button": "Shutdown Server",
    },
}


def _safe_set_label(handle: object, value: str) -> None:
    """
    GUI ハンドルのラベル文字列を安全に更新する．

    `label` または `text` 属性が存在する場合にのみ値を設定し，
    失敗時は例外を握りつぶす．

    Args:
        handle: 更新対象の GUI ハンドル．
        value: 設定する文字列．
    """
    for attr in ("label", "text"):
        try:
            if hasattr(handle, attr):
                setattr(handle, attr, value)
                return
        except Exception:
            pass


def _safe_sigint_shutdown() -> None:
    """
    現在のプロセスへ SIGINT を送り，サーバー停止を試みる．
    """
    print("[INFO] Shutting down server (SIGINT)...")
    os.kill(os.getpid(), signal.SIGINT)


def _compute_transform(bounds: np.ndarray, target_size: float = 1.5) -> tuple[np.ndarray, float]:
    """
    境界ボックスから中心移動量とスケール倍率を計算する．

    Args:
        bounds: 最小座標と最大座標を含む 2×3 配列．
        target_size: 正規化後に収めたい最大サイズ．

    Returns:
        原点付近へ移動するための平行移動量と，正規化スケール倍率の組．
    """
    mins = bounds[0]
    maxs = bounds[1]
    center = (mins + maxs) / 2.0
    extent = np.maximum(maxs - mins, 1e-8)
    scale = float(target_size / np.max(extent))
    return -center, scale


def _normalize_scene_geometry(scene: trimesh.Scene) -> trimesh.Scene:
    """
    シーン全体をコピーし，中心化とスケーリングを行う．

    Args:
        scene: 正規化対象のシーン．

    Returns:
        正規化後のシーン．
    """
    scene = scene.copy()
    translation, scale = _compute_transform(scene.bounds)

    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] *= scale
    transform[:3, 3] = translation * scale
    scene.apply_transform(transform)
    return scene


def _rotate_scene_x_180(scene: trimesh.Scene) -> trimesh.Scene:
    """
    シーンを X 軸周りに 180 度回転させる．

    Args:
        scene: 回転対象のシーン．

    Returns:
        回転後のシーン．
    """
    scene = scene.copy()
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = tf.SO3.from_x_radians(np.pi).as_matrix().astype(np.float32)
    scene.apply_transform(transform)
    return scene


def _rotate_scene_z_180(scene: trimesh.Scene) -> trimesh.Scene:
    """
    シーンを Z 軸周りに 180 度回転させる．

    Args:
        scene: 回転対象のシーン．

    Returns:
        回転後のシーン．
    """
    scene = scene.copy()
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = tf.SO3.from_z_radians(np.pi).as_matrix().astype(np.float32)
    scene.apply_transform(transform)
    return scene


def load_asset(path: Path) -> trimesh.Trimesh | trimesh.Scene:
    """
    メッシュまたはシーンアセットをファイルから読み込む．

    Args:
        path: 読み込むファイルのパス．

    Returns:
        読み込まれた `trimesh.Trimesh` または `trimesh.Scene`．

    Raises:
        FileNotFoundError: ファイルが存在しない場合．
        ValueError: 読み込みに失敗した場合．
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    asset = trimesh.load(path, force=None)
    if asset is None:
        raise ValueError(f"Failed to load asset: {path}")
    return asset


def add_asset_to_viser(
    server: viser.ViserServer,
    asset: trimesh.Trimesh | trimesh.Scene,
    flip_updown: bool = False,
    flip_leftright: bool = False,
) -> None:
    """
    アセットを Viser シーンへ追加する．

    必要に応じてシーンの正規化と上下左右反転を行ったうえで，
    Viser 上へメッシュを登録する．

    Args:
        server: 追加先の Viser サーバー．
        asset: 追加するメッシュまたはシーン．
        flip_updown: True のとき上下反転補正を適用する．
        flip_leftright: True のとき左右反転補正を適用する．
    """
    server.scene.set_up_direction("+z")
    server.scene.add_frame("/world", axes_length=0.25, axes_radius=0.01)

    scene = trimesh.Scene(asset) if isinstance(asset, trimesh.Trimesh) else asset
    scene = _normalize_scene_geometry(scene)

    if flip_updown:
        scene = _rotate_scene_x_180(scene)
    if flip_leftright:
        scene = _rotate_scene_z_180(scene)

    server.scene.add_mesh_trimesh("/asset", scene)


def _camera_rotation(camera: viser.CameraHandle) -> np.ndarray:
    """
    カメラのクォータニオンから回転行列を取得する．

    Args:
        camera: 対象カメラ．

    Returns:
        3×3 の回転行列．
    """
    return tf.SO3(np.asarray(camera.wxyz, dtype=np.float32)).as_matrix().astype(np.float32)


def _camera_forward(camera: viser.CameraHandle) -> np.ndarray:
    """
    カメラの forward ベクトルを返す．

    Args:
        camera: 対象カメラ．

    Returns:
        正規化済みの forward ベクトル．
    """
    pos = np.asarray(camera.position, dtype=np.float32)
    look = np.asarray(camera.look_at, dtype=np.float32)
    forward = look - pos
    norm = np.linalg.norm(forward)
    if norm < 1e-6:
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)
    return (forward / norm).astype(np.float32)


def _move_camera_world(client: viser.ClientHandle, delta: np.ndarray) -> None:
    """
    カメラ位置と注視点をワールド座標系で平行移動する．

    Args:
        client: 対象クライアント．
        delta: 移動量ベクトル．
    """
    pos = np.asarray(client.camera.position, dtype=np.float32)
    look = np.asarray(client.camera.look_at, dtype=np.float32)
    with client.atomic():
        client.camera.position = tuple((pos + delta).tolist())
        client.camera.look_at = tuple((look + delta).tolist())


def _set_camera_direction(client: viser.ClientHandle, forward: np.ndarray) -> None:
    """
    カメラ位置を維持したまま注視方向を更新する．

    Args:
        client: 対象クライアント．
        forward: 新しい forward ベクトル．
    """
    pos = np.asarray(client.camera.position, dtype=np.float32)
    look = np.asarray(client.camera.look_at, dtype=np.float32)
    distance = float(np.linalg.norm(look - pos))
    if distance < 1e-6:
        distance = 1.0
    forward = forward / max(np.linalg.norm(forward), 1e-6)
    with client.atomic():
        client.camera.look_at = tuple((pos + forward * distance).tolist())


def _yaw_camera_world_z(client: viser.ClientHandle, yaw_deg: float) -> None:
    """
    カメラの注視方向をワールド Z 軸周りに yaw 回転させる．

    Args:
        client: 対象クライアント．
        yaw_deg: 回転角度［deg］．
    """
    forward = _camera_forward(client.camera)
    yaw = tf.SO3.from_z_radians(np.deg2rad(yaw_deg)).as_matrix().astype(np.float32)
    _set_camera_direction(client, yaw @ forward)


def _pitch_camera_local(client: viser.ClientHandle, pitch_deg: float) -> None:
    """
    カメラのローカル右軸周りに pitch 回転を適用する．

    Args:
        client: 対象クライアント．
        pitch_deg: 回転角度［deg］．
    """
    forward = _camera_forward(client.camera)
    rotation = _camera_rotation(client.camera)
    right = rotation[:, 0]
    pitch = tf.SO3.exp(right * np.deg2rad(pitch_deg)).as_matrix().astype(np.float32)
    candidate = pitch @ forward
    candidate = candidate / max(np.linalg.norm(candidate), 1e-6)
    if abs(float(candidate[2])) > 0.98:
        candidate[2] = np.sign(candidate[2]) * 0.98
        candidate = candidate / max(np.linalg.norm(candidate), 1e-6)
    _set_camera_direction(client, candidate)


def _roll_camera_local(client: viser.ClientHandle, roll_deg: float) -> None:
    """
    カメラの forward 軸周りに roll 回転を適用する．

    Args:
        client: 対象クライアント．
        roll_deg: 回転角度［deg］．
    """
    pos = np.asarray(client.camera.position, dtype=np.float32)
    look = np.asarray(client.camera.look_at, dtype=np.float32)
    wxyz = np.asarray(client.camera.wxyz, dtype=np.float32)
    roll = tf.SO3.exp(_camera_forward(client.camera) * np.deg2rad(roll_deg))
    with client.atomic():
        client.camera.wxyz = tuple((roll @ tf.SO3(wxyz)).wxyz.tolist())
        client.camera.position = tuple(pos.tolist())
        client.camera.look_at = tuple(look.tolist())


def main() -> None:
    """
    メッシュまたは GLB などの 3D アセットを表示する簡易 Viser viewer を起動する．

    CLI 引数を解釈してアセットを読み込み，GUI から再読み込み，
    反転補正，カメラ移動，姿勢変更，サーバー停止などを行えるようにする．
    """
    parser = argparse.ArgumentParser(description="Simple viser viewer for polygon meshes or GLB files.")
    parser.add_argument("--input", type=Path, help="Path to a mesh file (.glb, .gltf, .obj, .ply, .stl, etc.)")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the viser server")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind the viser server")
    args = parser.parse_args()

    asset = load_asset(args.input)

    server = viser.ViserServer(host=args.host, port=args.port)
    lang_state = {"lang": "ja"}
    view_state = {"flip_updown": False, "flip_leftright": False}

    def t(key: str) -> str:
        """
        現在の言語設定に対応する UI 文字列を返す．

        Args:
            key: I18N 辞書のキー．

        Returns:
            対応する UI 文字列．
        """
        return I18N[lang_state["lang"]][key]

    def _reload_asset() -> None:
        """
        入力アセットを再読み込みし，現在の反転設定を適用して再登録する．
        """
        try:
            reloaded = load_asset(args.input)
            add_asset_to_viser(
                server,
                reloaded,
                flip_updown=view_state["flip_updown"],
                flip_leftright=view_state["flip_leftright"],
            )
            print(
                "[INFO] asset reloaded. "
                f"flip_updown={view_state['flip_updown']}, "
                f"flip_leftright={view_state['flip_leftright']}"
            )
        except Exception as e:
            print(f"[WARN] failed to reload asset: {e}")

    add_asset_to_viser(
        server,
        asset,
        flip_updown=view_state["flip_updown"],
        flip_leftright=view_state["flip_leftright"],
    )

    lang_dropdown = server.gui.add_dropdown(
        I18N["ja"]["lang_label"],
        options=[I18N["ja"]["lang_ja"], I18N["en"]["lang_en"]],
        initial_value=I18N["ja"]["lang_ja"],
    )
    help_md = server.gui.add_markdown(t("help_md"))
    reload_fix_ud_button = server.gui.add_button(t("reload_fix_ud"))
    reload_fix_lr_button = server.gui.add_button(t("reload_fix_lr"))

    with server.gui.add_folder(t("camera_folder")) as camera_folder:
        move_speed_gui = server.gui.add_slider(t("move_speed"), min=0.01, max=2.0, step=0.01, initial_value=0.10)
        yaw_speed_gui = server.gui.add_slider(t("yaw_deg"), min=1.0, max=30.0, step=1.0, initial_value=8.0)
        pitch_speed_gui = server.gui.add_slider(t("pitch_deg"), min=1.0, max=30.0, step=1.0, initial_value=6.0)
        roll_speed_gui = server.gui.add_slider(t("roll_deg"), min=1.0, max=30.0, step=1.0, initial_value=8.0)

        x_axis_buttons = server.gui.add_button_group(t("x_axis"), ("+X", "-X"))
        y_axis_buttons = server.gui.add_button_group(t("y_axis"), ("+Y", "-Y"))
        z_axis_buttons = server.gui.add_button_group(t("z_axis"), ("+Z", "-Z"))

        yaw_buttons = server.gui.add_button_group(t("yaw"), (t("yaw_left"), t("yaw_right")))
        pitch_buttons = server.gui.add_button_group(t("pitch"), (t("pitch_up"), t("pitch_down")))
        roll_buttons = server.gui.add_button_group(t("roll"), (t("roll_left"), t("roll_right")))

    with server.gui.add_folder(t("server_folder")) as server_folder:
        confirm_shutdown = server.gui.add_checkbox(t("confirm_shutdown"), initial_value=False)
        shutdown_button = server.gui.add_button(t("shutdown_button"))
        shutdown_button.disabled = True

    def _client_or_broadcast() -> list[viser.ClientHandle]:
        """
        現在接続中のクライアント一覧を返す．

        Returns:
            接続中クライアントのリスト．
        """
        return list(server.get_clients().values())

    @lang_dropdown.on_update
    def _on_lang_change(_event: viser.GuiEvent) -> None:
        """
        UI の表示言語を切り替え，更新可能なラベルを再設定する．

        Args:
            _event: GUI 更新イベント．
        """
        lang_state["lang"] = "ja" if lang_dropdown.value == I18N["ja"]["lang_ja"] else "en"

        help_md.content = t("help_md")

        _safe_set_label(lang_dropdown, I18N["ja"]["lang_label"])
        _safe_set_label(reload_fix_ud_button, t("reload_fix_ud"))
        _safe_set_label(reload_fix_lr_button, t("reload_fix_lr"))

        _safe_set_label(camera_folder, t("camera_folder"))
        _safe_set_label(server_folder, t("server_folder"))

        _safe_set_label(move_speed_gui, t("move_speed"))
        _safe_set_label(yaw_speed_gui, t("yaw_deg"))
        _safe_set_label(pitch_speed_gui, t("pitch_deg"))
        _safe_set_label(roll_speed_gui, t("roll_deg"))

        _safe_set_label(x_axis_buttons, t("x_axis"))
        _safe_set_label(y_axis_buttons, t("y_axis"))
        _safe_set_label(z_axis_buttons, t("z_axis"))
        _safe_set_label(yaw_buttons, t("yaw"))
        _safe_set_label(pitch_buttons, t("pitch"))
        _safe_set_label(roll_buttons, t("roll"))

        _safe_set_label(confirm_shutdown, t("confirm_shutdown"))
        _safe_set_label(shutdown_button, t("shutdown_button"))

    @reload_fix_ud_button.on_click
    def _on_reload_fix_ud(_event: viser.GuiEvent) -> None:
        """
        上下反転フラグを切り替えてアセットを再読み込みする．

        Args:
            _event: ボタンクリックイベント．
        """
        view_state["flip_updown"] = not view_state["flip_updown"]
        _reload_asset()

    @reload_fix_lr_button.on_click
    def _on_reload_fix_lr(_event: viser.GuiEvent) -> None:
        """
        左右反転フラグを切り替えてアセットを再読み込みする．

        Args:
            _event: ボタンクリックイベント．
        """
        view_state["flip_leftright"] = not view_state["flip_leftright"]
        _reload_asset()

    @confirm_shutdown.on_update
    def _on_confirm_shutdown(_event: viser.GuiEvent) -> None:
        """
        確認チェックボックスの状態に応じて停止ボタンの有効状態を切り替える．

        Args:
            _event: GUI 更新イベント．
        """
        shutdown_button.disabled = not bool(confirm_shutdown.value)

    @shutdown_button.on_click
    def _on_shutdown(_event: viser.GuiEvent) -> None:
        """
        別スレッドで少し遅延してからサーバー停止処理を呼び出す．

        Args:
            _event: ボタンクリックイベント．
        """
        def worker() -> None:
            """
            短時間待機したあとに SIGINT による停止を実行する．
            """
            time.sleep(0.1)
            _safe_sigint_shutdown()

        threading.Thread(target=worker, daemon=True).start()

    @x_axis_buttons.on_click
    def _on_x_axis(event: viser.GuiEvent) -> None:
        """
        カメラを X 軸方向へ平行移動する．

        Args:
            event: ボタングループのクリックイベント．
        """
        step = np.float32(move_speed_gui.value)
        delta = np.array([step, 0.0, 0.0], dtype=np.float32) if event.target.value == "+X" else np.array([-step, 0.0, 0.0], dtype=np.float32)
        for client in _client_or_broadcast():
            _move_camera_world(client, delta)

    @y_axis_buttons.on_click
    def _on_y_axis(event: viser.GuiEvent) -> None:
        """
        カメラを Y 軸方向へ平行移動する．

        Args:
            event: ボタングループのクリックイベント．
        """
        step = np.float32(move_speed_gui.value)
        delta = np.array([0.0, step, 0.0], dtype=np.float32) if event.target.value == "+Y" else np.array([0.0, -step, 0.0], dtype=np.float32)
        for client in _client_or_broadcast():
            _move_camera_world(client, delta)

    @z_axis_buttons.on_click
    def _on_z_axis(event: viser.GuiEvent) -> None:
        """
        カメラを Z 軸方向へ平行移動する．

        Args:
            event: ボタングループのクリックイベント．
        """
        step = np.float32(move_speed_gui.value)
        delta = np.array([0.0, 0.0, step], dtype=np.float32) if event.target.value == "+Z" else np.array([0.0, 0.0, -step], dtype=np.float32)
        for client in _client_or_broadcast():
            _move_camera_world(client, delta)

    @yaw_buttons.on_click
    def _on_yaw(event: viser.GuiEvent) -> None:
        """
        カメラに yaw 回転を適用する．

        Args:
            event: ボタングループのクリックイベント．
        """
        for client in _client_or_broadcast():
            if event.target.value in (I18N["ja"]["yaw_left"], I18N["en"]["yaw_left"]):
                _yaw_camera_world_z(client, yaw_speed_gui.value)
            else:
                _yaw_camera_world_z(client, -yaw_speed_gui.value)

    @pitch_buttons.on_click
    def _on_pitch(event: viser.GuiEvent) -> None:
        """
        カメラに pitch 回転を適用する．

        Args:
            event: ボタングループのクリックイベント．
        """
        for client in _client_or_broadcast():
            if event.target.value in (I18N["ja"]["pitch_up"], I18N["en"]["pitch_up"]):
                _pitch_camera_local(client, -pitch_speed_gui.value)
            else:
                _pitch_camera_local(client, pitch_speed_gui.value)

    @roll_buttons.on_click
    def _on_roll(event: viser.GuiEvent) -> None:
        """
        カメラに roll 回転を適用する．

        Args:
            event: ボタングループのクリックイベント．
        """
        for client in _client_or_broadcast():
            if event.target.value in (I18N["ja"]["roll_left"], I18N["en"]["roll_left"]):
                _roll_camera_local(client, roll_speed_gui.value)
            else:
                _roll_camera_local(client, -roll_speed_gui.value)

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Stopping viewer...")


if __name__ == "__main__":
    main()