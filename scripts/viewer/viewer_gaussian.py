from __future__ import annotations

import argparse
import os
import signal
import threading
import time
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import numpy.typing as npt
from plyfile import PlyData

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
        "reload_fix_ud": "上下反転を補正して再レンダリング",
        "reload_fix_lr": "左右反転を補正して再レンダリング",
        "camera_folder": "カメラ設定",
        "render_folder": "レンダリング設定",
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
        "scale_multiplier": "Gaussianスケール倍率",
        "max_scale_percentile": "Gaussianスケール上限",
        "opacity_threshold": "Gaussian削除閾値",
        "quat_order": "PLYクォータニオン読み込み順",
        "rerender_button": "再レンダリング",
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
        "reload_fix_ud": "Re-render with upside-down fix",
        "reload_fix_lr": "Re-render with left-right fix",
        "camera_folder": "Camera Settings",
        "render_folder": "Render Settings",
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
        "scale_multiplier": "Gaussian Scale Multiplier",
        "max_scale_percentile": "Gaussian Scale Upper Limit",
        "opacity_threshold": "Gaussian Removal Threshold",
        "quat_order": "PLY Quaternion Loading Order",
        "rerender_button": "Re-render",
        "confirm_shutdown": "Confirm server shutdown",
        "shutdown_button": "Shutdown Server",
    },
}


class SplatFile(TypedDict):
    centers: npt.NDArray[np.floating]
    rgbs: npt.NDArray[np.floating]
    opacities: npt.NDArray[np.floating]
    covariances: npt.NDArray[np.floating]


class RawGaussianData(TypedDict):
    positions: npt.NDArray[np.floating]
    scales: npt.NDArray[np.floating]
    quaternions: npt.NDArray[np.floating]
    colors: npt.NDArray[np.floating]
    opacities: npt.NDArray[np.floating]


def _safe_set_label(handle: object, value: str) -> None:
    for attr in ("label", "text"):
        try:
            if hasattr(handle, attr):
                setattr(handle, attr, value)
                return
        except Exception:
            pass


def _safe_sigint_shutdown() -> None:
    print("[INFO] Shutting down server (SIGINT)...")
    os.kill(os.getpid(), signal.SIGINT)


def _sigmoid(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


def _normalize_quaternions(quats: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return (quats / norms).astype(np.float32)


def _rotation_x_180() -> npt.NDArray[np.float32]:
    return tf.SO3.from_x_radians(np.pi).as_matrix().astype(np.float32)


def _rotation_z_180() -> npt.NDArray[np.float32]:
    return tf.SO3.from_z_radians(np.pi).as_matrix().astype(np.float32)


def _apply_rotation_to_splat(
    splat: SplatFile,
    rotation: npt.NDArray[np.float32],
) -> SplatFile:
    centers = (splat["centers"] @ rotation.T).astype(np.float32)
    covariances = np.einsum(
        "ij,njk,lk->nil",
        rotation,
        splat["covariances"],
        rotation,
    ).astype(np.float32)

    return SplatFile(
        centers=centers,
        rgbs=splat["rgbs"],
        opacities=splat["opacities"],
        covariances=covariances,
    )


def load_raw_ply_file(ply_file_path: Path, center: bool = False) -> RawGaussianData:
    sh_c0 = 0.28209479177387814
    start_time = time.time()

    plydata = PlyData.read(ply_file_path)
    if "vertex" not in plydata:
        raise ValueError("PLY file does not contain a vertex element.")

    v = plydata["vertex"]
    names = set(v.data.dtype.names or [])
    required = {
        "x", "y", "z",
        "f_dc_0", "f_dc_1", "f_dc_2",
        "opacity",
        "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3",
    }
    missing = sorted(required - names)
    if missing:
        raise ValueError(f"PLY is missing required Gaussian fields: {missing}")

    positions = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)
    if center:
        positions -= np.mean(positions, axis=0, keepdims=True)

    scales = np.exp(
        np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1).astype(np.float32)
    )
    quats = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1).astype(np.float32)
    colors = 0.5 + sh_c0 * np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1).astype(np.float32)
    colors = np.clip(colors, 0.0, 1.0)
    opacities = _sigmoid(v["opacity"].astype(np.float32)[:, None])

    print(f"Loaded {len(v)} gaussians from {ply_file_path.name} in {time.time() - start_time:.3f}s")
    return RawGaussianData(
        positions=positions,
        scales=scales,
        quaternions=quats,
        colors=colors,
        opacities=opacities,
    )


def make_splat_file(
    raw: RawGaussianData,
    *,
    quat_order: Literal["wxyz", "xyzw"],
    scale_multiplier: float,
    max_scale_percentile: float,
    opacity_threshold: float,
    max_points: int,
    random_seed: int,
) -> SplatFile:
    positions = raw["positions"].astype(np.float32).copy()
    scales = raw["scales"].astype(np.float32).copy()
    quats = raw["quaternions"].astype(np.float32).copy()
    colors = raw["colors"].astype(np.float32)
    opacities = raw["opacities"].astype(np.float32).copy()

    if quat_order == "xyzw":
        quats = quats[:, [3, 0, 1, 2]]
    quats = _normalize_quaternions(quats)

    clip_value = np.percentile(scales, max_scale_percentile, axis=0)
    clip_value = np.maximum(clip_value, 1e-6)
    scales = np.minimum(scales, clip_value[None, :])
    scales *= np.float32(scale_multiplier)
    scales = np.maximum(scales, 1e-6)

    keep_mask = opacities[:, 0] >= np.float32(opacity_threshold)
    if not np.any(keep_mask):
        keep_mask[:] = True

    positions = positions[keep_mask]
    scales = scales[keep_mask]
    quats = quats[keep_mask]
    colors = colors[keep_mask]
    opacities = np.clip(opacities[keep_mask], 0.0, 1.0)

    if len(positions) > max_points:
        rng = np.random.default_rng(random_seed)
        ids = rng.choice(len(positions), size=max_points, replace=False)
        positions = positions[ids]
        scales = scales[ids]
        quats = quats[ids]
        colors = colors[ids]
        opacities = opacities[ids]

    Rs = tf.SO3(quats).as_matrix().astype(np.float32)
    covariances = np.einsum(
        "nij,njk,nlk->nil",
        Rs,
        np.eye(3, dtype=np.float32)[None, :, :] * scales[:, None, :] ** 2,
        Rs,
    ).astype(np.float32)

    return SplatFile(
        centers=positions,
        rgbs=colors,
        opacities=opacities,
        covariances=covariances,
    )


def _camera_rotation(camera: viser.CameraHandle) -> np.ndarray:
    return tf.SO3(np.asarray(camera.wxyz, dtype=np.float32)).as_matrix().astype(np.float32)


def _camera_forward(camera: viser.CameraHandle) -> np.ndarray:
    pos = np.asarray(camera.position, dtype=np.float32)
    look = np.asarray(camera.look_at, dtype=np.float32)
    forward = look - pos
    norm = np.linalg.norm(forward)
    if norm < 1e-6:
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)
    return (forward / norm).astype(np.float32)


def _move_camera_world(client: viser.ClientHandle, delta: np.ndarray) -> None:
    pos = np.asarray(client.camera.position, dtype=np.float32)
    look = np.asarray(client.camera.look_at, dtype=np.float32)
    with client.atomic():
        client.camera.position = tuple((pos + delta).tolist())
        client.camera.look_at = tuple((look + delta).tolist())


def _set_camera_direction(client: viser.ClientHandle, forward: np.ndarray) -> None:
    pos = np.asarray(client.camera.position, dtype=np.float32)
    look = np.asarray(client.camera.look_at, dtype=np.float32)
    distance = float(np.linalg.norm(look - pos))
    if distance < 1e-6:
        distance = 1.0
    forward = forward / max(np.linalg.norm(forward), 1e-6)
    with client.atomic():
        client.camera.look_at = tuple((pos + forward * distance).tolist())


def _yaw_camera_world_z(client: viser.ClientHandle, yaw_deg: float) -> None:
    forward = _camera_forward(client.camera)
    yaw = tf.SO3.from_z_radians(np.deg2rad(yaw_deg)).as_matrix().astype(np.float32)
    _set_camera_direction(client, yaw @ forward)


def _pitch_camera_local(client: viser.ClientHandle, pitch_deg: float) -> None:
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
    pos = np.asarray(client.camera.position, dtype=np.float32)
    look = np.asarray(client.camera.look_at, dtype=np.float32)
    wxyz = np.asarray(client.camera.wxyz, dtype=np.float32)
    roll = tf.SO3.exp(_camera_forward(client.camera) * np.deg2rad(roll_deg))
    with client.atomic():
        client.camera.wxyz = tuple((roll @ tf.SO3(wxyz)).wxyz.tolist())
        client.camera.position = tuple(pos.tolist())
        client.camera.look_at = tuple(look.tolist())


def main(
    input: tuple[Path, ...],
    host: str = "0.0.0.0",
    port: int = 8080,
    center: bool = True,
    quat_order: Literal["wxyz", "xyzw"] = "wxyz",
    scale_multiplier: float = 1.0,
    max_scale_percentile: float = 90.0,
    opacity_threshold: float = 0.05,
    max_points: int = 200000,
    random_seed: int = 0,
) -> None:
    server = viser.ViserServer(host=host, port=port)
    server.scene.set_up_direction("+z")
    server.scene.configure_fog(near=1.0, far=2.0, enabled=False)

    lang_state = {"lang": "ja"}
    view_state = {"flip_updown": False, "flip_leftright": False}

    def t(key: str) -> str:
        return I18N[lang_state["lang"]][key]

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

    with server.gui.add_folder(t("render_folder")) as render_folder:
        scale_gui = server.gui.add_slider(t("scale_multiplier"), min=0.001, max=1.0, step=0.001, initial_value=scale_multiplier)
        perc_gui = server.gui.add_slider(t("max_scale_percentile"), min=50.0, max=100.0, step=1.0, initial_value=max_scale_percentile)
        opacity_gui = server.gui.add_slider(t("opacity_threshold"), min=0.0, max=0.5, step=0.005, initial_value=opacity_threshold)
        quat_gui = server.gui.add_dropdown(t("quat_order"), ("wxyz", "xyzw"), initial_value=quat_order)
        rerender_button = server.gui.add_button(t("rerender_button"))

    with server.gui.add_folder(t("server_folder")) as server_folder:
        confirm_shutdown = server.gui.add_checkbox(t("confirm_shutdown"), initial_value=False)
        shutdown_button = server.gui.add_button(t("shutdown_button"))
        shutdown_button.disabled = True

    raw_data_list: list[RawGaussianData] = []
    handles: list[viser.GaussianSplatsHandle] = []

    for ply_path in input:
        if ply_path.suffix.lower() != ".ply":
            raise SystemExit("Please provide only Gaussian-splat PLY files.")
        raw_data_list.append(load_raw_ply_file(ply_path, center=center))

    def render_all() -> None:
        nonlocal handles

        for h in handles:
            h.remove()
        handles = []

        rot_x = _rotation_x_180()
        rot_z = _rotation_z_180()

        for i, raw in enumerate(raw_data_list):
            splat_data = make_splat_file(
                raw,
                quat_order=quat_gui.value,
                scale_multiplier=scale_gui.value,
                max_scale_percentile=perc_gui.value,
                opacity_threshold=opacity_gui.value,
                max_points=max_points,
                random_seed=random_seed,
            )

            if view_state["flip_updown"]:
                splat_data = _apply_rotation_to_splat(splat_data, rot_x)
            if view_state["flip_leftright"]:
                splat_data = _apply_rotation_to_splat(splat_data, rot_z)

            handle = server.scene.add_gaussian_splats(
                f"/{i}/gaussian_splats",
                centers=splat_data["centers"],
                rgbs=splat_data["rgbs"],
                opacities=splat_data["opacities"],
                covariances=splat_data["covariances"],
            )
            handles.append(handle)

    def _client_or_broadcast() -> list[viser.ClientHandle]:
        return list(server.get_clients().values())

    @lang_dropdown.on_update
    def _on_lang_change(_event: viser.GuiEvent) -> None:
        lang_state["lang"] = "ja" if lang_dropdown.value == I18N["ja"]["lang_ja"] else "en"

        help_md.content = t("help_md")

        _safe_set_label(lang_dropdown, I18N["ja"]["lang_label"])
        _safe_set_label(reload_fix_ud_button, t("reload_fix_ud"))
        _safe_set_label(reload_fix_lr_button, t("reload_fix_lr"))

        _safe_set_label(camera_folder, t("camera_folder"))
        _safe_set_label(render_folder, t("render_folder"))
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

        _safe_set_label(scale_gui, t("scale_multiplier"))
        _safe_set_label(perc_gui, t("max_scale_percentile"))
        _safe_set_label(opacity_gui, t("opacity_threshold"))
        _safe_set_label(quat_gui, t("quat_order"))
        _safe_set_label(rerender_button, t("rerender_button"))

        _safe_set_label(confirm_shutdown, t("confirm_shutdown"))
        _safe_set_label(shutdown_button, t("shutdown_button"))

    @reload_fix_ud_button.on_click
    def _on_reload_fix_ud(_event: viser.GuiEvent) -> None:
        view_state["flip_updown"] = not view_state["flip_updown"]
        render_all()

    @reload_fix_lr_button.on_click
    def _on_reload_fix_lr(_event: viser.GuiEvent) -> None:
        view_state["flip_leftright"] = not view_state["flip_leftright"]
        render_all()

    @confirm_shutdown.on_update
    def _on_confirm_shutdown(_event: viser.GuiEvent) -> None:
        shutdown_button.disabled = not bool(confirm_shutdown.value)

    @shutdown_button.on_click
    def _on_shutdown(_event: viser.GuiEvent) -> None:
        def worker() -> None:
            time.sleep(0.1)
            _safe_sigint_shutdown()

        threading.Thread(target=worker, daemon=True).start()

    @rerender_button.on_click
    def _on_rerender(_event: viser.GuiEvent) -> None:
        render_all()

    @x_axis_buttons.on_click
    def _on_x_axis(event: viser.GuiEvent) -> None:
        step = np.float32(move_speed_gui.value)
        delta = np.array([step, 0.0, 0.0], dtype=np.float32) if event.target.value == "+X" else np.array([-step, 0.0, 0.0], dtype=np.float32)
        for client in _client_or_broadcast():
            _move_camera_world(client, delta)

    @y_axis_buttons.on_click
    def _on_y_axis(event: viser.GuiEvent) -> None:
        step = np.float32(move_speed_gui.value)
        delta = np.array([0.0, step, 0.0], dtype=np.float32) if event.target.value == "+Y" else np.array([0.0, -step, 0.0], dtype=np.float32)
        for client in _client_or_broadcast():
            _move_camera_world(client, delta)

    @z_axis_buttons.on_click
    def _on_z_axis(event: viser.GuiEvent) -> None:
        step = np.float32(move_speed_gui.value)
        delta = np.array([0.0, 0.0, step], dtype=np.float32) if event.target.value == "+Z" else np.array([0.0, 0.0, -step], dtype=np.float32)
        for client in _client_or_broadcast():
            _move_camera_world(client, delta)

    @yaw_buttons.on_click
    def _on_yaw(event: viser.GuiEvent) -> None:
        for client in _client_or_broadcast():
            if event.target.value in (I18N["ja"]["yaw_left"], I18N["en"]["yaw_left"]):
                _yaw_camera_world_z(client, yaw_speed_gui.value)
            else:
                _yaw_camera_world_z(client, -yaw_speed_gui.value)

    @pitch_buttons.on_click
    def _on_pitch(event: viser.GuiEvent) -> None:
        for client in _client_or_broadcast():
            if event.target.value in (I18N["ja"]["pitch_up"], I18N["en"]["pitch_up"]):
                _pitch_camera_local(client, -pitch_speed_gui.value)
            else:
                _pitch_camera_local(client, pitch_speed_gui.value)

    @roll_buttons.on_click
    def _on_roll(event: viser.GuiEvent) -> None:
        for client in _client_or_broadcast():
            if event.target.value in (I18N["ja"]["roll_left"], I18N["en"]["roll_left"]):
                _roll_camera_local(client, roll_speed_gui.value)
            else:
                _roll_camera_local(client, -roll_speed_gui.value)

    render_all()

    try:
        while True:
            time.sleep(10.0)
    except KeyboardInterrupt:
        print("Stopping viewer...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--center", action="store_true")
    parser.add_argument("--quat_order", type=str, choices=["wxyz", "xyzw"], default="wxyz")
    parser.add_argument("--scale_multiplier", type=float, default=1.0)
    parser.add_argument("--max_scale_percentile", type=float, default=90.0)
    parser.add_argument("--opacity_threshold", type=float, default=0.05)
    parser.add_argument("--max_points", type=int, default=200000)
    parser.add_argument("--random_seed", type=int, default=0)

    args = parser.parse_args()

    main(
        input=tuple(args.input),
        host=args.host,
        port=args.port,
        center=args.center,
        quat_order=args.quat_order,
        scale_multiplier=args.scale_multiplier,
        max_scale_percentile=args.max_scale_percentile,
        opacity_threshold=args.opacity_threshold,
        max_points=args.max_points,
        random_seed=args.random_seed,
    )