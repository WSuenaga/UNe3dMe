# Copyright 2022 the Regents of the University of California,
# Nerfstudio Team and contributors.
#
# This file is derived from the Nerfstudio viewer implementation.
# Modifications:
#   - Custom Viser GUI
#   - Viewer launcher adapted for local repository layout
#   - Custom UI language switch (JA/EN) for this file's UI only
#   - Added a Shutdown button to stop the viewer process
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Custom Nerfstudio Viewer launcher:
- Rendering logic is 100% Nerfstudio (Viewer + RenderStateMachine).
- Only Viser GUI is customized (added), default GUI is hidden where possible.
- Custom UI can switch language (Japanese/English) ONLY for the custom UI.
- Adds a Shutdown button (SIGINT).
"""

from __future__ import annotations

import os
import signal
import sys
import time
import threading
from dataclasses import dataclass, field, fields
from pathlib import Path
from threading import Lock
from typing import Literal, Optional

import tyro

# ------------------------------------------------------------
# ✅ Make sure "nerfstudio" is imported from your local repo copy
# (Prevents class/type mismatch and ensures you run your repo version)
# ------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
DEMO_ROOT = THIS_FILE.parent  # .../Demo
NS_REPO_ROOT = DEMO_ROOT / "models" / "nerfstudio"  # .../Demo/models/nerfstudio

# Put repo root first so `import nerfstudio` resolves to .../models/nerfstudio/nerfstudio
sys.path.insert(0, str(NS_REPO_ROOT))

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils import writer
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.viewer.viewer import Viewer as NsViewer
from nerfstudio.viewer_legacy.server.viewer_state import ViewerLegacyState


# ----------------------------
# GUI customized Viewer
# ----------------------------
class CustomViewer(NsViewer):
    """
    Nerfstudio の Viewer を継承し，独自 GUI を追加したカスタム Viewer．

    レンダリング処理そのものは Nerfstudio 本体の実装をそのまま利用し，
    Viser 上の GUI だけを追加・調整する．
    """

    def __init__(self, *args, **kwargs):
        """
        CustomViewer を初期化し，独自 GUI を構築する．

        Args:
            *args: 親クラスへ渡す位置引数．
            **kwargs: 親クラスへ渡すキーワード引数．
        """
        super().__init__(*args, **kwargs)  # ✅ Nerfstudio does everything (render, statemachine, etc.)
        self._install_custom_gui()

    def _hide_default_gui_bits(self) -> None:
        """
        利用可能な場合に限り，Nerfstudio 標準 GUI の一部を非表示にする．

        バージョン差異により対象ハンドルが存在しない場合があるため，
        安全に無視できるようにしている．
        """
        for name in [
            "pause_train",
            "resume_train",
            "hide_images",
            "show_images",
            "stats_markdown",
        ]:
            h = getattr(self, name, None)
            try:
                if h is not None:
                    h.visible = False
            except Exception:
                pass

    def _safe_sigint_shutdown(self) -> None:
        """
        自プロセスへ SIGINT を送信し，Viewer の停止を試みる．

        Ctrl+C 相当の停止を行うための補助メソッドである．
        """
        print("[INFO] Shutting down server (SIGINT)...")
        os.kill(os.getpid(), signal.SIGINT)

    @staticmethod
    def _safe_set_label(handle, value: str) -> None:
        """
        GUI ハンドルのラベル文字列を安全に更新する．

        viser のバージョン差異により，`.label` または `.text` を
        用いる場合があるため，利用可能な属性へ設定する．
        更新できない場合は例外を無視する．

        Args:
            handle: ラベル更新対象の GUI ハンドル．
            value: 設定する文字列．
        """
        for attr in ("label", "text"):
            try:
                if hasattr(handle, attr):
                    setattr(handle, attr, value)
                    return
            except Exception:
                pass

    def _install_custom_gui(self) -> None:
        """
        独自タブと各種 GUI 部品を追加し，イベントコールバックを登録する．

        言語切り替え，再レンダリング，学習カメラ表示切り替え，
        最大解像度変更，サーバ停止などの UI を構築する．
        """
        gui = self.viser_server.gui

        # Hide some default widgets (tabs themselves may remain; safest approach)
        self._hide_default_gui_bits()

        # Theme tweak (optional)
        try:
            gui.configure_theme(
                control_layout="collapsible",
                dark_mode=True,
                brand_color=(80, 180, 255),
            )
        except Exception:
            pass

        # ----------------------------
        # i18n (custom UI only)
        # ----------------------------
        I18N = {
            "ja": {
                "tab_title": "拡張機能 / Extensions",
                "language_label": "言語 / Language",
                "language_ja": "日本語",
                "language_en": "English",
                "header": "### 拡張機能",

                "render_description": "- 現在のカメラ位置からシーンを再レンダリング",
                "render": "再レンダリング",

                "show_train_cams": "学習カメラを表示",
                "max_res": "最大解像度",
                "max_res_unavailable": "> この環境では最大解像度の変更は利用できません。",
                "step_unknown": "Step: (不明)",

                "server_control": "### サーバ停止",
                "confirm_shutdown": "サーバを停止する",
                "shutdown": "サーバ停止",
            },
            "en": {
                "tab_title": "拡張機能 / Extensions",
                "language_label": "言語 / Language",
                "language_ja": "日本語",
                "language_en": "English",
                "header": "### Extensions",

                "render_description": "- Re-render the scene from the current camera position",
                "render": "Re-render",

                "show_train_cams": "Show Training Cameras",
                "max_res": "Max Resolution",
                "max_res_unavailable": "> Max resolution control is not available in this environment.",
                "step_unknown": "Step: (unknown)",

                "server_control": "### Server Shutdown",
                "confirm_shutdown": "Confirm server shutdown",
                "shutdown": "Shutdown Server",
            },
        }
        lang_state = {"lang": "ja"}  # default

        def t(key: str) -> str:
            """
            現在の言語状態に応じた UI 文字列を返す．

            Args:
                key: I18N 辞書のキー．

            Returns:
                対応する UI 文字列．
            """
            return I18N.get(lang_state["lang"], I18N["en"]).get(key, key)

        # Add a new tab group for your UI
        tabs = gui.add_tab_group()
        my_tab = tabs.add_tab(t("tab_title"), icon=None)

        with my_tab:
            # Language selector (custom UI only)
            lang_sel = gui.add_dropdown(
                t("language_label"),
                options=[I18N["ja"]["language_ja"], I18N["en"]["language_en"]],
                initial_value=I18N["ja"]["language_ja"],
            )

            header_md = gui.add_markdown(t("header"))

            render_desc_md = gui.add_markdown(t("render_description"))
            rerender_btn = gui.add_button(t("render"), icon=None)

            # Toggle train cameras visibility using Nerfstudio's existing method
            show_train_chk = gui.add_checkbox(t("show_train_cams"), initial_value=True)

            # Max res slider (optional)
            slider = None
            max_res_note_md = None
            try:
                current = int(self.control_panel.max_res)
                slider = gui.add_slider(t("max_res"), min=64, max=2048, step=64, initial_value=current)
            except Exception:
                max_res_note_md = gui.add_markdown(t("max_res_unavailable"))

            # Step label
            step_md = gui.add_markdown(t("step_unknown"))

            # Shutdown (single button + confirm checkbox to reduce misclicks)
            server_md = gui.add_markdown(t("server_control"))
            confirm_chk = gui.add_checkbox(t("confirm_shutdown"), initial_value=False)
            shutdown_btn = gui.add_button(t("shutdown"), icon=None)
            shutdown_btn.disabled = True

            # ---- callbacks ----
            def _update_step(_=None):
                """
                現在の step 情報を GUI 表示へ反映する．
                """
                try:
                    step_md.content = f"Step: {int(getattr(self, 'step', 0))}"
                except Exception:
                    pass

            def _rerender(_):
                """
                再レンダリングを実行し，step 表示を更新する．
                """
                self._trigger_rerender()
                _update_step()

            rerender_btn.on_click(_rerender)

            def _toggle_traincams(_):
                """
                学習カメラの表示・非表示を切り替える．
                """
                try:
                    self.set_camera_visibility(bool(show_train_chk.value))
                except Exception:
                    pass

            show_train_chk.on_update(_toggle_traincams)

            if slider is not None:
                def _set_res(_):
                    """
                    最大解像度を更新し，再レンダリングを行う．
                    """
                    try:
                        self.control_panel.max_res = int(slider.value)
                        self._trigger_rerender()
                    except Exception:
                        pass

                slider.on_update(_set_res)

            def _update_enabled(_):
                """
                確認チェックボックスの状態に応じて停止ボタンの有効状態を切り替える．
                """
                shutdown_btn.disabled = not bool(confirm_chk.value)

            confirm_chk.on_update(_update_enabled)

            def _shutdown(_):
                """
                少し遅延させた別スレッドでサーバ停止処理を呼び出す．
                """
                # Avoid killing inside GUI callback: do it shortly later on a daemon thread
                def worker():
                    """
                    短時間待機してから SIGINT による停止を行うワーカースレッド．
                    """
                    time.sleep(0.1)
                    self._safe_sigint_shutdown()

                threading.Thread(target=worker, daemon=True).start()

            shutdown_btn.on_click(_shutdown)

            # ---- language switch: update what we can safely update ----
            def _on_lang_change(_):
                """
                カスタム UI の表示言語を切り替え，更新可能なラベルを再設定する．
                """
                lang_state["lang"] = "ja" if lang_sel.value == I18N["ja"]["language_ja"] else "en"

                # Update markdown 
                header_md.content = t("header")
                server_md.content = t("server_control")
                render_desc_md.content = t("render_description")

                # Update labels if possible (depends on viser version)
                self._safe_set_label(lang_sel, t("language_label"))
                self._safe_set_label(rerender_btn, t("render"))
                self._safe_set_label(show_train_chk, t("show_train_cams"))
                self._safe_set_label(confirm_chk, t("confirm_shutdown"))
                self._safe_set_label(shutdown_btn, t("shutdown"))

                if slider is not None:
                    self._safe_set_label(slider, t("max_res"))
                if max_res_note_md is not None:
                    max_res_note_md.content = t("max_res_unavailable")

                # Step label keeps numeric value; just leave as-is or update prefix
                _update_step()

            lang_sel.on_update(_on_lang_change)


# ----------------------------
# run_viewer.py equivalent
# ----------------------------
@dataclass
class ViewerConfigWithoutNumRays(ViewerConfig):
    """
    CLI から `num_rays_per_chunk` を隠すための ViewerConfig 派生クラス．
    """
    num_rays_per_chunk: tyro.conf.Suppress[int] = -1

    def as_viewer_config(self):
        """
        現在の設定値から通常の ViewerConfig を生成して返す．

        Returns:
            ViewerConfig に変換した設定オブジェクト．
        """
        return ViewerConfig(**{x.name: getattr(self, x.name) for x in fields(self)})


@dataclass
class RunViewer:
    """
    チェックポイントを読み込み，eval モードで Viewer を起動するための設定クラス．

    カスタム GUI を用いた viewer と legacy viewer の両方を扱える．
    """
    load_config: Path
    viewer: ViewerConfigWithoutNumRays = field(default_factory=ViewerConfigWithoutNumRays)
    vis: Literal["viewer", "viewer_legacy"] = "viewer"

    def main(self) -> None:
        """
        設定ファイルを読み込み，必要な Viewer 設定を補正したうえで Viewer を起動する．
        """
        config, pipeline, _, step = eval_setup(
            self.load_config,
            eval_num_rays_per_chunk=None,
            test_mode="test",
        )
        num_rays_per_chunk = config.viewer.num_rays_per_chunk
        assert self.viewer.num_rays_per_chunk == -1

        config.vis = self.vis
        config.viewer = self.viewer.as_viewer_config()
        config.viewer.num_rays_per_chunk = num_rays_per_chunk

        _start_viewer(config, pipeline, step)

    def save_checkpoint(self, *args, **kwargs):
        """
        Trainer 互換のために用意されたダミーメソッド．

        Args:
            *args: 未使用の位置引数．
            **kwargs: 未使用のキーワード引数．
        """
        pass


def _start_viewer(config: TrainerConfig, pipeline: Pipeline, step: int):
    """
    設定とパイプラインを受け取り，Viewer を初期化して実行し続ける．

    Args:
        config: Trainer 全体の設定．
        pipeline: 評価用に初期化済みのパイプライン．
        step: 現在の学習ステップ数．
    """
    base_dir = config.get_base_dir()
    viewer_log_path = base_dir / config.viewer.relative_log_filename
    viewer_callback_lock = Lock()

    if config.vis == "viewer_legacy":
        viewer_state = ViewerLegacyState(
            config.viewer,
            log_filename=viewer_log_path,
            datapath=pipeline.datamanager.get_datapath(),
            pipeline=pipeline,
            train_lock=viewer_callback_lock,
        )
        banner_messages = [f"Legacy viewer at: {viewer_state.viewer_url}"]
    else:
        viewer_state = CustomViewer(
            config.viewer,
            log_filename=viewer_log_path,
            datapath=pipeline.datamanager.get_datapath(),
            pipeline=pipeline,
            share=config.viewer.make_share_url,
            train_lock=viewer_callback_lock,
        )
        banner_messages = viewer_state.viewer_info

    # We don't need logging, but writer.GLOBAL_BUFFER needs to be populated
    config.logging.local_writer.enable = False
    writer.setup_local_writer(config.logging, max_iter=config.max_num_iterations, banner_messages=banner_messages)

    assert pipeline.datamanager.train_dataset is not None
    viewer_state.init_scene(
        train_dataset=pipeline.datamanager.train_dataset,
        train_state="completed",
        eval_dataset=pipeline.datamanager.eval_dataset,
    )
    if isinstance(viewer_state, ViewerLegacyState):
        viewer_state.viser_server.set_training_state("completed")

    viewer_state.update_scene(step=step)

    while True:
        time.sleep(0.01)


def entrypoint():
    """
    CLI エントリポイント．

    tyro の見た目設定を行い，RunViewer を CLI から実行する．
    """
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(tyro.conf.FlagConversionOff[RunViewer]).main()


if __name__ == "__main__":
    entrypoint()