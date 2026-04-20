import os
import tempfile

import local_backend
import methods
from demo import main_demo

if __name__ == '__main__':
    with tempfile.TemporaryDirectory(prefix='GradioDemo_') as tmpdir:
        # 作業ディレクトリの表示
        title = "Working Directory"
        max_width = max(len(title), len(tmpdir))
        GREEN = "\033[0;32m"
        RESET = "\033[0m"
        top_bottom = GREEN + "=" * (max_width + 2) + RESET
        title_line = GREEN + f" {title.ljust(max_width)} " + RESET
        path_line = GREEN + f" {tmpdir.ljust(max_width)} " + RESET
        print(f"\n{top_bottom}\n{title_line}\n{path_line}\n{top_bottom}\n")

        # 入力，出力，ログを保存するディレクトリの作成
        datasets = os.path.join(tmpdir, "datasets")
        os.mkdir(datasets)
        outputs = os.path.join(tmpdir, "outputs")
        os.mkdir(outputs)
        log_dir = os.path.join(tmpdir, "logs")
        os.mkdir(log_dir)

        # 作業ディレクトリの場所を教える
        local_backend.TMPDIR = methods.TMPDIR = tmpdir
        
        # Web UI 起動
        main_demo(tmpdir, datasets, outputs)
