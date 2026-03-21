import os
import local_backend
import methods
import tempfile
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

        # 入力・出力を保存するディレクトリの作成
        datasets = os.path.join(tmpdir, "datasets")
        os.mkdir(datasets)
        outputs = os.path.join(tmpdir, "outputs")
        os.mkdir(outputs)

        local_backend.TMPDIR = methods.TMPDIR =tmpdir
        
        # Gradio Demo起動
        main_demo(tmpdir, datasets, outputs)
