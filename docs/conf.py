# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'UNe3dMe'
copyright = '2026, WSuenaga'
author = 'WSuenaga'
release = 'v1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc', # Python の docstring から説明文を自動生成する
    'sphinx.ext.napoleon', # Google 形式や NumPy 形式の docstring をきれいに解釈する
    'sphinx.ext.githubpages', # GitHub Pages 用の .nojekyll などを出力する
    "myst_parser", # Markdown ファイルを Sphinx で扱えるようにする
    'sphinxcontrib.mermaid', # フローチャート作成用
]

# どの拡張子をどの文書形式として読むかを指定
source_suffix = {
    ".md": "markdown",
}

# ウェルカムページ
root_doc = "index"

# 独自テンプレートを置くディレクトリ
templates_path = ['_templates']
# ビルド対象から除外するものを指定
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

language = 'ja'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# テーマ
html_theme = 'sphinx_rtd_theme'

# 各ページの左サイドバーに何を表示するかを指定
#  globaltoc.html：全体目次
#  relations.html：前へ／次へ
#  searchbox.html：検索欄
html_sidebars = {
    '**': ['globaltoc.html', 'relations.html', 'searchbox.html'],
}

# テーマ固有の設定
#  globaltoc_includehidden = True ; :hidden: を付けた toctree も全体目次に含める
#  globaltoc_collapse = False ; 左の目次を折りたたみすぎず，展開気味に表示する
html_theme_options = {
    'globaltoc_includehidden': True,
    'globaltoc_collapse': False,
}

# 独自 CSS や画像などの静的ファイル置き場
html_static_path = ['_static']

