from setuptools import setup, find_packages

setup(
    name="kinoco",  # プロジェクトの名前
    version="25.04.18",   # バージョン
    packages=find_packages(),  # すべてのパッケージを自動検出
    include_package_data=True,
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "adjustText"
    ],  # 依存パッケージ
)

