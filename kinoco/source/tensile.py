#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
引張試験データの読み込みとプロット。

データ: Mg-1Ca-0.5Zn 合金を HPT (6 GPa, 1 rpm, N = 50) 加工後、
200 / 250 / 300 °C で焼鈍した試料の引張試験 (島津製作所)。
元の xlsx から実データ部分だけを CSV に変換して kinoco/source/data/tensile/ に置いている。

列の意味
    time           時間 (s)
    force          試験力 (N)
    stroke         ストローク (mm)
    stress         応力 (N/mm2 = MPa)
    strain_stroke  ストロークから求めたひずみ (%)
    disp           変位 (mm)
    strain_disp    変位から求めたひずみ (%)
"""
import os
import re
import glob
import pandas as pd
from kinoco.source.matplotlib_condition import plt, Cmap

module_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(module_dir, 'data', 'tensile')

COLUMNS = {
    'time': '時間 (s)',
    'force': '試験力 (N)',
    'stroke': 'ストローク (mm)',
    'stress': '応力 (MPa)',
    'strain_stroke': 'ストロークひずみ (%)',
    'disp': '変位 (mm)',
    'strain_disp': '変位ひずみ (%)',
}


def list_tensile() -> list[str]:
    """利用できるデータ名 (ファイル名から .csv を除いたもの) を返す"""
    paths = sorted(glob.glob(os.path.join(data_dir, '*.csv')))
    return [os.path.splitext(os.path.basename(p))[0] for p in paths]


def read_tensile(name: str) -> pd.DataFrame:
    """データ名を指定して 1 試料分の DataFrame を返す"""
    path = os.path.join(data_dir, f'{name}.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f'{name} がない。list_tensile() で確認: {list_tensile()}')
    return pd.read_csv(path)


def temperature_of(name: str) -> int:
    """データ名から焼鈍温度 (°C) を取り出す。例: '...-250C-N50' → 250"""
    match = re.search(r'-(\d+)C-', name)
    if match is None:
        raise ValueError(f'{name} から温度を読み取れない')
    return int(match.group(1))


def read_tensile_all() -> dict[int, pd.DataFrame]:
    """全データを {焼鈍温度: DataFrame} の辞書で返す"""
    return {temperature_of(name): read_tensile(name) for name in list_tensile()}


def plot_stress_strain(data: dict[int, pd.DataFrame], strain: str = 'strain_disp'):
    """応力–ひずみ曲線を温度ごとに重ねて描く

    data   : read_tensile_all() の戻り値
    strain : 横軸に使う列。'strain_disp' または 'strain_stroke'
    """
    _, edge = Cmap.get_pair_my_tab()
    for i, (temp, df) in enumerate(sorted(data.items())):
        plt.plot(df[strain], df['stress'], '-', color=edge[i], label=f'{temp} °C')
    plt.xlabel('Strain (%)')
    plt.ylabel('Stress (MPa)')
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.legend()


def summarize(data: dict[int, pd.DataFrame], strain: str = 'strain_disp') -> pd.DataFrame:
    """引張強さ (UTS) と破断ひずみの一覧表を返す"""
    rows = []
    for temp, df in sorted(data.items()):
        rows.append({'T_anneal (°C)': temp,
                     'UTS (MPa)': df['stress'].max(),
                     'strain at UTS (%)': df.loc[df['stress'].idxmax(), strain],
                     'max strain (%)': df[strain].max()})
    return pd.DataFrame(rows)
