#!/usr/bin/python
# -*- coding:  utf-8 -*-
"""
共有のグローバル変数を保管。
"""
import os
import re
from functools import reduce
import glob
import pandas as pd

module_dir = os.path.dirname(os.path.abspath(__file__))

def read_data_csv(fname):
    df = None
    for skip in [18, 33]:
        try:
            df = pd.read_csv(fname, skiprows=skip, encoding='shift_jis')
            if 'Time' in df.columns or df.columns[0] == '番号':
                break
        except Exception:
            continue
    if df is None:
        raise ValueError(f"読み込み失敗: {fname}")
    # 先頭行がヘッダーになっていない場合に対応
    if df.columns[0] == '番号':
        df = df.iloc[1:].reset_index(drop=True)
    # 列名の置換
    new_columns = {}
    for col in df.columns:
        match = re.match(r'CH-(\d+)\[℃\]', col)
        if match:
            new_columns[col] = f'CH{match.group(1)}'
        elif col == '番号':
            new_columns[col] = 'Time'
    df.rename(columns=new_columns, inplace=True)
    # Time列の文字列→数値変換 (変換できるものだけ)
    if 'Time' in df.columns:
        df['Time'] = pd.to_numeric(df['Time'], errors='coerce').astype('Int64')  # 欠損にも対応
    # CH列を float に変換（BURNOUT などに対応）
    ch_cols = [col for col in df.columns if col.startswith('CH')]
    df[ch_cols] = df[ch_cols].apply(pd.to_numeric, errors='coerce')
    return df

def read_data_exp(date, group_num):
    list_path = glob.glob(os.path.join(module_dir, 'data', date, group_num, '*.CSV'))
    list_path_xlsx = glob.glob(os.path.join(module_dir, 'data', date, group_num, '*.xlsx'))
    df_raw = pd.read_excel(list_path_xlsx[0], header=None)
    df_raw
    df_map = df_raw.iloc[2:].copy()
    df_map.columns = ['熱電対番号', '1回目', '2回目']
    df_map.replace('-', pd.NA, inplace=True)
    map_ch_1 = {f'CH{int(row["熱電対番号"])}': f'sample_{int(row["1回目"])}'
                for i, row in df_map.iterrows()
                if pd.notna(row['1回目'])}
    map_ch_2 = {f'CH{int(row["熱電対番号"])}': f'sample_{int(row["2回目"])}'
                for i, row in df_map.iterrows()
                if pd.notna(row['2回目'])}

    df_list = []
    for path in list_path:
        df = read_data_csv(path)

        if '1回目' in path:
            df = df.rename(columns=map_ch_1)
            df['回'] = 1
        elif '2回目' in path:
            df = df.rename(columns=map_ch_2)
            df['回'] = 2
        sample_cols = ['Time'] + [col for col in df.columns if col.startswith('sample_')] + ['回']
        df = df[sample_cols]
        df_list.append(df)
    df_merged_by_time = reduce(lambda left, right: pd.merge(left, right, on='Time', how='outer'), df_list)

    # sample_番号順に並び替え
    sample_cols_sorted = sorted([col for col in df_merged_by_time.columns if col.startswith('sample_')],
                                key=lambda x: int(x.split('_')[1]))
    final_columns = ['Time'] + sample_cols_sorted 
    df_final = df_merged_by_time[final_columns]

    return df_final

# 読み込み
exp_data = {}
dates = ['250415']
groups = ['group1', 'group2']

for dd in dates:
    for gg in groups:
        exp_data.update({f'{dd}_{gg}': read_data_exp(dd, gg)})

