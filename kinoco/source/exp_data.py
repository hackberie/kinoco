#!/usr/bin/python
# -*- coding:  utf-8 -*-
"""
共有のグローバル変数を保管。
"""
import os
import re
from functools import reduce
import glob
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from adjustText import adjust_text


from scipy.spatial import Delaunay
import plotly.graph_objects as go
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import RBF


from kinoco.utility.equilibrium import GibbsTriangle as GT
from kinoco.utility.equilibrium import GibbsTrianglePlotly as GTP
from kinoco.source.matplotlib_condition import plt, Cmap

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
        # 欠損にも対応
        df['Time'] = pd.to_numeric(df['Time'], errors='coerce').astype('Int64')  
    # CH列を float に変換（BURNOUT などに対応）
    ch_cols = [col for col in df.columns if col.startswith('CH')]
    df[ch_cols] = df[ch_cols].apply(pd.to_numeric, errors='coerce')
    return df


def read_data_exp(date, group_num):
    list_path = glob.glob(os.path.join(
        module_dir, 'data', date, group_num, '*.CSV'))
    list_path_xlsx = glob.glob(os.path.join(
        module_dir, 'data', date, group_num, '*.xlsx'))
    # print(list_path, list_path_xlsx)
    df_raw = pd.read_excel(list_path_xlsx[0], header=None)
    df_raw
    df_map = df_raw.iloc[2:].copy()
    df_map.columns = ['熱電対番号', '1回目', '2回目']
    pd.set_option('future.no_silent_downcasting', True)
    df_map.replace('-', pd.NA, inplace=True)
    # df_map = df_map.infer_objects(copy=False)
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


def plot_cooling_curve_smooth(pd_dataframe, data_name, wo_plot=False):
    x = pd_dataframe['Time'].values # .values は ndarray に変換
    # y = pd_dataframe[data_name].values # .values は ndarray に変換
    # 文字列にも対応
    y = pd.to_numeric(pd_dataframe[data_name], errors='coerce').values
    mask = ~np.isnan(y)
    x = x[mask]
    y = y[mask]

    yy = y[y == y.max()]
    xx = x[y == y.max()]
    xxx = x[x > xx[0]]
    yyy = y[x > xx[0]]
    xxxx = xxx[yyy < 250]
    original = xxxx[0]
    xxxx = xxxx - original # 初期値を 0 に
    yyyy = yyy[yyy < 250]

    dTdt = savgol_filter(yyyy, window_length=51, polyorder=3, deriv=1, delta=1)
    peaks, _ = find_peaks(dTdt, prominence=0.01, distance=50)

    new_peaks = []
    for pp in peaks:
        if dTdt[pp] < 0:
            new_peaks.append(pp)
            continue
        local_y = dTdt[pp:pp+30]
        local_x = xxxx[pp:pp+30]
        ## 0 に近いところを探す
        dTdt0 = local_y[
            (local_y)**2 == ((local_y)**2).min()][0]
        new_peaks.append(xxxx[dTdt == dTdt0][0])
    peaks = np.array(new_peaks, dtype=int)
    if not wo_plot:
        print(
            f"ピーク位置の時間 original (sec)： {', '.join(map(str, xxxx[peaks]+original))}")
        print(f"ピーク位置の時間 (sec)： {', '.join(map(str, xxxx[peaks]))}")
        print(f"ピーク位置の温度 (℃)： {', '.join(map(str, yyyy[peaks]))}")
    plt.plot(xxxx, yyyy, label=data_name)

    c1, c2 = Cmap.get_pair_my_tab()
    texts = []
    for i, ppp in enumerate(peaks):
        plt.plot(xxxx[ppp], yyyy[ppp], 'x')
        texts.append(plt.text(xxxx[ppp], yyyy[ppp]+5,
                              s=yyyy[ppp], size=15, color=c2[i+1]))
    adjust_text(texts, only_move={'points':'none', 'texts':'xy'})

    plt.xlabel('Time (s)')
    plt.ylabel('Temperature (℃)')

    ## 現在の axis を習得 (get carrent axis)
    ax1 = plt.gca()
    ## 右側の y 軸を使ったプロットに変更 (twinx:ダブルy軸にする)
    ax2 = ax1.twinx()
    plt.plot(xxxx, dTdt, color=c2[1], label=r'$dT/dt$')
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)
    plt.ylim(-2, 10)
    plt.ylabel(r'$dT/dt$ (℃/s)')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    if wo_plot:
        plt.close()
    return yyyy[peaks[0]]


def plot_cooling_curve_all_data(df):
    """
    全データの可視化
    sample_** のデータを表示する
    """
    samples = [x for x in df.columns if 'sample_' in x]
    nn = len(samples)
    midle = round(nn/2) - 1
    vv = midle+1

    fig, axes = plt.subplots(vv, 2, figsize=(10, vv/2*5), 
                             constrained_layout=True)
    axes = axes.T.reshape(-1)
    data_tm = []
    for i in range(nn):
        print(f'sample_{i+1}')
        plt.sca(axes[i])
        tm = plot_cooling_curve_smooth(df, f'sample_{i+1}')
        data_tm.append({'sample': f'sample_{i+1}', 'T_m': tm})
        plt.sca(axes[i])
        plt.ylabel(f'$T (℃)$')
        plt.ylim(0, 300)
        if (not i == midle) and (not i == 2*midle+1):
          axes[i].set_xticklabels([])
          plt.xlabel('')
        plt.xlim(0, 1000)

    plt.sca(axes[-1])
    plt.xlim(0, 1000)
    plt.ylim(0, 300)
    plt.xlabel('Time (s)')

    data_tm = pd.DataFrame(data_tm)
    return data_tm


def mass2atomic(mass_Bi, mass_In, mass_Sn):
    M_Bi = 208.98
    M_In = 114.82
    M_Sn = 118.71
    n_Bi = mass_Bi / M_Bi
    n_In = mass_In / M_In
    n_Sn = mass_Sn / M_Sn
    n_total = n_Bi + n_In + n_Sn
    x_Bi = n_Bi/n_total
    x_In = n_In/n_total
    x_Sn = n_Sn/n_total
    return x_Bi, x_In, x_Sn
  

def mass2atomic(mass_Bi, mass_In, mass_Sn):
    M_Bi = 208.98
    M_In = 114.82
    M_Sn = 118.71
    n_Bi = mass_Bi / M_Bi
    n_In = mass_In / M_In
    n_Sn = mass_Sn / M_Sn
    n_total = n_Bi + n_In + n_Sn
    x_Bi = n_Bi/n_total
    x_In = n_In/n_total
    x_Sn = n_Sn/n_total
    return x_Bi, x_In, x_Sn
  

def atomic2mass(n_Bi, n_In, n_Sn):
    M_Bi = 208.98
    M_In = 114.82
    M_Sn = 118.71

    mass_Bi = n_Bi * M_Bi
    mass_In = n_In * M_In
    mass_Sn = n_Sn * M_Sn

    mass_total = mass_Bi + mass_In + mass_Sn

    w_Bi = mass_Bi/mass_total
    w_In = mass_In/mass_total
    w_Sn = mass_Sn/mass_total
    target_total_mass = 5 # (5g)
    return w_Bi*target_total_mass, w_In*target_total_mass, w_Sn*target_total_mass


def predict3d_plotly(
    src_x, src_y, nu=3/2, alpha=0.005, beta=1.96,
    show=True, new_data=[]):
    """ plotly を使用する """
    if len(new_data):
        src_x = np.r_[src_x, new_data[:, :3]]
        src_y = np.r_[src_y, new_data[:, -1]]
    kernel = Matern(nu=nu)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha,
                                n_restarts_optimizer=20, normalize_y=True)
    gp.fit(src_x, src_y)
    mesh = 101
    x = np.linspace(0, 1, mesh)
    y = np.linspace(0, 1, mesh)
    xx, yy = np.meshgrid(x, y)
    xx = xx.flatten()
    yy = yy.flatten()
    mask = (xx + yy <= 1.0)
    xx = xx[mask]
    yy = yy[mask]
    zz = 1-xx-yy
    x_grid = np.c_[xx, yy, zz]
    y_pred, sigma = gp.predict(x_grid, return_std=True)

    ## 獲得関数 (LCB)
    y_min = np.min(src_y)  # 既存データの最小値
    lcb = y_pred - beta * sigma # 信頼度 95%
    ## LCB が最小の x_grid の値
    suggest = x_grid[lcb == lcb.min()][0]

    just = y_pred[lcb == lcb.min()][0]
    err = sigma[lcb == lcb.min()][0]
    predict = {'just': float(just), 
               'upper': float(just+err), 
               'lower': float(just-err)}

    if not show:
        return suggest, float(y_min - lcb.min()), predict

    gtp = GTP()
    gtp.make_frame('Bi', 'In', 'Sn')

    x = src_x[:, 0]
    y = src_x[:, 1]
    z = src_y

    x, y = GT.convert_triangle(x, y)

    fig_plotly = gtp.fig.add_trace(go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(size=5,
                    color='blue'),
        showlegend = False
        )
    )
    if len(new_data):
        dd = new_data
        x = dd[:, 0]
        y = dd[:, 1]
        z = dd[:, 3]
        x, y = GT.convert_triangle(x, y)

        fig_plotly = gtp.fig.add_trace(go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(size=5,
                        color='red'),
            showlegend = False
            )
        )

    x, y = GT.convert_triangle(xx, yy)
    # 散らばった点群から三角形をつなぐ
    # 三角分割法 (できるだけ鋭角ができない三角形で面を張る)
    triangles = Delaunay(np.column_stack([x, y])).simplices

    fig_plotly = gtp.fig.add_trace(go.Mesh3d(
        x=x,
        y=y,
        z=y_pred,
        i=triangles[:, 0], j=triangles[:, 1], k=triangles[:, 2],
        color='cyan',
        opacity=0.5,
        )
    )

    z = y_pred - 1.96 * sigma
    fig_plotly = gtp.fig.add_trace(go.Mesh3d(
        x=x,
        y=y,
        z=z,
        i=triangles[:, 0], j=triangles[:, 1], k=triangles[:, 2],
        opacity=0.5, colorscale='Plotly3', intensity=sigma,
        showscale=False 
    ))

    z = y_pred + 1.96 * sigma
    fig_plotly = gtp.fig.add_trace(go.Mesh3d(
        x=x,
        y=y,
        z=z,
        i=triangles[:, 0], j=triangles[:, 1], k=triangles[:, 2],
        # color='blue',
        opacity=0.5, colorscale='Plotly3', intensity=sigma,
        showscale=False 
    ))

    gtp.fig.update_layout(
        scene_camera=dict(projection=dict(type="orthographic"))
    )

    score = y_min - lcb
    score[score < 0] = 0
    fig_plotly = gtp.fig.add_trace(go.Mesh3d(
        x=x,
        y=y,
        z=x*0,
        i=triangles[:, 0], j=triangles[:, 1], k=triangles[:, 2],
        # color='blue',
        opacity=0.5, colorscale='Viridis', intensity=score,
        # showscale=False 
    ))

    gtp.fig.update_layout(
        scene=dict(
            xaxis=dict(
                title='',
                type='linear',  # 線形スケールを指定
                range=[0, 1]  # x軸の範囲を設定
            ),
            yaxis=dict(
                title='',
                type='linear',  # 線形スケールを指定
                range=[0, 1]  # y軸の範囲を設定
            )))


    gtp.show()

    just = y_pred[lcb == lcb.min()][0]
    err = sigma[lcb == lcb.min()][0]
    predict = {'just': float(just), 
               'upper': float(just+err), 
               'lower': float(just-err)}

    return suggest, float(y_min - lcb.min()), predict


def set_atomic(df_orig):
    df = df_orig.copy()
    x_Bi, x_In, x_Sn = mass2atomic(df['w_Bi'], df['w_In'], df['w_Sn'])
    df['x_Bi'] = x_Bi
    df['x_In'] = x_In
    df['x_Sn'] = x_Sn
    return df

def set_mass(df_orig):
    df = df_orig.copy()
    w_Bi, w_In, w_Sn = atomic2mass(df['x_Bi'], df['x_In'], df['x_Sn'])
    df['w_Bi'] = w_Bi
    df['w_In'] = w_In
    df['w_Sn'] = w_Sn
    return df


# 読み込み group_A
exp_data = {}
dates = ['250415']
groups = ['group1', 'group2']

for dd in dates:
    for gg in groups:
        exp_data.update({f'{dd}_{gg}': read_data_exp(dd, gg)})

# 読み込み group_D
dates = ['250520']
groups = ['group1', 'group2']

for dd in dates:
    for gg in groups:
        fname = glob.glob(os.path.join(module_dir, 'data', dd, gg, 
                                       'Group_D_*.csv'))[0]
        exp_data.update({f'{dd}_{gg}': pd.read_csv(fname)})

# 読み込み
exp_data_tm = {}
dates = ['250415', '250520']
groups = ['group1', 'group2']
df_pure = [{'w_Bi':5, 'w_In':0, 'w_Sn':0, 
            'T_m_script': 271.4, 'T_m_excel': 271.4, 'sample': 'sample_Bi'},
           {'w_Bi':0, 'w_In':5, 'w_Sn':0, 
            'T_m_script': 156.6, 'T_m_excel': 156.6, 'sample': 'sample_In'},
           {'w_Bi':0, 'w_In':0, 'w_Sn':5, 
            'T_m_script': 231.9, 'T_m_excel': 231.9, 'sample': 'sample_Sn'}]
df_pure = pd.DataFrame(df_pure)
for dd in dates:
    for i, gg in enumerate(groups):
        df = pd.read_csv(
                os.path.join(module_dir, 'data', dd, gg, f'data_tm{i+1}.csv'))
        df = pd.concat([df, df_pure])
        df = set_atomic(df)
        exp_data_tm.update({f'{dd}_{gg}': df})

# 読み込み group_B
# exp_data = 
