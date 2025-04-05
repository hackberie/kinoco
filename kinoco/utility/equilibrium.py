#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
平衡計算用のモジュール
"""
import sys
import copy

import numpy as np
import pandas as pd
import matplotlib.ticker as ticker
import plotly.graph_objects as go

from kinoco.source.matplotlib_condition import plt

class GibbsTriangle:
    """
    Gibbs Triangle をプロットする
    font_size パラメータを追加
    label のフォーマット (tex, sort_alphabet) を追加
    """
    def __init__(self, ax, points=None, pairs=None, labels=None, font_size=12, 
                 labels_tex=False, labels_sort_alphabet=False):
        self.ax = ax
        self.points = points
        self.pairs = pairs
        self.labels = labels
        self.font_size = font_size
        self.labels_tex = labels_tex
        self.labels_sort_alphabet = labels_sort_alphabet

    @classmethod
    def from_dict_gs(cls, ax, dict_gs):
        """
        dict から object 生成
        ax は 必要
        """
        return cls(ax, dict_gs['points'], dict_gs['pairs'], dict_gs['labels'])

    @staticmethod
    def convert_triangle(array_x, array_y):
        """
        三角図の座標に変換する
        (1, 0, 0) -> (0.5, √3/2)
        (0, 1, 0) -> (0, 0)
        (0, 0, 1) -> (1, 0)
        """
        triangle_x = (1 - 0.5 * array_x - array_y)
        triangle_y = (np.sqrt(3) * 0.5 * array_x)
        return triangle_x, triangle_y

    def plot_line(self, x, y, color, lw, ls='-', alpha=1):
        """
        plot single line
        """
        tx, ty = self.convert_triangle(x, y)
        self.ax.plot(tx, ty, color=color, lw=lw, ls=ls, alpha=alpha)

    def plot_point(self, x, y, s, facecolors, edgecolors, marker='o',
                   zorder=1, alpha=1):
        """
        plot point
        """
        tx, ty = self.convert_triangle(x, y)
        self.ax.scatter(tx, ty, s=s,
                        facecolors=facecolors,
                        edgecolors=edgecolors,
                        marker=marker, zorder=zorder, alpha=alpha)

    def plot_text(self, x, y, text, size=None,  zorder=200, **args):
        """
        plot text
        """
        if not size:
            size = self.font_size
        tx, ty = self.convert_triangle(x, y)
        self.ax.text(tx, ty, text, size=size, zorder=zorder, **args)

    def plot_points(self, s=50, facecolors='cyan',
                    edgecolors='blue', marker='o', zorder=100):
        """
        plot points
        """
        x, y, _ = np.array(self.points).T  #pylint: disable=E0633
        self.plot_point(x, y, s=s,
                        facecolors=facecolors,
                        edgecolors=edgecolors,
                        marker=marker, zorder=zorder)

    def plot_pairs(self, color='blue', lw=1, ls='-'):
        """
        plot pairs
        """
        for p in self.pairs:
            x, y, _ = p.T
            self.plot_line(x, y, color, lw, ls)

    def plot_labels(self):
        """
        plot labels
        """
        n = 0
        labels = []
        for l in self.labels:
            l.tex = self.labels_tex
            l.sort_alphabet = self.labels_sort_alphabet
            labels.append(l)
        for p, ll in zip(self.points, labels):
            if not ll:
                continue
            l = str(ll)
            x, y, _ = p
            if x == 1:
                d = 0.03
                self.plot_text(x+d, y-d/2, l, horizontalalignment='center')
            elif y == 1:
                d = -0.03
                self.plot_text(x+d, y-d/2, l, horizontalalignment='right',
                               verticalalignment='top')
            elif x+y == 0:
                d = -0.03
                self.plot_text(x+d, y-d/2, l, horizontalalignment='left',
                               verticalalignment='top')
            elif x == 0:
                d = -0.02 - 0.05 * n
                self.plot_text(x+d, y-d/2, l, verticalalignment='top',
                               horizontalalignment='center')
                n += 1
            elif x+y == 1:
                self.plot_text(x, y+0.03, l, verticalalignment='center',
                               horizontalalignment='right')
            elif y == 0:
                self.plot_text(x, y-0.03, l, verticalalignment='center',
                               horizontalalignment='left')
            else:
                self.plot_text(x+0.03, y, l)

    def plot_labels_wo_ss(self):
        """
        ss_ (solid solute) から始まるラベル以外をプロットする
        """
        rsv_labels = copy.deepcopy(self.labels)
        rsv_points = copy.deepcopy(self.points)
        self.labels, self.points = [], []
        for l, p in zip(rsv_labels, rsv_points):
            if str(l)[:3] != 'SS_':
                self.labels.append(l)
                self.points.append(p)
        self.plot_labels()
        self.labels, self.points = rsv_labels, rsv_points

    def plot_points_wo_ss(self):
        """
        ss_ (solid solute) から始まるラベル以外をプロットする
        """
        rsv_labels = copy.deepcopy(self.labels)
        rsv_points = copy.deepcopy(self.points)
        self.labels, self.points = [], []
        for l, p in zip(rsv_labels, rsv_points):
            if str(l)[:3] != 'SS_':
                self.labels.append(l)
                self.points.append(p)
        self.plot_points()
        self.labels, self.points = rsv_labels, rsv_points

    def plot_points_ss(self):
        """
        ss_ (solid solute) から始まるラベル以外をプロットする
        aqua, lime, violet, black
        """
        colors = ('aqua', 'lime', 'violet', 'black')
        rsv_labels = copy.deepcopy(self.labels)
        rsv_points = copy.deepcopy(self.points)
        set_label_ss = [x for x in set([str(y) for y in self.labels]) 
                        if x[:3] == 'SS_']
        for i, l_ss in enumerate(set_label_ss):
            self.labels, self.points = [], []
            for l, p in zip(rsv_labels, rsv_points):
                if l == l_ss:
                    self.labels.append(l)
                    self.points.append(p)
            self.plot_points(s=20,
                             facecolors=colors[i],
                             edgecolors=colors[i],
                             marker='o', zorder=90)
            self.plot_point(0.8-0.1*i, 0.8+0.05*i, s=20,
                            facecolors=colors[i],
                            edgecolors=colors[i],
                            marker='o', zorder=90)
            self.plot_text(0.8-0.1*i, 0.8+0.05*i-0.03,
                           l_ss[3:], verticalalignment='center')
            # self.plot_point(0.7, 0.85, s=20,
            #                 facecolors=colors[i+1],
            #                 edgecolors=colors[i+1],
            #                 marker='o', zorder=90)
            # self.plot_point(0.6, 0.9, s=20,
            #                 facecolors=colors[i+2],
            #                 edgecolors=colors[i+2],
            #                 marker='o', zorder=90)
        self.labels, self.points = rsv_labels, rsv_points

    def make_frame(self, axis10=None, axis01=None, axis00=None, axis3d=False):
        """
        0.1 刻みのメッシュ作成
        axis** の軸は convert_triangle の引数と対応する (**=xy)
        3DAxis にも対応するように修正 
        """
        for i in range(0, 10):
            x = np.array([1 - i*0.1, 0])
            y = np.array([0, 1 - i*0.1])
            self.plot_line(x, y, color='gray', lw=0.5)

            x = np.array([i*0.1, i*0.1])
            y = np.array([0, 1 - i*0.1])
            self.plot_line(x, y, color='gray', lw=0.5)

            x = np.array([0, 1 - i*0.1])
            y = np.array([i*0.1, i*0.1])
            self.plot_line(x, y, color='gray', lw=0.5)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['bottom'].set_visible(False)
        self.ax.spines['left'].set_visible(False)
        self.ax.tick_params(axis='x', which='both',
                            top=False, bottom=False, labelbottom=False)
        self.ax.tick_params(axis='y', which='both',
                            left=False, right=False, labelleft=False)
        if not axis3d:
            self.ax.set_aspect('equal', adjustable='box')
        else: # 軸と pane を消去
            self.ax.xaxis.pane.set_edgecolor('none')
            self.ax.yaxis.pane.set_edgecolor('none')
            self.ax.zaxis.pane.set_edgecolor('none')
            self.ax.xaxis.pane.set_facecolor("none")
            self.ax.yaxis.pane.set_facecolor("none")
            self.ax.zaxis.pane.set_facecolor("none")
            self.ax.xaxis.set_major_locator(ticker.NullLocator())
            self.ax.xaxis.set_minor_locator(ticker.NullLocator())
            self.ax.yaxis.set_major_locator(ticker.NullLocator())
            self.ax.yaxis.set_minor_locator(ticker.NullLocator())
            self.ax.zaxis.set_major_locator(ticker.NullLocator())
            self.ax.zaxis.set_minor_locator(ticker.NullLocator())
            self.ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            self.ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            self.ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

        if axis10:
            if axis3d:
                self.ax.text(0.5, np.sqrt(3)/2+0.025, 0, axis10, ha='center')
            else:
                self.ax.text(0.5, np.sqrt(3)/2+0.025, axis10, ha='center')
        if axis01:
            if axis3d:
                self.ax.text(0.0, 0.0-0.025, 0, axis01, ha='center', va='top')
            else:
                self.ax.text(0.0, 0.0-0.025, axis01, ha='center', va='top')
        if axis00:
            if axis3d:
                self.ax.text(1.0, 0.0-0.025, 0, axis00, ha='center', va='top')
            else:
                self.ax.text(1.0, 0.0-0.025, axis00, ha='center', va='top')


    def plot_all(self):
        """
        全てプロットする。
        SS (固溶体) に対応するよう修正
        """
        self.make_frame()
        self.plot_pairs()
        # self.plot_points()
        # self.plot_labels()
        self.plot_points_wo_ss()
        self.plot_labels_wo_ss()
        self.plot_points_ss()


    def plot_all_wo_labels(self):
        """
        全てプロットする。
        SS (固溶体) に対応するよう修正
        """
        self.make_frame()
        self.plot_pairs()
        self.plot_points_wo_ss()
        # self.plot_labels_wo_ss()
        self.plot_points_ss()


class GibbsTrianglePlotly:
    '''
    Gibbs Triangle をプロットする
    '''
    def __init__(self):
        self.fig = go.Figure()

    @staticmethod
    def convert_triangle(x, y):
        return GibbsTriangle.convert_triangle(x, y)

    def plot_line(self, x, y, color='gray', lw=0.5):
        ''' gibbs triangle の補助線 '''
        x, y = self.convert_triangle(x, y)
        self.fig.add_trace(go.Scatter3d(
            x=x, y=y, z=[0, 0],  # ここではz=0と仮定
            mode='lines',
            line=dict(color=color, width=lw),
            showlegend=False
        ))

    def make_frame(self, axis10='', axis01='', axis00=''):
        for i in range(0, 10):
            x = np.array([1 - i * 0.1, 0])
            y = np.array([0, 1 - i * 0.1])
            self.plot_line(x, y, color='gray', lw=0.5)

            x = np.array([i * 0.1, i * 0.1])
            y = np.array([0, 1 - i * 0.1])
            self.plot_line(x, y, color='gray', lw=0.5)

            x = np.array([0, 1 - i * 0.1])
            y = np.array([i * 0.1, i * 0.1])
            self.plot_line(x, y, color='gray', lw=0.5)

        # 軸と軸ラベルの非表示
        self.fig.update_layout(
            scene=dict(
                xaxis=dict(showticklabels=False, zeroline=False, showgrid=False),
                yaxis=dict(showticklabels=False, zeroline=False, showgrid=False),
                zaxis=dict(showticklabels=False, zeroline=False, showgrid=False)
            ),
        )

        # 3D軸にテキストを追加する
        self.fig.add_trace(go.Scatter3d(
            x=[0.5, 0, 1], y=[np.sqrt(3)/2, 0, 0], z=[0, 0, 0],
            mode='text',
            text=[axis10, axis01, axis00]))


    def show(self):
        self.fig.update_layout(
            plot_bgcolor='rgba(0, 0, 0, 0)',  # プロットエリアの背景を透明に
            paper_bgcolor='rgba(0, 0, 0, 0)',  # 図全体の背景を透明に
            scene=dict(
                xaxis=dict(
                    showbackground=False,  # x軸の背景を非表示
                    showticklabels=False,  # x軸の目盛りラベルを非表示
                    title=''  # x軸のタイトル（文字）を非表示
                ),
                yaxis=dict(
                    showbackground=False,  # y軸の背景を非表示
                    showticklabels=False,  # y軸の目盛りラベルを非表示
                    title=''  # y軸のタイトル（文字）を非表示
                ),
                zaxis=dict(
                    showbackground=False,  # z軸の背景を非表示
                    showticklabels=True,  # z軸の目盛りラベルを非表示
                    title='Tm'  # z軸のタイトル（文字）を非表示
                ),
                bgcolor='rgba(0, 0, 0, 0)'  # 3Dシーンの背景を透明に
            )
        )
        self.fig.show()




class BinaryConvexHull:
    """
    x-E の関係でプロットする
    equib を受け取る
    温度は渡す equib 側で指定しておく 
    """
    def __init__(self, ax, equib, elements):
        self.ax = ax
        self.equib = equib
        self.elements = elements

    def plot_gs(self, **args):
        """ gs をプロットする """
        dd = self.equib.data[
            self.equib.data[f"conc_{self.elements[0]}"] + 
            self.equib.data[f"conc_{self.elements[1]}"] == 1]
        dd = dd[dd.from_gs == 0]
        dd = dd.sort_values(f"conc_{self.elements[1]}")
        plt.sca(self.ax)
        plt.plot(
            dd[f"conc_{self.elements[1]}"], dd.formation_energy, 'o-', **args)


    def plot_all_points(self, prop='formation_energy', **args):
        """
        全ての点をプロットする
        """
        dd = self.equib.data[
            self.equib.data[f"conc_{self.elements[0]}"] + 
            self.equib.data[f"conc_{self.elements[1]}"] == 1]            
        plt.plot(
            dd[f"conc_{self.elements[1]}"], dd[prop], '.', **args)

    def make_frame(self):
        """
        軸を設定
        """
        plt.sca(self.ax)
        plt.xlim(0, 1)
        plt.xlabel(f"Atomic fraction of {self.elements[1]}")
        plt.ylabel(f"Formation energy (eV/atom)")
        plt.xlim(0, 1)

    def plot_labels_gs(self):
        """
        ss_ (solid solute) から始まるラベル以外をプロットする
        """
        dd = self.equib.data[
            self.equib.data[f"conc_{self.elements[0]}"] + 
            self.equib.data[f"conc_{self.elements[1]}"] == 1]
        dd = dd[dd.from_gs == 0]
        dd = dd.sort_values(f"conc_{self.elements[1]}")
        plt.sca(self.ax)
        for _, ddd in dd.iterrows():
            plt.text(
                ddd[f"conc_{self.elements[1]}"], 
                ddd.formation_energy, str(ddd.poscar), va='top', size=10)


class BinaryPhaseDiagram:
    """
    Binary Phase Diagram をプロットする。
    三元系と実装を変えて data (pd.DataFrame) を受け取る。
    data には {conc, temperature, label} が格納されている。
    conc は elements[1] の濃度。
    """
    def __init__(self, ax, data, elements, dt, tmax=2000):
        self.ax = ax
        self.data = data
        self.list_elements = elements
        self.dt = dt
        self.tmax = tmax

    def plot_wo_ss(self):
        """
        固溶体以外のデータを plot
        """
        labels = [x for x in set(self.data['label'].values)
                  if x[:3] != 'SS_']
        dt = self.dt
        # 相境界となる端点の組成を抽出
        for l in labels:
            d = self.data[self.data['label'] == l]
            temps = d['temperature'].values
            # 端点の温度 2 * n  
            d = d[d['temperature'].map(
                lambda x, v=temps: x not in v+dt or x not in v-dt)]
            d = d.sort_values('temperature', ignore_index=True)
            # 縦の線、およびラベル、点
            for i in range(len(d) // 2):
                self.ax.plot([d.loc[i*2, 'conc']+0.01*i, d.loc[i*2+1, 'conc']],
                             [d.loc[i*2, 'temperature']-dt/2,
                              d.loc[i*2+1, 'temperature']+dt/2], 'bo-')
            
                self.ax.text(d.loc[i*2, 'conc']+0.01,
                             d.loc[[i*2, i*2+1], 'temperature'].mean(),
                             d.loc[i*2, 'label'], rotation=90,
                             va='center', size=12)
            # 横の線
            for i, t in enumerate(d['temperature'].values):
                dd = self.data
                dd = dd[dd['temperature'] == t]
                dd = dd.sort_values('conc', ignore_index=True)
                idx = dd[dd['conc'] == d['conc'].values[0]].index.values[0]
                idx = list(range(max(0, idx-1), min(len(dd), idx+2)))
                dd = dd.iloc[idx, :]
                self.ax.plot(dd['conc'], dd['temperature'] + (i*2-1)*dt/2, 'b-')

    def plot_ss(self):
        """ 固溶体のデータを plot """
        labels = [x for x in set(self.data['label'].values)
                  if x[:3] == 'SS_']
        colors = ('aqua', 'lime', 'violet', 'gray', 'blue')
        # colors = ('lime', 'violet', 'black')

        for i, l in enumerate(labels):
            d = self.data[self.data['label'] == l]
            self.ax.scatter(d['conc'], d['temperature'], s=30, marker='o',
                            facecolors=colors[i], edgecolors=colors[i])
            self.ax.scatter([0], [-1], s=20,
                            marker='o', facecolors=colors[i],
                            edgecolors=colors[i], label=l[3:])
        if labels:
            return True
        return False

    def plot_all(self):
        """
        全てプロットする。
        SS (固溶体) に未対応。
        """
        # 相ごと (labels) に処理する
        self.plot_wo_ss()
        if self.plot_ss():
            self.ax.legend(bbox_to_anchor=(0, 1.025),
                           loc='lower left', borderaxespad=0)
        self.ax.set_ylim(0, self.tmax)
        self.ax.set_title('{}-{}'.format(self.list_elements[0], self.list_elements[1]))
        self.ax.set_xlabel('Atomic fraction of {}'.format(self.list_elements[1]))
        self.ax.set_ylabel('Temperature (K)')

