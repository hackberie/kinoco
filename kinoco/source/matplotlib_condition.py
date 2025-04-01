#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
matplotlib の condtion
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
# from matplotlib.pyplot import scatter as plt_scatter

import seaborn

plt.rcParams.update(
    {'font.size': 15, 
     'figure.figsize': (5.0, 5.0),
     # 適宜 plt.figure(figsize=(16,4)) などで変更
     'savefig.dpi': 80,
     'axes.titlesize': 18, 'axes.labelsize': 18,
     'xtick.major.pad': 8, 'legend.numpoints': 1,
     'figure.autolayout': False, # True にすると図がはみ出さなくなる、あるいは plt.tight_layout() を使う
     'xtick.bottom': True, 'xtick.top': True, 'xtick.major.size' : 4,
     'xtick.direction': 'in',
     'ytick.left': True, 'ytick.right': True, 'ytick.major.size' : 4,
     'ytick.direction': 'in'
    })

if ('IPAexGothic' in 
        {f.name for f in font_manager.fontManager.ttflist}):
    plt.rcParams.update({'font.family': 'IPAexGothic'})# 'Times New Roman'

## よく使う marker の種類
MARKERS = 'o^sXvD*'

## よく使う cmap の設定

class Cmap:
    """ cmap のメソッド集 """
    @staticmethod
    def set_husl(nn):
        """ seaborn husl を set """
        cmap = seaborn.color_palette('husl', nn)[::-1]
        plt.rcParams["axes.prop_cycle"] = plt.cycler("color", cmap)

    @staticmethod
    def get_husl(nn):
        """ seaborn husl """
        cmap = seaborn.husl_palette(nn)[::-1]
        return cmap
        
    @staticmethod
    def get_clear_husl(nn):
        """ seaborn husl clear (edge 用) """
        cmap = seaborn.husl_palette(nn, l=0.5, s=1)[::-1]
        return cmap

    @classmethod
    def get_pair_husl(cls, nn):
        """
        seaborn husl 
        seaborn husl clear (edge 用) 
        のペアを返す
        """
        return (cls.get_husl(nn), cls.get_clear_husl(nn))

    @classmethod
    def set_my_tab(cls):
        """ tab20 の順番を変更したものを set """
        cmap = cls.get_my_tab()
        plt.rcParams["axes.prop_cycle"] = plt.cycler("color", cmap)

    @staticmethod
    def get_my_tab():
        """ tab20 の順番を変更したものを get """
        c = seaborn.color_palette('tab20', 20)
        c[4] = tuple(np.array(c[4]) + 0.15)
        # 18, 12, 2, 14 のみは濃い方を使う残りは薄いものを採用
        # したがって get_pair_my_tab とは濃淡が異なるため注意
        cmap = [c[18], c[12], c[5], c[2], c[14], c[9], c[1], c[11], c[7], c[17]]
        cmap += [c[19], c[13], c[4], c[3], c[15], c[8], c[0], c[10], c[6], c[16]]
        return cmap

    @classmethod
    def get_pair_my_tab(cls):
        """ tab20 の順番を変更したものを get 
        return (facecolor, edgecolor)
        """
        c = seaborn.color_palette('tab20', 20)
        c[4] = tuple(np.array(c[4]) + 0.15)
        cmap1 = [c[19], c[13], c[5], c[3], c[15], c[9], c[1], c[11], c[7], c[17]]
        cmap2 = [c[18], c[12], c[4], c[2], c[14], c[8], c[0], c[10], c[6], c[16]]
        return (cmap1, cmap2)

    @classmethod
    def set_tab(cls):
        """ tab20 の順番を変更したものを set """
        cmap = cls.get_tab()
        plt.rcParams["axes.prop_cycle"] = plt.cycler("color", cmap)

    @staticmethod
    def get_tab():
        """ tab20 の順番を変更したものを get """
        c = seaborn.color_palette('tab20', 20)
        cmap = c[1::2] + c[::2]
        return cmap

    @staticmethod
    def get_pair_tab():
        """ tab20 の順番を変更したものを get 
        return (facecolor, edgecolor)
        """
        c = seaborn.color_palette('tab20', 20)
        cmap1 = c[1::2]
        cmap2 = c[::2]
        return (cmap1, cmap2)

Cmap.set_my_tab()
    
def my_plot(x, y, size=10, marker='o', colors=None, with_line=True, **kargs):
    if not colors:
        c1, c2 = Cmap.get_pair_my_tab()
        i = len(plt.gca().lines)
        i = len(plt.gca().collections)
        c1, c2 = c1[i%10], c2[i%10]
    else:
        c1, c2 = colors
    if with_line:
        plt.plot(x, y, color=c1)
    plt.scatter(x, y, s=size, marker=marker, 
                facecolor=c1, edgecolor=c2, zorder=10, **kargs)

plt.my_plot = my_plot
