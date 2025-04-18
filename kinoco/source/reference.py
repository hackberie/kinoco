#!/usr/bin/python
# -*- coding:  utf-8 -*-
"""
データを保管。
"""
import os
import pandas as pd

module_dir = os.path.dirname(os.path.abspath(__file__))
DB_Tm = pd.read_csv(os.path.join(module_dir, "tm_BiInSn.csv"))
