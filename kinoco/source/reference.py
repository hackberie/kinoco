#!/usr/bin/python
# -*- coding:  utf-8 -*-
"""
データを保管。
"""
import os
import pandas as pd

module_dir = os.path.dirname(os.path.abspath(__file__))
DB_Tm = pd.read_csv(os.path.join(module_dir, "tm_BiInSn.csv"))

ELEMENTS = {
    'Vac': {'Z': 0, 'Rwigs': 1.5, 'Pomass': 0.0},
    'H': {'Z': 1, 'Rwigs': 1.39, 'Pomass':1.0},
    'He': {'Z': 2, 'Rwigs': 2.55, 'Pomass':4.0},
    'Li': {'Z': 3, 'Rwigs': 3.04, 'Pomass':7.01},
    'Be': {'Z': 4, 'Rwigs': 2.27, 'Pomass':9.013},
    'B': {'Z': 5, 'Rwigs': 1.96, 'Pomass':10.811},
    'C': {'Z': 6, 'Rwigs': 1.66, 'Pomass':12.011},
    'N': {'Z': 7, 'Rwigs': 1.9, 'Pomass':14.001},
    'O': {'Z': 8, 'Rwigs': 1.9, 'Pomass':16.0},
    'F': {'Z': 9, 'Rwigs': 2.17, 'Pomass':18.998},
    'Ne': {'Z': 10, 'Rwigs': 2.89, 'Pomass':20.18},
    'Na': {'Z': 11, 'Rwigs': 3.76, 'Pomass':22.99},
    'Mg': {'Z': 12, 'Rwigs': 3.25, 'Pomass':24.305},
    'Al': {'Z': 13, 'Rwigs': 2.95, 'Pomass':26.981},
    'Si': {'Z': 14, 'Rwigs': 2.63, 'Pomass':28.085},
    'P': {'Z': 15, 'Rwigs': 2.56, 'Pomass':30.974},
    'S': {'Z': 16, 'Rwigs': 2.7, 'Pomass':32.066},
    'Cl': {'Z': 17, 'Rwigs': 2.85, 'Pomass':35.453},
    'Ar': {'Z': 18, 'Rwigs': 3.71, 'Pomass':39.949},
    'K': {'Z': 19, 'Rwigs': 4.66, 'Pomass':39.098},
    'Ca': {'Z': 20, 'Rwigs': 3.88, 'Pomass':40.078},
    'Sc': {'Z': 21, 'Rwigs': 3.31, 'Pomass':44.956},
    'Ti': {'Z': 22, 'Rwigs': 2.99, 'Pomass':47.88},
    'V': {'Z': 23, 'Rwigs': 2.76, 'Pomass':50.941},
    'Cr': {'Z': 24, 'Rwigs': 2.64, 'Pomass':51.996},
    'Mn': {'Z': 25, 'Rwigs': 2.57, 'Pomass':54.938},
    'Fe': {'Z': 26, 'Rwigs': 2.52, 'Pomass':55.847},
    'Co': {'Z': 27, 'Rwigs': 2.52, 'Pomass':58.933},
    'Ni': {'Z': 28, 'Rwigs': 2.55, 'Pomass':58.69},
    'Cu': {'Z': 29, 'Rwigs': 2.62, 'Pomass':63.546},
    'Zn': {'Z': 30, 'Rwigs': 2.78, 'Pomass':65.39},
    'Ga': {'Z': 31, 'Rwigs': 2.75, 'Pomass':69.723},
    'Ge': {'Z': 32, 'Rwigs': 2.79, 'Pomass':72.61},
    'As': {'Z': 33, 'Rwigs': 2.83, 'Pomass':74.922},
    'Se': {'Z': 34, 'Rwigs': 2.94, 'Pomass':78.96},
    'Br': {'Z': 35, 'Rwigs': 3.13, 'Pomass':79.904},
    'Kr': {'Z': 36, 'Rwigs': 4.32, 'Pomass':83.8},
    'Rb': {'Z': 37, 'Rwigs': 4.95, 'Pomass':85.468},
    'Sr': {'Z': 38, 'Rwigs': 4.22, 'Pomass':87.62},
    'Y': {'Z': 39, 'Rwigs': 3.61, 'Pomass':88.906},
    'Zr': {'Z': 40, 'Rwigs': 3.28, 'Pomass':91.224},
    'Nb': {'Z': 41, 'Rwigs': 3.03, 'Pomass':92.0},
    'Mo': {'Z': 42, 'Rwigs': 2.91, 'Pomass':95.94},
    'Tc': {'Z': 43, 'Rwigs': 2.82, 'Pomass':98.906},
    'Ru': {'Z': 44, 'Rwigs': 2.77, 'Pomass':101.07},
    'Rh': {'Z': 45, 'Rwigs': 2.78, 'Pomass':102.906},
    'Pd': {'Z': 46, 'Rwigs': 2.84, 'Pomass':106.42},
    'Ag': {'Z': 47, 'Rwigs': 2.95, 'Pomass':107.868},
    'Cd': {'Z': 48, 'Rwigs': 3.14, 'Pomass':112.411},
    'In': {'Z': 49, 'Rwigs': 3.3, 'Pomass':114.82},
    'Sn': {'Z': 50, 'Rwigs': 3.45, 'Pomass':118.71},
    'Sb': {'Z': 51, 'Rwigs': 3.3, 'Pomass':121.75},
    'Te': {'Z': 52, 'Rwigs': 3.31, 'Pomass':127.6},
    'I': {'Z': 53, 'Rwigs': 3.5, 'Pomass':126.904},
    'Xe': {'Z': 54, 'Rwigs': 4.31, 'Pomass':131.294},
    'Cs': {'Z': 55, 'Rwigs': 5.3, 'Pomass':132.9},
    'Ba': {'Z': 56, 'Rwigs': 4.2, 'Pomass':137.327},
    'La': {'Z': 57, 'Rwigs': 3.91, 'Pomass':138.9},
    'Ce': {'Z': 58, 'Rwigs': 3.8, 'Pomass':140.115},
    'Pr': {'Z': 59, 'Rwigs': 3.75, 'Pomass':140.907},
    'Nd': {'Z': 60, 'Rwigs': 3.7, 'Pomass':144.24},
    'Pm': {'Z': 61, 'Rwigs': 3.65, 'Pomass':146.915},
    'Sm': {'Z': 62, 'Rwigs': 3.6, 'Pomass':150.36},
    'Eu': {'Z': 63, 'Rwigs': 3.55, 'Pomass':151.965},
    'Gd': {'Z': 64, 'Rwigs': 3.52, 'Pomass':157.25},
    'Tb': {'Z': 65, 'Rwigs': 3.61, 'Pomass':158.925},
    'Dy': {'Z': 66, 'Rwigs': 3.67, 'Pomass':162.5},
    'Ho': {'Z': 67, 'Rwigs': 3.7, 'Pomass':164.93},
    'Er': {'Z': 68, 'Rwigs': 3.73, 'Pomass':167.26},
    'Tm': {'Z': 69, 'Rwigs': 3.75, 'Pomass':168.93},
    'Yb': {'Z': 70, 'Rwigs': 3.56, 'Pomass':173.04},
    'Lu': {'Z': 71, 'Rwigs': 3.44, 'Pomass':174.967},
    'Hf': {'Z': 72, 'Rwigs': 3.23, 'Pomass':178.49},
    'Ta': {'Z': 73, 'Rwigs': 3.04, 'Pomass':180.948},
    'W': {'Z': 74, 'Rwigs': 2.93, 'Pomass':183.85},
    'Re': {'Z': 75, 'Rwigs': 2.86, 'Pomass':186.207},
    'Os': {'Z': 76, 'Rwigs': 2.82, 'Pomass':190.2},
    'Ir': {'Z': 77, 'Rwigs': 2.83, 'Pomass':192.22},
    'Pt': {'Z': 78, 'Rwigs': 2.88, 'Pomass':195.08},
    'Au': {'Z': 79, 'Rwigs': 2.98, 'Pomass':196.966},
    'Hg': {'Z': 80, 'Rwigs': 3.27, 'Pomass':200.59},
    'Tl': {'Z': 81, 'Rwigs': 3.57, 'Pomass':204.38},
    'Pb': {'Z': 82, 'Rwigs': 3.62, 'Pomass':207.2},
    'Bi': {'Z': 83, 'Rwigs': 3.37, 'Pomass':208.98},
    'Po': {'Z': 84, 'Rwigs': 3.46, 'Pomass':208.942},
    'At': {'Z': 85, 'Rwigs': 3.63, 'Pomass':209.987},
    'Rn': {'Z': 86, 'Rwigs': 4.44, 'Pomass':22.017},
    'Fr': {'Z': 87, 'Rwigs': 5.81, 'Pomass':223.02},
    'Ra': {'Z': 88, 'Rwigs': 4.3, 'Pomass':226.025},
    'Ac': {'Z': 89, 'Rwigs': 3.84, 'Pomass':227.028},
    'Th': {'Z': 90, 'Rwigs': 3.52, 'Pomass':232.039},
    'Pa': {'Z': 91, 'Rwigs': 3.32, 'Pomass':231.036},
    'U': {'Z': 92, 'Rwigs': 3.13, 'Pomass':238.029},
    'Np': {'Z': 93, 'Rwigs': 3.02, 'Pomass':237.048},
    'Pu': {'Z': 94, 'Rwigs': 2.96, 'Pomass':244.064},
    'Am': {'Z': 95, 'Rwigs': 2.93, 'Pomass':243.061},
    'Cm': {'Z': 96, 'Rwigs': 2.93, 'Pomass':247.0},
    'Cf': {'Z': 98, 'Rwigs': 2.99, 'Pomass':247.0}}
