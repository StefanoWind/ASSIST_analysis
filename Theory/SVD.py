# -*- coding: utf-8 -*-
"""
Singula value decomposition
"""
import os
cd=os.path.dirname(__file__)
import sys
sys.path.append('C:/Users/SLETIZIA/OneDrive - NREL/Desktop/PostDoc/utils')
import utils as utl
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 14

#%% Inputs


K=np.array([[utl.cosd(30),0,0],
            [0,1,0],
            [0,0,1],
            [utl.sind(30),0,0]])


#%% Main
U, S, VT = np.linalg.svd(K, full_matrices=False)
            
            
