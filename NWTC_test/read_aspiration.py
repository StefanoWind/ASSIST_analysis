# -*- coding: utf-8 -*-
'''
Read aspiration flags
'''
import os
cd=os.path.dirname(__file__)
import numpy as np
import xarray as xr
import glob
import warnings
from matplotlib import pyplot as plt
import matplotlib
import glob
import matplotlib.dates as mdates
plt.close('all')
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 14
warnings.filterwarnings('ignore')

#%% Inputs
source=os.path.join(cd,'data/nwtc/nwtc.m5_asp.c1/*')

#%% Initialization
data=xr.open_mfdataset(glob.glob(source)).compute()

#%% Main