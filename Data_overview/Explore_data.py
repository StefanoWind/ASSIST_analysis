# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 10:13:07 2024

@author: sletizia
"""

# -*- coding: utf-8 -*-
'''
Plot 3rd BB results
'''
import os
cd=os.path.dirname(__file__)

import sys
sys.path.append('C:/Users/SLETIZIA/OneDrive - NREL/Desktop/PostDoc/Custom_functions')
    
import myFunctions as SL
import numpy as np
from matplotlib import pyplot as plt
import warnings
import matplotlib 
import xarray as xr
import glob
import matplotlib.dates as mdates

warnings.filterwarnings('ignore')
plt.close('all')

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 16

#%% Inputs
source='data/test/assist-11/sb.assist.z01.00.20230719.000042.assistsummary.cdf'


#%% Initialization
Data=xr.open_dataset(source)