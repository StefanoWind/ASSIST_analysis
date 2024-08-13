# -*- coding: utf-8 -*-
"""
Effect on spectrum of cut function
"""

import os
cd=os.getcwd()
import sys
sys.path.append('C:/Users/SLETIZIA/OneDrive - NREL/Desktop/PostDoc/utils')
import numpy as np
import utils as utl
from matplotlib import pyplot as plt
import xarray as xr
import warnings
import matplotlib
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


#%% Inputs
Ts=np.arange(-273.15,5000,10)+273.15#[K] ambient temperature range

k=1.380649*10**-23#[J/Kg] Boltzman's constant
h=6.62607015*10**-34#[J s] Plank's constant
c=299792458#[m/s] speed of light

wnum_laser=15798.02#[cm^-1]
perc_sel=1#[%]

#%% Initalization
wnum=np.arange(1,30000,10)+0.0
perc_aliasing=[]

#%% Main
for T in Ts:
    B=2*h*c**2*(wnum*100)**3/(np.exp(h*c*(wnum*100)/(k*T))-1)*10**5
    perc_aliasing=np.append(perc_aliasing,np.sum(B[wnum>wnum_laser/2])/np.sum(B))

T_thresh=Ts[np.where(perc_aliasing*100>perc_sel)[0][0]]


#%% Plots
plt.figure()
plt.plot(Ts,perc_aliasing*100,'k')
plt.xlabel('$T$ [$^\circ$C]')
plt.ylabel('BB energy above Nyquist wavenumber [%]')
plt.grid()
