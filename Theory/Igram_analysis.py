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
from scipy.fft import fft, fftshift, fftfreq



#%% Inputs
source=os.path.join(cd,'data/20240810_000126_chA_SCENE.nc')
i_sel=0
clipping=0.75
T_amb=273.15+32

dx=1/15798.02

k=1.380649*10**-23#[J/Kg] Boltzman's constant
h=6.62607015*10**-34#[J s] Plank's constant
c=299792458#[m/s] speed of light

clips=[0.9,0.5,0.25]

#%% Functions
def phase(c):
    c[np.abs(c)<10**-10]=np.nan
    return np.angle(c)

#%% Initalization
Data=xr.open_dataset(source).sortby('time')
N0=len(Data['x_axis'])
I=Data['y_data'].values[i_sel,:]
n=Data['x_axis'].values
k=n.copy()
N=len(n)


#%% Main
# NN,KK=np.meshgrid(n,k)
# DFM=np.exp(-1j*KK*NN*2*np.pi/N)

B_ds = fftshift(fft(I))
wnum_ds = fftshift(fftfreq(N,d=dx))

B=((B_ds+B_ds[::-1])/2)[int(N/2):]
wnum=wnum_ds[int(N/2):]

#%% Plots
plt.figure()
plt.plot(wnum,np.abs(B))