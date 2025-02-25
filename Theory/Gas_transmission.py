# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 18:15:38 2024

@author: sletizia
"""


import os
cd=os.getcwd()
import sys
sys.path.append('C:/Users/SLETIZIA/OneDrive - NREL/Desktop/PostDoc/utils')
import numpy as np
import utils as utl
from matplotlib import pyplot as plt
import xarray as xr
import matplotlib
import netCDF4 as nc
import glob

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 16


#%% Inputs

source_abs='data/66c3ed96.par'

cp=1005#[J/Kg K]
g=9.81 #[m/s^2]
T0=30#[C]

k=1.380649*10**-23#[J/Kg] Boltzman's constant
h=6.62607015*10**-34#[J s] Plank's constant
c=299792458.0#[m/s] speed of light

wnum=np.arange(2000)+0.0
z=np.arange(0,10000,10)+0.0


a=np.zeros(len(wnum))+0.01
a[wnum<750]=1
a[wnum>1000]=1


#%% Initialization

Z,W=np.meshgrid(z,wnum)
_,A=np.meshgrid(z,a)
T=T0-(z[-1]-Z)*g/cp

S=[]
wnum_tr=[]
with open(source_abs, 'r') as file:
    for line in file:
        # Skip comments and empty lines

        # Split line into key and value
        values = np.array(line.split(' '))
        length=np.array([len(v) for v in values])
        values=values[length>0]
        wnum_tr=np.append(wnum_tr,np.float64(values[1]))
        S=np.append(S,np.float64(values[2]))
         
#%% Main
B0=2*h*c**2*W**3/(np.exp(h*c*W*100/(k*(273.15+T)))-1)*10**11

conv=A*np.exp(-A*(z[-1]-Z))
B=np.sum(B0*conv,axis=1)*(z[1]-z[0])

#%% Plots
plt.figure()
plt.plot(wnum,B,'k')
plt.grid()
plt.ylabel(r'$B$ [r.u.]')
plt.xlabel(r'$\tilde{\nu}$ [cm$^{-1}$]')
    