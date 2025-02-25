# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 10:59:25 2024

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
t=np.arange(0,1,0.001)

k=1.380649*10**-23#[J/Kg] Boltzman's constant
h=6.62607015*10**-34#[J s] Plank's constant
c=299792458#[m/s] speed of light

T=40
wnum=np.arange(1,2001)+0.0#[cm^-1]

#%% Main
B=2*h*c**2*(wnum*100)**3/(np.exp(h*c*(wnum*100)/(k*(273.15+T)))-1)*10**5

wnum_folded=np.concatenate([-np.flip(wnum),wnum])

B_folded=2*h*c**2*(np.abs(wnum_folded)*100)**3/(np.exp(h*c*(np.abs(wnum_folded)*100)/(k*(273.15+T)))-1)*10**5/2


I=np.fft.ifft(B_folded)

I_folded=np.concatenate([I[int(len(B_folded)/2):],I[:int(len(B_folded)/2)]])*(wnum[1]-wnum[0])
D=np.arange(len(I_folded))
D_folded=D-D[int(len(D)/2)]

#%% Plots
plt.figure(figsize=(14,6))
plt.subplot(1,2,2)
plt.plot(wnum_folded,B_folded,'--k')
plt.plot(wnum,B,'-k')
plt.grid()
plt.xlabel(r'$\tilde{\nu}$ [cm$^{-1}$]')
plt.ylabel(r'$B(\tilde{\nu})$ [r.u.]')

plt.subplot(1,2,1)
plt.plot(D_folded,np.real(I_folded),'-k',label='Re')
plt.plot(D_folded,np.imag(I_folded),'-r',label='Im')
plt.grid()
plt.xlabel('Mirror displacement')
plt.ylabel(r'$I$ [$W/$m$^{2}$ sr]')
plt.legend()
