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



#%% Inputs
source=os.path.join(cd,'data/sb.assist.z01.00.20230824.000041.assistcha.cdf')
hour_sel=15
T_amb=273.15+32

k=1.380649*10**-23#[J/Kg] Boltzman's constant
h=6.62607015*10**-34#[J s] Plank's constant
c=299792458#[m/s] speed of light

clips=[0.9,0.5,0.25]
samplings=[1,2,3,5]

#%% Functions
def phase(c):
    c[np.abs(c)<10**-10]=np.nan
    return np.angle(c)

#%% Initalization
Data=xr.open_dataset(source).sortby('time')
wnum=Data.wnum.values
time=Data.time.values

#%% Main
t_sel=time[np.argmin(np.abs(time-hour_sel*3600))]
B=Data['mean_rad'].sel(time=t_sel).values

#mirorring and filling of the spectrum
dwnum=np.nanmedian(np.diff(wnum))
wnum_ds=np.arange(-wnum[-1],wnum[-1]+dwnum,dwnum)
B_ds=2*h*c**2*(np.abs(wnum_ds*100))**3/(np.exp(h*c*(np.abs(wnum_ds*100))/(k*T_amb))-1)/2*10**5
B_ds[0:len(wnum)]=B[::-1]/2
B_ds[-len(wnum):]=B/2
B_ds[np.isnan(B_ds)]=0

#FT
N=len(wnum_ds)
n=np.arange(N)-np.floor(N/2)
k=np.arange(N)-np.floor(N/2)
NN,KK=np.meshgrid(n,k)
DFM=np.exp(-1j*KK*NN*2*np.pi/N)

#build igram
dx=1/dwnum/N
x=n*dx
I_ds=np.matmul(1/DFM,B_ds)/N

#%% Plots
plt.figure(figsize=(18,10))
plt.plot(wnum,np.abs(B),'-k')

ctr=1
for s in samplings:
    plt.subplot(2,len(samplings),ctr)
    for c in clips:
        hn=np.zeros(len(x))
        hn[::s]=1
        hn[np.abs(x)>=x[-1]*c-10**-10]=0
        B_ds_samp=np.matmul(DFM.T,I_ds*hn)
        
        #demirroring
        B_samp=(B_ds_samp+B_ds_samp[::-1])[wnum_ds>=wnum[0]]
        
        plt.plot(wnum,np.abs(B_samp))
    ctr+=1


