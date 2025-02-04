# -*- coding: utf-8 -*-
"""
Test numerically effect on wnum error on bias
"""

import os
cd=os.getcwd()
import sys
sys.path.append('C:/Users/SLETIZIA/OneDrive - NREL/Desktop/Main/utils')
import numpy as np
import utils as utl
from matplotlib import pyplot as plt
import xarray as xr
import matplotlib

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 14

#%% Inputs
source=os.path.join(cd,'data','20220510.20220824.irs_with_cbh.nc')
T_amb=273.15+25#[C] ambient temperature to fill gap in spectrum

k=1.380649*10**-23#[J/Kg] Boltzman's constant
h=6.62607015*10**-34#[J s] Plank's constant
c=299792458#[m/s] speed of light

clip=0.5
skip=100

gamma1=1
gamma2=1+4*10**-5
perc_lim=[1,99]


#%% Functions
def phase(c):
    c[np.abs(c)<10**-10]=np.nan
    return np.angle(c)

#%% Initalization
Data=xr.open_dataset(source).sortby('time').isel(time=slice(None, None, skip))

wnum=Data.wnum.values
time=Data.time.values

#%% Main
rad1=Data.rad.sel(channel='awaken/nwtc.assist.z02.00')
rad2=Data.rad.sel(channel='awaken/nwtc.assist.z03.00')
B=xr.apply_ufunc(utl.filt_stat,rad1.where(Data.cloud_flag==0),
                    kwargs={"func": np.nanmean,'perc_lim': perc_lim},
                    input_core_dims=[["time"]],  # Operate along the 'space' dimension
                    vectorize=True)

rad_diff=rad2-rad1
bias=xr.apply_ufunc(utl.filt_stat,rad_diff.where(Data.cloud_flag==0),
                    kwargs={"func": np.nanmean,'perc_lim': perc_lim},
                    input_core_dims=[["time"]],  # Operate along the 'space' dimension
                    vectorize=True)

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
I_ds=np.matmul(1/DFM.T,B_ds)/N

#define heavyside
hn=np.zeros(len(x))+1
hn[np.abs(x)>=x[-1]*clip-10**-10]=0

#convoluted spectra
B_ds_opd=np.matmul(DFM,I_ds*hn)

B_clip=[]
for gamma in [gamma1,gamma2]:
    
    #distorted grid
    B_ds_clip=np.interp(wnum_ds*gamma,wnum_ds,B_ds_opd)
    
    #demirroring
    B_clip=utl.vstack(B_clip,(B_ds_clip+B_ds_clip[::-1])[wnum_ds>=wnum[0]])

dB_dwnum=np.gradient(B,wnum)
dB1_dwnum=np.gradient(B_clip[0,:],wnum)
dB2_dwnum=np.gradient(B_clip[1,:],wnum)

bias=dB2_dwnum*(gamma2-gamma1)*wnum

#%% Plots
plt.close('all')
plt.figure(figsize=(18,8))
ax=plt.subplot(2,1,1)
plt.plot(wnum,B,'k',alpha=0.25,label='Original')
plt.plot(wnum,B_clip[0,:],'k',label=r'$B_1$')
plt.plot(wnum,B_clip[1,:],'r',label=r'$B_2$')
plt.grid()
plt.ylabel(r'$B$ [r.u.]')

ax=plt.subplot(2,1,2)
plt.plot(wnum,B_clip[1,:]-B_clip[0,:],'b',label='Data')
plt.plot(wnum,bias,'g',label='Model')
plt.grid()
plt.xlabel(r'$\tilde{\nu}$ [cm$^{-1}$]')
plt.ylabel(r'$B_1-B_2$ [r.u.]')


plt.figure()
utl.plot_lin_fit(B_clip[1,:]-B_clip[0,:], bias)