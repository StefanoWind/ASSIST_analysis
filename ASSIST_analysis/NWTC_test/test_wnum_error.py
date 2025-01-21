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
import warnings
import matplotlib
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import integrate

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 18

#%% Inputs
source=os.path.join(cd,'data','20220510.20220824.irs_with_cbh.nc')
hour_sel=15#select hour
T_amb=273.15+32#[C] ambient temperature to fill gap in spectrum

k=1.380649*10**-23#[J/Kg] Boltzman's constant
h=6.62607015*10**-34#[J s] Plank's constant
c=299792458#[m/s] speed of light

clip=1
skip=10

gamma1=1
gamma2=1+3.64*10**-5
perc_lim=[1,99]

wnum_laser=15798.02#[cm^-1]
N_real=32768#number of real samples

#graphics
zoom=[1000,1200,0,100]

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
B_clip=[]
for gamma in [gamma1,gamma2]:
    
    #distorted grid
    dx_r=1/dwnum/N/gamma
    x_r=n*dx_r
    I_ds_r=np.interp(x_r,x,I_ds)
    
    #FT
    B_ds_clip=np.matmul(DFM,I_ds_r*hn)
    
    #demirroring
    B_clip=utl.vstack(B_clip,(B_ds_clip+B_ds_clip[::-1])[wnum_ds>=wnum[0]])

dB_dwnum=np.gradient(B,wnum)
dB1_dwnum=np.gradient(B_clip[0,:],wnum)
dB2_dwnum=np.gradient(B_clip[1,:],wnum)
print(np.nanmean(np.abs(B_clip[1,:]-B_clip[0,:])))

dB_dwnum_sm=xr.DataArray(data=np.abs(dB_dwnum),coords={'wnum':wnum}).rolling(wnum=100,center=True,min_periods=10).mean().values
bias_sm=np.abs(bias).rolling(wnum=100,center=True,min_periods=10).mean().values

bias2_sm=xr.DataArray(data=np.abs(B_clip[1,:]-B_clip[0,:]),coords={'wnum':wnum}).rolling(wnum=100,center=True,min_periods=10).mean().values

#%% Plots
plt.figure()
plt.plot(wnum,B,'k',alpha=0.25)
plt.plot(wnum,B_clip[0,:],'k')
plt.plot(wnum,B_clip[1,:],'r')
plt.plot(wnum,B_clip[1,:]-B_clip[0,:],'b')
plt.plot(wnum,bias,'g')

plt.figure()
utl.plot_lin_fit((gamma2-gamma1)*wnum*dB_dwnum_sm,bias2_sm)
# plt.plot((gamma2-gamma1)*wnum*dB1_dwnum,B_clip[1,:]-B_clip[0,:],'.k')
# plt.plot((gamma2-gamma1)*wnum*dB2_dwnum,B_clip[1,:]-B_clip[0,:],'.r')