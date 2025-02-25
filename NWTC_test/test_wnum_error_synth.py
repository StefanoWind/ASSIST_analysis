# -*- coding: utf-8 -*-
"""
Test numerically effect on wnum error on bias trhough a numerical example (no real spectra)s
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
dwnum=1
cycles=50
sigma=100
amp=0.5
smax=1.037

gamma1=1
gamma2=1.01

#%% Initalization
wnum=np.arange(-5*sigma,5*sigma+dwnum,dwnum)
B=np.exp(-wnum**2/(2*sigma**2))*(1+np.sin(2*np.pi/np.ptp(wnum)*wnum*cycles)*amp)

#%% Main

#FT
N=len(wnum)
n=np.arange(N)-np.floor(N/2)
k=np.arange(N)-np.floor(N/2)
NN,KK=np.meshgrid(n,k)
DFM=np.exp(-1j*KK*NN*2*np.pi/N)

#build igram
ds=1/dwnum/N
s=n*ds
I=np.matmul(1/DFM.T,B)/N

#define heavyside
hn=np.zeros(len(s))+1
hn[np.abs(s)>=smax]=0

#convoluted spectra
B_opd=np.matmul(DFM,I*hn)


B_clip=[]
for gamma in [gamma1,gamma2]:
    
    #distorted grid
    B_r=np.interp(wnum*gamma,wnum,B_opd)
    
    #FT
    B_clip=utl.vstack(B_clip,B_r)
    
dB_dwnum=np.gradient(B,wnum)
dB1_dwnum=np.gradient(B_clip[0,:],wnum)
dB2_dwnum=np.gradient(B_clip[1,:],wnum)

bias=dB2_dwnum*(gamma2-1)*wnum


#%% Plots
plt.figure(figsize=(18,8))
ax=plt.subplot(2,1,1)
plt.plot(wnum,B,'k',alpha=0.25,label='Original')
plt.plot(wnum,B_clip[0,:],'k',label=r'$B_1$')
plt.plot(wnum,B_clip[1,:],'r',label=r'$B_2$')
ax=plt.subplot(2,1,2)
plt.plot(wnum,B_clip[1,:]-B_clip[0,:],'b',label='Data')
plt.plot(wnum,bias,'g',label='Model')
plt.grid()
plt.xlabel(r'$\tilde{\nu}$ [cm$^{-1}$]')
plt.ylabel(r'$B_1-B_2$ [r.u.]')
