# -*- coding: utf-8 -*-
"""
Plot sample spectra
"""

import os
cd=os.getcwd()
import numpy as np
from matplotlib import pyplot as plt
import xarray as xr
import matplotlib

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 30

#%% Inputs
source_pri=os.path.join(cd,'data/Xa_Sa_datafile.sgp.55_levels.month_08.cdf')
source_cha=os.path.join(cd,'data/sb.assist.z01.00.20230825.000154.assistcha.cdf')
source_trp=os.path.join(cd,'data/sb.assist.z01.c0.20230825.000015.nc')

hour_sel=15#select hour

#%% Initalization

#channel A data
Data_cha=xr.open_dataset(source_cha).sortby('time')
time_sel_cha=Data_cha.time.values[np.argmin(np.abs(Data_cha.time.values-hour_sel*3600))]
Data_cha_sel=Data_cha.sel(time=time_sel_cha)

#prior
Data_pri=xr.open_dataset(source_pri)

#tropoe
Data_trp=xr.open_dataset(source_trp)

#%% Plots
fig=plt.figure()
plt.plot(Data_cha.wnum,Data_cha_sel.mean_rad,'k',linewidth=2)
for nu in Data_trp.attrs['VIP_spectral_bands'].split(','):
    nu1=np.float32(nu[:4])
    nu2=np.float32(nu[6:])
    plt.fill_between([nu1,nu2], [0,0],[200,200], color='g',alpha=0.5)
    
plt.gca().set_xticks([])
plt.gca().set_yticks([])
plt.xlabel(r'$\tilde{\nu}$')
plt.ylabel(r'$B$')

plt.figure()
plt.plot(Data_pri.mean_temperature,Data_pri.height,'r',linewidth=2,label='$T_a$')
plt.gca().fill_betweenx(Data_pri.height,Data_pri.mean_temperature-np.diag(Data_pri.covariance_prior.values[:55,:55])**0.5,
                        Data_pri.mean_temperature+np.diag(Data_pri.covariance_prior.values[:55,:55])**0.5,
                        color='r',alpha=0.25)

plt.plot(Data_pri.mean_mixingratio,Data_pri.height,'b',linewidth=2,label='$r_a$')
plt.gca().fill_betweenx(Data_pri.height,Data_pri.mean_mixingratio-np.diag(Data_pri.covariance_prior.values[:55,:55])**0.5,
                        Data_pri.mean_mixingratio+np.diag(Data_pri.covariance_prior.values[:55,:55])**0.5,
                        color='b',alpha=0.25)
plt.legend(draggable=True)

plt.gca().set_xticks([])
plt.gca().set_yticks([])
plt.ylabel(r'$z$')
fig=plt.figure(figsize=(18,10))
CS=plt.contourf(Data_trp.time,Data_trp.height,Data_trp.temperature.T,np.arange(25,37,0.5),cmap='hot',extend='both')
plt.contour(Data_trp.time,Data_trp.height,Data_trp.temperature.T,np.arange(25,37,0.5),colors='k',alpha=0.25,linewidths=1)
plt.ylim([0,1])
plt.gca().set_xticks([])
plt.gca().set_yticks([])
plt.ylabel(r'$z$')
plt.xlabel('Time')
cb=plt.colorbar(CS,label=r'$T$')
cb.set_ticklabels([])



