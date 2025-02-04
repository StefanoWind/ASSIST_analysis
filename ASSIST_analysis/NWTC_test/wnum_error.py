# -*- coding: utf-8 -*-
"""
Compare spectra
"""
import os
cd=os.path.dirname(__file__)
import numpy as np
import sys
import xarray as xr
import matplotlib
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt
import yaml
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 14

#%% Inputs
source_config=os.path.join(cd,'configs','config.yaml')
source=os.path.join(cd,'data','20220510.20220824.irs_with_cbh.nc')
skip=100#how many timestep to skip

k=1.380649*10**-23#[J/Kg] Boltzman's constant
h=6.62607015*10**-34#[J s] Plank's constant
c=299792458.0#[m/s] speed of light

perc_lim=[1,99]#[%] limits for outlier filter (acts at each wavenumber independetly)

dgamma=4*10**-5
dgamma_c=4*10**-5

#graphics
name={'awaken/nwtc.assist.z02.00':'11',
      'awaken/nwtc.assist.z03.00':'12'}

#%% Initialization
#config
with open(source_config, 'r') as fid:
    config = yaml.safe_load(fid)
    
#imports
sys.path.append(config['path_utils'])
import utils as utl

Data=xr.open_dataset(source).isel(time=slice(None, None, skip))

#%% Main
rad1=Data.rad.sel(channel='awaken/nwtc.assist.z02.00')
rad2=Data.rad.sel(channel='awaken/nwtc.assist.z03.00')
rad_diff=rad2-rad1

bias=xr.apply_ufunc(utl.filt_stat,rad_diff.where(Data.cloud_flag==0),
                    kwargs={"func": np.nanmean,'perc_lim': perc_lim},
                    input_core_dims=[["time"]],  # Operate along the 'space' dimension
                    vectorize=True)

bias_c=xr.apply_ufunc(utl.filt_stat,rad_diff.where(Data.cloud_flag==1),
                    kwargs={"func": np.nanmean,'perc_lim': perc_lim},
                    input_core_dims=[["time"]],  # Operate along the 'space' dimension
                    vectorize=True)

mean_rad1=xr.apply_ufunc(utl.filt_stat,rad1.where(Data.cloud_flag==0),
                    kwargs={"func": np.nanmean,'perc_lim': perc_lim},
                    input_core_dims=[["time"]],  # Operate along the 'space' dimension
                    vectorize=True)

mean_rad1_c=xr.apply_ufunc(utl.filt_stat,rad1.where(Data.cloud_flag==1),
                    kwargs={"func": np.nanmean,'perc_lim': perc_lim},
                    input_core_dims=[["time"]],  # Operate along the 'space' dimension
                    vectorize=True)

dmean_rad1=xr.DataArray(
    np.gradient(mean_rad1.values, mean_rad1.wnum.values),
    coords={"wnum": rad1.wnum}
)

dmean_rad1_c=xr.DataArray(
    np.gradient(mean_rad1_c.values, mean_rad1_c.wnum.values),
    coords={"wnum": rad1.wnum}
)

# dgamma=np.polyfit(dmean_rad1*dmean_rad1.wnum,bias,1)[0]
# dgamma_c=np.polyfit(dmean_rad1_c*dmean_rad1_c.wnum,bias_c,1)[0]

#%% Plots

fig=plt.figure(figsize=(18,10))
gs = gridspec.GridSpec(3, 1, height_ratios=[1,1,1])
ax=fig.add_subplot(gs[0,0])
plt.plot(mean_rad1.wnum,mean_rad1,'-k',label='No clouds')
plt.plot(mean_rad1.wnum,mean_rad1_c,'--k',label='Clouds')
plt.ylabel(r'$B$ [r.u.]')
plt.grid()

ax=fig.add_subplot(gs[1,0])
plt.plot(dmean_rad1.wnum,dmean_rad1,'-k')
plt.plot(dmean_rad1_c.wnum,dmean_rad1_c,'--k')
plt.ylabel(r'$\frac{\partial B}{\partial \tilde{\nu}}$ [r.u. cm]')
plt.ylim([-100,100])
plt.grid()
    
ax=fig.add_subplot(gs[2,0])
plt.plot(bias.wnum,bias,'-k',label='Data (no clouds)')
plt.plot(dmean_rad1.wnum,dmean_rad1*dmean_rad1.wnum*dgamma,'-r',label='Model (no clouds)')
plt.plot(bias_c.wnum,bias_c,'--k',label='Data (clouds)')
plt.plot(dmean_rad1_c.wnum,dmean_rad1_c*dmean_rad1_c.wnum*dgamma_c,'--r',label='Model (clouds)')
plt.ylim([-3,3])
plt.grid()
plt.xlabel(r'$\tilde{\nu}$ [cm$^{-1}$]')
plt.ylabel(r'$\Delta B$ ('+name['awaken/nwtc.assist.z03.00']+'-'+name['awaken/nwtc.assist.z02.00']+') [r.u.]')
plt.legend(draggable=True)

plt.figure(figsize=(12,12))
plt.subplot(2,3,1)
plt.plot(dmean_rad1,bias,'.k')
plt.ylabel(r'$\Delta B$ ('+name['awaken/nwtc.assist.z03.00']+'-'+name['awaken/nwtc.assist.z02.00']+') [r.u.]')
plt.title('No clouds')
plt.grid()

plt.subplot(2,3,2)
plt.plot(bias.wnum,bias,'.k')
plt.title('No clouds')
plt.grid()

plt.subplot(2,3,3)
plt.plot(dmean_rad1*bias.wnum,bias,'.k')
plt.plot(dmean_rad1*bias.wnum,dmean_rad1*bias.wnum*dgamma,'r')
plt.title('No clouds')
plt.grid()

plt.subplot(2,3,4)
plt.plot(dmean_rad1_c,bias_c,'.k')
plt.xlabel(r'$\frac{\partial B}{\partial \tilde{\nu}}$ [r.u. cm]')
plt.ylabel(r'$\Delta B$ ('+name['awaken/nwtc.assist.z03.00']+'-'+name['awaken/nwtc.assist.z02.00']+') [r.u.]')
plt.title('Clouds')
plt.grid()

plt.subplot(2,3,5)
plt.plot(bias_c.wnum,bias_c,'.k')
plt.xlabel(r'$\tilde{\nu}$ [cm$^{-1}$]')
plt.title('Clouds')
plt.grid()

plt.subplot(2,3,6)
plt.plot(dmean_rad1_c*bias_c.wnum,bias_c,'.k')
plt.plot(dmean_rad1_c*bias_c.wnum,dmean_rad1_c*bias_c.wnum*dgamma_c,'r')
plt.xlabel(r'$\frac{\partial B}{\partial \tilde{\nu}} \tilde{\nu}$  [r.u.]')
plt.title('Clouds')
plt.grid()
plt.tight_layout()