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
import matplotlib.dates as mdates
import yaml
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 12

#%% Inputs
source_config=os.path.join(cd,'configs','config.yaml')
source=os.path.join(cd,'data','20220510.20220824.irs_with_cbh.nc')
skip=100

T_amb=20#[C]

k=1.380649*10**-23#[J/Kg] Boltzman's constant
h=6.62607015*10**-34#[J s] Plank's constant
c=299792458.0#[m/s] speed of light

max_err=0.01#fraction of ambient BB
perc_lim=[1,99]


tropoe_bands= np.array([[612.0,618.0],
                        [624.0,660.0],
                        [674.0,713.0],
                        [713.0,722.0],
                        [538.0,588.0],
                        [793.0,804.0],
                        [860.1,864.0],
                        [872.2,877.5],
                        [898.2,905.4]])#[cm^-1]

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

estd=xr.apply_ufunc(utl.filt_stat,rad_diff.where(Data.cloud_flag==0),
                    kwargs={"func": np.nanstd,'perc_lim': perc_lim},
                    input_core_dims=[["time"]],  # Operate along the 'space' dimension
                    vectorize=True)

bias_c=xr.apply_ufunc(utl.filt_stat,rad_diff.where(Data.cloud_flag==1),
                    kwargs={"func": np.nanmean,'perc_lim': perc_lim},
                    input_core_dims=[["time"]],  # Operate along the 'space' dimension
                    vectorize=True)
estd_c=xr.apply_ufunc(utl.filt_stat,rad_diff.where(Data.cloud_flag==1),
                    kwargs={"func": np.nanstd,'perc_lim': perc_lim},
                    input_core_dims=[["time"]],  # Operate along the 'space' dimension
                    vectorize=True)

mean_rad1=xr.apply_ufunc(utl.filt_stat,rad1.where(Data.cloud_flag==0),
                    kwargs={"func": np.nanmean,'perc_lim': perc_lim},
                    input_core_dims=[["time"]],  # Operate along the 'space' dimension
                    vectorize=True)

mean_rad2=xr.apply_ufunc(utl.filt_stat,rad2.where(Data.cloud_flag==0),
                    kwargs={"func": np.nanmean,'perc_lim': perc_lim},
                    input_core_dims=[["time"]],  # Operate along the 'space' dimension
                    vectorize=True)


drad1=xr.DataArray(
    np.gradient(rad1.values, rad1.wnum.values,axis=1),
    coords={"time":rad1.time,"wnum": rad1.wnum}
)

dmean_rad1=xr.DataArray(
    np.gradient(mean_rad1.values, mean_rad1.wnum.values),
    coords={"wnum": rad1.wnum}
)

dmean_rad2=xr.DataArray(
    np.gradient(mean_rad2.values, mean_rad2.wnum.values),
    coords={"wnum": rad2.wnum}
)



#%% Plots
plt.figure()
plt.plot(Data.wnum,rad1.isel(time=100),'.-k')
plt.plot(Data.wnum,rad2.isel(time=100),'.-r')
plt.plot(rad_diff.wnum,rad_diff.isel(time=100),'.-b')
plt.plot(drad1.wnum,drad1.isel(time=100),'.-g')
plt.grid()

plt.figure()
plt.plot(mean_rad1.wnum,mean_rad1,'.-k')
plt.plot(mean_rad2.wnum,mean_rad2,'.-r')
plt.plot(mean_rad1.wnum,(mean_rad2-mean_rad1),'.-b')
plt.plot(dmean_rad1.wnum,dmean_rad1,'.-g')
plt.grid()