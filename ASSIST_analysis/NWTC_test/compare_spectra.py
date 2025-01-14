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
from matplotlib import pyplot as plt
import yaml
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 12

#%% Inputs
source_config=os.path.join(cd,'configs','config.yaml')
source=os.path.join(cd,'data','20220515.20220801.irs.cbh.nc')

T_amb=20#[C]

k=1.380649*10**-23#[J/Kg] Boltzman's constant
h=6.62607015*10**-34#[J s] Plank's constant
c=299792458.0#[m/s] speed of light

max_err=0.01#fraction of ambient BB

tropoe_bands= np.array([[612.0,618.0],
                        [624.0,660.0],
                        [674.0,713.0],
                        [713.0,722.0],
                        [538.0,588.0],
                        [793.0,804.0],
                        [860.1,864.0],
                        [872.2,877.5],
                        [898.2,905.4]])#[cm^-1]


#%% Initialization
#config
with open(source_config, 'r') as fid:
    config = yaml.safe_load(fid)
    
#imports
sys.path.append(config['path_utils'])
import utils as utl

Data=xr.open_dataset(source)

#%% Main
rad_diff=Data.rad.sel(channel='awaken/nwtc.assist.z03.00')-Data.rad.sel(channel='awaken/nwtc.assist.z02.00')
bias=rad_diff.where(Data.cloud_flag==0).mean(dim='time')
estd=rad_diff.where(Data.cloud_flag==0).std(dim='time')

bias_c=rad_diff.where(Data.cloud_flag==1).mean(dim='time')
estd_c=rad_diff.where(Data.cloud_flag==1).std(dim='time')

B=2*h*c**2*Data.wnum**3/(np.exp(h*c*Data.wnum*100/(k*(273.15+T_amb)))-1)*10**11

err_sel=rad_diff.sel(wnum=600,method='nearest')

#%% Plots
fig=plt.figure(figsize=(18,8))
ax=fig.add_subplot(1,1,1)
for tb in tropoe_bands:
    rect = Rectangle((tb[0], -5), tb[1]-tb[0], 15, edgecolor='b', facecolor='b', linewidth=2,alpha=0.25)
    ax.add_patch(rect)
ax.fill_between(Data.wnum,max_err*B,-max_err*B, edgecolor='g', facecolor='g', linewidth=2,alpha=0.25)
plt.plot(Data.wnum,bias,'k')
plt.plot(Data.wnum,estd,'r')
plt.plot(Data.wnum,bias_c,'--k')
plt.plot(Data.wnum,estd_c,'--r')

plt.xlabel(r'$\tilde{\nu}$ [cm $^{-1}$]')
plt.ylabel(r'Radiance error [r.u.]')
plt.grid()