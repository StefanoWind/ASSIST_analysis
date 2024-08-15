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
import matplotlib
import netCDF4 as nc
import glob

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 16

#%% Inputs
source_met=os.path.join(cd,'data/sb.met.z01.b0')

source_cha=os.path.join(cd,'data/sb.assist.z01.00.20230824.000041.assistcha.cdf')
source_chb=os.path.join(cd,'data/sb.assist.z01.00.20230824.000041.assistchb.cdf')
source_ceil=os.path.join(cd,'data/sa1.ceil.z01.b0.20230824.000000.nc')
hour_sel=6#select hour

k=1.380649*10**-23#[J/Kg] Boltzman's constant
h=6.62607015*10**-34#[J s] Plank's constant
c=299792458.0#[m/s] speed of light

#%% Initalization

#channel A data
Data_cha=xr.open_dataset(source_cha).sortby('time')
time_sel_cha=Data_cha.time.values[np.argmin(np.abs(Data_cha.time.values-hour_sel*3600))]
Data_cha_sel=Data_cha.sel(time=time_sel_cha)

#channel B data
Data_chb=xr.open_dataset(source_chb).sortby('time')
time_sel_chb=Data_chb.time.values[np.argmin(np.abs(Data_chb.time.values-hour_sel*3600))]
Data_chb_sel=Data_chb.sel(time=time_sel_chb)

tnum_sel=(time_sel_cha+np.float64(Data_cha.base_time)/1000+time_sel_chb+np.float64(Data_chb.base_time)/1000)/2

#cloud data
Data_ceil=nc.Dataset(source_ceil)
tnum_ceil=np.float64(Data_ceil['time'][:])
first_cbh=np.float64(Data_ceil['cloud_data'][:])[:,0] #[m]
Data_ceil.close()
cbh_sel=np.interp(tnum_sel,tnum_ceil,first_cbh)

#met data
file_met=glob.glob(os.path.join(source_met,'*'+utl.datestr(tnum_sel,'%Y%m%d.%H0000.nc')))[0]
Data_met=xr.open_dataset(file_met)
T_sel=np.float64(Data_met.temperature.interp({'time':utl.num_to_dt64(tnum_sel)}).values)
RH_sel=np.float64(Data_met.relative_humidity.interp({'time':utl.num_to_dt64(tnum_sel)}).values)

wnum=np.arange(500,3025)+0.0

#%% Main

B=2*h*c**2*wnum**3/(np.exp(h*c*wnum*100/(k*(273.15+T_sel)))-1)*10**11


#%% Plots
plt.figure()
plt.plot(Data_cha.wnum,Data_cha_sel.mean_rad,'b',label='Channel A',linewidth=1)
plt.plot(Data_chb.wnum,Data_chb_sel.mean_rad,'r',label='Channel B',linewidth=1)
plt.plot(wnum,B,'k',label=r'BB at $T='+str(T_sel)+'^\circ$C')

plt.xlabel(r'$\tilde{\nu}$')
plt.ylabel(r'$B$ [r.u.]')
plt.grid()
plt.title(utl.datestr(tnum_sel,'%Y-%m-%d %H:%M UTC'))
if cbh_sel>0:
    plt.text(2000,120,r'$T_s='+str(T_sel)+'^\circ$C'+'\n'+r'RH$_s='+str(int(RH_sel))+'$ %' +'\n'+ 'CBH $='+str(int(cbh_sel))+' m')
else:
    plt.text(2000,120,r'$T_s='+str(T_sel)+'^\circ$C'+'\n'+r'RH$_s='+str(int(RH_sel))+'$ %' +'\n'+ 'No clouds')
plt.legend(draggable=True)
plt.tight_layout()