# -*- coding: utf-8 -*-
"""
Estimate impact of representativeness error
"""

import os
cd=os.path.dirname(__file__)
import sys
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
import warnings
from datetime import datetime, timedelta
import glob
from multiprocessing import Pool
import matplotlib.dates as mdates
import matplotlib.cm as cm
import yaml
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['savefig.dpi'] = 500

warnings.filterwarnings('ignore')
plt.close('all')


#%% Inputs
source_met_sta=os.path.join(cd,'data/nwtc/nwtc.m5.c1/*nc')#source of met stats
source_config=os.path.join(cd,'configs','config.yaml')
source_waked=os.path.join(cd,'data/turbine_wakes.nc')

#site
unit='ASSIST11'#assist id
sel_height=87#[m] height to select wind conditions
var_trp='temperature'#selected variable in TROPoe data
var_met='temperature'#selected variable in M5 data
var_sf='air_temp_rec'#selected structure function variable in M5 data
wd_align=230#[deg] direction of alignment (met tower based)


#stats
p_value=0.05#for CI
max_height=200#[km]
max_f=40#[C] max threshold of selected variable
min_f=-5#[C] min threshold of selected variable
max_time_diff=10#[s] maximum difference in time between met and TROPoe
perc_lim=[1,99] #[%] percentile fitler before feature selection
wd_lim=10#[deg] maximum misalignment
 
#graphics
cmap = plt.get_cmap("viridis")

#%% Initialization
#config
with open(source_config, 'r') as fid:
    config = yaml.safe_load(fid)
    
#read and align data
Data_trp=xr.open_dataset(os.path.join(cd,'data',f'tropoe.{unit}.bias.nc'))
Data_met=xr.open_dataset(os.path.join(cd,'data',f'met.a1.{unit}.nc'))

Data_trp,Data_met=xr.align(Data_trp,Data_met,join="inner",exclude=["height"])

#read wake data
waked=xr.open_dataset(source_waked)

#read met stats
files=glob.glob(source_met_sta)
Data_met_sta=xr.open_mfdataset(files)

#%% Main

#save cbh
cbh=Data_trp.cbh.where(Data_trp.cbh!=np.nanpercentile(Data_trp.cbh,10)).where(Data_trp.cbh!=np.nanpercentile(Data_trp.cbh,90))

#height interpolation
Data_trp=Data_trp.interp(height=Data_met.height)

#QC
Data_trp=Data_trp.where(Data_trp.qc==0)
print(f"{int(np.sum(Data_trp.qc!=0))} points fail QC in TROPoe")

Data_met=Data_met.where(Data_met.time_diff<=max_time_diff)
print(f"{int(np.sum(Data_met.time_diff>max_time_diff))} points fail max_time_diff")

#remove wake
Data_trp['waked']=waked['Site 3.2'].interp(time=Data_trp.time)
f_trp=Data_trp[var_trp].where(Data_trp['waked'].sum(dim='turbine')==0).sel(height=slice(0,max_height))
sigma_trp=Data_trp[f"sigma_{var_trp}"].where(Data_trp['waked'].sum(dim='turbine')==0).sel(height=slice(0,max_height))
print(f"{int(np.sum(Data_trp['waked'].sum(dim='turbine')>0))} wake events at Site 3.2 excluded")

Data_met['waked']=waked['M5'].interp(time=Data_met.time)
f_met=Data_met[var_met].where(Data_met['waked'].sum(dim='turbine')==0).sel(height=slice(0,max_height))
print(f"{int(np.sum(Data_met['waked'].sum(dim='turbine')>0))} wake events at M5 excluded")

#remove outliers
f_trp=f_trp.where(f_trp>=min_f).where(f_trp<=max_f)
f_met=f_met.where(f_met>=min_f).where(f_met<=max_f)
    
#extract coords
height=Data_met.height.values
time=Data_met.time.values

#T difference
diff=f_trp-f_met

#preconditioning
Ri=Data_met_sta.Ri_3_122.interp(time=Data_trp.time)
logRi=np.log10(np.abs(Ri)+1)*np.sign(Ri)

ws=Data_met_sta.ws.sel(height=sel_height).interp(time=Data_trp.time)

cos_wd=np.cos(np.radians(Data_met_sta.wd.sel(height=sel_height))).interp(time=Data_trp.time)
sin_wd=np.sin(np.radians(Data_met_sta.wd.sel(height=sel_height))).interp(time=Data_trp.time)
wd=np.degrees(np.arctan2(sin_wd,cos_wd))%360

ang_diff1=((wd - wd_align + 180) % 360) - 180
ang_diff2=((wd - wd_align+180 + 180) % 360) - 180
diff_sel=diff.where((np.abs(ang_diff1)<wd_lim)+(np.abs(ang_diff2)<wd_lim))

D=Data_met_sta[var_sf].interp(time=Data_trp.time)

