# -*- coding: utf-8 -*-
"""
Add error due to prior
"""
import os
cd=os.path.dirname(__file__)
import numpy as np
import xarray as xr
import matplotlib
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import yaml
from datetime import datetime
from datetime import timedelta

import glob
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 16

#%% Inputs
source_config=os.path.join(cd,'configs','config.yaml')
source='data/awaken/nwtc.assist.tropoe.z01.c0/*nc'
source_met=os.path.join(cd,'data/met_prior2.nc')
var='temperature_rec'

#graphics
max_z=3000#[m] maximum height ot plot
max_z_extr=500

#%% Initialization
#config
with open(source_config, 'r') as fid:
    config = yaml.safe_load(fid)
    
Prior=xr.open_dataset(source_met)
files=glob.glob(os.path.join(cd,source))

#%% Main
for f in files:
    
    Data=xr.open_dataset(f)
    tnum=np.float64(Data.time)/10**9
    hour=(tnum-np.floor(tnum/(3600*24))*3600*24)/3600
    month=int(str(Data.time.values[0])[5:7])
    Nz=len(Data.height)
    
    #calculate prior bias
    xa=Data.Xa[:Nz].rename({'arb_dim1':'height'}).assign_coords({'height':Data.height*1000})
    xa_int=Prior[var].sel(month=month).interp(height=Data.height.values*1000, kwargs={"fill_value": "extrapolate"}).interp(hour=hour, kwargs={"fill_value": "extrapolate"})
    
    dxa=(xa_int-xa).where(xa_int.height<max_z_extr,0).values.T
    
    I=np.eye(len(Data.height))
    A=Data['Akernal'].mean(dim='time').values[:Nz,:Nz].T
    
    bias=(A-I)@dxa
    
    #save covariances
    Data['bias']=xr.DataArray(data=bias,coords={'height':Data.height,'time':Data.time})

    #output
    os.makedirs(os.path.dirname(f.replace('c0','c2')),exist_ok=True)
    Data.to_netcdf(f.replace('c0','c2'))
    print(f)
    
  