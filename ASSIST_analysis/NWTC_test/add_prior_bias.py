# -*- coding: utf-8 -*-
"""
Add error due to prior
"""
import os
cd=os.path.dirname(__file__)
import numpy as np
import xarray as xr
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import yaml
import glob
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 16

#%% Inputs
source_config=os.path.join(cd,'configs','config.yaml')

#dataset
sources_trp={'ASSIST10':'data/awaken/nwtc.assist.tropoe.z01.c0/*nc',
             'ASSIST11':'data/awaken/nwtc.assist.tropoe.z02.c0/*nc',
             'ASSIST12':'data/awaken/nwtc.assist.tropoe.z03.c0/*nc'}
unit='ASSIST10'
height_assist=1#[m]

source_met=os.path.join(cd,'data/prior/Xa_Sa_datafile.nwtc.55_levels.month_{month:02}.cdf')

#graphics
max_z=3000#[m] maximum height ot plot
max_z_extr=500

#%% Initialization
#config
with open(source_config, 'r') as fid:
    config = yaml.safe_load(fid)
    

files=glob.glob(os.path.join(cd,sources_trp[unit]))

#%% Main
for f in files:
    
    #load TROPoe data
    data_trp=xr.open_dataset(f)
    tnum=np.float64(data_trp.time)/10**9
    hour=(tnum-np.floor(tnum/(3600*24))*3600*24)/3600
    month=int(str(data_trp.time.values[0])[5:7])
    Nz=len(data_trp.height)
    
    #load met prior data
    data_met=xr.open_dataset(glob.glob(source_met.format(month=month))[0])
    
    
    #calculate prior bias
    xa=data_trp.Xa[:Nz].rename({'arb_dim1':'height'}).assign_coords({'height':data_trp.height*1000})
    raise BaseException()
    # xa_int=data_met.mean_temperature.interp(height=data_trp.height*1000, kwargs={"fill_value": "extrapolate"}).interp(hour=hour, kwargs={"fill_value": "extrapolate"})
    
    # dxa=(xa_int-xa).where(xa_int.height<max_z_extr,0).values.T
    
    # I=np.eye(len(data_trp.height))
    # A=Data['Akernal'].mean(dim='time').values[:Nz,:Nz].T
    
    # bias=(A-I)@dxa
    
    # #save covariances
    # Data['bias']=xr.DataArray(data=bias,coords={'height':Data.height,'time':Data.time})

    # #output
    # os.makedirs(os.path.dirname(f.replace('c0','c2')),exist_ok=True)
    # Data.to_netcdf(f.replace('c0','c2'))
    # print(f)
    
  