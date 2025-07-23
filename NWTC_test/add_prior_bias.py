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
matplotlib.rcParams['font.size'] = 14

#%% Inputs
source_config=os.path.join(cd,'configs','config.yaml')
source=os.path.join(cd,'data/prior/Xa_Sa_datafile.nwtc.{unit}.55_levels.month_{month:02}.cdf')

#user
unit='ASSIST11'

#%% Initialization
#config
with open(source_config, 'r') as fid:
    config = yaml.safe_load(fid)
    
files=glob.glob(os.path.join(cd,config['sources_trp'][unit]))

data_trp_all=xr.open_dataset(os.path.join(cd,'data',f'tropoe.{unit}.nc'))

#zeroing
bias_all=[]
time_all=np.array([],dtype='datetime64')

os.makedirs(os.path.join(cd,'figures','bias'),exist_ok=True)

#%% Main
for f in files:
    
    #load TROPoe data
    data_trp=xr.open_dataset(f)
    tnum=np.float64(data_trp.time)/10**9
    hour=(tnum-np.floor(tnum/(3600*24))*3600*24)/3600
    month=int(str(data_trp.time.values[0])[5:7])
    height=data_trp.height.values*1000
    Nz=len(data_trp.height)
    
    #load met prior data
    data=xr.open_dataset(source.format(month=month,unit=unit)).interp(height=height)
    
    #calculate prior bias
    xa=data_trp.Xa[:Nz].rename({'arb_dim1':'height'}).assign_coords({'height':height})
    
    #blend met and TROPoe hourly mean
    data['mean_temperature_hourly']=data.mean_temperature_hourly_met.copy()
    missing=data.mean_temperature_hourly==0
    data['mean_temperature_hourly']=data.mean_temperature_hourly.where(~missing,data.mean_temperature_hourly_trp)
    
    #cycle hour
    hour_circle=np.concatenate([[data.hour.values[-1]-24],
                                 data.hour.values,
                                [data.hour.values[0]+24]])
    
    xa_met=np.zeros((len(data.height),len(hour_circle)))
    xa_met[:,0]=data.mean_temperature_hourly.values[:,-1]
    xa_met[:,1:-1]=data.mean_temperature_hourly.values
    xa_met[:,-1]=data.mean_temperature_hourly.values[:,0]
    
    data['Xa']=xr.DataArray(data=xa_met,coords={'height':height,'hour_ext':hour_circle})

    #calculatye prior bias
    xa_int=data.Xa.interp(hour_ext=hour)
    xa_int=xa_int.where(xa_int!=0)
    dxa=(xa_int-xa).values
    
    #calculate prior bias
    I=np.eye(len(height))
    bias=np.zeros_like(dxa)
    for i_t in range(len(data_trp.time)):
        A=data_trp['Akernal'].isel(time=i_t).values[:Nz,:Nz].T
        bias[:,i_t]=(A-I)@dxa[:,i_t]
        
    #store bias
    bias_all.append(bias.T)
    time_all=np.append(time_all,data_trp.time.values)
    print(f"{f} done")
    
    #%% Plots   
    plt.figure(figsize=(18,6))
    ax=plt.subplot(1,2,1)
    plt.pcolor(data_trp.time,height,dxa,vmin=-5,vmax=5,cmap='seismic')
    plt.xlabel('Time (UTC)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H%M'))
    plt.ylabel(r'$z$ [m]')
    plt.ylim([0,2000])
    plt.grid()
    plt.colorbar(label='$\Delta x_a$ [$^\circ$C]')
    
    ax=plt.subplot(1,2,2)
    plt.pcolor(data_trp.time,height,bias,vmin=-1,vmax=1,cmap='seismic')
    plt.xlabel('Time (UTC)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H%M'))
    plt.ylim([0,2000])
    plt.grid()
    plt.colorbar(label='$\Delta T$ [$^\circ$C]')
    plt.savefig(os.path.join(cd,'figures','bias',os.path.basename(f).replace('.nc','.bias.png')))
    
    plt.close()

#Output
output=data_trp_all.copy()
output['bias']=xr.DataArray(np.concatenate(bias_all),coords={'time':data_trp_all.time.values,'height':data_trp_all.height.values})
output.to_netcdf(os.path.join(cd,'data',f'tropoe.{unit}.bias.nc'))