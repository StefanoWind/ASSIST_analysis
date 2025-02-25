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

#dataset
sources_trp={'ASSIST10':'data/awaken/nwtc.assist.tropoe.z01.c0/*nc',
             'ASSIST11':'data/awaken/nwtc.assist.tropoe.z02.c0/*nc',
             'ASSIST12':'data/awaken/nwtc.assist.tropoe.z03.c0/*nc'}
unit='ASSIST10'

source_met=os.path.join(cd,'data/prior/Xa_Sa_datafile.nwtc.55_levels.month_{month:02}.cdf')

#stats
max_z_extr=1000#[m] max height at which prior difference goes to 0

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
    height=data_trp.height.values*1000
    Nz=len(data_trp.height)
    
    #load met prior data
    data_met=xr.open_dataset(glob.glob(source_met.format(month=month))[0]).interp(height=height)
    
    #calculate prior bias
    xa=data_trp.Xa[:Nz].rename({'arb_dim1':'height'}).assign_coords({'height':height})
    
    #cycle hour
    hour_met=np.concatenate([[data_met.hour.values[-1]-24],
                             data_met.hour.values,
                            [data_met.hour.values[0]+24]])
        
    xa_met=np.zeros((len(data_met.height),len(hour_met)))
    xa_met[:,0]=data_met.mean_temperature_hourly.values[:,-1]
    xa_met[:,1:-1]=data_met.mean_temperature_hourly.values
    xa_met[:,-1]=data_met.mean_temperature_hourly.values[:,0]
    
    #extend vertically
    xa_met[0,:]=xa_met[1,:]-(xa_met[2,:]-xa_met[1,:])/(height[2]-height[1])*(height[1]-height[0])
    data_met['Xa']=xr.DataArray(data=xa_met,coords={'height':height,'hour_ext':hour_met})

    #calculate difference
    xa_int=data_met.Xa.interp(hour_ext=hour)
    dxa=(xa_int-xa).where(xa.height<max_z_extr,0).values
    
    #taper difference to 0
    i_h1=np.where(data_met.Xa.mean(dim='hour_ext')!=0)[0][-1]
    i_h2=np.where(xa.height>max_z_extr)[0][1]
    for i_t in range(len(hour)):
        dxa[i_h1+1:i_h2,i_t]=dxa[i_h1,i_t]-dxa[i_h1,i_t]*(height[i_h1+1:i_h2]-height[i_h1])/(height[i_h2]-height[i_h1])
    
    #calculate prior bias
    I=np.eye(len(height))
    bias=np.zeros_like(dxa)
    for i_t in range(len(data_trp.time)):
        A=data_trp['Akernal'].isel(time=i_t).values[:Nz,:Nz].T
        bias[:,i_t]=(A-I)@dxa[:,i_t]
        
    #%% Output
    output=data_trp.copy()
    output['bias']=xr.DataArray(data=bias,coords={'height':height,'time':data_trp.time})
    os.makedirs(os.path.dirname(f.replace('c0','c2')),exist_ok=True)
    output.to_netcdf(f.replace('c0','c2'))
    print(f)
    
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
    plt.savefig(f.replace('c0','c2').replace('.nc','.bias.png'))

    
  