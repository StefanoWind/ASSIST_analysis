# -*- coding: utf-8 -*-
"""
Combine CBH from different locations at AWAKEN
"""

import os
cd=os.path.dirname(__file__)
import sys
import warnings
from matplotlib import pyplot as plt
import numpy as np
import glob
import matplotlib.dates as mdates
import xarray as xr
from datetime import datetime
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 14
warnings.filterwarnings('ignore')
plt.close('all')

#%% Inputs
path_save=os.path.join(cd,'data/awaken')
source_layout=os.path.join(cd,'data/20250225_AWAKEN_layout.nc')
sources_sel=['A1','H','C1','E37']
sources_sel=['A1','H']#,'C1','E37']
sites_sel=['B','C1a','G']
sdate='20230501'
edate='20231101'

#archive of sources
sources={'A1':os.path.join(path_save,'sa1.ceil.z01.cbh/*{datestr}*nc'),
          'H':os.path.join(path_save,'sgpceilS6.cbh/*{datestr}*nc')}

#time interpolation
dtime=60#[s] time step
max_time_diff=30#[s] maximum time difference

#stats
max_flat=0.5#ratio of points with 0 time derivative

#graphics
colors={'A1':'b','H':'g','E37':'m'}
   
#%% Initialization
dates=np.arange(np.datetime64(f"{sdate[:4]}-{sdate[4:6]}-{sdate[6:]}T00:00:00"),
                np.datetime64(f"{edate[:4]}-{edate[4:6]}-{edate[6:]}T00:00:00")+np.timedelta64(1,'s'),np.timedelta64(1,'D'))
time_offset=np.arange(0,3600*24+1,dtime)

dir_save={}
for site in sites_sel:  
    dir_save[site]=os.path.join(path_save,'s'+'.s'.join(s.lower() for s in sources_sel)+'_'+site.lower()+'.ceil.z01.cbh')
    os.makedirs(dir_save[site],exist_ok=True)

#load layout
Sites=xr.open_dataset(source_layout,group='ground_sites')

#%% Main

#averaging weights
distances=np.zeros((len(sources_sel),len(sites_sel)))
i1=0
for s1 in sources_sel:
    x1=Sites.x_utm.sel(site=s1).values
    y1=Sites.y_utm.sel(site=s1).values
    i2=0
    for s2 in sites_sel:
        x2=Sites.x_utm.sel(site=s2).values
        y2=Sites.y_utm.sel(site=s2).values
        distances[i1,i2]=((x1-x2)**2+(y1-y2)**2)**0.5/1000
        i2+=1
    i1+=1
        

for d in dates:
    cbh_all=np.zeros((len(time_offset),len(sources)))+np.nan
    data={}
    i_s=0
    time_np=d+np.timedelta64(1,'s')*time_offset
    
    found=0
    for source in sources_sel:
        files=glob.glob(sources[source].format(datestr=str(d)[:10].replace('-','')))
        
        if len(files)==1:
            found+=1
            
            #load data
            data=xr.open_dataset(files[0])
            
            #resolve duplicates
            if data.time.to_series().duplicated().any():
                data = data.groupby('time').mean()
    
            basetime=data.base_time.values
            
            #qc
            data=data.where(data['first_cbh']>0)
            data['d_first_cbh_dtime']=np.abs(data.first_cbh.differentiate("time"))
            if np.sum(data.d_first_cbh_dtime==0)/len(data.time)>max_flat:
                print(f'{os.path.basename(files[0])} has too many flat points, skipped',flush=True)
                found-=1
                continue
        
            #interpolate in time
            cbh_int=data.first_cbh.interp(time=time_offset)
            time_diff=data.time.interp(time=time_offset,method='nearest')-time_offset
            cbh_all[:,i_s]=cbh_int.where(np.abs(time_diff)<=max_time_diff)
    
            #plots
            if plt.get_fignums()==[]:
                plt.figure(figsize=(18,8))
            plt.plot(np.datetime64('1970-01-01T00:00:00')+np.timedelta64(1,'s')*(basetime+data.time),data.first_cbh,'.',color=colors[source],alpha=0.1)
            plt.plot(time_np,cbh_all[:,i_s],'-',color=colors[source],alpha=1,label=source)
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H%M'))
            plt.title('CBH on '+str(d)[:10])
            plt.xlim([d,d+np.timedelta64(1, 'D')])
            plt.ylim([0,12000])
            plt.xlabel('Time (UTC)')
            plt.ylabel(r'$z$ [m]')
        i_s+=1
    
    if found>0:
        
        #stats
        i_site=0
        cbh_all[np.isnan(cbh_all)]=-9999
        for site in  sites_sel:
            weights=np.tile(1/distances[:,i_site],(len(time_offset),1))
            weights[cbh_all==-9999]=0
            weights=weights/np.tile(np.sum(weights,axis=1),(len(sources_sel),1)).T
            cbh_sum=np.sum(cbh_all*weights,axis=1)
            real=~np.isnan(np.sum(cbh_all,axis=1))
            
            #Output
            Output=xr.Dataset()
            Output['first_cbh']=xr.DataArray(data=np.int32(np.nan_to_num(cbh_sum,nan=-9999)),
                                            coords={'time':np.int32(time_offset)},
                                            attrs={'description':f'First cloud base height combining sites {list(sources.keys())}','units':'m'})
            i_s=0
            for source in sources_sel:
                Output[f'first_cbh_{source}']=xr.DataArray(data=np.int32(cbh_all[:,i_s]),
                                                coords={'time':np.int32(time_offset)},
                                                attrs={'description':f'First cloud base height from site {source}','units':'m'})
                i_s+=1
    
            Output['base_time']=np.int64(d-np.datetime64('1970-01-01T00:00:00'))
            Output.attrs['comment']='created on '+datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')+' by stefano.letizia@nrel.gov'
            
            name_save=os.path.basename(dir_save[site])+'.'+datetime.utcfromtimestamp(Output['base_time'].values).strftime('%Y%m%d.%H%M%S')+'.nc'
            Output.to_netcdf(os.path.join(dir_save[site],name_save))
            print(f'{name_save} created',flush=True)
        
            #plots
            plot_sum,=plt.plot(time_np,cbh_sum,'-',color='r',alpha=1,label='Summary')
            leg=plt.legend(loc='upper right')
            plt.grid()
            plt.savefig(os.path.join(dir_save[site],name_save.replace('nc','png')))
            plot_sum.remove()
            leg.remove()
            i_site+=1
            
        plt.close()
            
        
        

