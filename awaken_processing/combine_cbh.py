# -*- coding: utf-8 -*-
"""
Combine CBH from different locations at AWAKEN
"""

import os
cd=os.path.dirname(__file__)
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

#dataset
sources={'A1':'/scratch/sletizia/data/awaken/sa1.ceil.z01.cbh/*{datestr}*nc',
         'H':'/scratch/sletizia/data/awaken/sgpceilS6.cbh/*{datestr}*nc'}

path_save='/scratch/sletizia/data/awaken'

sdate='20221001'
edate='20231031'

#time interpolation
dtime=60#[s]
max_time_diff=30#[s]

#graphics
colors={'A1':'b','H':'g','E37':'m'}
   
#%% Initialization
dates=np.arange(np.datetime64(f"{sdate[:4]}-{sdate[4:6]}-{sdate[6:]}T00:00:00"),
                np.datetime64(f"{edate[:4]}-{edate[4:6]}-{edate[6:]}T00:00:00")+np.timedelta64(1,'s'),np.timedelta64(1,'D'))
time_offset=np.arange(0,3600*24+1,dtime)

dir_save=os.path.join(path_save,'s'+'.s'.join(list(sources.keys())).lower()+'.ceil.z01.cbh')
os.makedirs(dir_save,exist_ok=True)

#%% Main
for d in dates:
    cbh_all=np.zeros((len(time_offset),len(sources)))+np.nan
    data={}
    i_s=0
    time_np=d+np.timedelta64(1,'s')*time_offset
    plt.figure(figsize=(18,8))
    for s in sources:
        files=glob.glob(sources[s].format(datestr=str(d)[:10].replace('-','')))
        
        basetime=0
        if len(files)==1:
            
            #load data
            data=xr.open_dataset(files[0])
            basetime=data.base_time.values
            data=data.where(data['first_cbh']>0)
            
            #interpolate in time
            cbh_int=data.first_cbh.interp(time=time_offset)
            time_diff=data.time.interp(time=time_offset,method='nearest')-time_offset
            cbh_all[:,i_s]=cbh_int.where(np.abs(time_diff)<=max_time_diff)
    
            #plots
            plt.plot(np.datetime64('1970-01-01T00:00:00')+np.timedelta64(1,'s')*(basetime+data.time),data.first_cbh,'.',color=colors[s],alpha=0.1)
            plt.plot(time_np,cbh_all[:,i_s],'-',color=colors[s],alpha=1,label=s)
        i_s+=1
    
    if basetime>0:
        #stats
        cbh_sum=np.nanmean(cbh_all,axis=1)
        cbh_sel=cbh_all[~np.isnan(np.sum(cbh_all,axis=1)),:]
        rho=np.corrcoef(cbh_sel[:,0],cbh_sel[:,1])[0,1]
        
        #Output
        Output=xr.Dataset()
        Output['first_cbh']=xr.DataArray(data=np.int32(np.nan_to_num(cbh_sum,nan=-9999)),
                                        coords={'time':np.int32(time_offset)},
                                        attrs={'description':f'First cloud base height combining sites {list(sources.keys())}','units':'m'})
        i_s=0
        for s in sources:
            Output[f'first_cbh_{s}']=xr.DataArray(data=np.int32(np.nan_to_num(cbh_all[:,i_s],nan=-9999)),
                                            coords={'time':np.int32(time_offset)},
                                            attrs={'description':f'First cloud base height from site {s}','units':'m'})
            i_s+=1

        Output['base_time']=np.int64(basetime)
        Output.attrs['comment']='created on '+datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')+' by stefano.letizia@nrel.gov'
        
        name_save=os.path.basename(dir_save)+'.'+datetime.utcfromtimestamp(basetime).strftime('%Y%m%d.%H%M%S')+'.nc'
        Output.to_netcdf(os.path.join(dir_save,name_save))
        
        #Plots
        plt.plot(time_np,cbh_sum,'-',color='r',alpha=1,label='Summary')
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H%M'))
        plt.title('CBH on '+str(d)[:10])
        plt.xlim([d,d+np.timedelta64(1, 'D')])
        plt.ylim([0,12000])
        plt.xlabel('Time (UTC)')
        plt.ylabel(r'$z$ [m]')
        plt.text(time_np[10],11500,r'$\rho='+str(np.round(rho,2))+'$',bbox={'facecolor':'w','alpha':0.5,'edgecolor':'k'})
        plt.legend(loc='upper right')
        plt.grid()
        
        plt.savefig(os.path.join(dir_save,name_save.replace('nc','png')))
        plt.close()
        
        

