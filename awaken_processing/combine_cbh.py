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

if len(sys.argv)==1:
    path_save=os.path.join(cd,'data/awaken')
    source1='A1'
    source2='H'
    sdate='20221001'
    edate='20231101'
else:
    path_save=sys.argv[1]
    source1=sys.argv[2]
    source2=sys.argv[3]
    sdate=sys.argv[4]
    edate=sys.argv[5]
    
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

dir_save=os.path.join(path_save,'s'+'.s'.join(list(sources.keys())).lower()+'.ceil.z01.cbh')
os.makedirs(dir_save,exist_ok=True)

sources_sel=[source1,source2]

#%% Main
for d in dates:
    cbh_all=np.zeros((len(time_offset),len(sources)))+np.nan
    data={}
    i_s=0
    time_np=d+np.timedelta64(1,'s')*time_offset
    
    found=0
    for s in sources_sel:
        files=glob.glob(sources[s].format(datestr=str(d)[:10].replace('-','')))
        
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
            plt.plot(np.datetime64('1970-01-01T00:00:00')+np.timedelta64(1,'s')*(basetime+data.time),data.first_cbh,'.',color=colors[s],alpha=0.1)
            plt.plot(time_np,cbh_all[:,i_s],'-',color=colors[s],alpha=1,label=s)
        i_s+=1
    
    if found>0:
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
        for s in sources_sel:
            Output[f'first_cbh_{s}']=xr.DataArray(data=np.int32(np.nan_to_num(cbh_all[:,i_s],nan=-9999)),
                                            coords={'time':np.int32(time_offset)},
                                            attrs={'description':f'First cloud base height from site {s}','units':'m'})
            i_s+=1

        Output['base_time']=np.int64(basetime)
        Output.attrs['comment']='created on '+datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')+' by stefano.letizia@nrel.gov'
        
        name_save=os.path.basename(dir_save)+'.'+datetime.utcfromtimestamp(basetime).strftime('%Y%m%d.%H%M%S')+'.nc'
        Output.to_netcdf(os.path.join(dir_save,name_save))
        print(f'{name_save} created',flush=True)
        
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
        
        

