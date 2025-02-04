'''
extract M5 data
'''

import os
cd=os.path.dirname(__file__)
import sys
import scipy.io as spio
import matplotlib.pyplot as plt
import glob
import datetime as dt
import xarray as xr
import numpy as np
# from multiprocessing import Pool
import yaml
import os
import warnings
plt.close('all')
warnings.filterwarnings("ignore")

#%% Inputs
source_config=os.path.join(cd,'configs','config.yaml')
source='Y:/Wind-data/Public/Projects/Met135/MetData/M5Twr'
sdate='2022-04-15'#[%Y-%m-%d] start date
edate='2022-05-16'#[%Y-%m-%d] end date
storage=os.path.join(cd,'data/nwtc.m5.a0')
replace=False

zero_datenum=719529#[days] 1970-01-01 in matlab time
variables={'air_temp':'Air_Temp_{z}m',
           'dewp_temp':'Dewpt_Temp_{z}m',
           'press':'Baro_Presr_{z}m',
           'precip':'Precip_TF'}

units={'air_temp':'C',
       'dewp_temp':'C',
       'press':'kPa',
       'precip':'unitless'}

height=[3,38,87,122]#[m] heights of the sensors

skip=20#downsampling ratio

#%% Functions
def extract_data(day,source,storage):
    files=glob.glob(os.path.join(source,day.strftime('%Y/%m/%d')+'/raw_data/*.mat'))
    Output=xr.Dataset()
    filename='nwtc.m5.therm.a0.'+day.strftime('%Y%m%d.%H%M%S')+'.nc'
    if replace==False and os.path.exists(os.path.join(storage,filename))==True:
        print(filename+' skipped')
    else:
        for f in files:
            Data=xr.Dataset()
            
            mat = spio.loadmat(f, struct_as_record=False, squeeze_me=True)
            tnum=(mat['time_UTC'].val-zero_datenum)*24*3600
            time=np.datetime64('1970-01-01T00:00:00')+np.timedelta64(1,'ms')*np.int64(tnum*1000)

            for v in variables:
                var=np.zeros((len(time[::skip]),len(height)))+np.nan
                i=0
                for h in height:
                    try:
                        var[:,i]=mat[variables[v].format(z=h)].val[::skip]
                    except:
                        pass
                    i+=1
                Data[v]=xr.DataArray(data=var,coords={'time':time[::skip],'height':height},attrs={'units':units[v]})
                                       
            T_rec=np.zeros((len(time[::skip]),len(height)))+np.nan
            
            #reconstructed air temperature
            T_rec[:,0]=mat['Air_Temp_{z}m'.format(z=height[0])].val[::skip]
            for i in range(len(height)-1):
                h1=height[i]
                h2=height[i+1]
                T_rec[:,i+1]=T_rec[:,i]+mat['DeltaT_{z2}_{z1}m'.format(z2=h2,z1=h1)].val[::skip]
            
            Data['air_temp_rec']=xr.DataArray(data=T_rec,coords={'time':time[::skip],'height':height},attrs={'units':units[v]})
            Data=Data.sortby('time')
            if 'time' in Output.dims:
                Output=xr.concat([Output,Data],dim='time')
            else:
                Output=Data
            print(f)
            
        Output.to_netcdf(os.path.join(storage,filename))
         
#%% Initialization
#config
with open(source_config, 'r') as fid:
    config = yaml.safe_load(fid)
    
#imports
sys.path.append(config['path_utils'])
import utils as utl

utl.mkdir(storage)

#generate days
d1=dt.datetime.utcfromtimestamp(utl.datenum(sdate,'%Y-%m-%d'))
d2=dt.datetime.utcfromtimestamp(utl.datenum(edate,'%Y-%m-%d'))

days=[]
d=d1
while d<=d2:
    days.append(d)
    d+=dt.timedelta(1,0,0)


#%% Main   
args = [(days[i],source,storage) for i in range(len(days))]
 
# with Pool(processes=1) as pool:
#     pool.starmap(extract_data, args)

for d in days:
    extract_data(d, source,storage)


            
        