'''
Format raw M5 data into a0 netCDF
'''

import os
cd=os.path.dirname(__file__)
import scipy.io as spio
import matplotlib.pyplot as plt
import glob
import xarray as xr
import numpy as np
from datetime import datetime, timedelta
import os
import warnings
plt.close('all')
warnings.filterwarnings("ignore")

#%% Inputs
source='Y:/Wind-data/Public/Projects/Met135/MetData/M5Twr'
sdate='2022-05-15'#[%Y-%m-%d] start date
edate='2022-05-16'#[%Y-%m-%d] end date
storage=os.path.join(cd,'data/nwtc/nwtc.m5.a0')#where to save
replace=False#replace existing files?

zero_datenum=719529#[days] 1970-01-01 in matlab time

#variables
variables={'air_temp':'Air_Temp_{z}m',
           'dewp_temp':'Dewpt_Temp_{z}m',
           'press':'Baro_Presr_{z}m',
           'precip':'Precip_TF',
           'u':'Sonic_x_clean_{z}m',
           'v':'Sonic_y_clean_{z}m',
           'w':'Sonic_z_clean_{z}m',
           'T':'Sonic_Temp_clean_{z}m'}

#units (in mat structures)
units={'air_temp':'C',
       'dewp_temp':'C',
       'press':'kPa',
       'precip':'unitless',
       'u':'m/s',
       'v':'m/s',
       'w':'m/s',
       'T':'K'}

#sensor heights a.g.l.
heights={'air_temp':[3,38,87,122],
       'dewp_temp':[3,38,87,122],
       'press':[3,38,87,122],
       'precip':[3],
       'u':[41,61,74,119],
       'v':[41,61,74,119],
       'w':[41,61,74,119],
       'T':[41,61,74,119]}
       
#type
variable_types={'air_temp':'therm',
               'dewp_temp':'therm',
               'press':'therm',
               'precip':'prec',
               'u':'kin',
               'v':'kin',
               'w':'kin',
               'T':'kin'}

#%% Functions
def extract_data(day,source,storage):
    files=sorted(glob.glob(os.path.join(source,day.strftime('%Y/%m/%d')+'/raw_data/*.mat')))
    
    for f in files:
        
        #comopse filename
        time_str = datetime.strptime(os.path.basename(f)[:-4], "%m_%d_%Y_%H_%M_%S_%f").strftime("%Y%m%d.%H%M%S")
        filename=f'nwtc.m5.a0.{time_str}.nc'
        
        if replace==False and os.path.exists(os.path.join(storage,filename))==True:
            print(f'{f} skipped')
        else:
            data=xr.Dataset()
            
            #read mat structure
            mat = spio.loadmat(f, struct_as_record=False, squeeze_me=True)
            
            #matlab datenum to numpy
            tnum=(mat['time_UTC'].val-zero_datenum)*24*3600
            time=np.datetime64('1970-01-01T00:00:00')+np.timedelta64(1,'ms')*np.int64(tnum*1000)
    
            #ingest variable
            for v in variables:
                var=np.zeros((len(time),len(heights[v])))+np.nan
                i=0
                for h in heights[v]:
                    try:
                        var[:,i]=mat[variables[v].format(z=h)].val
                    except:
                        pass
                    i+=1
                
                data[v]=xr.DataArray(data=var,coords={'time':time,f'height_{variable_types[v]}':heights[v]},attrs={'units':units[v]})
                                       
            #reconstruct air temperature from delta T
            T_rec=np.zeros((len(time),len(heights['air_temp'])))+np.nan
            T_rec[:,0]=mat['Air_Temp_{z}m'.format(z=heights['air_temp'][0])].val
            for i in range(len(heights['air_temp'])-1):
                h1=heights['air_temp'][i]
                h2=heights['air_temp'][i+1]
                T_rec[:,i+1]=T_rec[:,i]+mat['DeltaT_{z2}_{z1}m'.format(z2=h2,z1=h1)].val
            
            data['air_temp_rec']=xr.DataArray(data=T_rec,coords={'time':time,'height_therm':heights['air_temp']},attrs={'units':units[v]})
            
            #output
            data=data.sortby('time').to_netcdf(os.path.join(storage,filename))
            print(f'{filename} created')
         
#%% Initialization
os.makedirs(storage,exist_ok=True)

start = datetime.strptime(sdate, '%Y-%m-%d')
end = datetime.strptime(edate, '%Y-%m-%d')
days = [start + timedelta(days=i) for i in range((end - start).days + 1)]

#%% Main   
args = [(days[i],source,storage) for i in range(len(days))]

for d in days:
    extract_data(d, source,storage)


            
        