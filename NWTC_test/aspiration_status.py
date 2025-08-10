'''
Extract aspiration status from met data
'''

import os
cd=os.path.dirname(__file__)
import scipy.io as spio
import matplotlib.pyplot as plt
import glob
import xarray as xr
import yaml
import numpy as np
from datetime import datetime, timedelta
import os
import warnings
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['savefig.dpi'] = 50
warnings.filterwarnings('ignore')
plt.close('all')

#%% Inputs
path_config=os.path.join(cd,'configs/config.yaml')
source='Y:/Wind-data/Public/Projects/Met135/MetData/M5Twr'
sdate='2022-04-20' #[%Y-%m-%d] start date
edate='2022-08-25'#[%Y-%m-%d] end date
storage=os.path.join(cd,'data/nwtc/nwtc.m5_asp.c1')#where to save
replace=False#replace existing files?

#variables
variable='Raw_Asp_{z}_mean'

#sensor heights a.g.l.
heights=[3,38,87,122]

#%% Initialization
with open(path_config, 'r') as fid:
    config = yaml.safe_load(fid)
    
os.makedirs(storage,exist_ok=True)

start = datetime.strptime(sdate, '%Y-%m-%d')
end = datetime.strptime(edate, '%Y-%m-%d')
days = [start + timedelta(days=i) for i in range((end - start).days + 1)]

#%% Main   
for day in days:
    filename=os.path.join(storage,f'{os.path.basename(storage)}.{day.strftime("%Y%m%d.%H%M%S")}.nc')
    
    if os.path.isfile(filename):
        print(f'{filename} skipped')
    else:
        files=sorted(glob.glob(os.path.join(source,day.strftime('%Y/%m/%d')+'/summary_data/*.mat')))
        if len(files)>0:
            data=xr.Dataset()
            time=np.array([],dtype='datetime64')
            var=np.zeros((len(files),len(heights)))+np.nan
            
            i_f=0
            for f in files:
                
                #compose filename
                time_dt= datetime.strptime(os.path.basename(f)[:-4], "%m_%d_%Y_%H_%M_%S_%f")
                time_str =time_dt.strftime("%Y%m%d.%H%M%S")
               
                #read mat structure
                mat = spio.loadmat(f, struct_as_record=False, squeeze_me=True)
                
                #save time
                time=np.append(time,np.datetime64(time_dt.strftime("%Y-%m-%dT%H:%M:%S")))
        
                #ingest variable
                i_h=0
                for h in heights:
                    var[i_f,i_h]=mat[variable.format(z=h)].val
                    i_h+=1
                i_f+=1
                    
            #output
            data['aspiration']=xr.DataArray(data=var,coords={'time':time,'height':heights})
            data=data.sortby('time').to_netcdf(os.path.join(storage,filename))
            print(f'{filename} created')
