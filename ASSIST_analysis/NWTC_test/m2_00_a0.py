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
import pandas as pd
import yaml
import os
import warnings
plt.close('all')
warnings.filterwarnings("ignore")

#%% Inputs
source_config=os.path.join(cd,'configs','config.yaml')
source=os.path.join(cd,'data/nwtc.m2.00/*txt')
storage=os.path.join(cd,'data/nwtc.m2.a0')

replace=True
column_names = ["DATE", "MST", "Temp_02m", "Temp_50m", "Temp_80m","Ri"]
height=[2,50,80]#[m]
time_offset=7#[h]

#%% INitialization
files=glob.glob(source)
os.makedirs(storage,exist_ok=True)

#%% Main
for f in files:
    data = pd.read_csv(f, names=column_names, header=0, parse_dates=[["DATE", "MST"]])

    data["DATE_MST"]=data["DATE_MST"]+np.timedelta64(time_offset, 'h')
    
    data["Date"] = data["DATE_MST"].dt.date

    # Group by day and store in a dictionary
    daily_data = {date: group for date, group in data.groupby("Date")}
    
    for d in daily_data:
        time=daily_data[d]["DATE_MST"].to_numpy()
        filename=os.path.normpath(storage).split(os.sep)[-1]+'.'+str(time[0])[:-10].replace('-','').replace('T','.').replace(':','')+'.nc'
        if not os.path.isfile(filename) or replace==True:
            Output=xr.Dataset()
            T=np.zeros((len(time),len(height)))
            for i_h in range(len(height)):
                T[:,i_h]=daily_data[d][f"Temp_{height[i_h]:02}m"].values
            Output['temperature']=xr.DataArray(T,coords={'time':time,'height':height})
            Output['Ri']=xr.DataArray(daily_data[d]['Ri'].values,coords={'time':time})
            
            Output.to_netcdf(os.path.join(storage,filename))
        
    


            
        