# -*- coding: utf-8 -*-

import os
cd=os.path.dirname(__file__)
import sys
sys.path.append('/home/sletizia/codes')
import myFunctions as SL
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
import warnings
import matplotlib
import glob
import pandas as pd
from datetime import datetime
import re

warnings.filterwarnings('ignore')
plt.close('all')

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm' 
matplotlib.rcParams['font.size'] = 14

#%% Inputs

#dataset
IDs=[10,11,12]
sources={10:'/srv/data/nfs/assist/Assist-10/raw',
         11:'/srv/data/nfs/assist/Assist-11/raw',
         12:'/srv/data/nfs/assist/Assist-12/raw'}
start_date = '2022-10-01 00:00' # first date to process
end_date =   '2023-11-01 00:00' # last date to process

#%% Initialization


#%% Main
for ID in IDs:
    files_all = np.array(sorted(glob.glob(os.path.join(sources[ID],'*assistsummary*'))))
    t_file=[]
    for f in files_all:
        match = re.search(r'\d{8}\.\d{6}', f)
        t=datetime.strptime(match.group(0),'%Y%m%d.%H%M%S')
        t_file=np.append(t_file,t)

    sel_t=(t_file>=datetime.strptime(start_date,'%Y-%m-%d %H:%M'))*(t_file<datetime.strptime(end_date,'%Y-%m-%d %H:%M'))
    files_selected=files_all[sel_t]  
    
    if len(files_selected)>0:
        time=np.array([],dtype='datetime64')
        T_675=[]
        T_amb=[]
        T_abb=[]
        for f in files_selected:
            print('Processing '+f)
            Data=xr.open_dataset(f)
            Data=Data.sortby('time')
            
            time=np.append(time,Data['time'].values+Data['base_time'].values+np.datetime64('1970-01-01T00:00'))
            T_675=np.append(T_675,Data['mean_Tb_675_680'].values-273.15)
            T_amb=np.append(T_amb,Data['calibrationAmbientTemp'].values)
            T_abb=np.append(T_abb,Data['ABBapexTemp'].values)
            
        Output=pd.DataFrame()
        Output['T_675']=T_675
        Output['T_amb']=T_amb
        Output['T_abb']=T_abb
        Output['Time']=time
        
        Output.set_index('Time').to_csv('/home/sletizia/codes/ASSIST_analysis/Met_validation/data/Summary_T_'+str(ID)+'.csv')
            