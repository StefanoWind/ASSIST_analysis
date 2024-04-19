# -*- coding: utf-8 -*-

import os
cd=os.path.dirname(__file__)
import sys
sys.path.append('C:/Users/SLETIZIA/OneDrive - NREL/Desktop/PostDoc/Custom_functions')
sys.path.append('C:/Users/SLETIZIA/OneDrive - NREL/Desktop/PostDoc/Custom_functions/dap-py')
import myFunctions as SL
from doe_dap_dl import DAP
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
import warnings
import matplotlib
import glob
import pandas as pd
from datetime import datetime

warnings.filterwarnings('ignore')
plt.close('all')

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm' 
matplotlib.rcParams['font.size'] = 14

#%% Inputs

#dataset
download=True
username = 'sletizia'#ask DAP (dapteam@pnnl.gov)
password = 'pass_DAP1506@'
start_time='2023-05-08 00:00'
end_time='2023-10-16 00:00'

IDs=[10,11,12]
channels={10:'awaken/sg.assist.z01.00',
          11:'awaken/sb.assist.z01.00',
          12:'awaken/sc1.assist.z01.00'}

#%% Initialization
time_range_dap = [datetime.strftime(datetime.strptime(start_time, '%Y-%m-%d %H:%M'), '%Y%m%d%H%M%S'),
                  datetime.strftime(datetime.strptime(end_time, '%Y-%m-%d %H:%M'), '%Y%m%d%H%M%S')]

a2e = DAP('a2e.energy.gov',confirm_downloads=False)
a2e.setup_cert_auth(username=username, password=password)

#%% Main
for ID in IDs:
    if download:
        filter = {
            'Dataset': channels[ID],
            'date_time': {'between': time_range_dap},
            'file_type': 'cdf',
            'ext1':'assistsummary'
        }
        a2e.download_with_order(filter, path=os.path.join('data','assist-'+str(ID)), replace=False)
    
    files=glob.glob(os.path.join('data','assist-'+str(ID),'*assistsummary*'))
    
    if len(files)>0:
        time=np.array([],dtype='datetime64')
        T_675=[]
        T_amb=[]
        T_abb=[]
        for f in files:
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
        
        Output.set_index('Time').to_csv('data/Summary_T_'+str(ID)+'.csv')
            