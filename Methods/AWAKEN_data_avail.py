# -*- coding: utf-8 -*-
"""
Plot data availability for AWAKEN
"""

import os
cd=os.getcwd()
import sys
sys.path.append(os.path.join(cd,'../utils'))
import utils as utl
import numpy as np
from matplotlib import pyplot as plt
from doe_dap_dl import DAP
import warnings
import matplotlib.dates as mdates
from datetime import datetime
import matplotlib
import glob 
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm' 
matplotlib.rcParams['font.size'] = 14

plt.close('all')
warnings.filterwarnings('ignore')

#%% Inputs
username='sletizia'
password='pass_DAP1506@'
channels=['awaken/sb.assist.z01.00','awaken/sc1.assist.z01.00','awaken/sg.assist.z01.00',
          'awaken/sa1.ceil.z01.b0','awaken/arm.ceil.sgp_s6.cbh.b1','radiosondes',
          'awaken/sb.met.z01.b0',   'awaken/sc1.met.z01.b0', 'awaken/sg.met.z01.b0',
          'awaken/sb.assist.z01.c0','awaken/sc1.assist.z01.c0','awaken/sg.assist.z01.c0']
source_sondes='C:/Users/sletizia/OneDrive - NREL/Desktop/Main/ENDURA/ASSIST_analysis/Radiosondes_validation/data/awaken/sgpsondewnpnS6.b1/*cdf'

ext={'awaken/sb.assist.z01.00':'assistsummary',
     'awaken/sc1.assist.z01.00':'assistsummary',
     'awaken/sg.assist.z01.00':'assistsummary',
     'awaken/sa1.ceil.z01.b0':'',
     'awaken/arm.ceil.sgp_s6.cbh.b1':'',
     'awaken/sb.met.z01.b0':'', 
     'awaken/sc1.met.z01.b0':'',  
     'awaken/sg.met.z01.b0':'',
     'awaken/sb.assist.z01.c0':'',
     'awaken/sc1.assist.z01.c0':'',
     'awaken/sg.assist.z01.c0':''}

dtype={'awaken/sb.assist.z01.00':'cdf',
       'awaken/sc1.assist.z01.00':'cdf',
     'awaken/sg.assist.z01.00':'cdf',
     'awaken/sa1.ceil.z01.b0':'nc',
     'awaken/arm.ceil.sgp_s6.cbh.b1':'nc',
     'awaken/sb.met.z01.b0':'nc', 
     'awaken/sc1.met.z01.b0':'nc',  
     'awaken/sg.met.z01.b0':'nc',
     'awaken/sb.assist.z01.c0':'nc',
     'awaken/sc1.assist.z01.c0':'nc',
     'awaken/sg.assist.z01.c0':'nc'}


sdate='20220901000000'#start date for data search
edate='20231001000000'#end date for data search

#grpahics
colors={'awaken/sb.assist.z01.00':'r',
     'awaken/sc1.assist.z01.00':'r',
     'awaken/sg.assist.z01.00':'r',
     'awaken/sa1.ceil.z01.b0':'b',
     'awaken/arm.ceil.sgp_s6.cbh.b1':'b',
     'awaken/sb.met.z01.b0':'k', 
     'awaken/sc1.met.z01.b0':'k',  
     'awaken/sg.met.z01.b0':'k',
     'radiosondes':'w',
     'awaken/sb.assist.z01.c0':'orange',
     'awaken/sc1.assist.z01.c0':'orange',
     'awaken/sg.assist.z01.c0':'orange'}


labels={'awaken/sb.assist.z01.00':'South ASSIST',
     'awaken/sc1.assist.z01.00':'Middle ASSIST',
     'awaken/sg.assist.z01.00':'North ASSIST',
     'awaken/sa1.ceil.z01.b0':'CL51 ceilometer',
     'awaken/arm.ceil.sgp_s6.cbh.b1':'CL31 ceilometer',
     'awaken/sb.met.z01.b0':'South met', 
     'awaken/sc1.met.z01.b0':'Middle met',  
     'awaken/sg.met.z01.b0':'North met',
     'radiosondes':'Radiosondes',
     'awaken/sb.assist.z01.c0':'South TROPoe',
     'awaken/sc1.assist.z01.c0':'Middle TROPoe',
     'awaken/sg.assist.z01.c0':'North TROPoe'}


#%% Functions
def dap_search(channel,sdate,edate,file_type,ext1,time_search=30):
    '''
    Wrapper for a2e.search to avoid timeout:
        Inputs: channel name, start date, end date, file format, extention in WDH name, number of days scanned at each loop
        Outputs: list of files mathing the criteria
    '''
    dates_num=np.arange(utl.datenum(sdate,'%Y%m%d%H%M%S'),utl.datenum(edate,'%Y%m%d%H%M%S'),time_search*24*3600)
    dates=[utl.datestr(d,'%Y%m%d%H%M%S') for d in dates_num]+[edate]
    search_all=[]
    for d1,d2 in zip(dates[:-1],dates[1:]):
        
        if ext1!='':
            _filter = {
                'Dataset': channel,
                'date_time': {
                    'between': [d1,d2]
                },
                'file_type': file_type,
                'ext1':ext1, 
            }
        else:
            _filter = {
                'Dataset': channel,
                'date_time': {
                    'between': [d1,d2]
                },
                'file_type': file_type,
            }
        
        search=a2e.search(_filter)
        
        if search is None:
            print('Invalid authentication')
            return None
        else:
            search_all+=search
    
    return search_all


#%% Initalization
a2e = DAP('a2e.energy.gov',confirm_downloads=False)
a2e.setup_cert_auth(username=username, password=password)

files={}
time_file={}
    
#%% Main
for c in channels:
    try:
        files[c]=dap_search(c, sdate,edate, dtype[c], ext[c])
    
        time_file[c]=np.array([datetime.strptime(f["date_time"],"%Y%m%d%H%M%S") for f in files[c]])
    except:
        print(f"{c} is not a channel on WDH")

files_sondes=glob.glob(source_sondes)
dates_sondes=[f'{os.path.basename(f).split(".")[2]}{os.path.basename(f).split(".")[3]}' for f in files_sondes]
time_file['radiosondes']=np.array([datetime.strptime(d,"%Y%m%d%H%M%S") for d in dates_sondes])

#%% Plots
plt.figure(figsize=(16,7))
ctr=0
yticks=[]
ylabels=[]
for c in channels:
    for t in time_file[c]:
        plt.plot([t,t],[-ctr, -ctr+0.2],color=colors[c],linewidth=3)
    yticks.append(-ctr)
    ylabels.append(labels[c])
    ctr+=1
date_fmt = mdates.DateFormatter('%b %Y')
ax=plt.gca()
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))  # First day of each month
ax.xaxis.set_major_formatter(date_fmt) 
ax.set_yticks(yticks,ylabels)
plt.xticks(rotation=30)
ax.set_facecolor((0,1,0,0.25))
plt.grid()
plt.tight_layout()
