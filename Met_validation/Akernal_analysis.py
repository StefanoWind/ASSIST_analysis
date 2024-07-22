# -*- coding: utf-8 -*-
"""
Created on Thu May  9 14:33:35 2024

@author: sletizia
"""

# -*- coding: utf-8 -*-
"""
Plot deily cycles of temperature, detrended temperatures and their differences
"""
import os
cd=os.path.dirname(__file__)
import sys
sys.path.append('C:/Users/SLETIZIA/OneDrive - NREL/Desktop/PostDoc/utils')
import utils as utl
import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
import warnings
import matplotlib
import pandas as pd
import glob 

warnings.filterwarnings('ignore')
plt.close('all')

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm' 
matplotlib.rcParams['font.size'] = 12

#%% Inputs

#dataset
sources='data/assist-{ID}/*000015.nc'
IDs=[11,12,10]
sel_gate=1#selected TROPoe gates for extraction
zmax=2000#maximum height
timezone=-6#time zone in hours (CST)

site_names={10:'North',
            11:'South',
            12:'Middle'}

#%% Initialization
A_sel={}
time={}
A_avg={}
A_low={}
A_top={}

#%% Main
for ID in IDs:
    print('Reading ASSIST '+str(ID)+' data')
    files=np.array(glob.glob(sources.format(ID=ID)))
    A_sel[ID]=[]
    time[ID]=np.array([],dtype='datetime64')

    for f in files:
        Data=xr.open_dataset(f)
        A=Data['Akernal'].values
        z=Data['height'].values*1000
        i_max=np.where(z>zmax)[0][0]
        A_sel[ID]=utl.vstack(A_sel[ID],A[:,sel_gate,:i_max])
        time[ID]=np.append(time[ID],Data.time.values+np.timedelta64(timezone, 'h'))

    hour=np.array([pd.Timestamp(t).hour+pd.Timestamp(t).minute/60 for t in time[ID]])
    A_avg[ID]=np.zeros((24,i_max))
    A_low[ID]=np.zeros((24,i_max))
    A_top[ID]=np.zeros((24,i_max))
    
    for i in range(i_max):
        A_avg[ID][:,i],A_low[ID][:,i],A_top[ID][:,i]=utl.bins_unc(hour,A_sel[ID][:,i],bins=np.arange(-0.5,24))
  
#%% Plots
plt.close('all')
fig=plt.figure(figsize=(18,10))
for ID in IDs:
    ax = fig.add_subplot(len(IDs),1,np.where(ID==np.array(IDs))[0][0]+1)
    plt.sca(ax)

    plt.pcolor(time[10],z[:i_max],np.log10(np.abs(A_sel[10].T)),vmin=-2,vmax=0,cmap='hot')
    plt.colorbar(label=r'Log$_10$ of'+'\n A-kernel at '+str(z[sel_gate])+' m')
    plt.ylabel(r'$z$ [m.a.g.l]')
    plt.title(site_names[ID])
plt.xlabel('Time')  
plt.tight_layout()

fig=plt.figure(figsize=(18,10))
for ID in IDs:
    ax = fig.add_subplot(len(IDs),1,np.where(ID==np.array(IDs))[0][0]+1)
    plt.sca(ax)

    plt.pcolor(np.arange(24),z[:i_max],np.log10(np.abs(A_avg[10].T)),vmin=-2,vmax=0,cmap='hot')
    plt.colorbar(label=r'Log$_10$ of'+'\n A-kernel at '+str(z[sel_gate])+' m')
    plt.ylabel(r'$z$ [m.a.g.l]')
    plt.title(site_names[ID])
    plt.xticks([0,6,12,18],labels=['00','06','12','18'])
plt.xlabel('Hour')  
plt.tight_layout()