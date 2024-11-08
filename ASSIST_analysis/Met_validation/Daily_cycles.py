# -*- coding: utf-8 -*-
"""
Plot deily cycles of temperature, detrended temperatures and their differences
"""
import os
cd=os.path.dirname(__file__)
import sys
sys.path.append('C:/Users/SLETIZIA/OneDrive - NREL/Desktop/PostDoc/utils')
import utils as utl

import numpy as np
from matplotlib import pyplot as plt
import warnings
import matplotlib
import pandas as pd
warnings.filterwarnings('ignore')
plt.close('all')

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm' 
matplotlib.rcParams['font.size'] = 12

#%% Inputs

#user
shield_uncertainty=False

#dataset
source='data/All_T.csv'
IDs=[11,12,10] #IDs of the ASSISTs
timezone=-6#[hours] difference local time - UTC

#stats
max_sigma_T=5#[K] maximum uncertainty

# graphics
site_names={10:'North',
            11:'South',
            12:'Middle'}

colors={10:'g',11:'r',12:'b'}

#%% Functions
def met_uncertainty(T,WS,shield_uncertainty):
    unc_T1=np.zeros(len(T))
    unc_T2=np.zeros(len(T))
    
    unc_T1=0.005*np.abs(T-20)+0.2
    
    if shield_uncertainty:
        ws=np.array([0,1,2,3,6,100])
        unc_ws=np.array([1.51,1.51,0.7,0.4,0.2,0.2])
        unc_T2=np.interp(WS,ws,unc_ws)
    else:
        unc_T2=0
    
    return unc_T1+unc_T2

#%% Initialization
Data=pd.read_csv(os.path.join(cd,source))
Data['Time']=np.array([utl.num_to_dt64(utl.datenum(t,'%Y-%m-%d %H:%M:%S')+timezone*3600) for t in Data['Time'].values])
Data=Data.set_index('Time')

#remove high uncertainty
for ID in IDs:
    Data['T_'+str(ID)+'_0.0m'][Data['sigma_T_'+str(ID)+'_0.0m']>max_sigma_T]=np.nan
    Data['sigma_T_'+str(ID)+'_0.0m'][Data['sigma_T_'+str(ID)+'_0.0m']>max_sigma_T]=np.nan
    
    Data['T_'+str(ID)+'_10.0m'][Data['sigma_T_'+str(ID)+'_10.0m']>max_sigma_T]=np.nan
    Data['sigma_T_'+str(ID)+'_10.0m'][Data['sigma_T_'+str(ID)+'_10.0m']>max_sigma_T]=np.nan
    
    Data['sigma_T_'+str(ID)+'_met']=met_uncertainty(Data['T_'+str(ID)+'_met'],Data['WS_'+str(ID)+'_met'],shield_uncertainty)
    Data['T_'+str(ID)+'_met'][Data['sigma_T_'+str(ID)+'_met']>max_sigma_T]=np.nan
    Data['sigma_T_'+str(ID)+'_met'][Data['sigma_T_'+str(ID)+'_met']>max_sigma_T]=np.nan
    
#%% Main
Data['hour']=np.array([t.hour+t.minute/60 for t in Data.index])

dt=np.nanmedian(np.diff(Data.index))
assert np.nanmax(np.diff(Data.index))==np.nanmin(np.diff(Data.index))
Data_daily_avg=Data.rolling(window=int(np.timedelta64(1,'D')/dt)).mean()
Data_det=Data-Data_daily_avg

for ID in IDs:
    Data['T_unb_{ID}_0.0m'.format(ID=ID)]=Data['T_{ID}_0.0m'.format(ID=ID)]-np.nanmean(Data['T_{ID}_0.0m'.format(ID=ID)]-Data['T_{ID}_met'.format(ID=ID)])
    Data['T_unb_{ID}_10.0m'.format(ID=ID)]=Data['T_{ID}_10.0m'.format(ID=ID)]-np.nanmean(Data['T_{ID}_10.0m'.format(ID=ID)]-Data['T_{ID}_met'.format(ID=ID)])
    
T_met_avg={}
T_0m_avg={}
T_10m_avg={}
T_det_met_avg={}
T_det_0m_avg={}
T_det_10m_avg={}
SWR_avg={}
T_abb_avg={}
DT_0m_avg={}
DT_10m_avg={}
DT_unb_0m_avg={}
DT_unb_10m_avg={}

T_met_low={}
T_0m_low={}
T_10m_low={}
T_det_met_low={}
T_det_0m_low={}
T_det_10m_low={}
SWR_low={}
T_abb_low={}
DT_0m_low={}
DT_10m_low={}
DT_unb_0m_low={}
DT_unb_10m_low={}

T_met_top={}
T_0m_top={}
T_10m_top={}
T_det_met_top={}
T_det_0m_top={}
T_det_10m_top={}
SWR_top={}
T_abb_top={}
DT_0m_top={}
DT_10m_top={}
DT_unb_0m_top={}
DT_unb_10m_top={}

for ID in IDs:
    T_met_avg[ID],T_met_low[ID],T_met_top[ID]=utl.bins_unc(Data['hour'],Data['T_{ID}_met'.format(ID=ID)],bins=np.arange(-0.5,24))
    T_0m_avg[ID],T_0m_low[ID],T_0m_top[ID]=utl.bins_unc(Data['hour'],Data['T_{ID}_0.0m'.format(ID=ID)],bins=np.arange(-0.5,24))
    T_10m_avg[ID],T_10m_low[ID],T_10m_top[ID]=utl.bins_unc(Data['hour'],Data['T_{ID}_10.0m'.format(ID=ID)],bins=np.arange(-0.5,24))
    
    T_det_met_avg[ID],T_det_met_low[ID],T_det_met_top[ID]=utl.bins_unc(Data['hour'],Data_det['T_{ID}_met'.format(ID=ID)],bins=np.arange(-0.5,24))
    T_det_0m_avg[ID],T_det_0m_low[ID],T_det_0m_top[ID]=utl.bins_unc(Data['hour'],Data_det['T_{ID}_0.0m'.format(ID=ID)],bins=np.arange(-0.5,24))
    T_det_10m_avg[ID],T_det_10m_low[ID],T_det_10m_top[ID]=utl.bins_unc(Data['hour'],Data_det['T_{ID}_10.0m'.format(ID=ID)],bins=np.arange(-0.5,24))
    
    SWR_avg[ID],SWR_low[ID],SWR_top[ID]=utl.bins_unc(Data['hour'],Data['SWR_{ID}_met'.format(ID=ID)],bins=np.arange(-0.5,24))
    T_abb_avg[ID],T_abb_low[ID],T_abb_top[ID]=utl.bins_unc(Data['hour'],Data['T_abb_{ID}_sum'.format(ID=ID)],bins=np.arange(-0.5,24))
    
    DT_0m_avg[ID],DT_0m_low[ID],DT_0m_top[ID]=utl.bins_unc(Data['hour'],Data['T_{ID}_0.0m'.format(ID=ID)]-Data['T_{ID}_met'.format(ID=ID)],bins=np.arange(-0.5,24))
    DT_10m_avg[ID],DT_10m_low[ID],DT_10m_top[ID]=utl.bins_unc(Data['hour'],Data['T_{ID}_10.0m'.format(ID=ID)]-Data['T_{ID}_met'.format(ID=ID)],bins=np.arange(-0.5,24))
    
    DT_unb_0m_avg[ID],DT_unb_0m_low[ID],DT_unb_0m_top[ID]=utl.bins_unc(Data['hour'],Data['T_unb_{ID}_0.0m'.format(ID=ID)]-Data['T_{ID}_met'.format(ID=ID)],bins=np.arange(-0.5,24))
    DT_unb_10m_avg[ID],DT_unb_10m_low[ID],DT_unb_10m_top[ID]=utl.bins_unc(Data['hour'],Data['T_unb_{ID}_10.0m'.format(ID=ID)]-Data['T_{ID}_met'.format(ID=ID)],bins=np.arange(-0.5,24))
    
#%% Plots
plt.close('all')
#absolute temperature 
fig=plt.figure(figsize=(18,5))
for ID in IDs:
    ax=plt.subplot(1,len(IDs),np.where(ID==np.array(IDs))[0][0]+1)
    plt.plot(np.arange(24),T_met_avg[ID],'k',label='Met at 2 m')
    ax.fill_between(np.arange(24),T_met_low[ID],T_met_top[ID],color='k', alpha=0.25)
    plt.ylabel(r'$T$ [$^\circ$C]')
    
    plt.plot(np.arange(24),T_0m_avg[ID],'b',label='TROPoe at 0 m')
    ax.fill_between(np.arange(24),T_0m_low[ID],T_0m_top[ID],color='b', alpha=0.25)
    
    plt.plot(np.arange(24),T_10m_avg[ID],'r',label='TROPoe at 10 m')
    ax.fill_between(np.arange(24),T_10m_low[ID],T_10m_top[ID],color='r', alpha=0.25)
    
    plt.grid()
    plt.title(site_names[ID])
    plt.xticks([0,6,12,18],labels=['00','06','12','18'])
    plt.xlabel('Hour')
utl.remove_labels(fig)    
plt.legend()

#detrended temperature
fig=plt.figure(figsize=(18,5))
for ID in IDs:
    ax=plt.subplot(1,len(IDs),np.where(ID==np.array(IDs))[0][0]+1)
    plt.plot(np.arange(24),T_det_met_avg[ID],'k',label='Met at 2 m')
    ax.fill_between(np.arange(24),T_det_met_low[ID],T_det_met_top[ID],color='k', alpha=0.25)
    plt.ylabel(r'$\tilde{T}$ [$^\circ$C]')
    
    plt.plot(np.arange(24),T_det_0m_avg[ID],'b',label='TROPoe at 0 m')
    ax.fill_between(np.arange(24),T_det_0m_low[ID],T_det_0m_top[ID],color='b', alpha=0.25)
    
    plt.plot(np.arange(24),T_det_10m_avg[ID],'r',label='TROPoe at 10 m')
    ax.fill_between(np.arange(24),T_det_10m_low[ID],T_det_10m_top[ID],color='r', alpha=0.25)
    
    plt.grid()
    plt.title(site_names[ID])
    plt.xticks([0,6,12,18],labels=['00','06','12','18'])
    plt.xlabel('Hour')
utl.remove_labels(fig)
plt.legend()

#solar radiation, ABB
fig=plt.figure(figsize=(14,10))
for ID in IDs:
    ax1=plt.subplot(3,len(IDs),np.where(ID==np.array(IDs))[0][0]+1)
    plt.plot(np.arange(24),T_met_avg[ID],'k',label='Met at 2 m')
    ax1.fill_between(np.arange(24),T_met_low[ID],T_met_top[ID],color='k', alpha=0.25)
    plt.plot(np.arange(24),T_abb_avg[ID],'g',label='ABB')
    ax1.fill_between(np.arange(24),T_abb_low[ID],T_abb_top[ID],color='g', alpha=0.25)
    plt.ylabel(r'T [$^\circ$C]')
    plt.ylim([0,50])
    plt.yticks(np.arange(0,51,10))
    plt.xticks([0,6,12,18],labels=['00','06','12','18'])
    plt.grid()
    plt.title(site_names[ID])
    if ID==IDs[-1]:
        plt.legend()
    
    ax2=plt.subplot(3,len(IDs),np.where(ID==np.array(IDs))[0][0]+4)
    plt.plot(np.arange(24),SWR_avg[ID],'k')
    ax2.fill_between(np.arange(24),SWR_low[ID],SWR_top[ID],color='k', alpha=0.25)
    plt.ylabel(r'SW radiation [m $^{-2}$]')
    plt.ylim([0,1000])
    plt.yticks(np.arange(0,1001,200))
    plt.grid()
    
    plt.xticks([0,6,12,18],labels=['00','06','12','18'])
    
    ax2=plt.subplot(3,len(IDs),np.where(ID==np.array(IDs))[0][0]+7)
    plt.plot(np.arange(24),DT_0m_avg[ID],'k',label='Raw')
    ax2.fill_between(np.arange(24),DT_0m_low[ID],DT_0m_top[ID],color='k', alpha=0.25)
    plt.plot(np.arange(24),DT_unb_0m_avg[ID],'--k',label='Unbiased')
    ax2.fill_between(np.arange(24),DT_unb_0m_low[ID],DT_unb_0m_top[ID],color='k', alpha=0.25)
    plt.ylabel(r'$\Delta T$ (TROPoe 0 m - met 2 m) [$^\circ$C]')
    plt.ylim([-1,1])
    plt.yticks(np.arange(-1,1.01,0.5))
    plt.grid()
    plt.xticks([0,6,12,18],labels=['00','06','12','18'])
    if ID==IDs[-1]:
        plt.legend()
    
    plt.xlabel('Hour')
utl.remove_labels(fig)

#profiles (unbiased)
plt.figure(figsize=(18,9))
for ID in IDs:
    ax1=plt.subplot(len(IDs),1,np.where(ID==np.array(IDs))[0][0]+1)
    for h in np.arange(24):
        plt.plot(np.array([0,DT_unb_0m_avg[ID][h],DT_unb_10m_avg[ID][h]])+h,[2,0,10],'.-',color=plt.cm.coolwarm((T_met_avg[ID][h]-20)/12),markersize=12)
        
    ax1.set_ylim([0,10])
    ax1.set_xticks(np.arange(24))
    ax1.set_yticks(np.arange(2,11,2))
    ax1.grid(visible=True)
    ax1.set_ylabel(r'$z$ [m.a.g.l.]')
    
    ax2 = ax1.twinx()
    ax2.fill_between(np.arange(24),DT_unb_0m_avg[ID],np.arange(24)*0,color='k',alpha=0.25)
    ax2.set_ylabel(r'$\Delta T$ (TROPoe 0 m - met 2 m) [$^\circ$C]')
    ax2.set_ylim([-0.4,0.4])
    ax2.set_yticks(np.arange(-0.4,0.5,0.2))
    plt.title(site_names[ID])
ax1.set_xlabel('Hour')

plt.tight_layout()