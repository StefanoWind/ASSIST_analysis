# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 11:37:54 2024

@author: sletizia
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
from scipy import stats

warnings.filterwarnings('ignore')
plt.close('all')

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm' 
matplotlib.rcParams['font.size'] = 12

#%% Inputs

#user
calculate_importance=True
max_DT_met=2

#dataset
source='data/All_T.csv'
IDs=[11,12,10] #IDs of the ASSISTs
var_names=[r'$T$ (met at 2 m) [$^\circ$C]',r'$\hat{T}$ (met at 2 m) [$^\circ$C]',r'$\tilde{T}$ (met at 2 m) [$^\circ$C]',
           r'$\frac{\partial T }{\partial z}$ at the ground [$^\circ$C m$^{-1}$]','Hour',r'$\overline{u}$ [m s$^{-1}$]',r'$\overline{\theta}_w$ [$^\circ$]']

WS_cutin=3#[m/s] cutin wind speed (KP+AF)
WS_rated=12#[m/s] rated wind speed (KP+AF)
WS_cutout=20#[m/s] cutout wind speed (KP+AF)
max_sigma_T=5#[K] maximum uncertainty
timezone=-6#[hours] difference local time - UTC

#stats
n_features=7
N_bins=10

# graphics
skip=5
ID_comb=[[11,10],[11,12],[12,10]]
N_days_plot=7
site_names={10:'North',
            11:'South',
            12:'Middle'}

colors={10:'g',11:'r',12:'b'}
limits={0:[0,50],
        1:[0,50],
        2:[10,10],
        3:[-0.25,0.25],
        4:[0,23],
        5:[0,20],
        6:[0,360]}

xticks={0:[0,10,20,30,40,50],
        1:[0,10,20,30,40,50],
        2:[-10,-5,0,5,10],
        3:np.arange(-0.3,0.31,0.1),
        4:[0,6,12,18,24],
        5:np.arange(0,21,5),
        6:[0,90,180,270,360]}



#%% Initialization
Data=pd.read_csv(os.path.join(cd,source))
Data['Time']=np.array([utl.num_to_dt64(utl.datenum(t,'%Y-%m-%d %H:%M:%S')+timezone*3600) for t in Data['Time'].values])
Data=Data.set_index('Time')

dt=np.nanmedian(np.diff(Data.index))
assert np.nanmax(np.diff(Data.index))==np.nanmin(np.diff(Data.index))
Data_daily_avg=Data.rolling(window=int(np.timedelta64(1,'D')/dt)).mean()
Data_det=Data-Data_daily_avg

#remove high uncertainty
for ID in IDs:
    Data['T_'+str(ID)+'_0.0m'][Data['sigma_T_'+str(ID)+'_0.0m']>max_sigma_T]=np.nan
    Data['sigma_T_'+str(ID)+'_0.0m'][Data['sigma_T_'+str(ID)+'_0.0m']>max_sigma_T]=np.nan
    
    Data['T_'+str(ID)+'_10.0m'][Data['sigma_T_'+str(ID)+'_10.0m']>max_sigma_T]=np.nan
    Data['sigma_T_'+str(ID)+'_10.0m'][Data['sigma_T_'+str(ID)+'_10.0m']>max_sigma_T]=np.nan
    
#%% Main
WS=Data['Hub-height wind speed [m/s]']
WD=Data['Hub-height wind direction [degrees]']
U=(utl.cosd(270-WD)*WS)
V=(utl.sind(270-WD)*WS)
regionI=WS<WS_cutin
regionII=(WS>=WS_cutin)*(WS<WS_rated)
regionIII=(WS>=WS_rated)*(WS<WS_cutout)

hour=np.array([t.hour+t.minute/60 for t in Data.index])

if calculate_importance:
    importance={}
    importance_std={}
fig=plt.figure(figsize=(18,10))
ctr=0
for ID in IDs:
    DT=Data['T_'+str(ID)+'_0.0m']-Data['T_'+str(ID)+'_met']
    DT_dz=(Data['T_'+str(ID)+'_10.0m']-Data['T_'+str(ID)+'_0.0m'])/10
    
    X=np.hstack((Data['T_'+str(ID)+'_met'].values.reshape(-1,1),
                 Data_daily_avg['T_'+str(ID)+'_met'].values.reshape(-1,1),
                 Data_det['T_'+str(ID)+'_met'].values.reshape(-1,1),
                 DT_dz.values.reshape(-1,1),
                 hour.reshape(-1,1),
                 WS.values.reshape(-1,1),
                 WD.values.reshape(-1,1)))
    
    assert len(X[0,:])==n_features, "Wrong number of features"
    
    if calculate_importance:
        importance[ID],importance_std[ID],*_=utl.RF_feature_selector(X,DT.values)
    
    for i_x in range(n_features):
        plt.subplot(len(IDs),n_features,ctr*n_features+i_x+1)
        plt.plot(X[:,i_x],DT.values,'.k',alpha=0.05)
        plt.xlabel(var_names[i_x])
        plt.ylabel(r'$\Delta T$ (TROPoe-met) '+'\n'+ 'at '+site_names[ID]+' [$^\circ$C]')
        plt.grid()
        plt.xlim(limits[i_x])
        plt.ylim([-4,4])
        plt.xticks(xticks[i_x])
    ctr+=1
utl.remove_labels(fig)

    
#%% Plots

if calculate_importance:
    plt.figure()
    ctr=0
    for ID in IDs:
        plt.bar(np.arange(n_features)*3-0.5*(ctr-1),importance[ID],yerr=importance_std[ID],color=colors[ID],capsize=5,linewidth=2,width=0.5,label=site_names[ID])
        ctr+=1
    plt.legend()
    plt.xticks(np.arange(n_features)*3,var_names)
    plt.grid()

#daily cycles
plt.figure(figsize=(18,10))
ctr=0
for ID in IDs:
    DT=Data['T_'+str(ID)+'_0.0m']-Data['T_'+str(ID)+'_met']
    DT_dz=(Data['T_'+str(ID)+'_10.0m']-Data['T_'+str(ID)+'_0.0m'])/10
    
    plt.subplot(len(IDs),4,ctr*4+1)
    plt.plot(hour,Data['T_'+str(ID)+'_met'],'.k',alpha=0.05)
    plt.title(site_names[ID])
    
    plt.subplot(len(IDs),4,ctr*4+2)
    plt.plot(hour,Data_det['T_'+str(ID)+'_met'],'.k',alpha=0.05)
    
    plt.subplot(len(IDs),4,ctr*4+3)
    plt.plot(hour,DT_dz,'.k',alpha=0.05)
    
    plt.subplot(len(IDs),4,ctr*4+4)
    plt.plot(hour,DT,'.k',alpha=0.05)
    ctr+=1
    