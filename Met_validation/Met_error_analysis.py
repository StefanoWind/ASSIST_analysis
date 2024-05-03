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

warnings.filterwarnings('ignore')
plt.close('all')

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm' 
matplotlib.rcParams['font.size'] = 12

#%% Inputs

#user
calculate_importance=False
shield_uncertainty=False
active_features=[0,1,2,3,4,5,6,7,10]#feature to consider
pairs=[[0,3],#pairs of features for 2D plots
       [4,0],
       [4,3],
       [4,7]]

#dataset
source='data/All_T.csv'
IDs=[11,12,10] #IDs of the ASSISTs
_vars=['T_{ID}_met','T_daily_avg_{ID}_met','T_det_{ID}_met','DT_dz_{ID}','hour',
       'Hub-height wind speed [m/s]','Hub-height wind direction [degrees]','WS_{ID}_met',
       'T_abb_{ID}_sum','T_frontend_{ID}_sum','NEN_{ID}_sum']

WS_cutin=3#[m/s] cutin wind speed (KP+AF)
WS_rated=12#[m/s] rated wind speed (KP+AF)
WS_cutout=20#[m/s] cutout wind speed (KP+AF)
max_sigma_T=5#[K] maximum uncertainty
timezone=-6#[hours] difference local time - UTC

# graphics
skip=5
ID_comb=[[11,10],[11,12],[12,10]]
N_days_plot=7

site_names={10:'North',
            11:'South',
            12:'Middle'}

colors={10:'g',11:'r',12:'b'}

labels={'T_{ID}_met':r'$T$ (met at 2 m) [$^\circ$C]',
        'T_daily_avg_{ID}_met':r'$\hat{T}$ (met at 2 m) [$^\circ$C]',
        'T_det_{ID}_met':r'$\tilde{T}$ (met at 2 m) [$^\circ$C]',
        'DT_dz_{ID}':r'$\frac{\partial T}{\partial z}$ at the ground [$^\circ$C m$^{-1}$]',
        'hour':'Hour',
        'Hub-height wind speed [m/s]':r'$\overline{u}$ (hub height) [m s$^{-1}$]',
        'Hub-height wind direction [degrees]':r'$\overline{\theta}_w$ (hub height)[$^\circ$]',
        'WS_{ID}_met':r'$\overline{u}$ (met at 3 m) [m s$^{-1}$]',
        'T_abb_{ID}_sum':r'$T$ (ABB) [$^\circ$C]',
        'T_frontend_{ID}_sum':r'$T$ (frontend) [$^\circ$C]',
        'NEN_{ID}_sum':'NEN'}

limits={'T_{ID}_met':[0,50],
        'T_daily_avg_{ID}_met':[0,50],
        'T_det_{ID}_met':[10,10],
        'DT_dz_{ID}':[-0.25,0.25],
        'hour':[0,23],
        'Hub-height wind speed [m/s]':[0,20],
        'Hub-height wind direction [degrees]':[0,360],
        'WS_{ID}_met':[0,10],
        'T_abb_{ID}_sum':[0,50],
        'T_frontend_{ID}_sum':[0,50],
        'NEN_{ID}_sum':[0,1],}

xticks={'T_{ID}_met':[0,10,20,30,40,50],
        'T_daily_avg_{ID}_met':[0,10,20,30,40,50],
        'T_det_{ID}_met':[-10,-5,0,5,10],
        'DT_dz_{ID}':[-0.25,0,0.25],
        'hour':[0,6,12,18,24],
        'Hub-height wind speed [m/s]':np.arange(0,21,5),
        'Hub-height wind direction [degrees]':[0,90,180,270,360],
        'WS_{ID}_met':[0,5,10],
        'T_abb_{ID}_sum':[0,10,20,30,40,50],
        'T_frontend_{ID}_sum':[0,10,20,30,40,50],
        'NEN_{ID}_sum':[0,0.5,1]}

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

_vars=np.array(_vars)[active_features]

#remove high uncertainty
for ID in IDs:
    Data['T_'+str(ID)+'_0.0m'][Data['sigma_T_'+str(ID)+'_0.0m']>max_sigma_T]=np.nan
    Data['sigma_T_'+str(ID)+'_0.0m'][Data['sigma_T_'+str(ID)+'_0.0m']>max_sigma_T]=np.nan
    
    Data['T_'+str(ID)+'_10.0m'][Data['sigma_T_'+str(ID)+'_10.0m']>max_sigma_T]=np.nan
    Data['sigma_T_'+str(ID)+'_10.0m'][Data['sigma_T_'+str(ID)+'_10.0m']>max_sigma_T]=np.nan
    
    Data['sigma_T_'+str(ID)+'_met']=met_uncertainty(Data['T_'+str(ID)+'_met'],Data['WS_'+str(ID)+'_met'],shield_uncertainty)
    Data['T_'+str(ID)+'_met'][Data['sigma_T_'+str(ID)+'_met']>max_sigma_T]=np.nan
    Data['sigma_T_'+str(ID)+'_met'][Data['sigma_T_'+str(ID)+'_met']>max_sigma_T]=np.nan
    
n_features=len(_vars)

#%% Main

#add missing features
Data['hour']=np.array([t.hour+t.minute/60 for t in Data.index])

dt=np.nanmedian(np.diff(Data.index))
assert np.nanmax(np.diff(Data.index))==np.nanmin(np.diff(Data.index))
Data_daily_avg=Data.rolling(window=int(np.timedelta64(1,'D')/dt)).mean()
Data_det=Data-Data_daily_avg

for ID in IDs:
    Data['DT_{ID}'.format(ID=ID)]=Data['T_'+str(ID)+'_0.0m']-Data['T_'+str(ID)+'_met']
    Data['DT_dz_{ID}'.format(ID=ID)]=(Data['T_'+str(ID)+'_10.0m']-Data['T_'+str(ID)+'_0.0m'])/10
    Data['T_daily_avg_{ID}_met'.format(ID=ID)]=Data_daily_avg['T_{ID}_met'.format(ID=ID)]
    Data['T_det_{ID}_met'.format(ID=ID)]=Data_det['T_{ID}_met'.format(ID=ID)]

if calculate_importance:
    importance={}
    importance_std={}

corr={}
for ID in IDs:
    X=[]
    for v in _vars:
        if X==[]:
            X=Data[v.format(ID=ID)].values.reshape(-1,1)
        else:
            X=np.hstack((X,Data[v.format(ID=ID)].values.reshape(-1,1)))
    y= Data['DT_'+str(ID)].values
    corr[ID]=utl.nancorrcoef(X.T)
    if calculate_importance:
        importance[ID],importance_std[ID],*_=utl.RF_feature_selector(X,y)

    
#%% Plots
if calculate_importance:
    plt.figure()
    ctr=0
    for ID in IDs:
        plt.bar(np.arange(n_features)*3-0.5*(ctr-1),importance[ID],yerr=importance_std[ID],color=colors[ID],capsize=5,linewidth=2,width=0.5,label=site_names[ID])
        ctr+=1
    plt.legend()
    plt.xticks(np.arange(n_features)*3,[labels[v] for v in +_vars])
    plt.grid()
    
#correlation matrix
fig=plt.figure(figsize=(18,10))
for ID in IDs:
    ax=plt.subplot(1,len(IDs),np.where(ID==np.array(IDs))[0][0]+1)
    ax=plt.pcolor(corr[ID],cmap='seismic',vmin=-1,vmax=1)
    plt.xticks(np.arange(n_features)+0.5,[labels[v] for v in +_vars],rotation=90)
    plt.yticks(np.arange(n_features)+0.5,[labels[v] for v in +_vars],rotation=0)
    for i in range(corr[ID].shape[0]):
        for j in range(corr[ID].shape[1]):
            plt.text(j + 0.5, i + 0.5, f'{corr[ID][i, j]:.2f}', ha='center', va='center', color='black',fontsize=8)
            
    plt.title('Correlation between features at '+site_names[ID])
    utl.axis_equal()
utl.remove_labels(fig) 
plt.tight_layout()

#1-D plots
fig=plt.figure(figsize=(18,6))
for ID in IDs:
    ctr=1
    for v in _vars:
        plt.subplot(len(IDs),n_features,np.where(ID==np.array(IDs))[0][0]*n_features+ctr)
        utl.simple_bins(Data[v.format(ID=ID)].values,Data['DT_{ID}'.format(ID=ID)].values,bins=25)
        plt.xlabel(labels[v],rotation=20)
        plt.ylabel(r'$\Delta T$ (TROPoe-met) '+'\n'+ 'at '+site_names[ID]+' [$^\circ$C]')
        plt.grid()
        plt.xlim(limits[v])
        plt.ylim([-2,2])
        plt.xticks(xticks[v])
        ctr+=1
utl.remove_labels(fig)
plt.tight_layout()
    
#2D plots
for p in pairs:
    vx=_vars[p[0]]
    vy=_vars[p[1]]
    fig=plt.figure(figsize=(18,6))
    for ID in IDs:
        ax=plt.subplot(1,len(IDs),np.where(ID==np.array(IDs))[0][0]+1)
        utl.simple_bins_2d(Data[vx.format(ID=ID)], Data[vy.format(ID=ID)],Data['DT_'+str(ID)].values)
        plt.xlabel(labels[vx])
        plt.ylabel(labels[vy])
        plt.title(site_names[ID])
        plt.grid()
    utl.remove_labels(fig)
