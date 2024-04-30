# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 11:37:54 2024

@author: sletizia
"""
import os
cd=os.path.dirname(__file__)
import sys
sys.path.append('C:/Users/SLETIZIA/OneDrive - NREL/Desktop/PostDoc/Custom_functions')
import utils as utl

import numpy as np
from matplotlib import pyplot as plt
import warnings
import matplotlib
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings('ignore')
plt.close('all')

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm' 
matplotlib.rcParams['font.size'] = 14

#%% Inputs
source='data/All_T.csv'
IDs=[11,12,10] #IDs of the ASSISTs

p_value=0.05 #p-vakue ofr confidence interval
WS_cutin=3#[m/s] cutin wind speed (KP+AF)
WS_rated=12#[m/s] rated wind speed (KP+AF)
WS_cutout=25#[m/s] cutout wind speed (KP+AF)
max_sigma_T=5#[K] maximum uncertainty
timezone=-6#[hours] difference local time - UTC

# graphics
skip=5
ID_comb=[[11,10],[11,12],[12,10]]
N_days_plot=7
site_names={10:'North',
            11:'South',
            12:'Middle'}

#%% Functions
def met_uncertainty(T,WS):
    unc_T1=np.zeros(len(T))
    unc_T2=np.zeros(len(T))
    
    unc_T1=0.005*np.abs(T-20)+0.2
    
    ws=np.array([0,1,2,3,6,100])
    unc_ws=np.array([1.51,1.51,0.7,0.4,0.2,0.2])
    unc_T2=np.interp(WS,ws,unc_ws)
    
    return unc_T1+unc_T2
    
#%% Initialization
Data=pd.read_csv(os.path.join(cd,source))
Data['Time']=np.array([utl.num_to_dt64(utl.datenum(t,'%Y-%m-%d %H:%M:%S')+timezone*3600) for t in Data['Time'].values])
Data=Data.set_index('Time')

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


#%% Plots
plt.close('all')

# #all temperature time series
time_bins=np.arange(Data.index[0],Data.index[-1],np.timedelta64(N_days_plot, 'D'))

for t1,t2 in zip(time_bins[:-1],time_bins[1:]):
    sel=(Data.index>t1)*(Data.index<=t2)
    fig=plt.figure(figsize=(18,10))
    gs = fig.add_gridspec(4, 1, width_ratios=[1], height_ratios=[0.75, 2,2,2], wspace=0.5, hspace=0.25,bottom=0.05, top=0.95)
    
    ax = fig.add_subplot(gs[0, 0])
    plt.barbs(Data.index.values[sel*regionI][::skip],  np.zeros(len(U.values[sel*regionI][::skip])),  U.values[sel*regionI][::skip]*1.94,  V.values[sel*regionI][::skip]*1.94,  length=6,alpha=0.25)
    plt.barbs(Data.index.values[sel*regionII][::skip], np.zeros(len(U.values[sel*regionII][::skip])), U.values[sel*regionII][::skip]*1.94, V.values[sel*regionII][::skip]*1.94, length=6,alpha=1)
    plt.barbs(Data.index.values[sel*regionIII][::skip],np.zeros(len(U.values[sel*regionIII][::skip])),U.values[sel*regionIII][::skip]*1.94,V.values[sel*regionIII][::skip]*1.94,length=6,alpha=0.5)
    ax.set_xticklabels([])
    plt.yticks([])
    plt.ylim([-1.5,1.5])
    ctr=1
    for ID in IDs:
        ax = fig.add_subplot(gs[ctr, 0])   
        plt.plot(Data['T_'+str(ID)+'_met'][sel],'k',label='Met at 2 m')
        plt.plot(Data['T_'+str(ID)+'_0.0m'][sel],'b',label='TROPoe at 0 m')
        plt.fill_between(Data.index[sel], Data['T_'+str(ID)+'_0.0m'].values[sel]-(-norm.ppf(p_value/2))*Data['sigma_T_'+str(ID)+'_0.0m'].values[sel],
                                          Data['T_'+str(ID)+'_0.0m'].values[sel]+(-norm.ppf(p_value/2))*Data['sigma_T_'+str(ID)+'_0.0m'].values[sel],
                                          color='b',alpha=0.25)
        plt.plot(Data['T_'+str(ID)+'_10.0m'][sel],'r',label='TROPoe at 10 m')
        plt.fill_between(Data.index[sel], Data['T_'+str(ID)+'_10.0m'].values[sel]-(-norm.ppf(p_value/2))*Data['sigma_T_'+str(ID)+'_10.0m'].values[sel],
                                          Data['T_'+str(ID)+'_10.0m'].values[sel]+(-norm.ppf(p_value/2))*Data['sigma_T_'+str(ID)+'_10.0m'].values[sel],
                                          color='r',alpha=0.25)
      
        plt.ylabel(r'$T$ [$^\circ$C]')
        
        plt.title(site_names[ID])
        plt.grid()
        plt.xlim([t1,t2])
        if ctr<3:
            ax.set_xticklabels([])
        ctr+=1
    plt.legend()
    plt.xlabel('Time (CST)')
    
    if not os.path.exists(os.path.join(cd,'figures','Met_comparison_T')):
        os.mkdir(os.path.join(cd,'figures','Met_comparison_T'))
    plt.savefig(os.path.join(cd,'figures','Met_comparison_T',utl.datestr(utl.dt64_to_num(t1),'%Y%m%d')+'-'+utl.datestr(utl.dt64_to_num(t2),'%Y%m%d')+'_T_met_comparison.png'))
    plt.close()


for t1,t2 in zip(time_bins[:-1],time_bins[1:]):
    sel=(Data.index>t1)*(Data.index<=t2)
    fig=plt.figure(figsize=(18,10))
    gs = fig.add_gridspec(4, 1, width_ratios=[1], height_ratios=[0.75, 2,2,2], wspace=0.5, hspace=0.25,bottom=0.05, top=0.95)
    
    ax = fig.add_subplot(gs[0, 0])
    plt.barbs(Data.index.values[sel*regionI][::skip],  np.zeros(len(U.values[sel*regionI][::skip])),  U.values[sel*regionI][::skip]*1.94,  V.values[sel*regionI][::skip]*1.94,  length=6,alpha=0.25)
    plt.barbs(Data.index.values[sel*regionII][::skip], np.zeros(len(U.values[sel*regionII][::skip])), U.values[sel*regionII][::skip]*1.94, V.values[sel*regionII][::skip]*1.94, length=6,alpha=1)
    plt.barbs(Data.index.values[sel*regionIII][::skip],np.zeros(len(U.values[sel*regionIII][::skip])),U.values[sel*regionIII][::skip]*1.94,V.values[sel*regionIII][::skip]*1.94,length=6,alpha=0.5)
    ax.set_xticklabels([])
    plt.yticks([])
    plt.ylim([-1.5,1.5])
    ctr=1

    for IDc in ID_comb:
        
        ID1=IDc[0]
        ID2=IDc[1]
        
        ax = fig.add_subplot(gs[ctr, 0]) 
        
        DT=Data['T_'+str(ID2)+'_met'][sel]-Data['T_'+str(ID1)+'_met'][sel]
        plt.plot(DT,'k',label='Met at 2 m')
        
        DT=Data['T_'+str(ID2)+'_0.0m'][sel]-Data['T_'+str(ID1)+'_0.0m'][sel]
        plt.plot(DT,'b',label='TROPoe at 0 m')
        sigma_T_diff=(Data['sigma_T_'+str(ID1)+'_0.0m'].values[sel]**2+Data['sigma_T_'+str(ID2)+'_0.0m'].values[sel]**2)**0.5
        plt.fill_between(Data.index[sel], DT.values-(-norm.ppf(p_value/2))*sigma_T_diff,
                                          DT.values+(-norm.ppf(p_value/2))*sigma_T_diff,
                                          color='b',alpha=0.25)
        
        DT=Data['T_'+str(ID2)+'_10.0m'][sel]-Data['T_'+str(ID1)+'_10.0m'][sel]
        plt.plot(DT,'r',label='TROPoe at 10 m')
        sigma_T_diff=(Data['sigma_T_'+str(ID1)+'_10.0m'].values[sel]**2+Data['sigma_T_'+str(ID2)+'_10.0m'].values[sel]**2)**0.5
        plt.fill_between(Data.index[sel], DT.values-(-norm.ppf(p_value/2))*sigma_T_diff,
                                          DT.values+(-norm.ppf(p_value/2))*sigma_T_diff,
                                          color='r',alpha=0.25)

        plt.ylabel(r'$\Delta T$ [$^\circ$C]')
        
        plt.title(site_names[ID2]+' - '+site_names[ID1])
        plt.grid()
        plt.xlim([t1,t2])
        plt.ylim([-5,5])
        if ctr<3:
            ax.set_xticklabels([])
        ctr+=1
    plt.legend()
    plt.xlabel('Time (CST)')
    
    if not os.path.exists(os.path.join(cd,'figures','Met_comparison_DeltaT')):
        os.mkdir(os.path.join(cd,'figures','Met_comparison_DeltaT'))
    plt.savefig(os.path.join(cd,'figures','Met_comparison_DeltaT',utl.datestr(utl.dt64_to_num(t1),'%Y%m%d')+'-'+utl.datestr(utl.dt64_to_num(t2),'%Y%m%d')+'_DeltaT_met_comparison.png'))
    plt.close()
    
#linear fits of temperature
plt.figure(figsize=(18,5))
ctr=1
for ID in IDs:
    plt.subplot(1,3,ctr)
    utl.plot_lin_fit(Data['T_'+str(ID)+'_met'], Data['T_'+str(ID)+'_0.0m'],0,50, '$^\circ$C')
    plt.xlabel(r'$T$ (met at 2 m) [$^\circ$C]')
    plt.ylabel(r'$T$  (TROPoe at 0 m) [$^\circ$C]')
    plt.title(site_names[ID])
    ctr+=1

#linear fit of temperature differences
plt.figure(figsize=(18,5))
ctr=1
for IDc in ID_comb:
    ID1=IDc[0]
    ID2=IDc[1]
    plt.subplot(1,3,ctr)
    utl.plot_lin_fit(Data['T_'+str(ID2)+'_met']- Data['T_'+str(ID1)+'_met'], Data['T_'+str(ID2)+'_0.0m']-Data['T_'+str(ID1)+'_0.0m'],-5,5, '$^\circ$C')
    plt.xlabel(r'$\Delta T$ (met at 2 m) [$^\circ$C]')
    plt.ylabel(r'$\Delta T$ (TROPoe at 0 m) [$^\circ$C]')
    plt.title(site_names[ID2]+' - '+site_names[ID1])
    ctr+=1
