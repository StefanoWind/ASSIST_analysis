# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 11:37:54 2024

@author: sletizia
"""

import os
cd=os.path.dirname(__file__)
import sys
sys.path.append('C:/Users/SLETIZIA/OneDrive - NREL/Desktop/PostDoc/Custom_functions')
import myFunctions as SL

import numpy as np
from matplotlib import pyplot as plt
import warnings
import matplotlib
import pandas as pd
from scipy import stats
from scipy.stats import norm

warnings.filterwarnings('ignore')
plt.close('all')

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm' 
matplotlib.rcParams['font.size'] = 14

#%% Inputs
source='data/All_T.csv'
IDs=[10,11,12]

ws_range=[5,12]#[m/s]
wd_range=[150,210]#[deg]
p_value=0.05
bin_WS=np.array([0,4,15])
dWD=15
dhour=4

# graphics
skip=5
ID_comb=[[11,10],[11,12],[12,10]]

#%% Initialization
Data=pd.read_csv(source)
Data['Time']=np.array([SL.num_to_dt64(SL.datenum(t,'%Y-%m-%d %H:%M:%S')) for t in Data['Time'].values])
Data=Data.set_index('Time')


#%% Main
U=(SL.cosd(270-Data['Hub-height wind direction [degrees]'])*Data['Hub-height wind speed [m/s]']).values
V=(SL.sind(270-Data['Hub-height wind direction [degrees]'])*Data['Hub-height wind speed [m/s]']).values

regionII=(Data['Hub-height wind speed [m/s]'].values>ws_range[0])*(Data['Hub-height wind speed [m/s]'].values<ws_range[1])
south=(Data['Hub-height wind direction [degrees]'].values>wd_range[0])*(Data['Hub-height wind direction [degrees]'].values<wd_range[1])



bin_WD=np.arange(0,361,dWD)
hour=np.array([t.hour for t in Data.index])
bin_hour=np.arange(-0.5,23.6,dhour)
samples=[hour,Data['Hub-height wind direction [degrees]'].values,Data['Hub-height wind speed [m/s]'].values]
bins=[bin_hour,bin_WD,bin_WS]
err_T_avg={}
for ID in IDs:
    err_T=Data['T_'+str(ID)+'_0.0m']- Data['T_'+str(ID)+'_met']
    err_T_avg[ID]=stats.binned_statistic_dd(samples, err_T,statistic=lambda x:np.nanmean(x),bins=bins)[0]
    
DT_avg={}
DT_met_avg={}
for IDc in ID_comb:
    ID1=IDc[0]
    ID2=IDc[1]
    DT=Data['T_'+str(ID2)+'_0.0m']- Data['T_'+str(ID1)+'_0.0m']
    DT_avg[str(ID2)+'-'+str(ID1)]=stats.binned_statistic_dd(samples, DT,statistic=lambda x:np.nanmean(x),bins=bins)[0]    
    
    DT_met=Data['T_'+str(ID2)+'_met']- Data['T_'+str(ID1)+'_met']
    DT_met_avg[str(ID2)+'-'+str(ID1)]=stats.binned_statistic_dd(samples, DT_met,statistic=lambda x:np.nanmean(x),bins=bins)[0]    
    
    

#%% Plots
plt.close('all')


#selected time series
for ID in IDs:
    plt.figure(figsize=(18,8))
    plt.plot(Data['T_'+str(ID)+'_met'],'k',label='Met at 2 m')
    plt.plot(Data['T_'+str(ID)+'_0.0m'],'b',label='TROPoe at 0 m')
    plt.fill_between(Data.index, Data['T_'+str(ID)+'_0.0m'].values-(-norm.ppf(p_value/2))*Data['sigma_T_'+str(ID)+'_0.0m'].values,
                                 Data['T_'+str(ID)+'_0.0m'].values+(-norm.ppf(p_value/2))*Data['sigma_T_'+str(ID)+'_0.0m'].values,
                                 color='b',alpha=0.25)
    plt.plot(Data['T_'+str(ID)+'_10.0m'],'r',label='TROPoe at 10 m')
    plt.fill_between(Data.index, Data['T_'+str(ID)+'_10.0m'].values-(-norm.ppf(p_value/2))*Data['sigma_T_'+str(ID)+'_10.0m'].values,
                                 Data['T_'+str(ID)+'_10.0m'].values+(-norm.ppf(p_value/2))*Data['sigma_T_'+str(ID)+'_10.0m'].values,
                                 color='r',alpha=0.25)
    plt.plot(Data['T_675_'+str(ID)+'_sum'],'g',label=r'$T_b$ at $\tilde{\nu}=675$ cm$^{-1}$')
    plt.quiver(Data.index.values[regionII][::skip],np.zeros(len(U[regionII][::skip]))+45,U[regionII][::skip],V[regionII][::skip])
    
    plt.xlabel('Time (UTC)')
    plt.ylabel(r'$T$ [$^\circ$C]')
    plt.legend()
    plt.title('ASSIST-'+str(ID))
    plt.grid()
    plt.xlim([np.datetime64('2023-07-27T00:00:00'),np.datetime64('2023-08-03T00:00:00')])
    plt.ylim([20,48])
    

for IDc in ID_comb:
   
    ID1=IDc[0]
    ID2=IDc[1]
    
    plt.figure(figsize=(18,5))
    plt.plot(Data['T_'+str(ID2)+'_met']-    Data['T_'+str(ID1)+'_met'],'k',label='Met at 2 m')
    
    DT=Data['T_'+str(ID2)+'_0.0m']-   Data['T_'+str(ID1)+'_0.0m']
    plt.plot(DT,'b',label='TROPoe at 0 m')
    sigma_T_diff=(Data['sigma_T_'+str(ID1)+'_0.0m'].values**2+Data['sigma_T_'+str(ID2)+'_0.0m'].values**2)**0.5
    plt.fill_between(Data.index, DT.values-(-norm.ppf(p_value/2))*sigma_T_diff,
                                 DT.values+(-norm.ppf(p_value/2))*sigma_T_diff,
                                 color='b',alpha=0.25)
    
    DT=Data['T_'+str(ID2)+'_10.0m']-   Data['T_'+str(ID1)+'_10.0m']
    plt.plot(DT,'r',label='TROPoe at 10 m')
    sigma_T_diff=(Data['sigma_T_'+str(ID1)+'_10.0m'].values**2+Data['sigma_T_'+str(ID2)+'_10.0m'].values**2)**0.5
    plt.fill_between(Data.index, DT.values-(-norm.ppf(p_value/2))*sigma_T_diff,
                                 DT.values+(-norm.ppf(p_value/2))*sigma_T_diff,
                                 color='r',alpha=0.25)
    
    plt.plot(Data['T_675_'+str(ID2)+'_sum']-Data['T_675_'+str(ID1)+'_sum'],'g',label=r'$T_b$ at $\tilde{\nu}=675$ cm$^{-1}$')
    plt.quiver(Data.index.values[regionII][::skip],np.zeros(len(U[regionII][::skip]))+3.75,U[regionII][::skip],V[regionII][::skip])
    
    plt.xlabel('Time (UTC)')
    plt.ylabel(r'$\Delta T$ [$^\circ$C]')
    plt.legend()
    plt.title('ASSIST-'+str(ID2)+' - '+'ASSIST-'+str(ID1))
    plt.grid()
    plt.xlim([np.datetime64('2023-07-27T00:00:00'),np.datetime64('2023-08-03T00:00:00')])
    plt.ylim([-4.5,4.5])
    

#linear fits
plt.figure(figsize=(18,5))
ctr=1
for ID in IDs:
    plt.subplot(1,3,ctr)
    SL.plot_linfit_v3(Data['T_'+str(ID)+'_met'], Data['T_'+str(ID)+'_0.0m'])
    plt.xlim([0,50])
    plt.ylim([0,50])
    plt.xlabel(r'$T$ (met) [$^\circ$C]')
    plt.ylabel(r'$T$ (ASSIST-'+str(ID)+') [$^\circ$C]')
    ctr+=1

plt.figure(figsize=(18,5))
ctr=1
for IDc in ID_comb:
    ID1=IDc[0]
    ID2=IDc[1]
    
    plt.subplot(1,3,ctr)
    SL.plot_linfit_v3(Data['T_'+str(ID2)+'_met']-    Data['T_'+str(ID1)+'_met'], Data['T_'+str(ID2)+'_0.0m']-   Data['T_'+str(ID1)+'_0.0m'])
    plt.xlim([-5,5])
    plt.ylim([-5,5])
    plt.xticks(np.arange(-5,6))
    plt.yticks(np.arange(-5,6))
    plt.xlabel(r'$T$ (met at 2 m) [$^\circ$C]')
    plt.ylabel(r'$\Delta T$ (TROPoe at 0 m) [$^\circ$C]')
    plt.title('ASSIST-'+str(ID2)+' - '+'ASSIST-'+str(ID1))
    ctr+=1
    
    
WD,H=np.meshgrid(bin_WD,np.arange(len(bin_hour)))
for ID in IDs:
    plt.figure(figsize=(18,8))
    for i_WS in range(len(bin_WS)-1):
        plt.subplot(1,len(bin_WS)-1,i_WS+1)
        plt.pcolor(H*SL.cosd(90-WD),H*SL.sind(90-WD),err_T_avg[ID][:,:,i_WS],vmin=-1,vmax=1,cmap='seismic')
        SL.axis_equal()
        plt.title(r'$U_\infty \in ('+str(bin_WS[i_WS])+','+str(bin_WS[i_WS+1])+r']$ [m s$^{-1}$]')
        plt.grid()
        plt.colorbar(label=r'$\Delta T$ (ASSIST-'+str(ID)+r' - met) [$^\circ$C]')
        
        
for IDc in ID_comb:
    ID1=IDc[0]
    ID2=IDc[1]
    plt.figure(figsize=(18,8))
    for i_WS in range(len(bin_WS)-1):
        plt.subplot(1,len(bin_WS)-1,i_WS+1)
        plt.pcolor(H*SL.cosd(90-WD),H*SL.sind(90-WD),DT_avg[str(ID2)+'-'+str(ID1)][:,:,i_WS],vmin=-1,vmax=1,cmap='seismic')
        SL.axis_equal()
        plt.title(r'$U_\infty \in ('+str(bin_WS[i_WS])+','+str(bin_WS[i_WS+1])+r']$ [m s$^{-1}$]')
        plt.grid()
        plt.colorbar(label=r'$\Delta T$ (ASSIST-'+str(ID2)+' - '+'ASSIST-'+str(ID1)+') [$^\circ$C]')
        
        
for IDc in ID_comb:
    ID1=IDc[0]
    ID2=IDc[1]
    plt.figure(figsize=(18,8))
    for i_WS in range(len(bin_WS)-1):
        plt.subplot(1,len(bin_WS)-1,i_WS+1)
        plt.pcolor(H*SL.cosd(90-WD),H*SL.sind(90-WD),DT_met_avg[str(ID2)+'-'+str(ID1)][:,:,i_WS],vmin=-1,vmax=1,cmap='seismic')
        SL.axis_equal()
        plt.title(r'$U_\infty \in ('+str(bin_WS[i_WS])+','+str(bin_WS[i_WS+1])+r']$ [m s$^{-1}$]')
        plt.grid()
        plt.colorbar(label=r'$\Delta T$ (met-'+str(ID2)+' - '+'met-'+str(ID1)+') [$^\circ$C]')