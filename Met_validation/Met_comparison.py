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
from scipy import stats
from scipy.stats import norm
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore')
plt.close('all')

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm' 
matplotlib.rcParams['font.size'] = 14

#%% Inputs
source='data/All_T.csv'
IDs=[11,12,10]

ws_range=[5,12]#[m/s]
wd_range=[150,210]#[deg]
p_value=0.05
WS_cutin=4#[m/s]
WS_rated=12#[m/s]
WS_cutout=25#[m/s]
dWD=15
dhour=4
max_sigma_T=5#[K]
timezone=-6#[hours]

# graphics
skip=5
ID_comb=[[11,10],[11,12],[12,10]]
N_days_plot=7
site_names={10:'North',
            11:'South',
            12:'Middle'}

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
regionI=Data['Hub-height wind speed [m/s]']<WS_cutin
regionII=(Data['Hub-height wind speed [m/s]']>=WS_cutin)*(Data['Hub-height wind speed [m/s]']<WS_rated)
regionIII=(Data['Hub-height wind speed [m/s]']>=WS_rated)*(Data['Hub-height wind speed [m/s]']<WS_cutout)

#bin-averaged error
bin_WD=np.arange(0,361,dWD)
hour=np.array([t.hour+t.minute/60 for t in Data.index])
bin_WS=[0,WS_cutin,WS_rated,WS_cutout]
bin_hour=np.arange(-0.01,23.99,dhour)
samples=[hour,Data['Hub-height wind direction [degrees]'].values,Data['Hub-height wind speed [m/s]'].values]
bins=[bin_hour,bin_WD,bin_WS]
err_T_avg={}
for ID in IDs:
    err_T=Data['T_'+str(ID)+'_0.0m']- Data['T_'+str(ID)+'_met']
    err_T_avg[ID]=stats.binned_statistic_dd(samples, err_T,statistic=lambda x:np.nanmean(x),bins=bins)[0]

#daily cycles
dt=np.mean(np.diff(Data.index))
assert np.max(np.diff(Data.index))-np.min(np.diff(Data.index))==0, "Time is not equally spaced"
Data_daily_avg=Data.rolling(window=int(np.timedelta64(1,'D')/dt)).mean()
Data_daily_std=Data.rolling(window=int(np.timedelta64(1,'D')/dt)).std()

#detrended daily temperature
Data_det=Data-Data_daily_avg
Data_det['bin_hour']=pd.cut(hour,bins=np.arange(-0.001,23.99))
Data_det_avg=Data_det.groupby('bin_hour').mean()

err_T_daily_avg={}
for ID in IDs:
    err_T=Data['T_'+str(ID)+'_0.0m']- Data['T_'+str(ID)+'_met']
    err_T_daily_avg[ID]=stats.binned_statistic(hour, err_T,statistic=lambda x:np.nanmean(x),bins=np.arange(-0.001,23.99))[0]
    
#%% Plots
plt.close('all')

#all temperature time series
time_bins=np.arange(Data.index[0],Data.index[-1],np.timedelta64(N_days_plot, 'D'))

for t1,t2 in zip(time_bins[:-1],time_bins[1:]):
    sel=(Data.index>t1)*(Data.index<=t2)
    fig=plt.figure(figsize=(18,10))
    gs = fig.add_gridspec(4, 1, width_ratios=[1], height_ratios=[0.75, 2,2,2], wspace=0.5, hspace=0.25,bottom=0.05, top=0.95)
    
    ax = fig.add_subplot(gs[0, 0])
    plt.barbs(Data.index.values[sel*regionI][::skip],np.zeros(len(U.values[sel*regionI][::skip])),U.values[sel*regionI][::skip]*1.94,V.values[sel*regionI][::skip]*1.94,length=6,alpha=0.25)
    plt.barbs(Data.index.values[sel*regionII][::skip],np.zeros(len(U.values[sel*regionII][::skip])),U.values[sel*regionII][::skip]*1.94,V.values[sel*regionII][::skip]*1.94,length=6,alpha=1)
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
    
    plt.quiver(Data.index.values[::skip],np.zeros(len(U[::skip]))+3.75,U[::skip],V[::skip])
    
    plt.xlabel('Time (UTC)')
    plt.ylabel(r'$\Delta T$ [$^\circ$C]')
    plt.legend()
    plt.title('ASSIST-'+str(ID2)+' - '+'ASSIST-'+str(ID1))
    plt.grid()
    # plt.xlim([np.datetime64('2023-07-27T00:00:00'),np.datetime64('2023-08-03T00:00:00')])
    # plt.ylim([-4.5,4.5])
    

#linear fits of temperature
plt.figure(figsize=(18,5))
ctr=1
for ID in IDs:
    plt.subplot(1,3,ctr)
    utl.plot_lin_fit(Data['T_'+str(ID)+'_met'], Data['T_'+str(ID)+'_0.0m'],0,50, '$^\circ$C')
    plt.xlabel(r'$T$ (met) [$^\circ$C]')
    plt.ylabel(r'$T$ (ASSIST-'+str(ID)+') [$^\circ$C]')
    ctr+=1

#linead fit of temperature differences
plt.figure(figsize=(18,5))
ctr=1
for IDc in ID_comb:
    ID1=IDc[0]
    ID2=IDc[1]
    plt.subplot(1,3,ctr)
    utl.plot_lin_fit(Data['T_'+str(ID2)+'_met']- Data['T_'+str(ID1)+'_met'], Data['T_'+str(ID2)+'_0.0m']-Data['T_'+str(ID1)+'_0.0m'],-5,5, '$^\circ$C')
    plt.xlabel(r'$T$ (met at 2 m) [$^\circ$C]')
    plt.ylabel(r'$\Delta T$ (TROPoe at 0 m) [$^\circ$C]')
    plt.title('ASSIST-'+str(ID2)+' - '+'ASSIST-'+str(ID1))
    ctr+=1

#daily cycles




# error statistics
WD,H=np.meshgrid(bin_WD,np.arange(len(bin_hour)))
for ID in IDs:
    plt.figure(figsize=(18,8))
    for i_WS in range(len(bin_WS)-1):
        plt.subplot(1,len(bin_WS)-1,i_WS+1)
        plt.pcolor(H*utl.cosd(90-WD),H*utl.sind(90-WD),err_T_avg[ID][:,:,i_WS],vmin=-1,vmax=1,cmap='seismic')
        utl.axis_equal()
        plt.title(r'$U_\infty \in ('+str(bin_WS[i_WS])+','+str(bin_WS[i_WS+1])+r']$ m s$^{-1}$')
        plt.grid()
        plt.colorbar(label=r'$\Delta T$ (ASSIST-'+str(ID)+r' - met) [$^\circ$C]')
        

# plt.figure(figsize=(18,8))
# ctr=1
# for ID in IDs:
#     plt.subplot(1,3,ctr)
#     err_T=Data['T_'+str(ID)+'_0.0m']- Data['T_'+str(ID)+'_met']
#     DT_dz=(Data['T_'+str(ID)+'_10.0m']-Data['T_'+str(ID)+'_0.0m'])/10
#     plt.plot()
#     plt.plot(DT_dz,err_T,'.k',alpha=0.1)
#     plt.xlim([-0.3,0.3])
#     plt.ylim([-2,2])
#     ctr+=1


#detrended daily cycle
for ID in IDs:
    plt.figure(figsize=(18,8))
    plt.plot(np.arange(23),Data_det_avg['T_'+str(ID)+'_met'],'k',label='Met at 2 m')
    plt.plot(np.arange(23),Data_det_avg['T_'+str(ID)+'_0.0m'],'b',label='TROPoe at 0 m')
    # plt.fill_between(Data.index, Data['T_'+str(ID)+'_0.0m'].values-(-norm.ppf(p_value/2))*Data['sigma_T_'+str(ID)+'_0.0m'].values,
    #                               Data['T_'+str(ID)+'_0.0m'].values+(-norm.ppf(p_value/2))*Data['sigma_T_'+str(ID)+'_0.0m'].values,
    #                               color='b',alpha=0.25)
    plt.plot(np.arange(23),Data_det_avg['T_'+str(ID)+'_10.0m'],'r',label='TROPoe at 10 m')
    # plt.fill_between(Data.index, Data['T_'+str(ID)+'_10.0m'].values-(-norm.ppf(p_value/2))*Data['sigma_T_'+str(ID)+'_10.0m'].values,
    #                              Data['T_'+str(ID)+'_10.0m'].values+(-norm.ppf(p_value/2))*Data['sigma_T_'+str(ID)+'_10.0m'].values,
    #                              color='r',alpha=0.25)
# 
    plt.ylim([-7,7])
    plt.xlabel('Hour (UTC)')
    plt.ylabel(r'$T^\prime$ [$^\circ$C]')
    plt.legend()
    plt.title('ASSIST-'+str(ID))
    

#error daily cycle
plt.figure(figsize=(18,8))
for ID in IDs:
    plt.plot(np.arange(23),err_T_daily_avg[ID],label='ASSIST-'+str(ID))
    plt.xlabel('Hour (UTC)')
    plt.ylabel(r'$\Delta T$ (ASSIST - met) [$^\circ$C]')
    plt.legend()
    plt.title('ASSIST-'+str(ID))
    
    