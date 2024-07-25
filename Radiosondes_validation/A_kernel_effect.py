# -*- coding: utf-8 -*-
"""
A-kernel effect on radiosonde profiles
"""
import os
cd=os.path.dirname(__file__)
import sys
sys.path.append('C:/Users/SLETIZIA/OneDrive - NREL/Desktop/PostDoc/utils')
import utils as utl
import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import cumtrapz
import warnings
import matplotlib
import pandas as pd
import glob 
from scipy.stats import binned_statistic

warnings.filterwarnings('ignore')
plt.close('all')

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm' 
matplotlib.rcParams['font.size'] = 16

#%% Inputs
source_prior='data/prior/Xa_Sa_datafile.sgp.55_levels.month_{month:02d}.cdf'
source_temp='data/All_T.csv'
source_sonde='data/sgpsondewnpnC1.b1/*2024{month:02d}*cdf'
source_tropoe='data/assist-10/nreltropoe_10.c0.20230508.000015.nc'

#dataset
height_met=2#[m]
timezone=-6#[hours]
z0_sonde=0#[m]
IDs=np.array([11,12,10])

#stats
overheating=0.5#[C]

# bias={10:0,
#       11:0,
#       12:0}

bias={10:0,
      11:0.5,
      12:0.7}

bin_hour=np.arange(-1.5,26,3)
bin_hour[0]=0
bin_hour[-1]=24

#graphics
site_names={10:'North',
            11:'South',
            12:'Middle'}
max_height_plot=250#[m]

#%% Initialization

#load tropoe temperature data
Data_temp=pd.read_csv(os.path.join(cd,source_temp))
tnum_temp=np.array([utl.datenum(t,'%Y-%m-%d %H:%M:%S')+timezone*3600 for t in Data_temp['Time'].values])
hour_temp=(tnum_temp-utl.floor(tnum_temp,3600*24))/3600
Data_temp['Time']=np.array([utl.num_to_dt64(utl.datenum(t,'%Y-%m-%d %H:%M:%S'))+np.timedelta64(timezone,'h') for t in Data_temp['Time'].values])
Data_temp=Data_temp.set_index('Time')

#load a sample TROPoe
Data_tropoe=xr.open_dataset(os.path.join(cd,source_tropoe))
height=Data_tropoe.height.values*10**3
A=np.nanmean(Data_tropoe.Akernal.values,axis=0)[:len(height),:len(height)].T


#zeroing
T_sonde=[]
T_sonde_smooth=[]
tnum_sonde=[]
hour_sonde=[]
T_met_eq=[]

#%% Main
for month in range(1,13):
    files_sonde=glob.glob(os.path.join(cd,source_sonde.format(month=month)))
    Prior=xr.open_dataset(os.path.join(cd,source_prior.format(month=month)))
    T_mean_prior=Prior['mean_temperature'].values
    for f in files_sonde:
        Data=xr.open_dataset(f)
        time=Data['time'].values+np.timedelta64(timezone,'h')
        asc=Data['asc'].values
        T=Data.tdry.values
        tnum=np.float64(time)/10**9
        height_sonde=cumtrapz(asc,tnum,initial=z0_sonde)
        
        tnum_sonde=np.append(tnum_sonde,np.nanmean(tnum))
        hour_sonde=np.append(hour_sonde,(np.nanmean(tnum)-utl.floor(np.nanmean(tnum),3600*24))/3600)
        
        T_interp=np.interp(height,height_sonde,T)
        T_sonde=utl.vstack(T_sonde,T_interp)
        T_sonde_smooth=utl.vstack(T_sonde_smooth,np.matmul(A,(T_interp-T_mean_prior))+T_mean_prior)
        DT_overheating=(np.abs(12-hour_sonde[-1])<8)*overheating
        T_met_eq=np.append(T_met_eq,np.interp(height_met,height_sonde,T)+DT_overheating)
        
#daily cycles
T_sonde_avg=np.zeros((len(bin_hour)-1,len(height)))
T_sonde_smooth_avg=np.zeros((len(bin_hour)-1,len(height)))
DT_avg=np.zeros((len(bin_hour)-1,len(height)))
DT_smooth_avg=np.zeros((len(bin_hour)-1,len(height)))

for i_z in range(len(height)):
    T_sonde_avg[:,i_z]=binned_statistic(hour_sonde, T_sonde[:,i_z],statistic=lambda x: utl.filt_mean(x),bins=bin_hour)[0]
    T_sonde_smooth_avg[:,i_z]=binned_statistic(hour_sonde, T_sonde_smooth[:,i_z],statistic=lambda x: utl.filt_mean(x),bins=bin_hour)[0]
    DT_avg[:,i_z]=binned_statistic(hour_sonde, T_sonde[:,i_z]-T_met_eq,statistic=lambda x: utl.filt_mean(x),bins=bin_hour)[0]
    DT_smooth_avg[:,i_z]=binned_statistic(hour_sonde, T_sonde_smooth[:,i_z]-T_met_eq,statistic=lambda x: utl.filt_mean(x),bins=bin_hour)[0]
    

DT_tropoe_avg_all=np.zeros((len(bin_hour)-1,2,len(IDs)))
for ID in IDs:
    DT_tropoe_avg_all[:,0,np.where(ID==IDs)[0][0]]=binned_statistic(hour_temp, Data_temp[f'T_{ID}_0.0m']-bias[ID]-Data_temp[f'T_{ID}_met'],statistic=lambda x: utl.filt_mean(x),bins=bin_hour)[0]
    DT_tropoe_avg_all[:,1,np.where(ID==IDs)[0][0]]=binned_statistic(hour_temp, Data_temp[f'T_{ID}_10.0m']-Data_temp[f'T_{ID}_met'],statistic=lambda x: utl.filt_mean(x),bins=bin_hour)[0]

#%% Plots

#daily cycles of differences
plt.figure(figsize=(18,10))
for ID in IDs:
    plt.subplot(2,len(IDs),np.where(ID==IDs)[0][0]+1)
    plt.plot(utl.mid(bin_hour),DT_avg[:,0],'.-k',markersize=10,label=f'$z={int(height[0])}$ m, RS')
    plt.plot(utl.mid(bin_hour),DT_smooth_avg[:,0],'.--k',markersize=10,label=f'$z={int(height[0])}$ m, RS smoothed')
    plt.plot(utl.mid(bin_hour),DT_tropoe_avg_all[:,0,np.where(ID==IDs)[0][0]],'^-k',markersize=10,label=f'$z={int(height[0])}$ m, TROPoe')
    plt.gca().fill_between(utl.mid(bin_hour),utl.mid(bin_hour)*0,DT_tropoe_avg_all[:,0,np.where(ID==IDs)[0][0]]-DT_avg[:,0],color='r',alpha=0.5)
    
    plt.title(site_names[ID])
    plt.ylim([-1.5,1.5])
    plt.grid()
    if ID==IDs[0]:
        plt.legend()
        plt.ylabel(r'$\Delta T $ (TROPoe - met) [$^\circ$C]')
        
    plt.subplot(2,len(IDs),np.where(ID==IDs)[0][0]+1+len(IDs))
    plt.plot(utl.mid(bin_hour),DT_avg[:,1],'.-b',markersize=10,label=f'$z={int(height[1])}$ m, RS')
    plt.plot(utl.mid(bin_hour),DT_smooth_avg[:,1],'.--b',markersize=10,label=f'$z={int(height[1])}$ m, RS smoothed')
    plt.plot(utl.mid(bin_hour),DT_tropoe_avg_all[:,1,np.where(ID==IDs)[0][0]],'^-b',markersize=10,label=f'$z={int(height[1])}$ m, TROPoe')
    plt.xlabel('Hour (CST)')
    plt.ylim([-1.5,1.5])
    plt.grid()
    if ID==IDs[0]:
        plt.legend()
        plt.ylabel(r'$\Delta T $ (TROPoe - met) [$^\circ$C]')
    
# daily profiles
sel_hour=~np.isnan(T_sonde_avg[:,0])>0
plt.figure(figsize=(18,4))
ctr=1
for i_hour in np.where(sel_hour)[0]:
    plt.subplot(1,np.sum(sel_hour),ctr)
    plt.plot(T_sonde_avg[i_hour,height<max_height_plot].T,height[height<max_height_plot],'k',label='RS')
    plt.plot(T_sonde_smooth_avg[i_hour,height<max_height_plot].T,height[height<max_height_plot],'--k',label='RS smoothed')
    plt.grid()
    hour1=int(bin_hour[i_hour])
    hour2=int(bin_hour[i_hour+1])
    plt.title(f'{hour1:02d}-{hour2:02d}')
    if i_hour>0:
        plt.gca().set_yticklabels([])
    else:
        plt.ylabel(r'$z$ [m.a.g.l.]')
    plt.xlabel(r'$T$ [$^\circ$C]')
    ctr+=1
plt.tight_layout()
    

plt.figure(figsize=(18,4))
plt.hist(hour_sonde,bins=24*6,color='k')
for hour in bin_hour:
    plt.plot([hour,hour],[0,60],'--r')
plt.xlabel('Hour (CST)')
plt.ylabel('Occurrence')
plt.xticks(bin_hour)
plt.grid()
plt.tight_layout()
    