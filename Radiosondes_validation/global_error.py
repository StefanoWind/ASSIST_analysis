# -*- coding: utf-8 -*-
"""
Global error stats
"""
import os
cd=os.path.dirname(__file__)
import sys
sys.path.append('../utils')
import utils as utl
import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import cumtrapz
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import warnings
import matplotlib
import pandas as pd
import glob 
from scipy.stats import binned_statistic

warnings.filterwarnings('ignore')
plt.close('all')

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm' 
matplotlib.rcParams['font.size'] = 14

#%% Inputs
source_trp=os.path.join(cd,'C:/Users/sletizia/OneDrive - NREL/Desktop/Main/ENDURA/ASSIST_analysis/awaken_processing/data/awaken/sg.assist.tropoe.z01.c0/*nc')
source_rsn=os.path.join(cd,'data/awaken/sgpsondewnpnS6.b1/*cdf')
var_sel=['temperature','sigma_temperature','waterVapor','sigma_waterVapor']

#dataset
z0_sonde=0#[m] initial height of radiosondes
unc_rsn=0.15#[C] 1-sigma uncertainty of radiosondes [https://www.arm.gov/publications/tech_reports/handbooks/sonde_handbook.pdf]
sampling_rate=14#[s]
height_assist=1#[m] height ASSIST a.g.l.


#qc
max_gamma=1#maximum gamma
max_rmsa=5#maximum rmsa
min_lwp=5#[g/cm^2] minimum LWP for clouds

#stats
perc_lim=[5,95]
p_value=0.05

#graphics
max_height=2000#[m]
i_height_sel=[0,7,24]

#%% Initialization
files_rsn=glob.glob(source_rsn)
files_trp=glob.glob(source_trp)

height=xr.open_dataset(files_trp[0]).height.values*1000+height_assist
height=height[height<=max_height+100]

#zeroing
T_rsn=[]
r_rsn=[]

T_trp=[]
sigmaT_trp=[]
r_trp=[]
sigmar_trp=[]

time=np.array([],dtype='datetime64')


#%% Main
for f in files_rsn:
    #load radiosondes
    Data_rsn=xr.open_dataset(f)
    
    #extract profiles
    time_rsn=Data_rsn['time'].values
    asc=Data_rsn['asc'].values
    
    dt_rsn=np.median(np.diff(time_rsn))/np.timedelta64(1,'s')
    Data_rsn['temperature']=Data_rsn['tdry'].where(Data_rsn['qc_tdry']==0).rolling(time=int(sampling_rate/dt_rsn),center=True).mean()
    
    #calculate mixing ratio
    e_s=6.112*np.exp(17.67*Data_rsn['temperature']/(Data_rsn['temperature']+243.5))*100
    e=Data_rsn.rh/100*e_s
    r=e*0.622/(Data_rsn.pres*100-e)*1000
    
    Data_rsn['waterVapor']=r.where(Data_rsn.qc_rh==0).where(Data_rsn.qc_pres==0).rolling(time=int(sampling_rate/dt_rsn),center=True).mean()
    
    tnum_rsn=np.float64(time_rsn)/10**9
    height_rsn=cumtrapz(asc,tnum_rsn,initial=z0_sonde)
    
    #time matching
    date1=utl.datestr(np.nanmin(tnum_rsn[height_rsn<=max_height]),'%Y%m%d')
    date2=utl.datestr(np.nanmin(tnum_rsn[height_rsn<=max_height])+24*3600,'%Y%m%d')
    match=result = [s for s in files_trp if date1 in s or date2 in s]
        
    if len(match)>0:
        #load tropoe
        Data_trp=xr.open_mfdataset(match)
        
        #qc
        Data_trp['cbh'][(Data_trp['lwp']<min_lwp).compute()]=Data_trp['height'].max()#remove clouds with low lwp
     
        qc_gamma=Data_trp['gamma']<=max_gamma
        qc_rmsa=Data_trp['rmsa']<=max_rmsa
        qc_cbh=Data_trp['height']<Data_trp['cbh']
        qc=qc_gamma*qc_rmsa*qc_cbh
        Data_trp['qc']=~qc+0
    
        #interpolation
        Data_trp=Data_trp.assign_coords(height=Data_trp.height*1000+height_assist)
        Data_trp_sel=Data_trp[var_sel].interp(time=('points', time_rsn[height_rsn<=max_height]),
                                        height=('points', height_rsn[height_rsn<=max_height])).compute()
        
        tnum_avg=np.nanmean(tnum_rsn[height_rsn<=max_height])
        time_avg=np.datetime64('1970-01-01 00:00:00')+tnum_avg*np.timedelta64(1,'s')
        
        #stack data
        if np.sum(~np.isnan(Data_trp_sel.temperature)>0):
            T_trp=utl.vstack(T_trp,np.interp(height,Data_trp_sel.height,Data_trp_sel['temperature'].values))
            sigmaT_trp=utl.vstack(sigmaT_trp,np.interp(height,Data_trp_sel.height,Data_trp_sel['sigma_temperature'].values))
            
            r_trp=utl.vstack(r_trp,np.interp(height,Data_trp_sel.height,Data_trp_sel['waterVapor'].values))
            sigmar_trp=utl.vstack(sigmar_trp,np.interp(height,Data_trp_sel.height,Data_trp_sel['sigma_waterVapor'].values))
            
            T_rsn=utl.vstack(T_rsn,np.interp(height,height_rsn,Data_rsn['temperature'].values))
            r_rsn=utl.vstack(r_rsn,np.interp(height,height_rsn,Data_rsn['waterVapor'].values))
            
            time=np.append(time,time_avg)
    print(f'{f} done, {np.round(np.sum(qc==0).values/qc.size*100,1)} % rejected')

Diff=xr.Dataset()
Diff['DT']=xr.DataArray(T_trp-T_rsn,coords={'time':time,'height':height})
Diff['sigmaDT']=xr.DataArray(sigmaT_trp,coords={'time':time,'height':height})
Diff['Dr']=xr.DataArray(r_trp-r_rsn,coords={'time':time,'height':height})
Diff['sigmaDr']=xr.DataArray(sigmar_trp,coords={'time':time,'height':height})

#mean error
bias_avg=xr.apply_ufunc(utl.filt_stat,Diff['DT'],
                        input_core_dims=[["time"]],  
                        kwargs={"func": np.nanmean, 'perc_lim': perc_lim},
                        vectorize=True)

bias_low=xr.apply_ufunc(utl.filt_BS_stat,Diff['DT'],
                    kwargs={"func": np.nanmean,'p_value':p_value*100/2,'perc_lim': perc_lim},
                    input_core_dims=[["time"]],  
                    vectorize=True)

bias_top=xr.apply_ufunc(utl.filt_BS_stat,Diff['DT'],
                    kwargs={"func": np.nanmean,'p_value':(1-p_value/2)*100,'perc_lim': perc_lim},
                    input_core_dims=[["time"]],  
                    vectorize=True)

#error stdev
rmsd_avg=xr.apply_ufunc(utl.filt_stat,Diff['DT']**2,
                    input_core_dims=[["time"]],  
                    kwargs={"func": np.nanmean, 'perc_lim': [0,100]},
                    vectorize=True)**0.5

rmsd_low=xr.apply_ufunc(utl.filt_BS_stat,Diff['DT']**2,
                    kwargs={"func": np.nanmean,'p_value':p_value*100/2,'perc_lim': [0,100]},
                    input_core_dims=[["time"]],  
                    vectorize=True)**0.5

rmsd_top=xr.apply_ufunc(utl.filt_BS_stat,Diff['DT']**2,
                    kwargs={"func": np.nanmean,'p_value':(1-p_value/2)*100,'perc_lim': [0,100]},
                    input_core_dims=[["time"]],  
                    vectorize=True)**0.5

rmsd_th=(xr.apply_ufunc(utl.filt_stat,Diff['sigmaDT'],
                    input_core_dims=[["time"]],  
                    kwargs={"func": np.nanmean, 'perc_lim': perc_lim},
                    vectorize=True)**2+unc_rsn**2)**0.5



#mean error
bias_r_avg=xr.apply_ufunc(utl.filt_stat,Diff['Dr'],
                        input_core_dims=[["time"]],  
                        kwargs={"func": np.nanmean, 'perc_lim': perc_lim},
                        vectorize=True)

bias_r_low=xr.apply_ufunc(utl.filt_BS_stat,Diff['Dr'],
                    kwargs={"func": np.nanmean,'p_value':p_value*100/2,'perc_lim': perc_lim},
                    input_core_dims=[["time"]],  
                    vectorize=True)

bias_r_top=xr.apply_ufunc(utl.filt_BS_stat,Diff['Dr'],
                    kwargs={"func": np.nanmean,'p_value':(1-p_value/2)*100,'perc_lim': perc_lim},
                    input_core_dims=[["time"]],  
                    vectorize=True)

#error stdev
rmsd_r_avg=xr.apply_ufunc(utl.filt_stat,Diff['Dr']**2,
                    input_core_dims=[["time"]],  
                    kwargs={"func": np.nanmean, 'perc_lim': [0,100]},
                    vectorize=True)**0.5

rmsd_r_low=xr.apply_ufunc(utl.filt_BS_stat,Diff['Dr']**2,
                    kwargs={"func": np.nanmean,'p_value':p_value*100/2,'perc_lim': [0,100]},
                    input_core_dims=[["time"]],  
                    vectorize=True)**0.5

rmsd_r_top=xr.apply_ufunc(utl.filt_BS_stat,Diff['Dr']**2,
                    kwargs={"func": np.nanmean,'p_value':(1-p_value/2)*100,'perc_lim': [0,100]},
                    input_core_dims=[["time"]],  
                    vectorize=True)**0.5

rmsd_r_th=(xr.apply_ufunc(utl.filt_stat,Diff['sigmaDr'],
                    input_core_dims=[["time"]],  
                    kwargs={"func": np.nanmean, 'perc_lim': perc_lim},
                    vectorize=True)**2+unc_rsn**2)**0.5

#%% Plots
index=np.arange(len(time))

plt.close('all')
fig=plt.figure(figsize=(18,8))
gs = gridspec.GridSpec(2, 5,height_ratios=[1,10],width_ratios=[2,2,5,5,5])
ax=fig.add_subplot(gs[1,0])
plt.plot(bias_avg,height,'-k',label='Data')
plt.fill_betweenx(height, bias_low,bias_top,color='k',alpha=0.25)
plt.plot(height*0,height,'-r',label='Theory')
plt.xlim([-0.35,0.35])
plt.ylim([0,max_height])
plt.grid()
plt.xlabel(r'Bias of $\Delta T$ [$^\circ$C]')
plt.ylabel(r'$z$ [m]')
plt.legend(draggable=True)

ax=fig.add_subplot(gs[1,1])
plt.plot(rmsd_avg,height,'k')
plt.fill_betweenx(height, rmsd_low,rmsd_top,color='k',alpha=0.25)
plt.plot(rmsd_th,height,'r')
plt.ylim([0,max_height])
ax.set_yticklabels([])
plt.grid()
plt.xlabel(r'RMS of $\Delta T$ [$^\circ$C]')

ax=fig.add_subplot(gs[1,2])
cf=plt.contourf(index,height,T_rsn.T,np.arange(10,37,0.5),cmap='hot',extend='both')
plt.contour(index,height,T_rsn.T,np.arange(10,37,0.5),colors='k',alpha=0.25,linewidths=1)
plt.xlabel('Sonde launch #')
ax.set_yticklabels([])
plt.ylim([0,max_height])

cax=fig.add_subplot(gs[0,2])
cb=plt.colorbar(cf,cax,orientation='horizontal',ticks=np.arange(10,36,5))
cb.set_label(label='$T$ (sondes) [$^\circ$C]', labelpad=-100)

ax=fig.add_subplot(gs[1,3])
cf=plt.contourf(index,height,T_trp.T,np.arange(10,37,0.5),cmap='hot',extend='both')
plt.contour(index,height,T_trp.T,np.arange(10,37,0.5),colors='k',alpha=0.25,linewidths=1)
plt.xlabel('Sonde launch #')
ax.set_yticklabels([])
plt.ylim([0,max_height])

cax=fig.add_subplot(gs[0,3])
cb=plt.colorbar(cf,cax,orientation='horizontal',ticks=np.arange(10,36,5))
cb.set_label(label=r'$T$ (TROPoe) [$^\circ$C]', labelpad=-100)

ax=fig.add_subplot(gs[1,4])
cf=plt.contourf(index,height,T_trp.T-T_rsn.T,np.arange(-2,2.1,0.25),cmap='seismic',extend='both')
plt.contour(index,height,T_trp.T-T_rsn.T,np.arange(-2,2.1,0.25),colors='k',alpha=0.25,linewidths=1)
plt.xlabel('Sonde launch #')
ax.set_yticklabels([])
plt.ylim([0,max_height])

cax=fig.add_subplot(gs[0,4])
cb=plt.colorbar(cf,cax,orientation='horizontal',ticks=np.arange(-2,2.1,1))
cb.set_label(label='$\Delta T$ (TROPoe-sondes) [$^\circ$C]', labelpad=-100)



fig=plt.figure(figsize=(18,8))
gs = gridspec.GridSpec(2, 5,height_ratios=[1,10],width_ratios=[2,2,5,5,5])
ax=fig.add_subplot(gs[1,0])
plt.plot(bias_r_avg,height,'-k',label='Data')
plt.fill_betweenx(height, bias_r_low,bias_r_top,color='k',alpha=0.25)
plt.plot(height*0,height,'-r',label='Theory')
plt.xlim([-0.35,0.35])
plt.ylim([0,max_height])
plt.grid()
plt.xlabel(r'Bias of $\Delta r$ [g kg$^{-1}$]')
plt.ylabel(r'$z$ [m]')
plt.legend(draggable=True)

ax=fig.add_subplot(gs[1,1])
plt.plot(rmsd_avg,height,'k')
plt.fill_betweenx(height, rmsd_low,rmsd_top,color='k',alpha=0.25)
plt.plot(rmsd_th,height,'r')
plt.ylim([0,max_height])
ax.set_yticklabels([])
plt.grid()
plt.xlabel(r'RMS of $\Delta r$ [g kg$^{-1}$]')

ax=fig.add_subplot(gs[1,2])
cf=plt.contourf(index,height,T_rsn.T,np.arange(0,10,0.5),cmap='Blues',extend='both')
plt.contour(index,height,T_rsn.T,np.arange(0,10,0.5),colors='k',alpha=0.25,linewidths=1)
plt.xlabel('Sonde launch #')
ax.set_yticklabels([])
plt.ylim([0,max_height])

cax=fig.add_subplot(gs[0,2])
cb=plt.colorbar(cf,cax,orientation='horizontal',ticks=np.arange(10,36,5))
cb.set_label(label='$r$ (sondes) [g kg$^{-1}$]', labelpad=-100)

ax=fig.add_subplot(gs[1,3])
cf=plt.contourf(index,height,T_trp.T,np.arange(0,10,0.5),cmap='Blues',extend='both')
plt.contour(index,height,T_trp.T,np.arange(0,10,0.5),colors='k',alpha=0.25,linewidths=1)
plt.xlabel('Sonde launch #')
ax.set_yticklabels([])
plt.ylim([0,max_height])

cax=fig.add_subplot(gs[0,3])
cb=plt.colorbar(cf,cax,orientation='horizontal',ticks=np.arange(0,11))
cb.set_label(label=r'$r$ (TROPoe) [g kg$^{-1}$]', labelpad=-100)

ax=fig.add_subplot(gs[1,4])
cf=plt.contourf(index,height,T_trp.T-T_rsn.T,np.arange(-2,2.1,0.25),cmap='seismic',extend='both')
plt.contour(index,height,T_trp.T-T_rsn.T,np.arange(-2,2.1,0.25),colors='k',alpha=0.25,linewidths=1)
plt.xlabel('Sonde launch #')
ax.set_yticklabels([])
plt.ylim([0,max_height])

cax=fig.add_subplot(gs[0,4])
cb=plt.colorbar(cf,cax,orientation='horizontal',ticks=np.arange(-2,2.1,1))
cb.set_label(label='$\Delta r$ (TROPoe-sondes) [g kg$^{-1}$]', labelpad=-100)
