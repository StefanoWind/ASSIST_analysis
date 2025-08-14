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
var_sel=['temperature','sigma_temperature','vres_temperature']

#dataset
z0_sonde=0#[m] initial height of radiosondes
unc_rsn=0.15#[C] 1-sigma uncertainty of radiosondes [https://www.arm.gov/publications/tech_reports/handbooks/sonde_handbook.pdf]
window=1#smoothing window
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
T_std_rsn=[]
T_trp=[]
sigmaT_trp=[]
time=np.array([],dtype='datetime64')


#%% Main
for f in files_rsn:
    #load radiosondes
    Data_rsn=xr.open_dataset(f)
    
    #extract profiles
    time_rsn=Data_rsn['time'].values
    asc=Data_rsn['asc'].values
    Data_rsn['temperature']=Data_rsn['tdry'].where(Data_rsn['qc_tdry']==0).rolling(time=window,center=True).mean().values
    
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
        
        # print(f'File {f}')
        # print(f'{np.round(np.sum(qc_gamma).values/qc_gamma.size*100,1)}% retained after gamma filter', flush=True)
        # print(f'{np.round(np.sum(qc_rmsa).values/qc_rmsa.size*100,1)}% retained after rmsa filter', flush=True)
        # print(f'{np.round(np.sum(qc_cbh).values/qc_cbh.size*100,1)}% retained after cbh filter', flush=True)
    
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
            
            T_rsn=utl.vstack(T_rsn,np.interp(height,height_rsn,Data_rsn['temperature'].values))
            time=np.append(time,time_avg)
    print(f'{f} done, {np.round(np.sum(qc==0).values/qc.size*100,1)} rejected')

unc=(np.nanmean(sigmaT_trp,axis=0)**2+unc_rsn**2)**0.5


Diff=xr.Dataset()
Diff['DT']=xr.DataArray(T_trp-T_rsn,coords={'time':time,'height':height})
Diff['sigmaDT']=xr.DataArray(sigmaT_trp,coords={'time':time,'height':height})

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

rmsd_th=xr.apply_ufunc(utl.filt_stat,Diff['sigmaDT'],
                    input_core_dims=[["time"]],  
                    kwargs={"func": np.nanmean, 'perc_lim': perc_lim},
                    vectorize=True)

#%% Plots
index=np.arange(len(time))

plt.close('all')
fig=plt.figure(figsize=(18,8))
gs = gridspec.GridSpec(2, 5,height_ratios=[1,10],width_ratios=[2,2,5,5,5])
ax=fig.add_subplot(gs[1,0])
plt.plot(bias_avg,height,'-k',label='Data')
plt.fill_betweenx(height, bias_low,bias_top,color='k',alpha=0.25)
plt.plot(height*0,height,'-r',label='Theory')
plt.xlim([-0.3,0.3])
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
cb=plt.colorbar(cf,cax,orientation='horizontal',ticks=np.arange(15,36,5))
cb.set_label(label='$T$ (sondes) [$^\circ$C]', labelpad=-100)

ax=fig.add_subplot(gs[1,3])
cf=plt.contourf(index,height,T_trp.T,np.arange(10,37,0.5),cmap='hot',extend='both')
plt.contour(index,height,T_trp.T,np.arange(10,37,0.5),colors='k',alpha=0.25,linewidths=1)
plt.xlabel('Sonde launch #')
ax.set_yticklabels([])
plt.ylim([0,max_height])

cax=fig.add_subplot(gs[0,3])
cb=plt.colorbar(cf,cax,orientation='horizontal',ticks=np.arange(15,36,5))
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


# plt.figure(figsize=(18,8))
# ctr=1
# for i_h in i_height_sel:
#     plt.subplot(len(i_height_sel),1,ctr)
#     plt.plot(index,T_rsn[:,i_h],'-k',label='Sondes')
#     plt.plot(index,T_trp[:,i_h],'-r',label='TROPoe')
#     plt.plot(index[hour<sunrise],T_rsn[hour<sunrise,i_h],'*k',markersize=10,alpha=0.75)
#     plt.plot(index[hour>=sunrise],T_rsn[hour>=sunrise,i_h],'.k',markersize=15,alpha=0.75)
#     plt.plot(index[hour<sunrise],T_trp[hour<sunrise,i_h],'*r',markersize=10,alpha=0.75)
#     plt.plot(index[hour>=sunrise],T_trp[hour>=sunrise,i_h],'.r',markersize=15,alpha=0.75)
#     RMSD=np.mean((T_trp[:,i_h]-T_rsn[:,i_h])**2)**0.5
#     plt.text(1,35,s=r'$z='+str(int(height[i_h]))+'$ m: RMSD$='+str(np.round(RMSD,1))+'^\circ$C',
#              bbox={'alpha':0.25,'color':'g','edgecolor':'k'})
#     ctr+=1
#     plt.xlim([0,len(index)])
#     plt.ylim([12,40])
#     plt.grid()
#     plt.ylabel(r'$T$ [$^\circ$C]')
#     if ctr<len(i_height_sel):
#         ax.set_yticklabels([])
    
# plt.xlabel('Launch number')
# plt.legend()
