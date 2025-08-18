# -*- coding: utf-8 -*-
"""
Global TROPoe vs. sondes error stats
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
import warnings
import matplotlib
import glob 
warnings.filterwarnings('ignore')
plt.close('all')

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm' 
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['savefig.dpi']=500

#%% Inputs
source_trp=os.path.join(cd,'C:/Users/sletizia/OneDrive - NREL/Desktop/Main/ENDURA/ASSIST_analysis/awaken_processing/data/awaken/sg.assist.tropoe.z01.c0/*nc')
source_rsn=os.path.join(cd,'data/awaken/sgpsondewnpnS6.b1/*cdf')
var_sel=['temperature','sigma_temperature','sigma_temperature_n','waterVapor','sigma_waterVapor','sigma_waterVapor_n']

#dataset
z0_sonde=0#[m] initial height of radiosondes
unc_rsn=0.15#[C] 1-sigma uncertainty of radiosondes [https://www.arm.gov/publications/tech_reports/handbooks/sonde_handbook.pdf]
unc_rsn_r=0
sampling_rate=14#[s] sampling rate of ASSIST
height_assist=1#[m] height ASSIST a.g.l.
H=90#[m] hub height
D=127#[m] rotor diamter

#qc
max_gamma=1#maximum gamma
max_rmsa=5#maximum rmsa
min_lwp=5#[g/cm^2] minimum LWP for clouds

#stats
perc_lim=[5,95] #percentile limits
p_value=0.05# p-value for c.i.

#graphics
max_height=2000#[m]

#%% Function
def rolling_mean(x,window):
    '''
    Rolling mean with tail treatment
    '''
    x_ma=x+np.nan
    for i in range(len(x)):
        i1=i-int(window/2)
        i2=i+int(window/2)
        if i1<0:
            i1=0
        if i2>len(x)-1:
            i2=len(x)-1

        x_ma[i]=np.nanmean(x[i1:i2])
        
    return x_ma
        

#%% Initialization
files_rsn=glob.glob(source_rsn)
files_trp=glob.glob(source_trp)

#defne common height
height_trp=xr.open_dataset(files_trp[0]).height.values*1000+height_assist
height=height_trp[height_trp<max_height+100]

#zeroing
T_rsn=[]
r_rsn=[]
T_rsn_smt=[]
r_rsn_smt=[]
ws_rsn=[]
wd_rsn=[]

T_trp=[]
sigmaT_trp=[]
sigmaTn_trp=[]

r_trp=[]
sigmar_trp=[]
sigmarn_trp=[]

time=np.array([],dtype='datetime64')

#%% Main
for f in files_rsn:
    #load radiosondes
    Data_rsn=xr.open_dataset(f)
    
    #calculate mixing ratio
    e_s=6.112*np.exp(17.67*Data_rsn.tdry/(Data_rsn.tdry+243.5))*100
    e=Data_rsn.rh/100*e_s
    r=0.622*e/(Data_rsn.pres*100-e)*1000
    
    #resampling at ASSIST rate
    time_rsn=Data_rsn['time'].values
    dt_rsn=np.median(np.diff(time_rsn))/np.timedelta64(1,'s')
    
    T_rsn_res=rolling_mean(Data_rsn.tdry.where(Data_rsn.qc_tdry==0).values,int(sampling_rate/dt_rsn))
    Data_rsn['temperature']=xr.DataArray(T_rsn_res,coords={'time':time_rsn})
    
    r_rsn_res=rolling_mean(r.where(Data_rsn.qc_rh==0).where(Data_rsn.qc_pres==0).where(Data_rsn.qc_tdry==0).values,int(sampling_rate/dt_rsn))
    Data_rsn['waterVapor']=xr.DataArray(r_rsn_res,coords={'time':time_rsn})
    
    #define coords
    tnum_rsn=np.float64(time_rsn)/10**9
    asc=Data_rsn['asc'].values
    height_rsn=cumtrapz(asc,tnum_rsn,initial=z0_sonde)
    
    #time matching
    date1=utl.datestr(np.nanmin(tnum_rsn[height_rsn<=max_height]),'%Y%m%d')
    date2=utl.datestr(np.nanmin(tnum_rsn[height_rsn<=max_height])+24*3600,'%Y%m%d')
    match=result = [s for s in files_trp if date1 in s or date2 in s]
        
    if len(match)>0:
        #load tropoe
        Data_trp=xr.open_mfdataset(match)
        
        #qc
        Data_trp['cbh'][(Data_trp['lwp']<min_lwp).compute()]=Data_trp['height'].max()+1000#remove clouds with low lwp
     
        qc_gamma=Data_trp['gamma']<=max_gamma
        qc_rmsa=Data_trp['rmsa']<=max_rmsa
        qc_cbh=Data_trp['height']<Data_trp['cbh']
        qc=qc_gamma*qc_rmsa*qc_cbh
        Data_trp['qc']=~qc+0
        
        Dat_trp=Data_trp.where(Data_trp.qc==0)
        
        #timestamp
        tnum_avg=np.nanmean(tnum_rsn[height_rsn<=max_height])
        time_avg=np.datetime64('1970-01-01 00:00:00')+tnum_avg*np.timedelta64(1,'s')
        
        #calculate noise covariance
        I=np.eye(len(Data_trp.arb_dim1))
        Ss=np.zeros((len(Data_trp.time),len(Data_trp.arb_dim1),len(Data_trp.arb_dim1)))
        if 'time' in Data_trp.Sa.dims:
            Sa=Data_trp.Sa.interp(time=time_avg).values
        else:
            Sa=Data_trp.Sa.values
            
        for it in range(len(Data_trp.time)):
            A=Data_trp['Akernal'].isel(time=it).values.T
            Ss[it,:,:]=(A-I)@Sa@(A-I).T
       
        Sn=Data_trp.Sop.values-Ss
        
        #extract diagonal terms
        Nz=len(Data_trp.height)
        Data_trp['sigma_temperature_n']=xr.DataArray(data=Sn[:,np.arange(Nz),np.arange(Nz)]**0.5,coords={'time':Data_trp.time,'height':Data_trp.height})
        Data_trp['sigma_waterVapor_n']= xr.DataArray(data=Sn[:,np.arange(Nz,2*Nz),np.arange(Nz,2*Nz)]**0.5,coords={'time':Data_trp.time,'height':Data_trp.height})

        #interpolation in sondes time
        Data_trp=Data_trp.assign_coords(height=Data_trp.height*1000+height_assist)
        Data_trp_sel=Data_trp[var_sel].interp(time=('points', time_rsn[height_rsn<=max_height]),
                                        height=('points', height_rsn[height_rsn<=max_height])).compute()
        
        #interpolation in TROPoe heights
        T_trp_int=       np.interp(height,Data_trp_sel.height,Data_trp_sel['temperature'].values)
        sigma_T_trp_int= np.interp(height,Data_trp_sel.height,Data_trp_sel['sigma_temperature'].values)
        sigma_Tn_trp_int=np.interp(height,Data_trp_sel.height,Data_trp_sel['sigma_temperature_n'].values)
        
        r_trp_int=       np.interp(height,Data_trp_sel.height,Data_trp_sel['waterVapor'].values)
        sigma_r_trp_int= np.interp(height,Data_trp_sel.height,Data_trp_sel['sigma_waterVapor'].values)
        sigma_rn_trp_int=np.interp(height,Data_trp_sel.height,Data_trp_sel['sigma_waterVapor_n'].values)
        
        T_rsn_int=       np.interp(height_trp,height_rsn,Data_rsn['temperature'].values)
        r_rsn_int=       np.interp(height_trp,height_rsn,Data_rsn['waterVapor'].values)
        
        #wind
        ws_rsn_int= np.interp(height,height_rsn,Data_rsn['wspd'].where(Data_rsn['qc_wspd']==0).values)
        
        c= np.cos(np.radians(np.interp(height,height_rsn,Data_rsn['deg'].where(Data_rsn['qc_deg']==0).values)))
        s= np.sin(np.radians(np.interp(height,height_rsn,Data_rsn['deg'].where(Data_rsn['qc_deg']==0).values)))
        wd_rsn_int=np.degrees(np.arctan2(s,c))%360
        
        #smoothing sondes
        A=Data_trp.Akernal.interp(time=time_avg).values.T
        if 'time' in Data_trp.Xa.dims:
            xa=Data_trp.Xa.interp(time=time_avg).values
        else:
            xa=Data_trp.Xa.values
        x_hat=Data_trp.Xop.interp(time=time_avg).values
        x=np.concatenate([T_rsn_int, r_rsn_int,x_hat[len(height_trp)*2:]])
        x_smt=xa+A@(x-xa)
        
        T_rsn_smt_int=x_smt[:Nz]
        r_rsn_smt_int=x_smt[Nz:2*Nz]

        #stack data
        if np.sum(~np.isnan(Data_trp_sel.temperature)>0):
            T_trp=      utl.vstack(T_trp,      T_trp_int)
            sigmaT_trp= utl.vstack(sigmaT_trp, sigma_T_trp_int)
            sigmaTn_trp=utl.vstack(sigmaTn_trp,sigma_Tn_trp_int)
            
            r_trp=      utl.vstack(r_trp,      r_trp_int)
            sigmar_trp= utl.vstack(sigmar_trp, sigma_r_trp_int)
            sigmarn_trp=utl.vstack(sigmarn_trp,sigma_rn_trp_int)
            
            T_rsn=      utl.vstack(T_rsn,T_rsn_int[height_trp<max_height+100])
            r_rsn=      utl.vstack(r_rsn,r_rsn_int[height_trp<max_height+100])
            
            T_rsn_smt=  utl.vstack(T_rsn_smt,T_rsn_smt_int[height_trp<max_height+100])
            r_rsn_smt=  utl.vstack(r_rsn_smt,r_rsn_smt_int[height_trp<max_height+100])
            
            ws_rsn=     utl.vstack(ws_rsn,ws_rsn_int)
            wd_rsn=     utl.vstack(wd_rsn,wd_rsn_int)
            
            time=np.append(time,time_avg)
        else:
            print(f'No valid TROPoe data in {f}')
            
        print(f'{f} done, {np.round(np.sum(qc==0).values/qc.size*100,1)} % rejected')
    
    else:
        print(f'No TROPoe data mathing {f}')
Diff=xr.Dataset()
Diff['DT']=xr.DataArray(T_trp-T_rsn,coords={'time':time,'height':height})
Diff['DT_smt']=xr.DataArray(T_trp-T_rsn_smt,coords={'time':time,'height':height})
Diff['sigmaDT']=xr.DataArray(sigmaT_trp,coords={'time':time,'height':height})
Diff['sigmaDT_n']=xr.DataArray(sigmaTn_trp,coords={'time':time,'height':height})
Diff['Dr']=xr.DataArray(r_trp-r_rsn,coords={'time':time,'height':height})
Diff['Dr_smt']=xr.DataArray(r_trp-r_rsn_smt,coords={'time':time,'height':height})
Diff['sigmaDr']=xr.DataArray(sigmar_trp,coords={'time':time,'height':height})
Diff['sigmaDr_n']=xr.DataArray(sigmar_trp,coords={'time':time,'height':height})

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


#mean error (smoothed)
bias_smt_avg=xr.apply_ufunc(utl.filt_stat,Diff['DT_smt'],
                        input_core_dims=[["time"]],  
                        kwargs={"func": np.nanmean, 'perc_lim': perc_lim},
                        vectorize=True)

#RMSD
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

#RMSD (smoothed)
rmsd_smt_avg=xr.apply_ufunc(utl.filt_stat,Diff['DT_smt']**2,
                    input_core_dims=[["time"]],  
                    kwargs={"func": np.nanmean, 'perc_lim': [0,100]},
                    vectorize=True)**0.5

rmsd_smt_th=(xr.apply_ufunc(utl.filt_stat,Diff['sigmaDT_n'],
                    input_core_dims=[["time"]],  
                    kwargs={"func": np.nanmean, 'perc_lim': perc_lim},
                    vectorize=True)**2+unc_rsn**2)**0.5


#mean error mix ratio
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

#RMSD mix ratio
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
                    vectorize=True)**2+unc_rsn_r**2)**0.5


#mean error mix ratio (smoothed)
bias_r_smt_avg=xr.apply_ufunc(utl.filt_stat,Diff['Dr_smt'],
                        input_core_dims=[["time"]],  
                        kwargs={"func": np.nanmean, 'perc_lim': perc_lim},
                        vectorize=True)

#RMSD mix ratio (smoothed)
rmsd_r_smt_avg=xr.apply_ufunc(utl.filt_stat,Diff['Dr_smt']**2,
                    input_core_dims=[["time"]],  
                    kwargs={"func": np.nanmean, 'perc_lim': [0,100]},
                    vectorize=True)**0.5

rmsd_r_smt_th=(xr.apply_ufunc(utl.filt_stat,Diff['sigmaDr_n'],
                    input_core_dims=[["time"]],  
                    kwargs={"func": np.nanmean, 'perc_lim': perc_lim},
                    vectorize=True)**2+unc_rsn_r**2)**0.5

#%% Plots
index=np.arange(len(time))

plt.close('all')

#temperature
fig=plt.figure(figsize=(18,8))
gs = gridspec.GridSpec(2, 5,height_ratios=[1,10],width_ratios=[2,2,5,5,5])
ax=fig.add_subplot(gs[1,0])
plt.plot(bias_avg,height,'-k',label='Data')
plt.plot(bias_smt_avg,height,'--k',label='Data (smoothed)')
plt.fill_betweenx(height, bias_low,bias_top,color='k',alpha=0.25)
plt.plot(height*0,height,'-r',label='Theory')
plt.plot(height*0,height,'--r',label='Theory (smoothed)')
plt.xlim([-0.5,0.5])
plt.ylim([0,max_height])
plt.grid()
plt.xlabel(r'Bias of $\Delta T$ [$^\circ$C]')
plt.ylabel(r'$z$ [m]')
plt.legend(draggable=True)

ax=fig.add_subplot(gs[1,1])
plt.plot(rmsd_avg,height,'k')
plt.plot(rmsd_smt_avg,height,'--k')
plt.fill_betweenx(height, rmsd_low,rmsd_top,color='k',alpha=0.25)
plt.plot(rmsd_th,height,'r')
plt.plot(rmsd_smt_th,height,'--r')
plt.xlim([0,2])
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


#mix ratio
fig=plt.figure(figsize=(18,8))
gs = gridspec.GridSpec(2, 5,height_ratios=[1,10],width_ratios=[2,2,5,5,5])
ax=fig.add_subplot(gs[1,0])
plt.plot(bias_r_avg,height,'-k',label='Data')
plt.plot(bias_r_smt_avg,height,'--k',label='Data (smoothed)')
plt.fill_betweenx(height, bias_r_low,bias_r_top,color='k',alpha=0.25)
plt.plot(height*0,height,'-r',label='Theory')
plt.plot(height*0,height,'--r',label='Theory (smoothed)')
plt.xlim([-1,1])
plt.ylim([0,max_height])
plt.grid()
plt.xlabel(r'Bias of $\Delta r$ [g kg$^{-1}$]')
plt.ylabel(r'$z$ [m]')
plt.legend(draggable=True)

ax=fig.add_subplot(gs[1,1])
plt.plot(rmsd_r_avg,height,'k')
plt.plot(rmsd_r_smt_avg,height,'--k')
plt.fill_betweenx(height, rmsd_r_low,rmsd_r_top,color='k',alpha=0.25)
plt.plot(rmsd_r_th,height,'r')
plt.plot(rmsd_r_smt_th,height,'--r')
plt.xlim([0,3])
plt.ylim([0,max_height])
ax.set_yticklabels([])
plt.grid()
plt.xlabel(r'RMS of $\Delta r$ [g kg$^{-1}$]')

ax=fig.add_subplot(gs[1,2])
cf=plt.contourf(index,height,T_rsn.T,np.arange(0,31),cmap='Blues',extend='both')
plt.contour(index,height,T_rsn.T,np.arange(0,31),colors='k',alpha=0.25,linewidths=1)
plt.xlabel('Sonde launch #')
ax.set_yticklabels([])
plt.ylim([0,max_height])

cax=fig.add_subplot(gs[0,2])
cb=plt.colorbar(cf,cax,orientation='horizontal',ticks=np.arange(0,31,5))
cb.set_label(label='$r$ (sondes) [g kg$^{-1}$]', labelpad=-100)

ax=fig.add_subplot(gs[1,3])
cf=plt.contourf(index,height,T_trp.T,np.arange(0,31),cmap='Blues',extend='both')
plt.contour(index,height,T_trp.T,np.arange(0,31),colors='k',alpha=0.25,linewidths=1)
plt.xlabel('Sonde launch #')
ax.set_yticklabels([])
plt.ylim([0,max_height])

cax=fig.add_subplot(gs[0,3])
cb=plt.colorbar(cf,cax,orientation='horizontal',ticks=np.arange(0,31,5))
cb.set_label(label=r'$r$ (TROPoe) [g kg$^{-1}$]', labelpad=-100)

ax=fig.add_subplot(gs[1,4])
cf=plt.contourf(index,height,r_trp.T-r_rsn.T,np.arange(-2,2.1,0.25),cmap='seismic',extend='both')
plt.contour(index,height,r_trp.T-r_rsn.T,np.arange(-2,2.1,0.25),colors='k',alpha=0.25,linewidths=1)
plt.xlabel('Sonde launch #')
ax.set_yticklabels([])
plt.ylim([0,max_height])

cax=fig.add_subplot(gs[0,4])
cb=plt.colorbar(cf,cax,orientation='horizontal',ticks=np.arange(-2,2.1,1))
cb.set_label(label='$\Delta r$ (TROPoe-sondes) [g kg$^{-1}$]', labelpad=-100)

#directional effects
matplotlib.rcParams['font.size'] = 12
wd=np.radians(np.nanmean(wd_rsn[:,(height>H-D/2)*(height<H+D/2)],axis=1))
ws=np.nanmean(ws_rsn[:,(height>H-D/2)*(height<H+D/2)],axis=1)
DT=Diff['DT'].where(Diff.height<H).mean(dim='height').values
fig, ax= plt.subplots(1, 1, constrained_layout=True,subplot_kw={'projection': 'polar'})
plt.plot(np.radians(np.arange(360)),np.radians(np.arange(360))*0+4,'--k')
plt.plot(np.radians(np.arange(360)),np.radians(np.arange(360))*0+11,'--k')
sc=plt.scatter(wd,ws,s=np.abs(DT)*50,c=DT,cmap='seismic',alpha=0.75,vmin=-1,vmax=1)
plt.grid()
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
ax.set_xticks([0,np.pi/2,np.pi,np.pi/2*3], labels=['N','E','S','W'])
plt.colorbar(sc,label='$\Delta T$ (TROPoe-sondes) [$^\circ$C]')
plt.grid()
ax.set_facecolor((0,0,0,0.1))
