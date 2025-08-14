# -*- coding: utf-8 -*-
"""
Compare tropoe retrievals
"""
import os
cd=os.path.dirname(__file__)
import sys
sys.path.append('../utils')
import utils as utl
import numpy as np
import xarray as xr
import matplotlib
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt
import yaml
import statsmodels.api as sm
from scipy.stats import norm
import glob
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 18
matplotlib.rcParams['savefig.dpi']=500

#%% Inputs
source_config=os.path.join(cd,'configs','config.yaml')
units=['ASSIST11','ASSIST12']
sources={'ASSIST10':'data/awaken/nwtc.assist.tropoe.z01.c1/*nc',
         'ASSIST11':'data/awaken/nwtc.assist.tropoe.z02.c1/*nc',
         'ASSIST12':'data/awaken/nwtc.assist.tropoe.z03.c1/*nc'}

#stats
max_height=2#[km] maximum height where data is selcted
p_value=0.05#for CI
perc_lim=[5,95]#percentile limits

#graphics
height_sel=[1000,100,10]#[m]

#%% Initialization
#config
with open(source_config, 'r') as fid:
    config = yaml.safe_load(fid)
    
#importsimport utils as utl

#load data
if not os.path.isfile(os.path.join(cd,'data',f'DT{units[1]}-{units[0]}.nc')):
    Data={}
    for u in units:
        files=glob.glob(os.path.join(cd,sources[u]))
        Data[u]=xr.open_mfdataset(files).sel(height=slice(0,max_height*1.2))
        
        #qc data
        Data[u]['cbh'][(Data[u]['lwp']<config['min_lwp']).compute()]=Data[u]['height'].max()*2#remove clouds with low lwp
        qc_gamma=Data[u]['gamma']<=config['max_gamma']
        qc_rmsa=Data[u]['rmsa']<=config['max_rmsa']
        qc_cbh=Data[u]['height']<Data[u]['cbh']
        qc=qc_gamma*qc_rmsa*qc_cbh
        Data[u]['qc']=~qc+0
            
        print(f'{u}: {np.round(np.sum(qc_gamma).values/qc_gamma.size*100,1)}% retained after gamma filter')
        print(f'{u}: {np.round(np.sum(qc_rmsa).values/qc_rmsa.size*100,1)}% retained after rmsa filter')
        print(f'{u}: {np.round(np.sum(qc_cbh).values/qc_cbh.size*100,1)}% retained after cbh filter')
        
    print('Computing temperature difference')
    DT=(Data[units[1]].temperature.where(Data[units[1]].qc==0)-\
        Data[units[0]].temperature.where(Data[units[0]].qc==0)).compute()
        
    print('Computing uncertainty on temperature difference')
    sigmaDT=((Data[units[1]].sigma_temperature_n.where(Data[units[1]].qc==0)**2+\
              Data[units[0]].sigma_temperature_n.where(Data[units[0]].qc==0)**2)**0.5).compute()
    
    Diff=xr.Dataset()
    Diff['DT']=DT
    Diff['sigmaDT']=sigmaDT
    Diff.to_netcdf(os.path.join(cd,'data',f'DT{units[1]}-{units[0]}.nc'))
else:
    Diff=xr.open_dataset(os.path.join(cd,'data',f'DT{units[1]}-{units[0]}.nc'))

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

#probability of exceedence
nexc=xr.apply_ufunc(np.nansum,np.abs(Diff['DT'])>Diff['sigmaDT']*norm.ppf(1-p_value/2,loc=0,scale=1),
                    input_core_dims=[["time"]],  
                    vectorize=True)

nall=xr.apply_ufunc(np.nansum,~np.isnan(Diff['DT']+Diff['sigmaDT']),
                    input_core_dims=[["time"]],  
                    vectorize=True)

#extract coordinates 
height=Diff.height.values*1000+config['height_assist']

#interpolated data for ACF
time_int=np.arange(Diff.time.values[0],Diff.time.values[-1],np.nanmedian(np.diff(Diff.time)))
DT_interp=Diff['DT'].interp(time=time_int)

#%% Plots
plt.close('all')

matplotlib.rcParams['font.size'] = 16

#profiles 
fig=plt.figure(figsize=(18,12))
gs = gridspec.GridSpec(1, 3, width_ratios=[1,1,1])
ax=fig.add_subplot(gs[0,0])
plt.plot(bias_avg,height,'k',label='Data',linewidth=2)
plt.plot(height*0,height,'r',label='Theory',linewidth=2)
plt.fill_betweenx(height, bias_low,bias_top,color='k',alpha=0.25)
for h in height_sel:
    plt.plot([-0.025,0.025],[h,h],'--b')
plt.xlim([-0.025,0.025])
plt.ylim([0,max_height*1000])
plt.xlabel(r'Bias of $\Delta \hat{T}$ [$^\circ$C]')
plt.ylabel('$z$ [m]')
plt.grid()
plt.legend(draggable=True)

ax=fig.add_subplot(gs[0,1])
plt.plot(rmsd_avg,height,'k')
plt.fill_betweenx(height, rmsd_low,rmsd_top,color='k',alpha=0.25,linewidth=2)
plt.plot(rmsd_th,height,'r',linewidth=2)
for h in height_sel:
    plt.plot([0,0.6],[h,h],'--b')
plt.xlabel(r'RMS of $\Delta \hat{T}$ [$^\circ$C]')
plt.xlim([0,0.6])
plt.ylim([0,max_height*1000])
ax.set_yticklabels([])
plt.grid()

ax=fig.add_subplot(gs[0,2])
plt.plot(nexc/nall*100,height,'k',label='Data',linewidth=2)
plt.plot(height**0*p_value*100,height,'-r',label='TROPoe',linewidth=2)
for h in height_sel:
    plt.plot([0,35],[h,h],'--b')
plt.xlabel(r'Probability of $\Delta \hat{T}$ '+f'exceeding {int((1-p_value)*100)}% c.i. [%]')
plt.xlim([0,35])
plt.ylim([0,max_height*1000])
ax.set_yticklabels([])
plt.grid()

bins=np.arange(-2,2+0.0001,0.01)
locs=['upper right','center right','lower right']
ctr=0 
for h in height_sel:
    ax_inset = inset_axes(ax, width="40%", height="30%", loc=locs[ctr])
                    # bbox_to_anchor=(1.05, 0.65, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
    ax_inset.hist(Diff['DT'].interp(height=h/1000),bins,color='k',density=True,alpha=0.5)
    ax_inset.plot(bins,norm.pdf(bins,loc=0,scale=Diff['sigmaDT'].interp(height=h/1000).mean()),'r',linewidth=2)
    ax_inset.fill_between(bins,norm.pdf(bins,loc=0,scale=Diff['sigmaDT'].interp(height=h/1000).min()),
                               norm.pdf(bins,loc=0,scale=Diff['sigmaDT'].interp(height=h/1000).max()),color='r',alpha=0.25)
    ax_inset.plot(bins,norm.pdf(bins,loc=Diff['DT'].interp(height=h/1000).mean(),scale=Diff['DT'].interp(height=h/1000).std()),'k',linewidth=2)
    ax_inset.text(-1.2+ctr*0.125,10,r'$z='+str(int(h))+'$ m',color='b',bbox={'facecolor':'b','alpha':0.25},fontsize=18)
    ax_inset.set_yscale('log')
    ax_inset.set_xlim([-2,2])
    ax_inset.set_ylim([1/len(Diff.time)/0.01/2,20])
    if ctr>0:
        ax_inset.set_xticklabels([])
        ax_inset.set_yticklabels([])
    else:
        ax_inset.set_xlabel('$\Delta \hat{T}$ [$^\circ$C]')
        ax_inset.set_ylabel('PDF')
    ax_inset.grid(True)
    ctr+=1
plt.tight_layout()

# ACF
plt.figure(figsize=(18,10))
ctr=0
for h in height_sel:
    ax=plt.subplot(len(height_sel),1,1+ctr)
    x=DT_interp.interp(height=h/1000).values
    sm.graphics.tsa.plot_acf(x, lags=144*2,ax=ax,markersize=3,missing='conservative',adjusted=True,color='k',vlines_kwargs={'colors': 'black','linewidth':1.5,'alpha':0.75})
    for poly in ax.collections:  # ax.collections contains the filled regions
        if not 'Line' in str(poly):
            poly.set_color('black')  # Change to black
            poly.set_alpha(0.25)  # Adjust transparency
    
    plt.title('')
    plt.gca().set_xscale('symlog')
    if h==height_sel[-1]:
        plt.xticks([0,1,6,144],['0','10 min','1 h','1 d'])
    else:
        ax.set_xticklabels([])
    
    plt.xlabel('Lag')
    plt.ylabel('ACF')
    plt.ylim([-0.1,1.05])
    plt.yticks([0,0.25,0.5,0.75,1])
    plt.grid()
    plt.text(100,0.875,r'$z='+str(int(h))+'$ m')
    ctr+=1

matplotlib.rcParams['font.size'] = 24

