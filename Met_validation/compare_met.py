# -*- coding: utf-8 -*-
"""
Compare tropoe retrievals to met station data
"""
import os
cd=os.path.dirname(__file__)
import sys
sys.path.append(os.path.join(cd,'../utils'))
import utils as utl
import numpy as np
import xarray as xr
import matplotlib
from matplotlib import pyplot as plt
import yaml
import pandas as pd
import seaborn as sns
from scipy.stats import norm
from matplotlib.ticker import NullFormatter
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['savefig.dpi']=500
plt.close("all")

#%% Inputs

#dataset
source_config=os.path.join(cd,'configs','config.yaml')
sources_snc={'A2':os.path.join(cd,'data/sonic.c0.A2.nc'),
             'A5':os.path.join(cd,'data/sonic.c0.A5.nc')}

sites=['B','C1a','G']
site_diff=['C1a-B','G-B','G-C1a']
sites_snc=['A2','A5']
sigma_met=0.25#[C] uncertainty of met measurements [NOAA, 2004]
height_met=2#[m a.g.l.]

#wd ranges at which sonic is not affetded by container
wd_range_snc={'A2':[0,270],
              'A5':[270,360]}

#stability classes [Ksirhnamurty et al., 2019]
stab_classes={'S':[10,200],
              'NS':[200, 500],
              'N1':[500,np.inf],
              'N2':[-np.inf,-500],
              'NU':[-500,-200],
              'U':[-200,-50]}

stab_class_uni=['S','NS','N','NU','U']

#stats
p_value=0.05#for CI
perc_lim=[5,95]#percentile filter
max_T=45#[C] max threshold of selected variable
min_T=-10#[C] min threshold of selected variable
max_time_diff=60#[s] maximum difference in time between met and TROPoe
max_time_diff_L=30*60#[s] maximum difference in tme between sonic and TROPoe

#graphics
site_names={'B':'South','C1a':'Middle','G':'North'}
site_diff_names={'C1a-B':'Middle-South','G-B':'North-South','G-C1a':'North-Middle'}

#%% Initialization
#config
with open(source_config, 'r') as fid:
    config = yaml.safe_load(fid)

#%% Main

#read sonic data
SNC={}
SNC_unw={}
for s in sites_snc:
    SNC[s]=xr.open_dataset(sources_snc[s])
    
SNC[sites_snc[0]],SNC[sites_snc[1]]=xr.align(SNC[sites_snc[0]],SNC[sites_snc[1]],join='outer')

#select unwaked sectors
for s in sites_snc:
    if wd_range_snc[s][1]>wd_range_snc[s][0]:
       SNC_unw[s]=SNC[s].where(SNC[s]['wd']>=wd_range_snc[s][0]).where(SNC[s]['wd']<wd_range_snc[s][1])
    else:
       SNC_unw[s]=SNC[s].where((SNC[s]['wd']<wd_range_snc[s][1]) | (SNC[s]['wd']>=wd_range_snc[s][0]))

#average L
L=(SNC[sites_snc[0]].L+SNC[sites_snc[1]].L)/2
tnum_L=(L.time-np.datetime64('1970-01-01T00:00:00'))/np.timedelta64(1,'s')

#read all temperature data
T_trp={}
sigma_trp={}
T_met={}
time_all={}
stab={}
for s in sites:
    #read and align data
    Data_trp=xr.open_dataset(os.path.join(cd,'data',f'tropoe.{s}.nc'))
    Data_met=xr.open_dataset(os.path.join(cd,'data',f'met.b0.{s}.nc'))
    
    Data_trp,Data_met=xr.align(Data_trp,Data_met,join="inner",exclude=["height"])
    
    #QC
    print(f"{int(np.sum(Data_trp.qc!=0))} points fail QC in TROPoe")
    Data_trp=Data_trp.where(Data_trp.qc==0)
    
    print(f"{int(np.sum(Data_met.time_diff>max_time_diff))} points fail max_time_diff")
    print(f"{int(np.sum(Data_met.qc_temperature!=0))} points fail met QC")
    Data_met=Data_met.where(np.abs(Data_met.time_diff)<=max_time_diff).where(Data_met.qc_temperature==0)
    
    #height interpolation
    T_trp[s]=Data_trp.temperature.interp(height=height_met)
    sigma_trp[s]=Data_trp.sigma_temperature.interp(height=height_met)
    T_met[s]=Data_met.temperature
    
    #thresholding
    T_trp[s]=T_trp[s].where(T_trp[s]>=min_T).where(T_trp[s]<=max_T)
    T_met[s]=T_met[s].where(T_met[s]>=min_T).where(T_met[s]<=max_T)
    
    #stability
    tnum_trp=(Data_trp.time-np.datetime64('1970-01-01T00:00:00'))/np.timedelta64(1,'s')
    time_diff=tnum_L.interp(time=Data_trp.time,method='nearest')-tnum_trp
    print(f"{int(np.sum(time_diff>max_time_diff_L))} points fail max_time_diff for L")
    L_int=L.interp(time=Data_trp.time).where(np.abs(time_diff)<max_time_diff_L)
    
    stab[s]=xr.DataArray(data=['null']*len(L_int.time),coords={'time':L_int.time})
    for sc in stab_classes.keys():
        sel=(L_int>=stab_classes[sc][0])*(L_int<stab_classes[sc][1])
        if sc=='N1' or sc=='N2':
            sc='N'
        stab[s]=stab[s].where(~sel,other=sc)

#calculate temperature differences (time)
diff={}
for s in sites:
    diff[s]=T_trp[s]-T_met[s]

#calculate temperature differences (space)
ctr=0
diff_trp={}
diff_met={}
diff_sigma_trp={}
diff_stab={}
for s1 in sites:
    for s2 in sites[ctr+1:]:
        diff_trp[f'{s2}-{s1}']=T_trp[s2]-T_trp[s1]
        diff_met[f'{s2}-{s1}']=T_met[s2]-T_met[s1]
        diff_sigma_trp[f'{s2}-{s1}']=(sigma_trp[s1]**2+sigma_trp[s2]**2)**0.5
        diff_stab[f'{s2}-{s1}']=stab[s1].where(diff_trp[f'{s2}-{s1}'])
    ctr+=1

#stability stats (time)
diff_avg=np.zeros((len(sites),len(stab_class_uni)))
for i_s in range(len(sites)):
    s=sites[i_s]
    for i_sc in range(len(stab_class_uni)):
        diff_sel=diff[s].where(stab[s]==stab_class_uni[i_sc]).values
        diff_avg[i_s,i_sc]=utl.filt_stat(diff_sel, np.nanmean,perc_lim=perc_lim)
      
#stability stats (space)
diff_diff_avg=np.zeros((len(sites),len(stab_class_uni)))
for i_s in range(len(site_diff)):
    s=site_diff[i_s]
    for i_sc in range(len(stab_class_uni)):
        diff_diff_sel=(diff_trp[s]-diff_met[s]).where(diff_stab[s]==stab_class_uni[i_sc]).values
        diff_diff_avg[i_s,i_sc]=utl.filt_stat(diff_diff_sel, np.nanmean,perc_lim=perc_lim)

#%% Plots
plt.close('all')
#time series of T
fig=plt.figure(figsize=(18,10))
ctr=1
for s in sites:
    ax=plt.subplot(len(sites),1,ctr)
    plt.plot(T_met[s].time,T_met[s],'-k',label='Met')
    plt.plot(T_trp[s].time,T_trp[s],'-r',label='TROPoe')
    plt.ylim([-10,45])
    plt.grid()
    plt.ylabel(r'$T$ [$^\circ$C]')
    if ctr==0:
        plt.xlabel('Time (UTC)')
    ctr+=1
    plt.title(s)
plt.legend()
plt.tight_layout()

#time series of DT
fig=plt.figure(figsize=(18,10))
ctr=1
for s in sites:
    ax=plt.subplot(len(sites),1,ctr)
    plt.plot(diff[s].time,diff[s],'-k')
    plt.ylim([-5,5])
    plt.grid()
    plt.ylabel(r'$\Delta T$ (TROPoe-met) [$^\circ$C]')
    if ctr==0:
        plt.xlabel('Time (UTC)')
    ctr+=1
    plt.title(s)
plt.tight_layout()

#time series of T difference in space
fig=plt.figure(figsize=(18,10))
ctr=1
for s in diff_trp.keys():
    ax=plt.subplot(len(diff_trp.keys()),1,ctr)
    plt.plot(diff_met[s].time,diff_met[s],'-k',label='Met')
    plt.plot(diff_trp[s].time,diff_trp[s],'-r',label='TROPoe')
    plt.ylim([-4,4])
    plt.grid()
    plt.ylabel(r'$\Delta T$ ('+s+r') [$^\circ$C]')
    if ctr==0:
        plt.xlabel('Time (UTC)')
    ctr+=1
    plt.title(s)
plt.legend()
plt.tight_layout()

#linear regression
matplotlib.rcParams['font.size'] = 14
bins=np.arange(-5,5.1,0.05)
fig=plt.figure(figsize=(18,10))
gs = gridspec.GridSpec(2,len(sites)+1,width_ratios=[1]*len(sites)+[0.05]) 
ctr=0
for s in sites:
    ax=fig.add_subplot(gs[0,ctr])
    if ctr==len(sites)-1:
        cax=fig.add_subplot(gs[0,ctr+1])
    else:
        cax=None
    utl.plot_lin_fit(T_met[s].values,T_trp[s].values,ax=ax,cax=cax,bins=100,legend=ctr==0,limits=[0,100])
    
    ax.set_xlim([-10,45])
    ax.set_ylim([-10,45])
    ax.set_xticks([-10,0,10,20,30,40])
    ax.set_yticks([-10,0,10,20,30,40])
    ax.grid(True)
    ax.set_xlabel(r'$T$ (met) [$^\circ$C]')
    if ctr==0:
        ax.set_ylabel(r'$T$ (TROPoe) [$^\circ$C]')
        plt.legend(draggable=True)
    else:
        ax.set_yticklabels([])
    
  
    ax=fig.add_subplot(gs[1,ctr])
    
    plt.hist(diff[s],bins=bins,color='k',alpha=0.25,density=True)
    plt.plot(bins,norm.pdf(bins,loc=diff[s].mean(),scale=diff[s].std()),'k',label='Data')
    plt.plot(bins,norm.pdf(bins,loc=0,scale=(sigma_trp[s].mean()**2+sigma_met**2)**0.5),'r',label='Theory')
    ax.fill_between(bins,norm.pdf(bins,loc=0,scale=(sigma_trp[s].min()**2+sigma_met**2)**0.5),
                         norm.pdf(bins,loc=0,scale=(sigma_trp[s].max()**2+sigma_met**2)**0.5),color='r',alpha=0.25)
    ax.set_yscale('log')
    plt.grid()
    if ctr==0:
        ax.set_ylabel('PDF')
        plt.legend(draggable=True)
    else:
        ax.yaxis.set_major_formatter(NullFormatter())
    
    plt.xlabel(r'$\Delta T$ (TROPoe-met) [$^\circ$C]')
    plt.xlim([-4,4])
    plt.ylim([0.01,10])
    ctr+=1    
    
#linear regression (differences)
matplotlib.rcParams['font.size'] = 14
bins=np.arange(-5,5.1,0.05)
fig=plt.figure(figsize=(18,10))
gs = gridspec.GridSpec(2,len(sites)+1,width_ratios=[1]*len(sites)+[0.05]) 
ctr=0
for s in site_diff:
    ax=fig.add_subplot(gs[0,ctr])
    if ctr==len(sites)-1:
        cax=fig.add_subplot(gs[0,ctr+1])
    else:
        cax=None
    utl.plot_lin_fit(diff_met[s].values, diff_trp[s].values,ax=ax,cax=cax,bins=100,legend=ctr==0,limits=[0,100])
    
    ax.set_xlim([-4,4])
    ax.set_ylim([-4,4])
    ax.set_xticks([-4,-2,0,2,4])
    ax.set_yticks([-4,-2,0,2,4])
    ax.grid(True)
    ax.set_xlabel(r'$\Delta T$ (met) [$^\circ$C]')
    if ctr==0:
        ax.set_ylabel(r'$\Delta T$ (TROPoe) [$^\circ$C]')
        plt.legend(draggable=True)
    else:
        ax.set_yticklabels([])
    
    ax=fig.add_subplot(gs[1,ctr])
    
    plt.hist(diff_trp[s]-diff_met[s],bins=bins,color='k',alpha=0.25,density=True)
    plt.plot(bins,norm.pdf(bins,loc=(diff_trp[s]-diff_met[s]).mean(),scale=(diff_trp[s]-diff_met[s]).std()),'k',label='Data')
    plt.plot(bins,norm.pdf(bins,loc=0,scale=(diff_sigma_trp[s].mean()**2+2*sigma_met**2)**0.5),'r',label='Theory')
    ax.fill_between(bins,norm.pdf(bins,loc=0,scale=(diff_sigma_trp[s].min()**2+2*sigma_met**2)**0.5),
                         norm.pdf(bins,loc=0,scale=(diff_sigma_trp[s].max()**2+2*sigma_met**2)**0.5),color='r',alpha=0.25)
    ax.set_yscale('log')
    plt.grid()
    if ctr==0:
        ax.set_ylabel('PDF')
        plt.legend(draggable=True)
    else:
        ax.yaxis.set_major_formatter(NullFormatter())
    
    ax.set_xlabel(r'$\Delta (\Delta T)$ (TROPoe-met) [$^\circ$C]')
    plt.xlim([-4,4])
    plt.ylim([0.01,10])
    ctr+=1    


#stability (time)
cmap=matplotlib.cm.get_cmap('coolwarm')
colors={}
ctr=0
for sc in stab_class_uni:
    colors[sc]=cmap(ctr/(len(stab_class_uni)-1))
    ctr+=1
colors['N']=tuple(x*0.8 for x in colors['N'][:-1])+(1,)    
plt.figure(figsize=(18,8))
i_s=0
for s in sites:  
    ax=plt.subplot(2,len(sites),i_s+1)
    df=pd.DataFrame({'sc':stab[s].where(stab[s]!='null',drop=True),'diff':diff[s].where(stab[s]!='null',drop=True)})
    sns.violinplot(x="sc", y="diff", data=df, inner="box", palette=colors,order=stab_class_uni)
    plt.plot(diff_avg[i_s,:],'.w',markersize=15,markeredgecolor='k')
    plt.grid()
    plt.ylim([-2,2])
    if i_s>0:
        plt.ylabel("")
        ax.set_yticklabels([])
    else:
        plt.ylabel(r'$\Delta T$ (TROPoe-met) [$^\circ$C]')
    ax.set_xlabel('')
    ax.set_xticklabels([])
    i_sc=0
    for sc in stab_class_uni:
        plt.text(i_sc+0.05,1.5,np.round(diff_avg[i_s,i_sc],2),color=colors[sc])
        i_sc+=1
    i_s+=1
    
i_s=0
for s in site_diff:  
    ax=plt.subplot(2,len(sites),i_s+len(sites)+1)
    df=pd.DataFrame({'sc':diff_stab[s].where(diff_stab[s]!='null',drop=True),'diff':(diff_trp[s]-diff_met[s]).where(diff_stab[s]!='null',drop=True)})
    sns.violinplot(x="sc", y="diff", data=df, inner="box", palette=colors,order=stab_class_uni)
    plt.plot(diff_diff_avg[i_s,:],'.w',markersize=15,markeredgecolor='k')
    plt.ylim([-2,2])
    plt.grid()
    if i_s>0:
        plt.ylabel("")
        ax.set_yticklabels([])
    else:
        ax.set_ylabel(r'$\Delta (\Delta T)$ (TROPoe-met) [$^\circ$C]')
    ax.set_xlabel('')
    i_sc=0
    for sc in stab_class_uni:
        plt.text(i_sc+0.05,1.5,np.round(diff_diff_avg[i_s,i_sc],2),color=colors[sc])
        i_sc+=1
    i_s+=1