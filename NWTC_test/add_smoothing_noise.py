# -*- coding: utf-8 -*-
"""
Add error due to noise and smoothing to TROPoe retreival
"""
import os
cd=os.path.dirname(__file__)
import numpy as np
import xarray as xr
import matplotlib
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import yaml
from datetime import datetime
from datetime import timedelta

import glob
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 16

#%% Inputs
source_config=os.path.join(cd,'configs','config.yaml')
source='data/awaken/nwtc.assist.tropoe.z03.c0/*nc'

#graphics
max_z=3000#[m] maximum height ot plot

#%% Initialization
#config
with open(source_config, 'r') as fid:
    config = yaml.safe_load(fid)
    
files=glob.glob(os.path.join(cd,source))

#%% Main
for f in files:
    
    Data=xr.open_dataset(f)
    
    #calculate smoothing error
    I=np.eye(len(Data.arb_dim1))
    Ss=np.zeros_like(Data.Sop)
    Sa=Data.Sa.values
    for it in range(len(Data.time)):
        A=Data['Akernal'].isel(time=it).values.T
        Ss[it,:,:]=(A-I)@Sa@(A-I).T
   
    #save covariances
    Data['Ss']=xr.DataArray(data=Ss,coords={'time':Data.time,'arb_dim1':Data.arb_dim1,'arb_dim2':Data.arb_dim2})
    Data['Sn']=Data.Sop-Data.Ss
    
    #extract diagonal terms
    Nz=len(Data.height)
    Data['sigma_temperature_s']=xr.DataArray(data=            Ss[:,np.arange(Nz),np.arange(Nz)]**0.5,         coords={'time':Data.time,'height':Data.height})
    Data['sigma_temperature_n']=xr.DataArray(data=Data.Sn.values[:,np.arange(Nz),np.arange(Nz)]**0.5,         coords={'time':Data.time,'height':Data.height})
    
    Data['sigma_waterVapor_s']=xr.DataArray(data=            Ss[:,np.arange(Nz,2*Nz),np.arange(Nz,2*Nz)]**0.5,coords={'time':Data.time,'height':Data.height})
    Data['sigma_waterVapor_n']=xr.DataArray(data=Data.Sn.values[:,np.arange(Nz,2*Nz),np.arange(Nz,2*Nz)]**0.5,coords={'time':Data.time,'height':Data.height})
    
    #output
    os.makedirs(os.path.dirname(f.replace('c0','c1')),exist_ok=True)
    Data.to_netcdf(f.replace('c0','c1'))
    
    #plots
    Data=Data.resample(time=str(np.median(np.diff(Data['time']))/np.timedelta64(1,'m'))+'min').nearest(tolerance='1min')

    time=np.array(Data['time'])
    date=str(Data.base_time.values)[:10].replace('-','')
    height0=np.array(Data['height'][:])*1000
    sel_z=height0<max_z
    height=height0[sel_z]

    qc=(Data['gamma']<=config['max_gamma'])*(Data['rmsa']<=config['max_rmsa'])
    
    sigma_temperature=np.array(Data['sigma_temperature'].where(qc)[:,sel_z])
    sigma_temperature_s=np.array(Data['sigma_temperature_s'].where(qc)[:,sel_z])
    sigma_temperature_n=np.array(Data['sigma_temperature_n'].where(qc)[:,sel_z])
    
    sigma_waterVapor=np.array(Data['sigma_waterVapor'].where(qc)[:,sel_z])
    sigma_waterVapor_s=np.array(Data['sigma_waterVapor_s'].where(qc)[:,sel_z])
    sigma_waterVapor_n=np.array(Data['sigma_waterVapor_n'].where(qc)[:,sel_z])
    
    cbh=np.array(Data['cbh'].where(qc))*1000#[m]
    lwp=np.array(Data['lwp'])
    cbh_sel=cbh.copy()
    cbh_sel[lwp<config['min_lwp']]=np.nan
    
    plt.close('all')
    fig=plt.figure(figsize=(25,15))
    gs= gridspec.GridSpec(2, 4, width_ratios=[1,1,1,0.05])
    
    ax=fig.add_subplot(gs[0,0])
    CS=plt.contourf(time,height,sigma_temperature.T,np.arange(0,1.1,0.1),cmap='RdYlGn_r',extend='both')
    plt.plot(time,cbh_sel,'.c',label='Cloud base height',markersize=10)
    if np.sum(~np.isnan(cbh_sel))>0:
        plt.legend(loc='center left')
    ax.set_ylabel(r'$z$ [m.a.g.l.]')
    ax.set_xlabel('Time (UTC)')
    ax.set_xlim([datetime.strptime(date,'%Y%m%d'),datetime.strptime(date,'%Y%m%d')+timedelta(days=1)])
    ax.set_ylim(0, np.max(height)+10)
    ax.grid()
    ax.tick_params(axis='both', which='major')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.set_facecolor((0.9,0.9,0.9))
    plt.text(datetime.strptime(date,'%Y%m%d')+timedelta(hours=10),max_z*0.95,'Total',weight='bold',bbox={'color':'w','alpha':0.5,'edgecolor':'k'})
    
    ax=fig.add_subplot(gs[0,1])
    CS=plt.contourf(time,height,sigma_temperature_s.T,np.arange(0,1.1,0.1),cmap='RdYlGn_r',extend='both')
    plt.plot(time,cbh_sel,'.c',label='Cloud base height',markersize=10)
    ax.set_xlabel('Time (UTC)')
    ax.set_xlim([datetime.strptime(date,'%Y%m%d'),datetime.strptime(date,'%Y%m%d')+timedelta(days=1)])
    ax.set_ylim(0, np.max(height)+10)
    ax.set_yticklabels([])
    ax.grid()
    ax.tick_params(axis='both', which='major')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.set_title('TROPoe retrieval at ' + Data.attrs['Site'] + ' on '+date+'\n File: '+os.path.basename(f), x=0.45)
    ax.set_facecolor((0.9,0.9,0.9))
    plt.text(datetime.strptime(date,'%Y%m%d')+timedelta(hours=10),max_z*0.95,'Smooth.',weight='bold',bbox={'color':'w','alpha':0.5,'edgecolor':'k'})
    
    ax=fig.add_subplot(gs[0,2])
    CS=plt.contourf(time,height,sigma_temperature_n.T,np.arange(0,1.1,0.1),cmap='RdYlGn_r',extend='both')
    plt.plot(time,cbh_sel,'.c',label='Cloud base height',markersize=10)
    ax.set_xlabel('Time (UTC)')
    ax.set_xlim([datetime.strptime(date,'%Y%m%d'),datetime.strptime(date,'%Y%m%d')+timedelta(days=1)])
    ax.set_ylim(0, np.max(height)+10)
    ax.set_yticklabels([])
    ax.grid()
    ax.tick_params(axis='both', which='major')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.set_facecolor((0.9,0.9,0.9))
    plt.text(datetime.strptime(date,'%Y%m%d')+timedelta(hours=10),max_z*0.95,'Noise',weight='bold',bbox={'color':'w','alpha':0.5,'edgecolor':'k'})

    ax=fig.add_subplot(gs[0,3])
    cb = fig.colorbar(CS, cax=ax, orientation='vertical')
    cb.set_label(r'$\sigma(T)$  [$^\circ$C]')
     
    ax=fig.add_subplot(gs[1,0])
    CS=plt.contourf(time,height,sigma_waterVapor.T,np.arange(0,1.1,0.1),cmap='RdYlGn_r',extend='both')
    plt.plot(time,cbh_sel,'.c',label='Cloud base height',markersize=10)
    if np.sum(~np.isnan(cbh_sel))>0:
        plt.legend(loc='center left')
    ax.set_ylabel(r'$z$ [m.a.g.l.]')
    ax.set_xlabel('Time (UTC)')
    ax.set_xlim([datetime.strptime(date,'%Y%m%d'),datetime.strptime(date,'%Y%m%d')+timedelta(days=1)])
    ax.set_ylim(0, np.max(height)+10)
    ax.grid()
    ax.tick_params(axis='both', which='major')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.set_facecolor((0.9,0.9,0.9))
    plt.text(datetime.strptime(date,'%Y%m%d')+timedelta(hours=10),max_z*0.95,'Total',weight='bold',bbox={'color':'w','alpha':0.5,'edgecolor':'k'})
    
    ax=fig.add_subplot(gs[1,1])
    CS=plt.contourf(time,height,sigma_waterVapor_s.T,np.arange(0,1.1,0.1),cmap='RdYlGn_r',extend='both')
    plt.plot(time,cbh_sel,'.c',label='Cloud base height',markersize=10)
    ax.set_xlabel('Time (UTC)')
    ax.set_xlim([datetime.strptime(date,'%Y%m%d'),datetime.strptime(date,'%Y%m%d')+timedelta(days=1)])
    ax.set_ylim(0, np.max(height)+10)
    ax.set_yticklabels([])
    ax.grid()
    ax.tick_params(axis='both', which='major')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.set_facecolor((0.9,0.9,0.9))
    plt.text(datetime.strptime(date,'%Y%m%d')+timedelta(hours=10),max_z*0.95,'Smooth.',weight='bold',bbox={'color':'w','alpha':0.5,'edgecolor':'k'})
    
    ax=fig.add_subplot(gs[1,2])
    CS=plt.contourf(time,height,sigma_waterVapor_n.T,np.arange(0,1.1,0.1),cmap='RdYlGn_r',extend='both')
    plt.plot(time,cbh_sel,'.c',label='Cloud base height',markersize=10)
    ax.set_xlabel('Time (UTC)')
    ax.set_xlim([datetime.strptime(date,'%Y%m%d'),datetime.strptime(date,'%Y%m%d')+timedelta(days=1)])
    ax.set_ylim(0, np.max(height)+10)
    ax.set_yticklabels([])
    ax.grid()
    ax.tick_params(axis='both', which='major')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.set_facecolor((0.9,0.9,0.9))
    plt.text(datetime.strptime(date,'%Y%m%d')+timedelta(hours=10),max_z*0.95,'Noise',weight='bold',bbox={'color':'w','alpha':0.5,'edgecolor':'k'})

    ax=fig.add_subplot(gs[1,3])
    cb = fig.colorbar(CS, cax=ax, orientation='vertical')
    cb.set_label(r'$\sigma(r)$ [g/kg]')
    plt.savefig(f.replace('c0','c1').replace('.nc','.unc.png'))
    
    Data.close()
    plt.close('all')
