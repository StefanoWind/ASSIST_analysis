# -*- coding: utf-8 -*-
"""
Comprehensive check of prior based on met data
"""
import os
cd=os.path.dirname(__file__)
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import chi2
from scipy.stats import norm
import matplotlib
import matplotlib.dates as mdates
from scipy.stats import binned_statistic
import pandas as pd
import glob
import xarray as xr
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 14

#%% Inputs
source_config=os.path.join(cd,'configs','config.yaml')

#user
unit='ASSIST10'

#dataset
source_pri=os.path.join(cd,'data/prior/Xa_Sa_datafile.denver.55_levels.month_{month:02}.cdf')
var_met='temperature'#selected temperature variable in M5 data
var_trp='temperature'#selected temperature variable in TROPoe data
start_prior=0#0-> temperature; 55-> water vapor

#stats
perc_lim=[0.1,99.9]#percentile filter limits
bin_hour=np.arange(25)-0.001#bins of hours

#graphics
cmap = plt.get_cmap("viridis")

#%% Initialization
os.makedirs(os.path.join(cd,'figures/prior_test'),exist_ok=True)

#load met data
data_met=xr.open_dataset(os.path.join(cd,'data',f'met.a1.{unit}.nc'))
month=np.array([int(str(t)[5:7]) for t in data_met.time.values])
hour=np.array([int(str(t)[11:13]) for t in data_met.time.values])
data_met['month']=xr.DataArray(month,coords={'time':data_met.time})
data_met['hour']=xr.DataArray(hour,coords={'time':data_met.time})

#load TROPoe data
data_trp=xr.open_dataset(os.path.join(cd,'data',f'tropoe.{unit}.nc'))
month=np.array([int(str(t)[5:7]) for t in data_trp.time.values])
hour=np.array([int(str(t)[11:13]) for t in data_trp.time.values])
data_trp['month']=xr.DataArray(month,coords={'time':data_trp.time})
data_trp['hour']=xr.DataArray(hour,coords={'time':data_trp.time})

hour_avg=(bin_hour[1:]+bin_hour[:-1])/2

#zeroing
chi={}
       
#%% Main
#load met data
for m in np.unique(month):
    
    #load prior data
    file_pri=glob.glob(source_pri.format(month=m))[0]
    data_pri=xr.open_dataset(file_pri)
    
    #fix heights and select met data
    data_pri=data_pri.assign_coords(height=data_pri.height*1000).assign_coords(height2=data_pri.height2*1000)
    height=data_pri.height.values
    data_met_sel=data_met.interp(height=height).where(data_met.month==m,drop=True)
    
    #load TROPoe data
    data_trp_sel=data_trp.where(data_trp.month==m,drop=True)
    
    #extract moments
    h_sel=np.where(~np.isnan(data_met_sel[var_met].mean(dim='time')))[0]
    S_a=data_pri.covariance_prior.values[h_sel+start_prior,:][:,h_sel+start_prior]
    x_a=data_pri.mean_prior.values[h_sel+start_prior]
    
    #calculate chi-square
    chi[m]=np.zeros(len(data_met_sel.time))
    S_a_inv=np.linalg.inv(S_a)
    f_met=data_met_sel[var_met].where(data_met_sel[var_met]>=np.nanpercentile(data_met_sel[var_met],perc_lim[0]))\
                               .where(data_met_sel[var_met]<=np.nanpercentile(data_met_sel[var_met],perc_lim[1])).values
                   
    for i in range(len(data_met_sel.time)):
        x=f_met[i,h_sel]
        chi[m][i]=(x-x_a).T@S_a_inv@(x-x_a)
        
    #mean
    x_a2=np.nanmean(f_met,axis=0)
       
    #covariance
    S_a2=np.zeros((len(height),len(height)))
    cov= pd.DataFrame(f_met[:,h_sel]).dropna().cov().values
    ctr1=0
    for i_h1 in h_sel:
        ctr2=0
        for i_h2 in h_sel:
            S_a2[i_h1,i_h2]=cov[ctr1,ctr2]
            ctr2+=1
        ctr1+=1
    
    #hourly stats (met)
    x_ah_met=np.zeros((len(height),len(bin_hour)-1))
    for i_h in h_sel:
        sel=~np.isnan(f_met[:,i_h])
        x_ah_met[i_h,:]=binned_statistic(data_met_sel.hour.values[sel],f_met[sel,i_h],statistic='mean',bins=bin_hour)[0]
    
    #hourly stats (TROPoe)
    f_trp=data_trp_sel[var_trp].where(data_trp_sel[var_trp]>=np.nanpercentile(data_trp_sel[var_trp],perc_lim[0]))\
                             .where(data_trp_sel[var_trp]<=np.nanpercentile(data_trp_sel[var_trp],perc_lim[1])).values
    x_ah_trp=np.zeros((len(height),len(bin_hour)-1))
    for i_h in range(len(data_trp.height)):
        sel=~np.isnan(f_trp[:,i_h])
        x_ah_trp[i_h,:]=binned_statistic(data_trp_sel.hour.values[sel],f_trp[sel,i_h],statistic='mean',bins=bin_hour)[0]

    #%% Output
    Output=xr.Dataset()
    Output['mean_temperature']=xr.DataArray(data=x_a2,coords={'height':height})
    Output['mean_temperature_hourly_met']=xr.DataArray(data=x_ah_met,coords={'height':height,'hour':hour_avg})
    Output['mean_temperature_hourly_trp']=xr.DataArray(data=x_ah_trp,coords={'height':height,'hour':hour_avg})
    Output['covariance_temperature']=xr.DataArray(data=S_a2,coords={'height1':height,'height2':height})
    Output.to_netcdf(os.path.join(cd,'data/prior',f'Xa_Sa_datafile.nwtc.{unit}.55_levels.month_{m:02}.cdf'))
        
    #%% Plots 
    plt.figure(figsize=(16,10))
    ctr=1
    for i_h in h_sel:
        ax=plt.subplot(len(h_sel),2,ctr*2-1)
        plt.plot(data_met_sel.time,f_met[:,i_h],'-k')
        plt.plot([data_met_sel.time.values[0],data_met_sel.time.values[-1]],[x_a2[i_h],x_a2[i_h]],'--r')
        plt.ylim([-10,40])
        plt.grid()
        plt.ylabel(r'$T$ [$^\circ$C]')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m%d'))
        if ctr==len(h_sel):
            plt.xlabel('Time (UTC)')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m%d'))
        else:
            ax.set_xticklabels([])
        plt.text(data_met_sel.time[10],25,r'$z='+str(height[i_h])+r'$ m',bbox={'alpha':0.5,'color':'w'})
        
        
        ax=plt.subplot(len(h_sel),2,ctr*2)
        
        plt.hist(f_met[:,i_h],np.arange(-10,40),density=True,color='k',alpha=0.5)
        plt.plot(np.arange(-10,40,0.1),norm.pdf(np.arange(-10,40,0.1),loc=x_a2[i_h],scale=S_a2[i_h,i_h]**0.5),color='k')
        
        plt.ylabel('p.d.f.')
        if ctr==len(h_sel):
            plt.xlabel(r'$T$ [$^\circ$C]')
        else:
            ax.set_xticklabels([])
            
        plt.grid()
        
        ctr+=1
    
    plt.savefig(os.path.join(cd,f'figures/prior_test/{m:02}.{unit}_met_check.png'))
    plt.close()
    
    #hourly prior
    plt.figure(figsize=(12,8))
    plt.pcolor(hour_avg,height[h_sel],x_ah_met[h_sel,:],cmap='hot')
    plt.xlabel('Hour (UTC)')
    plt.ylabel(r'$z$ [m]')
    plt.colorbar(label=r'Mean $T$ [$^\circ$C]')
    
    plt.savefig(os.path.join(cd,f'figures/prior_test/{m:02}{unit}_met_prior.png'))
    plt.close()
    
    plt.figure(figsize=(16,10))
    #mean
    ax=plt.subplot(2,2,1)
    plt.plot(x_a2[h_sel],height[h_sel],'.-k',label='Met')
    plt.plot(x_a,height[h_sel],'.-r',label='Prior')
    plt.xlabel(r'Mean $T$ [$^\circ$C]')
    plt.ylabel(r'$z$ [m]')
    plt.title(f'month: {m}')
    plt.xlim([10,24])
    plt.grid()
    plt.legend()
    
    #chi-square
    ax=plt.subplot(2,2,2)
    bins=np.linspace(0,np.nanpercentile(chi[m],95))
    plt.hist(chi[m],bins,density=True,color='k',alpha=0.5,label='Met+prior')
    plt.plot((bins[1:]+bins[:-1])/2,chi2.pdf((bins[1:]+bins[:-1])/2,len(h_sel)),'k',label='Ideal')
    plt.xlabel(r'$\chi^2$')
    plt.ylabel('p.d.f.')
    plt.grid()
    plt.legend()
    
    ax=plt.subplot(2,2,3)
    plt.pcolor(height[h_sel],height[h_sel],cov,cmap='hot')
    for i in range(cov.shape[0]):
        for j in range(cov.shape[1]):
            ax.text(height[h_sel][j], height[h_sel][i], f"{cov[i, j]:.1f}", 
                    ha='center', va='center', color='g', fontsize=10)
    plt.xlabel(r'$z$ [m]')
    plt.ylabel(r'$z$ [m]')
    plt.xticks(height[h_sel])
    plt.yticks(height[h_sel])
    plt.title(r'Covariance of $T$ (met) [$^\circ$C$^2$]')
    ax.set_aspect('equal')
    
    ax=plt.subplot(2,2,4)
    plt.pcolor(height[h_sel],height[h_sel],S_a,cmap='hot')
    for i in range(S_a.shape[0]):
        for j in range(S_a.shape[1]):
            ax.text(height[h_sel][j], height[h_sel][i], f"{S_a[i, j]:.1f}", 
                    ha='center', va='center', color='g', fontsize=10)
    plt.xlabel(r'$z$ [m]')
    plt.ylabel(r'$z$ [m]')
    plt.xticks(height[h_sel])
    plt.yticks(height[h_sel])
    plt.title(r'Covariance of $T$ (prior) [$^\circ$C$^2$]')
    ax.set_aspect('equal')
    
    plt.tight_layout()

    plt.savefig(os.path.join(cd,f'figures/prior_test/{m:02}{unit}_prior_test.png'))
    plt.close()
    data_pri.close()
    

