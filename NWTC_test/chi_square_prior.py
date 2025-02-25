# -*- coding: utf-8 -*-
"""
Comprehensive check of prior based on met data
"""
import os
cd=os.path.dirname(__file__)
import sys
sys.path.append('C:/Users/SLETIZIA/OneDrive - NREL/Desktop/Main/utils')
import utils as utl
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

#dataset
source_met=os.path.join(cd,'data/nwtc.m5.a0/*2022{month:02}*nc')
source_pri=os.path.join(cd,'data/prior/Xa_Sa_datafile.denver.55_levels.month_{month:02}.cdf')
var='temperature_rec'

#stats
perc_lim=[0.5,99.5]
bin_hour=np.arange(25)

#graphics
cmap = plt.get_cmap("viridis")

#%% Initialization
os.makedirs(os.path.join(cd,'figures/prior_test'),exist_ok=True)
chi={}
       
#%% Main
#load met data
for m in np.arange(1,13):
    files_met=glob.glob(source_met.format(month=m))
    if len(files_met)>0:
        
        #load met data
        data_met=xr.open_mfdataset(files_met).rename({"air_temp":"temperature"}).rename({"air_temp_rec":"temperature_rec"})
        h_max=data_met.height.max().values
        tnum=np.float64(data_met.time)/10**9
        hour=(tnum-np.floor(tnum/(3600*24))*3600*24)/3600
        
        #load prior data
        file_pri=glob.glob(source_pri.format(month=m))[0]
        data_pri=xr.open_dataset(file_pri)
        
        #fix heights
        data_pri=data_pri.assign_coords(height=data_pri.height*1000).assign_coords(height2=data_pri.height2*1000)
        height=data_pri.height.values
        data_met=data_met.interp(height=height)
        
        #extract moments
        h_sel=np.where(~np.isnan(data_met[var].mean(dim='time')))[0]
        S_a=data_pri.covariance_prior.values[h_sel,:][:,h_sel]
        x_a=data_pri.mean_prior.values[h_sel]
        
        #calculate chi-square
        chi[m]=np.zeros(len(data_met.time))
        S_a_inv=np.linalg.inv(S_a)
        T=data_met[var].where(data_met[var]>=np.nanpercentile(data_met[var],perc_lim[0]))\
                       .where(data_met[var]<=np.nanpercentile(data_met[var],perc_lim[1])).values
                       
        for i in range(len(data_met.time)):
            x=T[i,h_sel]
            chi[m][i]=(x-x_a).T@S_a_inv@(x-x_a)
            
        #mean
        x_a2=np.nanmean(T,axis=0)
           
        #covariance
        S_a2=np.zeros((len(height),len(height)))
        cov= pd.DataFrame(T[:,h_sel]).dropna().cov().values
        ctr1=0
        for i_h1 in h_sel:
            ctr2=0
            for i_h2 in h_sel:
                S_a2[i_h1,i_h2]=cov[ctr1,ctr2]
                ctr2+=1
            ctr1+=1
        
        #hourly stats
        x_ah=np.zeros((len(height),len(bin_hour)-1))
        for i_h in h_sel:
            sel=~np.isnan(T[:,i_h])
            x_ah[i_h,:]=binned_statistic(hour[sel],T[sel,i_h],statistic='mean',bins=bin_hour)[0]

        #%% Output
        Output=xr.Dataset()
        Output['mean_temperature']=xr.DataArray(data=x_a2,coords={'height':height})
        Output['mean_temperature_hourly']=xr.DataArray(data=x_ah,coords={'height':height,'hour':utl.mid(bin_hour)})
        Output['covariance_temperature']=xr.DataArray(data=S_a2,coords={'height1':height,'height2':height})
        Output.to_netcdf(os.path.join(cd,'data/prior',f'Xa_Sa_datafile.nwtc.55_levels.month_{m:02}.cdf'))
            
        #%% Plots 
        plt.figure(figsize=(16,10))
        ctr=1
        for i_h in h_sel:
            ax=plt.subplot(len(h_sel),2,ctr*2-1)
            plt.plot(data_met.time,T[:,i_h],'-k')
            plt.plot([data_met.time.values[0],data_met.time.values[-1]],[x_a2[i_h],x_a2[i_h]],'--r')
            plt.ylim([-10,40])
            plt.grid()
            plt.ylabel(r'$T$ [$^\circ$C]')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m%d'))
            if ctr==len(h_sel):
                plt.xlabel('Time (UTC)')
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m%d'))
            else:
                ax.set_xticklabels([])
            plt.text(data_met.time[10],25,r'$z='+str(height[i_h])+r'$ m',bbox={'alpha':0.5,'color':'w'})
            
            
            ax=plt.subplot(len(h_sel),2,ctr*2)
            
            plt.hist(T[:,i_h],np.arange(-10,40),density=True,color='k',alpha=0.5)
            plt.plot(np.arange(-10,40,0.1),norm.pdf(np.arange(-10,40,0.1),loc=x_a2[i_h],scale=S_a2[i_h,i_h]**0.5),color='k')
            
            plt.ylabel('p.d.f.')
            if ctr==len(h_sel):
                plt.xlabel(r'$T$ [$^\circ$C]')
            else:
                ax.set_xticklabels([])
                
            plt.grid()
            
            ctr+=1
        
        plt.savefig(os.path.join(cd,f'figures/prior_test/{m:02}_met_check.png'))
        plt.close()
        
        #hourly prior
        plt.figure(figsize=(12,8))
        plt.pcolor(utl.mid(bin_hour),height[h_sel],x_ah[h_sel,:],cmap='hot')
        plt.xlabel('Hour (UTC)')
        plt.ylabel(r'$z$ [m]')
        plt.colorbar(label=r'Mean $T$ [$^\circ$C]')
        
        plt.savefig(os.path.join(cd,f'figures/prior_test/{m:02}_met_prior.png'))
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
        plt.title('Covariance of $T$ (met) [$^\circ$C$^2$]')
        utl.axis_equal()
        
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
        plt.title('Covariance of $T$ (prior) [$^\circ$C$^2$]')
        utl.axis_equal()
        
        plt.tight_layout()

        plt.savefig(os.path.join(cd,f'figures/prior_test/{m:02}_prior_test.png'))
        plt.close()
        data_met.close()
        data_pri.close()
    

