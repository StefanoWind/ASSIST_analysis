# -*- coding: utf-8 -*-
"""
Chi^2 test of prior vs met tower
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
import glob
import xarray as xr
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 14


#%% Inputs
source_config=os.path.join(cd,'configs','config.yaml')
source_met=os.path.join(cd,'data/nwtc.m5.a0/*2022{month:02}*nc')
source_pri=os.path.join(cd,'data/prior/Xa_Sa_datafile.denver.55_levels.month_{month:02}.cdf')

var='temperature_rec'
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
        T=data_met[var].values
        for i in range(len(data_met.time)):
            x=T[i,h_sel]
            chi[m][i]=(x-x_a).T@S_a_inv@(x-x_a)
            
        S_a2=np.cov(T[:,h_sel].T)
        x_a2=np.nanmean(T[:,h_sel],axis=0)
            
        #%% Plots 
        plt.figure(figsize=(16,10))
        
        #mean
        ax=plt.subplot(2,3,1)
        plt.plot(x_a2,height[h_sel],'.-k',label='Met')
        plt.plot(x_a,height[h_sel],'.-r',label='Prior')
        plt.xlabel(r'Mean $T$ [$^\circ$C]')
        plt.ylabel(r'$z$ [m]')
        plt.title(f'month: {m}')
        plt.xlim([10,24])
        plt.grid()
        plt.legend()
        
        #pdf
        colors=[cmap(val) for val in np.linspace(0, 1, len(h_sel))]
        ax=plt.subplot(2,3,2)
        ctr=0
        for i_h in h_sel:
            hst=plt.hist(T[:,i_h],np.arange(-10,40),density=True,alpha=0.25,label=r'$z='+str(np.round(height[i_h],1))+'$ m',color=colors[ctr])
            plt.plot(np.arange(-10,40,0.1),norm.pdf(np.arange(-10,40,0.1),loc=x_a2[ctr],scale=S_a2[ctr,ctr]**0.5),color=colors[ctr])
            ctr+=1
        plt.xlabel(r'$T$ [$^\circ$C]')
        plt.ylabel('p.d.f.')
        plt.legend()
        plt.grid()
        
        #chi-square
        ax=plt.subplot(2,3,3)
        bins=np.linspace(0,np.nanpercentile(chi[m],95))
        plt.hist(chi[m],bins,density=True,color='k',alpha=0.5,label='Met+prior')
        plt.plot((bins[1:]+bins[:-1])/2,chi2.pdf((bins[1:]+bins[:-1])/2,len(h_sel)),'k',label='Ideal')
        plt.xlabel(r'$\chi^2$')
        plt.ylabel('p.d.f.')
        plt.grid()
        plt.legend()
        
        ax=plt.subplot(2,2,3)
        plt.pcolor(height[h_sel],height[h_sel],S_a2,cmap='hot')
        for i in range(S_a2.shape[0]):
            for j in range(S_a2.shape[1]):
                ax.text(height[h_sel][j], height[h_sel][i], f"{S_a2[i, j]:.1f}", 
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

