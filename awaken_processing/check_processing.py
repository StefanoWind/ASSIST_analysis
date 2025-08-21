# -*- coding: utf-8 -*-

"""
Check processing
"""
import os
cd=os.path.dirname(__file__)
import numpy as np
import xarray as xr
import matplotlib
from matplotlib import pyplot as plt
import glob
import warnings
warnings.filterwarnings('ignore')
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['savefig.dpi']=500
plt.close("all")

#%% Inputs
source_trp1='C:/Users/sletizia/OneDrive - NREL/Desktop/Main/ENDURA/ASSIST_analysis/NWTC_test/data/awaken/nwtc.assist.tropoe.z02.c0/*nc'
source_trp2='C:/Users/sletizia/OneDrive - NREL/Desktop/Main/ENDURA/ASSIST_analysis/awaken_processing/data/awaken/sb.assist.tropoe.z01.c0/*nc'

source1=os.path.join(cd,'data/tropoe.ASSIST11.nc')
source2=os.path.join(cd,'data/tropoe.B.nc')

source_pri1='C:/Users/sletizia/OneDrive - NREL/Desktop/Main/ENDURA/ASSIST_analysis/NWTC_test/data/prior/Xa_Sa_datafile.denver.55_levels.month_{m}.cdf'
source_pri2='C:/Users/sletizia/OneDrive - NREL/Desktop/Main/ENDURA/ASSIST_analysis/NWTC_test/data/prior/Xa_Sa_datafile.sgp.55_levels.month_{m}.cdf'

#%% Function
def mean_noise(ds):
    # keep only the variable you care about
    da = ds["obs_vector_uncertainty"]
    # average over time
    avg = da.mean(dim="time", keep_attrs=True)
    
    avg = avg.expand_dims(time=[0])  # or any value you want
   
    avg['time'] = [ds.time.values[0]]
    return avg.to_dataset(name="obs_vector_uncertainty")

#%% Initalization
Data1=xr.open_dataset(source1)
Data2=xr.open_dataset(source2)

Data2_sel=Data2.where(Data2.time>Data1.time.values[0]+np.timedelta64(365,'D'))\
               .where(Data2.time<Data1.time.values[-1]+np.timedelta64(365,'D'))

#%% Main

#check attributes
for a in Data1.attrs:
    if str(Data1.attrs[a]) != str(Data2.attrs[a]):
        print(a)

#avg posterior unertainty
sigma1=Data1.sigma_temperature.mean(dim='time')
sigma2=Data2.sigma_temperature.mean(dim='time')
sigma2_sel=Data2_sel.sigma_temperature.mean(dim='time')

#avg. prior covariance   
months1=np.array([str(t).split('-')[1] for t in Data1.time.values[::1000]])
months2=np.array([str(t).split('-')[1] for t in Data2.time.values[::1000]])

Sa1=0
Sa2_sel=0
for m in months1:
    Data_pri1=xr.open_dataset(source_pri1.format(m=m))
    Sa1+=Data_pri1.covariance_prior.values/len(months1)
    
    Data_pri2=xr.open_dataset(source_pri2.format(m=m))
    Sa2_sel+=Data_pri2.covariance_prior.values/len(months1)

Sa2=0
for m in months2:
    Data_pri2=xr.open_dataset(source_pri2.format(m=m))
    Sa2+=Data_pri2.covariance_prior.values/len(months2)
    
#load all unc data
files1=glob.glob(source_trp1)
Data_trp1=xr.open_mfdataset(files1,preprocess=mean_noise).compute()
print('TROPoe files 1 loaded')

files2=glob.glob(source_trp2)
Data_trp2=xr.open_mfdataset(files2,preprocess=mean_noise).compute()
print('TROPoe files 2 loaded')

Data_trp2_sel=Data_trp2.where(Data_trp2.time>Data_trp1.time.values[0]+np.timedelta64(365,'D'))\
                       .where(Data_trp2.time<Data_trp1.time.values[-1]+np.timedelta64(365,'D'))

#average unc
noise1=Data_trp1.obs_vector_uncertainty.mean(dim='time')
noise2=Data_trp2.obs_vector_uncertainty.mean(dim='time')
noise2_sel=Data_trp2_sel.obs_vector_uncertainty.mean(dim='time')

#%% Plots
plt.close("all")

#posterior uncertainty
plt.figure()
plt.plot(sigma1,sigma1.height,'k',label='NWTC')
plt.plot(sigma2,sigma2.height,'r',label='AWAKEN - All data')
plt.plot(sigma2_sel,sigma2_sel.height,'--r',label='AWAKEN - Same season')
plt.grid()
plt.xlim([0,2])
plt.ylim([0,2000])
plt.xlabel('$\sigma_T$ [$^\circ$C]')
plt.ylabel(r'$z$ [m .a.g.l.]')
plt.legend()

#pior covariance
plt.figure(figsize=(18,6))
plt.subplot(1,3,1)
plt.pcolor(Sa1,vmin=0,vmax=70,cmap='hot')
plt.xlabel(r'$z$ [m .a.g.l.]')
plt.ylabel(r'$z$ [m .a.g.l.]')
plt.title('NWTC')

plt.subplot(1,3,2)
plt.pcolor(Sa2,vmin=0,vmax=70,cmap='hot')
plt.xlabel(r'$z$ [m .a.g.l.]')
plt.ylabel(r'$z$ [m .a.g.l.]')
plt.title('AWAKEN - All data')

plt.subplot(1,3,3)
plt.pcolor(Sa2_sel,vmin=0,vmax=70,cmap='hot')
plt.xlabel(r'$z$ [m .a.g.l.]')
plt.ylabel(r'$z$ [m .a.g.l.]')
plt.title('AWAKEN - Same season')

#obs. uncertainty
plt.figure(figsize=(18,8))
plt.semilogy(noise1.obs_dim,noise1,'k',label='NWTC')
plt.semilogy(noise2.obs_dim,noise2,'r',label='AWAKEN - All data')
plt.semilogy(noise2_sel.obs_dim,noise2_sel,'--r',label='AWAKEN - Same season')
plt.ylabel('$\sigma_y$')
plt.xlabel('Obs. dimension')
plt.grid()
plt.legend()
