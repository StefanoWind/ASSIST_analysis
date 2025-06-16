# -*- coding: utf-8 -*-
"""
Extract climatology information
"""

import os
cd=os.path.dirname(__file__)
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
import warnings
import glob
from scipy import stats
from windrose import WindroseAxes
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 16

warnings.filterwarnings('ignore')
plt.close('all')

#%% Inputs
source=os.path.join(cd,'data/awaken/nwtc.m5.c0/*nc')
height_sel=74#[m] selected height

stab_class={'S':[0,200],
            'NS':[200,500],
            'N1':[500,np.inf],
            'N2':[-np.inf,-500],
            'NU':[-500,-200],
            'U':[-200,0]}#stability classes from Obukhov length [Hamilton and Debnath, 2019]

#stats
perc_lim=[5,95]#[%] percentile filter limit
p_value=0.05#p-value of uncertianty bands
bin_hour=np.arange(-0.5,24)#hour bins

#graphics
stab_class_uni=['S','NS','N','NU','U']

#%% Functions
def filt_stat(x,func,perc_lim=[5,95]):
    '''
    Statistic with percentile filter
    '''
    x_filt=x.copy()
    lb=np.nanpercentile(x_filt,perc_lim[0])
    ub=np.nanpercentile(x_filt,perc_lim[1])
    x_filt=x_filt[(x_filt>=lb)*(x_filt<=ub)]
       
    return func(x_filt)

def filt_BS_stat(x,func,p_value=5,M_BS=100,min_N=10,perc_lim=[5,95]):
    '''
    Statstics with percentile filter and bootstrap
    '''
    x_filt=x.copy()
    lb=np.nanpercentile(x_filt,perc_lim[0])
    ub=np.nanpercentile(x_filt,perc_lim[1])
    x_filt=x_filt[(x_filt>=lb)*(x_filt<=ub)]
    
    if len(x)>=min_N:
        x_BS=bootstrap(x_filt,M_BS)
        stat=func(x_BS,axis=1)
        BS=np.nanpercentile(stat,p_value)
    else:
        BS=np.nan
    return BS


def bootstrap(x,M):
    '''
    Bootstrap sample drawer
    '''
    i=np.random.randint(0,len(x),size=(M,len(x)))
    x_BS=x[i]
    return x_BS

#%% Initialization
files=glob.glob(source)
data=xr.open_mfdataset(files)
data=data.where(data.precip.isel(height_prec=0)==0)#excluding precipitation

#%% Main
ws=data.u_rot.sel(height_kin=height_sel).values
wd=data.wd.sel(height_kin=height_sel).values
L=data['L'].sel(height_kin=height_sel)

#stab classes
data['stab_class']=xr.DataArray(data=['null']*len(data.time),coords={'time':data.time})

for s in stab_class.keys():
    sel=(L>=stab_class[s][0])*(L<stab_class[s][1])
    if s=='N1' or s=='N2':
        s='N'
    data['stab_class']=data['stab_class'].where(~sel,other=s)
    
hour=[(t-np.datetime64(str(t)[:10]))/np.timedelta64(1,'h') for t in data.time.values]
data['hour']=xr.DataArray(data=hour,coords={'time':data.time.values})

#daily cycles
hour_avg=(bin_hour[:-1]+bin_hour[1:])/2

f_avg_all=np.zeros((len(hour_avg),len(data.height_therm)))
f_low_all=np.zeros((len(hour_avg),len(data.height_therm)))
f_top_all=np.zeros((len(hour_avg),len(data.height_therm)))
for i_h in range(len(data.height_therm)):
    f=data['air_temp_rec'].isel(height_therm=i_h).values
    real=~np.isnan(f)
    f_avg= stats.binned_statistic(data.hour.values[real], f[real],statistic=lambda x:   filt_stat(x,np.nanmean,perc_lim=perc_lim),                          bins=bin_hour)[0]
    f_low= stats.binned_statistic(data.hour.values[real], f[real],statistic=lambda x:filt_BS_stat(x,np.nanmean,perc_lim=perc_lim,p_value=p_value/2*100),    bins=bin_hour)[0]
    f_top= stats.binned_statistic(data.hour.values[real], f[real],statistic=lambda x:filt_BS_stat(x,np.nanmean,perc_lim=perc_lim,p_value=(1-p_value/2)*100),bins=bin_hour)[0]

    f_avg_all[:,i_h]=f_avg
    f_low_all[:,i_h]=f_low
    f_top_all[:,i_h]=f_top
    
data_avg=xr.Dataset()
data_avg['temp_avg']=xr.DataArray(data=f_avg_all,coords={'hour':hour_avg,'height':data.height_therm.values})
data_avg['temp_low']=xr.DataArray(data=f_low_all,coords={'hour':hour_avg,'height':data.height_therm.values})
data_avg['temp_top']=xr.DataArray(data=f_top_all,coords={'hour':hour_avg,'height':data.height_therm.values})
        
#%% Plots
plt.close('all')

#stability
cmap=matplotlib.cm.get_cmap('coolwarm')
colors = [cmap(i) for i in np.linspace(0,1,len(stab_class_uni))]

plt.figure(figsize=(16,8))

plt.subplot(1,2,1)
N_tot=stats.binned_statistic(data['hour'].where(data['stab_class']!='null'),
                             data['stab_class'].where(data['stab_class']!='null'),
                             statistic='count',bins=np.arange(-0.5,24,1))[0]
N_cum=0
for s,c in zip(stab_class_uni,colors):
    N=stats.binned_statistic(data['hour'].where(data['stab_class']==s),
                             data['stab_class'].where(data['stab_class']==s),
                             statistic='count',bins=np.arange(-0.5,24,1))[0]
    plt.bar(np.arange(24),N/N_tot*100,label=s,bottom=N_cum,color=c)
    N_cum+=N/N_tot*100
plt.xticks(np.arange(0,24),rotation=60)         
plt.yticks(np.arange(0,101,25)) 
plt.ylabel('Occurrence [%]')   
plt.legend(draggable='True')
plt.xlabel('Hour (UTC)')
plt.grid()

# temperature cycles
plt.subplot(1,2,2)
cmap=matplotlib.cm.get_cmap('plasma')
colors = [cmap(i) for i in np.linspace(0,1,len(data_avg.height))]
for h,c in zip(data_avg.height,colors):
    plt.plot(data_avg.hour,data_avg.temp_avg.sel(height=h),color=c,linewidth=2,label=r'$z='+str(h.values)+'$ m')
    plt.fill_between(data_avg.hour,data_avg.temp_low.sel(height=h), data_avg.temp_top.sel(height=h),color=c,alpha=0.25)
plt.grid()
plt.xticks(np.arange(0,24),rotation=60)     
plt.xlabel('Hour (UTC)')
plt.ylabel(r'Daily-averaged $T$')
plt.legend()

#windrose
cmap=matplotlib.cm.get_cmap('plasma')
real=~np.isnan(ws+wd)
ax = WindroseAxes.from_ax()
ax.bar(wd[real], ws[real], normed=True,opening=0.8,cmap=cmap,edgecolor="white",bins=((0,2,4,6,8,10,12)))
ax.set_rgrids(np.arange(0,12,2), np.arange(0,12,2))
for label in ax.get_yticklabels():
    label.set_backgroundcolor('white')   # Set background color
    label.set_color('black')             # Set text color
    label.set_fontsize(18)               # Optional: tweak size
    label.set_bbox(dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3',alpha=0.5))

plt.legend()