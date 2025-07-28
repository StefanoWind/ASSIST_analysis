# -*- coding: utf-8 -*-
"""
Extract climatology information
"""

import os
cd=os.path.dirname(__file__)
import sys
sys.path.append(os.path.join(cd,'../utils'))
import utils as utl
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
matplotlib.rcParams['font.size'] = 18
matplotlib.rcParams['savefig.dpi'] = 500

warnings.filterwarnings('ignore')
plt.close('all')

#%% Inputs
source=os.path.join(cd,'data/nwtc/nwtc.m5.c1/*nc')
source_waked=os.path.join(cd,'data/turbine_wakes.nc')
height_sel=87#[m] selected height for wind rose

#stats
perc_lim=[5,95]#[%] percentile filter limit
p_value=0.05#p-value of uncertianty bands
bin_hour_ws=np.arange(25)#hour bins for wind speed
bin_hour_wd=np.arange(0,25,2)#hour bins for wind direction
bin_Ri=np.array([-100,-0.25,-0.03,0.03,0.25,100])#bins in Ri [mix of Hamilton 2019 and Aitken 2014]
bin_wd=np.arange(0,361,30) #bins in wind direction
max_unc_temp_std=0.1#[C] maximum uncertainty of mean temperature std
max_unc_ti=10#[%] maximum uncertainty of TI

#graphics
stab_names={'S':4,'NS':3,'N':2,'NU':1,'U':0}
hour_sunrise=13#[h]
hour_sunset=1#[h]

#%% Initialization

#read met stats
files=glob.glob(source)
data=xr.open_mfdataset(files)

#read wake data
waked=xr.open_dataset(source_waked)

#%% Main

#QC
data=data.where(data.precip.isel(height=0)==0)#excluding precipitation
print(f"{int(np.sum(data.precip.isel(height=0)>0))} precipitation events excluded")
data['waked']=waked['M5'].interp(time=data.time)
print(f"{int(np.sum(data['waked'].sum(dim='turbine')>0))} wake events at M5 (stats) excluded")
data=data.where(data['waked'].sum(dim='turbine')==0)
    
#extract values
ws=data.ws.sel(height=height_sel).values
wd=data.wd.sel(height=height_sel).values
Ri=data.Ri_3_122
Ri=Ri.where((Ri>bin_Ri[0])*(Ri<bin_Ri[-1]))
ti=data.ws_std/data.ws*100

#hour
hour=np.array([(t-np.datetime64(str(t)[:10]))/np.timedelta64(1,'h') for t in data.time.values])
data['hour']=xr.DataArray(data=hour,coords={'time':data.time.values})

#daily cycles of temperature
hour_avg_ws=(bin_hour_ws[:-1]+bin_hour_ws[1:])/2

f_avg_all=np.zeros((len(hour_avg_ws),len(data.height)))
f_low_all=np.zeros((len(hour_avg_ws),len(data.height)))
f_top_all=np.zeros((len(hour_avg_ws),len(data.height)))
for i_h in range(len(data.height)):
    f_sel=data['air_temp_rec'].isel(height=i_h).values
    real=~np.isnan(f_sel)
    f_avg= stats.binned_statistic(hour[real], f_sel[real],statistic=lambda x: utl.filt_stat(x,   np.nanmean,perc_lim=perc_lim),                          bins=bin_hour_ws)[0]
    f_low= stats.binned_statistic(hour[real], f_sel[real],statistic=lambda x: utl.filt_BS_stat(x,np.nanmean,perc_lim=perc_lim,p_value=p_value/2*100),    bins=bin_hour_ws)[0]
    f_top= stats.binned_statistic(hour[real], f_sel[real],statistic=lambda x: utl.filt_BS_stat(x,np.nanmean,perc_lim=perc_lim,p_value=(1-p_value/2)*100),bins=bin_hour_ws)[0]

    f_avg_all[:,i_h]=f_avg
    f_low_all[:,i_h]=f_low
    f_top_all[:,i_h]=f_top
    
data_avg=xr.Dataset()
data_avg['temp_avg']=xr.DataArray(data=f_avg_all,coords={'hour':hour_avg_ws,'height':data.height.values})
data_avg['temp_low']=xr.DataArray(data=f_low_all,coords={'hour':hour_avg_ws,'height':data.height.values})
data_avg['temp_top']=xr.DataArray(data=f_top_all,coords={'hour':hour_avg_ws,'height':data.height.values})

#daily directional cycles of temperature std
hour_avg_wd=(bin_hour_wd[:-1]+bin_hour_wd[1:])/2
wd_avg=(bin_wd[:-1]+bin_wd[1:])/2

f_avg_all=np.zeros((len(hour_avg_wd),len(wd_avg),len(data.height)))
f_low_all=np.zeros((len(hour_avg_wd),len(wd_avg),len(data.height)))
f_top_all=np.zeros((len(hour_avg_wd),len(wd_avg),len(data.height)))
for i_h in range(len(data.height)):
    f_sel=data['air_temp_rec_std'].isel(height=i_h).values
    wd_sel=data.wd.isel(height=i_h).values
    real=~np.isnan(f_sel+wd_sel)
    f_avg= stats.binned_statistic_2d(hour[real], wd[real], f_sel[real],statistic=lambda x: utl.filt_stat(x,   np.nanmean,perc_lim=perc_lim),                          bins=[bin_hour_wd,bin_wd])[0]
    f_low= stats.binned_statistic_2d(hour[real], wd[real], f_sel[real],statistic=lambda x: utl.filt_BS_stat(x,np.nanmean,perc_lim=perc_lim,p_value=p_value/2*100),    bins=[bin_hour_wd,bin_wd])[0]
    f_top= stats.binned_statistic_2d(hour[real], wd[real], f_sel[real],statistic=lambda x: utl.filt_BS_stat(x,np.nanmean,perc_lim=perc_lim,p_value=(1-p_value/2)*100),bins=[bin_hour_wd,bin_wd])[0]

    f_avg_all[:,:,i_h]=f_avg
    f_low_all[:,:,i_h]=f_low
    f_top_all[:,:,i_h]=f_top
   
data_std=xr.Dataset()
data_std['temp_std_avg']=xr.DataArray(data=f_avg_all,coords={'hour':hour_avg_wd,'wd':wd_avg,'height':data.height.values})
data_std['temp_std_low']=xr.DataArray(data=f_low_all,coords={'hour':hour_avg_wd,'wd':wd_avg,'height':data.height.values})
data_std['temp_std_top']=xr.DataArray(data=f_top_all,coords={'hour':hour_avg_wd,'wd':wd_avg,'height':data.height.values})
data_std['temp_std_qc']=data_std['temp_std_avg'].where(data_std['temp_std_top']-data_std['temp_std_low']<=max_unc_temp_std)

#daily directional cycles of ti
f_avg_all=np.zeros((len(hour_avg_wd),len(wd_avg),len(data.height)))
f_low_all=np.zeros((len(hour_avg_wd),len(wd_avg),len(data.height)))
f_top_all=np.zeros((len(hour_avg_wd),len(wd_avg),len(data.height)))
for i_h in range(len(data.height)):
    f_sel=ti.isel(height=i_h).values
    wd_sel=data.wd.isel(height=i_h).values
    real=~np.isnan(f_sel+wd_sel)
    f_avg= stats.binned_statistic_2d(hour[real], wd[real], f_sel[real],statistic=lambda x: utl.filt_stat(x,   np.nanmean,perc_lim=perc_lim),                          bins=[bin_hour_wd,bin_wd])[0]
    f_low= stats.binned_statistic_2d(hour[real], wd[real], f_sel[real],statistic=lambda x: utl.filt_BS_stat(x,np.nanmean,perc_lim=perc_lim,p_value=p_value/2*100),    bins=[bin_hour_wd,bin_wd])[0]
    f_top= stats.binned_statistic_2d(hour[real], wd[real], f_sel[real],statistic=lambda x: utl.filt_BS_stat(x,np.nanmean,perc_lim=perc_lim,p_value=(1-p_value/2)*100),bins=[bin_hour_wd,bin_wd])[0]

    f_avg_all[:,:,i_h]=f_avg
    f_low_all[:,:,i_h]=f_low
    f_top_all[:,:,i_h]=f_top

data_std['ti_avg']=xr.DataArray(data=f_avg_all,coords={'hour':hour_avg_wd,'wd':wd_avg,'height':data.height.values})
data_std['ti_low']=xr.DataArray(data=f_low_all,coords={'hour':hour_avg_wd,'wd':wd_avg,'height':data.height.values})
data_std['ti_top']=xr.DataArray(data=f_top_all,coords={'hour':hour_avg_wd,'wd':wd_avg,'height':data.height.values})
data_std['ti_qc']=data_std['ti_avg'].where(data_std['ti_top']-data_std['ti_low']<=max_unc_ti)

#%% Plots
plt.close('all')

#stability
cmap=matplotlib.cm.get_cmap('coolwarm_r')
colors = [cmap(i) for i in np.linspace(0,1,len(bin_Ri)-1)]

plt.figure(figsize=(16,7))

plt.subplot(1,2,1)
N_tot=stats.binned_statistic(data['hour'].where(~np.isnan(Ri)),
                             Ri.where(~np.isnan(Ri)),
                             statistic='count',bins=bin_hour_ws)[0]
N_cum=0
for s in stab_names:
    i_Ri=stab_names[s]
    sel_Ri=(Ri>=bin_Ri[i_Ri])*(Ri<bin_Ri[i_Ri+1])
    N=stats.binned_statistic(data['hour'].where(sel_Ri),
                             Ri.where(sel_Ri),
                             statistic='count',bins=bin_hour_ws)[0]
    plt.bar(hour_avg_ws,N/N_tot*100,label=s,bottom=N_cum,color=colors[i_Ri])
    N_cum+=N/N_tot*100
plt.xticks(np.arange(25),rotation=60)         
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
plt.xticks(np.arange(25),rotation=60)     
plt.xlabel('Hour (UTC)')
plt.ylabel(r'$\overline{T}$ [$^\circ$C]')
plt.legend()
plt.tight_layout()

#wd-hour temperature std
theta_plot=np.radians(np.arange(360))
fig, axs = plt.subplots(1, 4, figsize=(18, 8), constrained_layout=True,subplot_kw={'projection': 'polar'})

theta=np.radians(np.append(wd_avg,wd_avg[0]+360))
for i_h in range(len(data.height)):
    plt.sca(axs[i_h])
    ax=axs[i_h]
    f_plot_qc=np.vstack([data_std['temp_std_qc'].values[:,:,i_h].T,data_std['temp_std_qc'].values[:,0,i_h]]).T
    f_plot=np.vstack([data_std['temp_std_avg'].values[:,:,i_h].T,data_std['temp_std_avg'].values[:,0,i_h]]).T
    cf=plt.contourf(theta,hour_avg_wd,f_plot_qc,np.arange(0.1,0.31,0.01),extend='both',cmap='hot')
    plt.contour(theta,hour_avg_wd,f_plot,np.arange(0.1,0.31,0.01),cmap='hot',alpha=1,linewidths=1)
    plt.contour(theta,hour_avg_wd,f_plot_qc,np.arange(0.1,0.31,0.01),colors='k',alpha=0.5,linewidths=0.1)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_rlim([0,22.5])
    ax.set_yticklabels([])
    ax.set_facecolor((0.8,0.8,0.8))
    ax.grid(False)
    ax.set_xticks([0,np.pi/2,np.pi,np.pi/2*3], labels=['N','E','S','W'])
    plt.plot(theta_plot,theta_plot*0+hour_sunrise,'.g',markersize=2)
    plt.plot(theta_plot,theta_plot*0+hour_sunset,'.g',markersize=2)
plt.tight_layout()
cbar_ax = fig.add_axes([0.1, 0.1, 0.8, 0.03])  # spans most of the width
cbar = fig.colorbar(cf, cax=cbar_ax, orientation='horizontal',ticks=np.arange(0.1,0.31,0.05))
cbar.set_label(r'$\sqrt{\overline{T^{\prime 2}}}$ [$^\circ$C]')

#wd-hour TI
fig, axs = plt.subplots(1, 4, figsize=(18, 8), constrained_layout=True,subplot_kw={'projection': 'polar'})
for i_h in range(len(data.height)):
    plt.sca(axs[i_h])
    ax=axs[i_h]
    f_plot_qc=np.vstack([data_std['ti_qc'].values[:,:,i_h].T,data_std['ti_qc'].values[:,0,i_h]]).T
    f_plot=np.vstack([data_std['ti_avg'].values[:,:,i_h].T,data_std['ti_avg'].values[:,0,i_h]]).T
    cf=plt.contourf(theta,hour_avg_wd,f_plot_qc,np.arange(10,41),extend='both',cmap='viridis')
    plt.contour(theta,hour_avg_wd,f_plot,np.arange(10,41),cmap='viridis',alpha=1,linewidths=1)
    plt.contour(theta,hour_avg_wd,f_plot_qc,np.arange(10,41),colors='k',alpha=0.5,linewidths=0.1)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_rlim([0,22.5])
    ax.set_yticklabels([])
    ax.set_facecolor((0.8,0.8,0.8))
    ax.grid(False)
    ax.set_xticks([0,np.pi/2,np.pi,np.pi/2*3], labels=['N','E','S','W'])
    plt.plot(theta_plot,theta_plot*0+hour_sunrise,'.r',markersize=2)
    plt.plot(theta_plot,theta_plot*0+hour_sunset,'.r',markersize=2)
plt.tight_layout()
cbar_ax = fig.add_axes([0.1, 0.1, 0.8, 0.03])  # spans most of the width
cbar = fig.colorbar(cf, cax=cbar_ax, orientation='horizontal',ticks=np.arange(10,41,5))
cbar.set_label('TI [%]')

#windrose
matplotlib.rcParams['font.size'] = 22
cmap=matplotlib.cm.get_cmap('viridis')
real=~np.isnan(ws+wd)
ax = WindroseAxes.from_ax()
ax.bar(wd[real], ws[real], normed=True,opening=0.8,cmap=cmap,edgecolor="white",bins=((0,2,4,6,8,10,12)))
ax.set_rgrids(np.arange(0,12,2), np.arange(0,12,2))
ax.set_xticks([0,np.pi/2,np.pi,np.pi/2*3], labels=['N','E','S','W'])
for label in ax.get_yticklabels():
    label.set_backgroundcolor('white')   # Set background color
    label.set_color('black')             # Set text color
    label.set_fontsize(18)               # Optional: tweak size
    label.set_bbox(dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3',alpha=0.5))

plt.legend(draggable=True)
