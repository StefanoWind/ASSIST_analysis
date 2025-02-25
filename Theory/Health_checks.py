
import sys
sys.path.append('C:/Users/SLETIZIA/OneDrive - NREL/Desktop/PostDoc/utils')
import xarray as xr
import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates

plt.close('all')
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 18

#%% Inputs
source='./data/rhod.assist.z01.a0.20240714.000055.nc'

#%% Initialization
Data=xr.open_dataset(source)
time=Data.time

#%% Plots
fig=plt.figure(figsize=(18,10))
gs = gridspec.GridSpec(6,1, height_ratios=[0.1,3,3,1,1,1])  # 0.1 row for the colorbars

ax=fig.add_subplot(gs[0,0])

#hatch
hopen=Data['hatch_status'].values==1
hclosed=Data['hatch_status'].values==0
hmove=Data['hatch_status'].values==-1

for i in range(10):
    plt.plot(time[hopen],np.zeros(np.sum(hopen))+i,'.g',markersize=0.5)
    plt.plot(time[hclosed],np.zeros(np.sum(hclosed))+i,'.r',markersize=0.5)
    plt.plot(time[hmove],np.zeros(np.sum(hmove))+i,'.b',markersize=0.5)
plt.xticks([])
plt.yticks([])
plt.title('ASSIST health check on '+str(Data.time.values[0])[:10]+' at '+Data.attrs['location_id']+'\n file: '+os.path.basename(source))
    
#temperatures
ax=fig.add_subplot(gs[1,0])
plt.plot(time,Data.hbb_apex_temperature,'r',label=r'HBB (apex)')
plt.plot(time,Data.abb_apex_temperature,'k',label=r'ABB (apex)')
plt.plot(time,Data.front_end_temperature,'g',label=r'Enclosure (front end)')
plt.plot(time,Data.mean_tb_675_680-273.15,color='orange',label=r'Tb ($675<\tilde{\nu}<680$ cm$^{-1}$)')
plt.plot(time,Data.mean_tb_985_990-273.15,'b',label=r'Tb ($985<\tilde{\nu}<990$ cm$^{-1}$)')
plt.plot(time,Data.interferometer_temperature,'m',label=r'Interferometer')
plt.plot(time,Data.cooler_block_temperature,'c',label=r'Cooler')
plt.ylabel(r'$T$ [$^\circ$C]')
plt.legend(draggable=True)
plt.grid()
ax.set_xticklabels([])

ax=fig.add_subplot(gs[2,0])
plt.plot(time,Data.hbb_apex_temperature-Data.hbb_top_temperature,'r',label='HBB (apex-top)')
plt.plot(time,Data.hbb_top_temperature-Data.hbb_bottom_temperature,'orange',label='HBB (top-bottom)')
plt.plot(time,Data.hbb_apex_temperature-Data.hbb_bottom_temperature,'m',label='HBB (apex-bottom)')
plt.plot(time,Data.abb_apex_temperature-Data.abb_top_temperature,'k',label='ABB (apex-top)')
plt.plot(time,Data.abb_top_temperature-Data.abb_bottom_temperature,'g',label='ABB (top-bottom)')
plt.plot(time,Data.abb_apex_temperature-Data.abb_bottom_temperature,'b',label='ABB (apex-bottom)')
plt.ylabel(r'$\Delta T$ [$^\circ$C]')
plt.legend(draggable=True)
plt.grid()
ax.set_xticklabels([])

ax=fig.add_subplot(gs[3,0])
plt.plot(time,Data.lw_responsivity,'k')
plt.ylabel(r'$\Re$',rotation=50, labelpad=35)
plt.grid()
ax.set_xticklabels([])
plt.ylim([0,3*10**5])

ax=fig.add_subplot(gs[4,0])
plt.plot(time,Data.mean_imaginary_rad_985_990,'k')
plt.ylabel(r'Imag$(B)$'+'\n'+r'($985<\tilde{\nu}<990$ cm$^{-1}$)',rotation=50, labelpad=35)
plt.grid()
ax.set_xticklabels([])
plt.ylim([-1,1])

ax=fig.add_subplot(gs[5,0])
plt.plot(time,Data.interferometer_humidity,'k')
plt.xlabel('Time (UTC)')
plt.ylabel('RH [%]',rotation=50, labelpad=35)
plt.grid()
date_format = mdates.DateFormatter('%H:%M') 
plt.ylim([0,15])
plt.gca().xaxis.set_major_formatter(date_format)