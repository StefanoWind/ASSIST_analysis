# -*- coding: utf-8 -*-
'''
Plot 3rd BB results
'''
import os
cd=os.path.dirname(__file__)

import sys
sys.path.append('C:/Users/SLETIZIA/OneDrive - NREL/Desktop/PostDoc/Custom_functions')
    
import myFunctions as SL
import numpy as np
from matplotlib import pyplot as plt
import warnings
import matplotlib 
import xarray as xr
import glob
from datetime import datetime
import matplotlib.dates as mdates

warnings.filterwarnings('ignore')
plt.close('all')

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 16

#%% Inputs
assist_id=11#instrument ID
date='2023-06-13'
setpoint=40#[C] temperature setpoint

tol=0.5#[C] tolerance of Tb around setpoint to define valid calibration
k=1.380649*10**-23#[J/Kg] Boltzman's constant
h=6.62607015*10**-34#[J s] Plank's constant
c=299792458#[m/s] speed of light


#%% Functions
def dt64_to_str(dt64,time_format='%Y-%m-%d %H:%M:%S'):
    return SL.datestr(SL.dt64_to_num(dt64),time_format)

#%% Initialization
file_sum=glob.glob(os.path.join('data','calib','assist-'+str(assist_id),
                                 '*'+datetime.strftime(datetime.strptime(date, '%Y-%m-%d'), '%Y%m%d')+'*summary*'))

file_cha=glob.glob(os.path.join('data','calib','assist-'+str(assist_id),
                                 '*'+datetime.strftime(datetime.strptime(date, '%Y-%m-%d'), '%Y%m%d')+'*cha*'))

assert len(file_sum)==1
assert len(file_cha)==1

#%% Main

#load data
Data_sum=xr.open_dataset(file_sum[0])

Data_cha=xr.open_dataset(file_cha[0]).sortby('time')
Data_cha=Data_cha.where(Data_cha['sceneMirrorAngle']==0)

#summary file
time_sum=np.datetime64('1970-01-01T00:00:00')+Data_sum['base_time'].values+Data_sum['time'].values
T_985=Data_sum['mean_Tb_985_990'].values-273.15

#define valid calibration period
steady=np.where(np.abs(T_985-setpoint)<tol)[0]
if np.sum(np.diff(steady)>1)>0:
    jumps=np.where(np.diff(steady)>1)[0]
    steady=steady[jumps[-1]+1:]

#channel A file
time_cha=np.datetime64('1970-01-01T00:00:00')+Data_cha['base_time'].values*np.timedelta64(1,'ms')+Data_cha['time'].values*np.timedelta64(1,'s')
wnum=Data_cha['wnum'].values

sel_time=np.where((time_cha>time_sum[steady[0]])&(time_cha<time_sum[steady[-1]]))[0]
mean_rad=Data_cha['mean_rad'].values

#error analysis
rad_BB=2*h*c**2*(wnum*100)**3/(np.exp(h*c*(wnum*100)/(k*(273.15+setpoint)))-1)*10**5
mean_rad_bias=np.nanmean(mean_rad[sel_time,:]-rad_BB,axis=0)
mean_rad_err_SD=np.nanstd(mean_rad[sel_time,:]-rad_BB,axis=0)

#%% Plots
plt.close('all')
date_fmt = mdates.DateFormatter('%H:%M')
plt.figure(figsize=(18,10))
plt.subplot(3,1,1)
patch = plt.Rectangle((time_sum[steady[0]],np.nanmin(T_985)), time_sum[steady[-1]]-time_sum[steady[0]],np.nanmax(T_985)-np.nanmin(T_985),
                      facecolor='red',alpha=0.5)
plt.gca().add_patch(patch)
plt.plot([time_sum[0],time_sum[-1]],[setpoint,setpoint],'-r')
plt.plot([time_sum[0],time_sum[-1]],[setpoint-tol,setpoint-tol],'--r')
plt.plot([time_sum[0],time_sum[-1]],[setpoint+tol,setpoint+tol],'--r')
plt.plot(time_sum,T_985,'.k')
plt.xlabel('Time (UTC)')
plt.ylabel(r'$T_b$ at 985 cm$^{-1}$ [$^\circ$C]')
plt.ylim([-100,50])
plt.grid()
plt.gca().xaxis.set_major_formatter(date_fmt)
plt.title('ASSIST '+str(assist_id)+' on '+dt64_to_str(time_sum[0],'%Y-%m-%d'))

plt.subplot(3,1,2)
plt.plot(wnum,wnum+np.nan,'k',label=str(len(sel_time))+' radiances')
plt.plot(wnum,mean_rad[sel_time,:].T,'k',alpha=0.1)
plt.plot(wnum,rad_BB,'--r',label='BB emission (Plank\'''s law)')
plt.grid()
plt.ylabel(r'Radiance [r.u.]')
plt.legend()
plt.title('Calibration period: ' + dt64_to_str(time_cha[sel_time[0]])+' - '+dt64_to_str(time_cha[sel_time[-1]]))

ax=plt.subplot(3,1,3)
plt.plot(wnum,mean_rad_bias,'k',label='Bias')
plt.plot(wnum,mean_rad_err_SD,'b',label='Error St.Dev.')
plt.plot(wnum,rad_BB*0.01,'--r',label='1% of BB emission')
plt.plot(wnum,-rad_BB*0.01,'--r')
plt.grid()
plt.xlabel(r'$\tilde{\nu}$ [cm$^{-1}$]')
plt.ylabel(r'Error in radiance [r.u.]')
plt.legend()

plt.tight_layout()
left, bottom, width, height = ax.get_position().bounds

ax.set_position([left, bottom+0.05, width, height])

plt.savefig(file_sum[0].replace('assistsummary.cdf','calib.png'))
