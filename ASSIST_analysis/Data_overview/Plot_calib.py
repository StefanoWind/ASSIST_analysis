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
import matplotlib.dates as mdates

warnings.filterwarnings('ignore')
plt.close('all')

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 16

#%% Inputs
assist_id=12#instrument ID
setpoint=40#[C] temperature setpoint

tol=0.5#[C] tolerance of Tb around setpoint to define valid calibration
k=1.380649*10**-23#[J/Kg] Boltzman's constant
h=6.62607015*10**-34#[J s] Plank's constant
c=299792458#[m/s] speed of light
wnum_sel=985.0#[cm^-1] wave number selected

#%% Functions
def dt64_to_str(dt64,time_format='%Y-%m-%d %H:%M:%S'):
    return SL.datestr(SL.dt64_to_num(dt64),time_format)

#%% Initialization
files=glob.glob(os.path.join('data','calib','assist-'+str(assist_id),'*cdf'))

#%% Main
for f in files:
    print('processing '+f)
    
    #load data
    Data=xr.open_dataset(f).sortby('time')
    Data=Data.where(Data['sceneMirrorAngle']==0,drop=True).where(~np.isnan(Data['base_time']),drop=True)
    time=np.datetime64('1970-01-01T00:00:00')+Data['base_time'].values*np.timedelta64(1,'ms')+Data['time'].values*np.timedelta64(1,'s')
    wnum=Data['wnum'].values
    mean_rad=Data['mean_rad'].values
    
    #calculate brightness temperature
    B_sel=Data['mean_rad'].interp(wnum=wnum_sel).values*10**-5
    T_sel=h*c*(wnum_sel*100)/k/(np.log(2*h*c**2*(wnum_sel*100)**3/B_sel+1))-273.15
    
    #define valid calibration period
    steady=np.where(np.abs(T_sel-setpoint)<tol)[0]
    
    assert len(steady)>0
    
    #error analysis
    rad_BB=2*h*c**2*(wnum*100)**3/(np.exp(h*c*(wnum*100)/(k*(273.15+setpoint)))-1)*10**5
    mean_rad_bias=np.nanmean(mean_rad[steady,:]-rad_BB,axis=0)
    mean_rad_err_SD=np.nanstd(mean_rad[steady,:]-rad_BB,axis=0)
    
    #%% Plots
    plt.close('all')
    date_fmt = mdates.DateFormatter('%H:%M')
    plt.figure(figsize=(18,10))
    plt.subplot(3,1,1)
    plt.plot([time[0],time[-1]],[setpoint,setpoint],'-r')
    plt.plot([time[0],time[-1]],[setpoint-tol,setpoint-tol],'--r')
    plt.plot([time[0],time[-1]],[setpoint+tol,setpoint+tol],'--r')
    plt.plot(time,T_sel,'.k')
    plt.plot(time[steady],T_sel[steady],'.g',markersize=2)
    plt.xlabel('Time (UTC)')
    plt.ylabel(r'$T_b$ at'+str(wnum_sel)+' cm$^{-1}$ [$^\circ$C]')
    plt.ylim([-100,50])
    plt.grid()
    plt.gca().xaxis.set_major_formatter(date_fmt)
    plt.title('ASSIST '+str(assist_id)+' on '+dt64_to_str(time[0],'%Y-%m-%d'))
    
    plt.subplot(3,1,2)
    plt.plot(wnum,wnum+np.nan,'k',label=str(len(steady))+' radiances')
    plt.plot(wnum,mean_rad[steady,:].T,'k',alpha=0.1)
    plt.plot(wnum,rad_BB,'--r',label='BB emission (Plank\'''s law)')
    plt.grid()
    plt.ylabel(r'Radiance [r.u.]')
    plt.ylim([0,200])
    plt.legend()
    plt.title('Calibration period: ' + dt64_to_str(time[steady[0]])+' - '+dt64_to_str(time[steady[-1]]))
    
    ax=plt.subplot(3,1,3)
    plt.plot(wnum,mean_rad_bias,'k',label='Bias')
    plt.plot(wnum,mean_rad_err_SD,'b',label='Error St.Dev.')
    plt.plot(wnum,rad_BB*0.01,'--r',label='1% of BB emission')
    plt.plot(wnum,-rad_BB*0.01,'--r')
    plt.ylim([-2,10])
    plt.grid()
    plt.xlabel(r'$\tilde{\nu}$ [cm$^{-1}$]')
    plt.ylabel(r'Error in radiance [r.u.]')
    plt.legend()
    
    plt.tight_layout()
    left, bottom, width, height = ax.get_position().bounds
    
    ax.set_position([left, bottom+0.05, width, height])
    
    plt.savefig(f.replace('assistcha.cdf','calib.png'))
