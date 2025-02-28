# -*- coding: utf-8 -*-
"""
Effect on spectrum of cut function
"""

import os
cd=os.getcwd()
import sys
sys.path.append('C:/Users/SLETIZIA/OneDrive - NREL/Desktop/PostDoc/utils')
import numpy as np
import utils as utl
from matplotlib import pyplot as plt
import xarray as xr
import warnings
import matplotlib
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import integrate

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 18

#%% Inputs
source=os.path.join(cd,'data/sb.assist.z01.00.20230824.000041.assistcha.cdf')
hour_sel=15#select hour
T_amb=273.15+32#[C] ambient temperature to fill gap in spectrum

k=1.380649*10**-23#[J/Kg] Boltzman's constant
h=6.62607015*10**-34#[J s] Plank's constant
c=299792458#[m/s] speed of light

clips=[0.9,0.5,0.1]#clipping levels

wnum_laser=15798.02#[cm^-1]
N_real=32768#number of real samples

#graphics
zoom=[1000,1200,0,100]

#%% Functions
def phase(c):
    c[np.abs(c)<10**-10]=np.nan
    return np.angle(c)

#%% Initalization
Data=xr.open_dataset(source).sortby('time')
wnum=Data.wnum.values
time=Data.time.values

#%% Main
t_sel=time[np.argmin(np.abs(time-hour_sel*3600))]
B=Data['mean_rad'].sel(time=t_sel).values

#mirorring and filling of the spectrum
dwnum=np.nanmedian(np.diff(wnum))
wnum_ds=np.arange(-wnum[-1],wnum[-1]+dwnum,dwnum)
B_ds=2*h*c**2*(np.abs(wnum_ds*100))**3/(np.exp(h*c*(np.abs(wnum_ds*100))/(k*T_amb))-1)/2*10**5
B_ds[0:len(wnum)]=B[::-1]/2
B_ds[-len(wnum):]=B/2
B_ds[np.isnan(B_ds)]=0

#FT
N=len(wnum_ds)
n=np.arange(N)-np.floor(N/2)
k=np.arange(N)-np.floor(N/2)
NN,KK=np.meshgrid(n,k)
DFM=np.exp(-1j*KK*NN*2*np.pi/N)

#build igram
dx=1/dwnum/N
x=n*dx
I_ds=np.matmul(1/DFM.T,B_ds)/N

dx_real=1/wnum_laser
dwnum_real=1/dx_real/N_real
xmax=1/(2*dwnum_real)

wnum_fine=np.arange(-100,100)*dwnum_real/10
H_real=2*np.sin(2*np.pi*wnum_fine*xmax)/(2*np.pi*wnum_fine)
H_real[wnum_fine==0]=xmax*2


#quantify smoothing
integ1, error1 = integrate.quad(lambda t: np.abs(2*np.sin(2*np.pi*t*xmax)/(2*np.pi*t)), 0,dwnum)
integ2, error2 = integrate.quad(lambda t: np.abs(2*np.sin(2*np.pi*t*xmax)/(2*np.pi*t)), dwnum,dwnum*2)
integ3, error3 = integrate.quad(lambda t: np.abs(2*np.sin(2*np.pi*t*xmax)/(2*np.pi*t)), dwnum*2,dwnum*3)
integ4, error4 = integrate.quad(lambda t: np.abs(2*np.sin(2*np.pi*t*xmax)/(2*np.pi*t)), dwnum*3,dwnum*4)
print(integ2/(integ1))
print(integ3/(integ1))
print(integ4/(integ1))

#%% Plots
fig=plt.figure(figsize=(18,10))
ctr=1
for c in clips:
    
    #define heavyside
    hn=np.zeros(len(x))+1
    hn[np.abs(x)>=x[-1]*c-10**-10]=0
    B_ds_clip=np.matmul(DFM,I_ds*hn)
    
    #heavyside FT
    Hk=np.matmul(DFM.T,hn)

    H_th=2*np.sin(2*np.pi*wnum_fine*xmax*c)/(2*np.pi*wnum_fine)
    H_th[wnum_fine==0]=xmax*c*2
    
    #demirroring
    B_clip=(B_ds_clip+B_ds_clip[::-1])[wnum_ds>=wnum[0]]
    
    ax=plt.subplot(2,len(clips),ctr)
    plt.plot(wnum,np.abs(B),'k',linewidth=1,label=r'$s_{max}='+str(np.round(xmax,3))+'$ cm')
    plt.plot(wnum,np.abs(B_clip),'r',linewidth=1,label=r'Reduced $s_{max}$')
    rectangle = patches.Polygon([[zoom[0],zoom[2]],
                                 [zoom[0],zoom[3]],
                                 [zoom[1],zoom[3]],
                                 [zoom[1],zoom[2]]],color='g',facecolor='g', closed=True, fill=True,alpha=0.25)
    ax.add_patch(rectangle)
    # plt.xlabel(r'$\tilde{\nu}$ [cm$^{-1}$]')
    if ctr==1:
        plt.ylabel(r'$B$ [r.u.]')
    plt.xlim([500,1850])
    plt.title(r'$s_{max}='+str(np.round(xmax*c,3))+'$ cm')
    plt.grid()
    # if ctr==len(clips):
        # plt.legend(draggable=True)
    
    inset_ax = inset_axes(ax, width="40%", height="30%", loc='upper right', borderpad=1)
    plt.plot(wnum,np.abs(B),'k',linewidth=1)
    plt.plot(wnum,np.abs(B_clip),'r',linewidth=1)
    plt.xlim([zoom[0],zoom[1]])
    plt.ylim([zoom[2],zoom[3]])
    plt.xticks([])
    plt.yticks([])
    
    plt.subplot(2,len(clips),ctr+len(clips))
    plt.plot(wnum_fine,H_real,'k',label=r'$s_{max}='+str(np.round(xmax,3))+'$ cm')
    plt.plot(wnum_fine,H_th,'r',alpha=1,label=r'Reduced $s_{max}$')
    # plt.plot(wnum_ds,np.real(Hk)*dx,'.r',label=r'Reduced $x_{max}$ (DFT)')
    
    plt.xlim([wnum_fine[0],wnum_fine[-1]])
    plt.xticks(np.arange(-5,5.1))
    plt.xlabel(r'$\tilde{\nu}$ [cm$^{-1}$]')
    if ctr==1:
        plt.ylabel(r'$H$ [cm]')
    plt.grid()
    ctr+=1

plt.legend(draggable=True)
plt.tight_layout()
