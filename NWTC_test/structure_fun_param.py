# -*- coding: utf-8 -*-
"""
Visualize structure function
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
import yaml
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['savefig.dpi'] = 500

warnings.filterwarnings('ignore')
plt.close('all')

#%% Inputs
source_met_sta=os.path.join(cd,'data/nwtc/nwtc.m5.c1/*nc')#source of met stats

us=[0.1,0.2,0.4,0.5]
Ls=[-100,-300,1000,300,100]
zs=[3,38,87,122]
min_ws=2

# DT_s=1

r=np.arange(500)
g=9.81
T0=20
k=0.41
z0=0.01

#%% Functions
def F(z_L):
    if z_L<=0:
        return 4.9*(1-7*z_L)**(-2/3)
    else:
        return 4.9*(1+2.4*(z_L)**(2/3))
    
def phi_h(z0_L):
    return 0.74*(1-9*z0_L)**(-0.5)

def psi_m(z_L):
    """Stability correction function for momentum (ψ_M)."""

    # Unstable case (z/L < 0)
    if z_L<0:
        x = (1 - 16 * z_L)**0.25
        psi= (
            2 * np.log((1 + x) / 2) +
            np.log((1 + x**2) / 2) -
            2 * np.arctan(x) +
            np.pi / 2
        )
    else:
        # Stable case (z/L > 0)
        beta = 5.0
        psi = -beta * z_L

    # Neutral case (z/L == 0): psi = 0, already set by zeros_like

    return psi

def psi_h(z_L):
    """Stability correction function for heat (ψ_H)."""
    
    if z_L<0:
        # Unstable case (z/L < 0)
        x = (1 - 16 * z_L)**0.25
        psi = 2 * np.log((1 + x**2) / 2)
    else:
        # Stable case (z/L > 0)
        beta = 5.0
        psi = -beta * z_L

    return psi

def f3(Ri):
    Ris=np.concatenate([np.arange(-2,-0.19,0.2),np.arange(-0.1,0.2,0.02)])
    f=np.array([3.62,
    3.50,
    3.37,
    3.22,
    3.06,
    2.89,
    2.68,
    2.44,
    2.14,
    1.75,
    1.48,
    1.42,
    1.35,
    1.27,
    1.19,
    1.09,
    0.81,
    0.63,
    0.50,
    0.39,
    0.30,
    0.22,
    0.15,
    0.098,
    0.051,
    0.015])
    
    return np.interp(Ri,Ris,f)
    


    
#%% Initialization
CT=np.zeros((len(us),len(Ls),len(zs)))

#read met stats
files=glob.glob(source_met_sta)
Data_met_sta=xr.open_mfdataset(files)

#%% Main

# i_u=0
# for u in us:
#     i_L=0
#     for L in Ls:
#         i_z=0
#         for z in zs:
#             # T_star=k*DT_s/(0.74*(np.log(z/z0)-psi_h(z/L)+2*phi_h(z0/L)/0.74))
#             T_star=-T0/(k*g)*u**2/L
#             CT[i_u,i_L,i_z]=T_star**2/z**(2/3)*F(z/L)
            
#             i_z+=1
#         i_L+=1
#     i_u+=1
    
height=Data_met_sta.height.values
time=Data_met_sta.time.values

#met stats synch
Data_met_sta=Data_met_sta.where(Data_met_sta.ws>min_ws)
 
D_T=Data_met_sta['D_air_temp_rec']
D_T_res=Data_met_sta['D_res_air_temp_rec']
r=(D_T.lag*Data_met_sta.ws).transpose('time','lag','height')

C_T=(D_T/r**(2/3)).where(D_T.lag>10).median(dim='lag')
C_T_res=(D_T_res/r**(2/3)).where(D_T.lag>10).median(dim='lag')

#gradients
dtheta_v_dz=np.zeros((len(time),len(height)-1))
dU_dz=np.zeros((len(time),len(height)-1))
z3=[]
i_h=0
for z1,z2 in zip(height[:-1],height[1:]):
    
    z3=np.append(z3,(z1*z2)**0.5)
    theta_v1=Data_met_sta.theta_v.sel(height=z1)
    theta_v2=Data_met_sta.theta_v.sel(height=z2)
    dtheta_v_dz[:,i_h]=(theta_v2-theta_v1)/(np.log(z2/z1)*z3[-1])
    
    U1=Data_met_sta.ws.sel(height=z1)
    U2=Data_met_sta.ws.sel(height=z2)
    dU_dz[:,i_h]=(U2-U1)/(np.log(z2/z1)*z3[-1])
    i_h+=1

dtheta_v_dz=xr.DataArray(dtheta_v_dz,coords={'time':time,'z3':z3})
dU_dz=xr.DataArray(dU_dz,coords={'time':time,'z3':z3})
z=dtheta_v_dz.z3
T=Data_met_sta.air_temp_rec.interp(height=z3).rename({'height':'z3'})+273.15
C_T_sca=(C_T.interp(height=z3).rename({'height':'z3'})/(z**(4/3)*dtheta_v_dz**2+10**-10))
Ri_z=9.81/T*dtheta_v_dz/(dU_dz**2+10**-10)

bin_Ri=np.nanpercentile(Ri_z.values.ravel(),np.arange(5,95,10))
f_avg=np.zeros((len(height),len(bin_Ri)-1))
for i_z in range(len(z)):
    i_Ri=0
    for Ri1,Ri2 in zip(bin_Ri[:-1],bin_Ri[1:]):
        sel_Ri=(Ri_z.isel(z3=i_z)>=Ri1)*(Ri_z.isel(z3=i_z)<Ri2)
      
        f_sel=np.log(C_T_sca.isel(z3=i_z).where(sel_Ri).values)
        f_avg[i_z,i_Ri]=np.exp(utl.filt_stat(f_sel, np.nanmean))
        i_Ri+=1
        print(i_Ri)
    

#%% Plots
# plt.figure(figsize=(18,8))
# for i_h in range(len(height)):
#     plt.subplot(2,2,i_h+1)
#     plt.loglog(r.isel(height=i_h),D_T.isel(height=i_h),'.b',alpha=0.1)
#     plt.plot(np.arange(500),np.arange(500)**(2/3),'--k')
#     plt.grid()
    
# plt.figure(figsize=(18,8))
# for i_h in range(len(height)):
#     ax=plt.subplot(2,2,i_h+1)
#     plt.plot(Ri,C_T.isel(height=i_h),'.b',alpha=0.01)
#     ax.set_xscale('symlog')
#     ax.set_yscale('log')
    
#     # plt.plot(np.arange(500),np.arange(500)**(2/3),'--k')
#     plt.grid()
    
    
    
# plt.figure(figsize=(18,8))
# for i_h in range(len(height)):
#     ax=plt.subplot(2,2,i_h+1)
#     plt.plot(ws.isel(height=i_h),C_T.isel(height=i_h),'.b',alpha=0.01)
#     # ax.set_xscale('symlog')
#     ax.set_yscale('log')
    
#     # plt.plot(np.arange(500),np.arange(500)**(2/3),'--k')
#     plt.grid()
    
plt.figure(figsize=(18,8))
for i_z in range(len(z)):
    ax=plt.subplot(2,2,i_z+1)
    plt.plot(Ri_z.isel(z3=i_z),C_T_sca.isel(z3=i_z),'.b',alpha=0.01)
    plt.plot((bin_Ri[:-1]+bin_Ri[1:])/2,f_avg[i_z,:],'.-r')
    plt.plot(np.arange(-10,10,0.01),f3(np.arange(-10,10,0.01)),'k')
    ax.set_xscale('symlog')
    ax.set_yscale('log')
    plt.xlim([-10,10])
    plt.grid()

