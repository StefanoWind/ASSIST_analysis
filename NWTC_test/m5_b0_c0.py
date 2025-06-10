'''
calculates 10-min stats of M5 QC'ed data
'''

import os
cd=os.path.dirname(__file__)
import numpy as np
import xarray as xr
import pandas as pd
from matplotlib import pyplot as plt
import warnings
import glob
import re
warnings.filterwarnings('ignore')
plt.close('all')

#%% Inputs
source=os.path.join(cd,'data/nwtc/nwtc.m5.b0/*nc')
source_tilt=os.path.join(cd,'data/20220512.000000-20220809.000000_m5_tilt.csv')
replace=False

z_s=3#[m] surface measurement height
R_a=287#[J/KgK] gas constant of dry air
g=9.81#[m/s^2] gravity acceleration
cp=1005#[J/KgK] specific heat of dry air
P_ref=101325#[Pa] reference pressure
wd_offset=8#[deg] wind direction offset from true North
min_data_avail=80#[%] minimum ratio of valid points


#%% Functions
def vapor_pressure(Td):
    #constants for T>=0 C
    A1=7.5
    B1=237.3
    
    #constants for T<0 C
    A2=9.5
    B2=265.5
    
    if Td.shape==():
        if Td>=0:
            e=6.11*10**((A1*Td)/(Td+B1))*100
        else:
            e=6.11*10**((A2*Td)/(Td+B2))*100
    else:
        e=Td.copy()+np.nan
        e.values[Td.values>=0]=6.11*10**((A1*Td[Td.values>=0])/(Td[Td.values>=0]+B1))*100
        e.values[Td.values<0]=6.11*10**((A2*Td[Td.values<0])/(Td[Td.values<0]+B2))*100
 
    return e

def lin_fit(x,y,min_real=2):
    #09/05/2023: finalized
    real=~np.isnan(y+x)
    if sum(real)>=min_real:
        LF=np.polyfit(x[real],y[real],1)
    else:
        LF=[np.nan,np.nan]
    return LF

#%% Initialization

files=glob.glob(source)
TILT=pd.read_csv(source_tilt)
TILT=TILT.set_index('z [m AGL]')

#%% Main
for f in files:
    data=xr.open_dataset(f)

    #tilt+rotation
    u_rot=np.zeros(np.shape(data.u))
    v_rot=np.zeros(np.shape(data.v))
    w_rot=np.zeros(np.shape(data.w))
    for i_h in range(len(data.height_kin)):
        h=data.u.height_kin.values[i_h]
        Um=data.u.isel(height_kin=i_h)
        Vm=data.v.isel(height_kin=i_h)
        Wm=data.w.isel(height_kin=i_h)

        pitch=TILT['Pitch [deg]'][h]
        roll=TILT['Roll [deg]'][h]
        bias=TILT['Bias [m/s]'][h]
    
        if np.isnan(pitch):
            pitch=0
        if np.isnan(roll):
            roll=0
        if np.isnan(bias):
            bias=0
            
        #tilt correction
        C=np.array([[1,                       0,                        0],
                    [0,np.cos(np.radians(roll)),-np.sin(np.radians(roll))],
                    [0,np.sin(np.radians(roll)), np.cos(np.radians(roll))]])
        
        D=np.array([[np.cos(np.radians(pitch)),0,np.sin(np.radians(pitch))],
                    [0,                        1,                       0],
                    [-np.sin(np.radians(pitch)),0,np.cos(np.radians(pitch))]])
       
        P=np.matmul(D.T,C.T)
        
        UVWm=np.array([Um,Vm,Wm-bias])
        UVWp=P@UVWm
        
        #rotation
        U_avg=np.nanmean(UVWp[0,:])
        V_avg=np.nanmean(UVWp[1,:])
        yaw=(270-np.degrees(np.arctan2(V_avg,U_avg)))%360
        M=np.array([[ np.cos(np.radians(270-yaw)),np.sin(np.radians(270-yaw)),0],
                    [-np.sin(np.radians(270-yaw)),np.cos(np.radians(270-yaw)),0],
                    [0,                     0,                       1]])
            
        UVW=M@UVWp
        
        u_rot[:,i_h]=UVW[0,:]
        v_rot[:,i_h]=UVW[1,:]
        w_rot[:,i_h]=UVW[2,:]
        
    data['u_rot']=xr.DataArray(data=u_rot,coords=data.u.coords)
    data['v_rot']=xr.DataArray(data=v_rot,coords=data.v.coords)
    data['w_rot']=xr.DataArray(data=w_rot,coords=data.w.coords)
    
    #mean
    data_avg=data.mean(dim='time')
    data_avail=(~np.isnan(data)).sum(dim='time')/len(data.time)*100
    data_avg=data_avg.where(data_avail>min_data_avail)
    
    data_avg['wd']=(270-np.degrees(np.arctan2(data_avg['v'],data_avg['u']))+wd_offset)%360
    
    #turbulent fluxes
    data_det=data-data_avg
    data_avg['uu_rot']=(data_det['u_rot']*data_det['u_rot']).mean(dim='time')
    data_avg['vv_rot']=(data_det['v_rot']*data_det['v_rot']).mean(dim='time')
    data_avg['ww_rot']=(data_det['w_rot']*data_det['w_rot']).mean(dim='time')
    data_avg['uv_rot']=(data_det['u_rot']*data_det['v_rot']).mean(dim='time')
    data_avg['uw_rot']=(data_det['u_rot']*data_det['w_rot']).mean(dim='time')
    data_avg['vw_rot']=(data_det['v_rot']*data_det['w_rot']).mean(dim='time')
    data_avg['uT_rot']=(data_det['u_rot']*data_det['T']).mean(dim='time')
    data_avg['vT_rot']=(data_det['v_rot']*data_det['T']).mean(dim='time')
    data_avg['wT_rot']=(data_det['w_rot']*data_det['T']).mean(dim='time')
    
    data_avg['ti']=data_avg['uu_rot']**0.5/data_avg['u']*100
    
    
    #pressure gradient
    e=vapor_pressure(data_avg['dewp_temp'])
    q_s=0.622*e.isel(height_therm=0)/(data_avg['press'].isel(height_therm=0)*100)
    Tv_s=(data_avg['air_temp_rec'].isel(height_therm=0)+273.15)*(1+0.61*q_s)
    dP_dz=-g*data_avg['press'].isel(height_therm=0)*100/(R_a*Tv_s) 
    
    #potential virtual temperature
    data_avg['press']=data_avg['press'].isel(height_therm=0)+(data.height_therm-data.height_therm[0])*dP_dz/100
    q=0.622*e/(data_avg['press']*100)
    data_avg['Tv']=(data_avg['air_temp_rec']+273.15)*(1+0.61*q)
    data_avg['theta_v']= data_avg['Tv']*(P_ref/(data_avg['press']*100))**(R_a/cp)
    
    #Obukhov length
    data_avg['u_star']=(data_avg['uw_rot']**2+data_avg['vw_rot']**2)**0.25
    data_avg['theta_v_int']=data_avg['theta_v'].interp(height_therm=data_avg.height_kin)
    data_avg['L']=-data_avg['theta_v_int']*data_avg['u_star']**3/(0.41*data_avg['wT_rot'])
    
    # data_avg['time']=xr.dataarray(data=data.time.mean())
    
    plt.figure(figsize=(18,8))
    plt.subplot(1,4,1)
    plt.plot(data_avg['u_rot'],data_avg.height_kin,'.k',markersize=10)
    plt.xlabel(r'$\overline{U}_w$ [m s$^{-1}$]')
    plt.ylabel(r'$z$ [m a.g.l.]')
    plt.ylim([0,120])
    plt.xlim([0,25])
    plt.grid()
    plt.text(1,10,s=F'Time: {str(data.time.mean().values).replace("T"," ")[:19]}')
    
    plt.subplot(1,4,2)
    plt.plot(data_avg['wd'],data_avg.height_kin,'.k',markersize=10)
    plt.xlabel(r'$\overline{\theta}_w$ [$^\circ$]')
    plt.ylim([0,120])
    plt.xlim([0,360])
    plt.grid()
    
    plt.subplot(1,4,3)
    plt.plot(data_avg['ti'],data_avg.height_kin,'.k',markersize=10)
    plt.xlabel(r'TI [%]')
    plt.ylim([0,120])
    plt.xlim([0,50])
    plt.grid()
    
    plt.subplot(1,4,4)
    plt.plot(data_avg['L'],data_avg.height_kin,'.k',markersize=10)
    plt.xlabel(r'$L$ [%]')
    plt.ylim([0,120])
    plt.xlim([-10**6,10**6])
    plt.gca().set_xscale('symlog')
    plt.grid()
    
    os.makedirs(os.path.basename(f.replace('b0', 'c0')),exist_ok=True)
    plt.savefig(f.replace('b0', 'c0').replace('nc','png'))
    plt.close()
    

