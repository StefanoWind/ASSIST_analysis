'''
calculates 10-min stats of M5 QC'ed data
'''

import os
cd=os.path.dirname(__file__)
import sys
import numpy as np
import xarray as xr
import pandas as pd
from matplotlib import pyplot as plt
import warnings
from datetime import datetime, timedelta
import glob
from multiprocessing import Pool
import matplotlib.dates as mdates
import matplotlib.cm as cm
import yaml
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 12

warnings.filterwarnings('ignore')
plt.close('all')

#%% Inputs

path_config=os.path.join(cd,'configs/config.yaml')
if len(sys.argv)==1:
    source=os.path.join(cd,'data/nwtc/nwtc.m5.a1')#location of a0 files
    replace=True#replace old files?
    sdate='2022-05-15'#start date
    edate='2024-08-25'#end date
    mode="serial"
else:
    source=sys.argv[1]
    replace=sys.argv[2]=="True"
    sdate=sys.argv[3]
    edate=sys.argv[4]
    mode=sys.argv[5]
    
source_tilt=os.path.join(cd,'data/20220512.000000-20220809.000000_m5_tilt.csv')#source of tower tilt correction

R_a=287#[J/KgK] gas constant of dry air
g=9.81#[m/s^2] gravity acceleration
cp=1005#[J/KgK] specific heat of dry air
P_ref=101325#[Pa] reference pressure
wd_offset=8#[deg] wind direction offset from true North
min_data_avail=80#[%] minimum ratio of valid points

#graphics
date_fmt = mdates.DateFormatter('%H:%M')
cmap = cm.get_cmap('viridis')

#%% Functions
def vapor_pressure(Td):
    """
    Partial vapor pressure from dewpoint temperature
    """
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
    """
    Nan-free linear gradient
    """
    real=~np.isnan(y+x)
    if sum(real)>=min_real:
        LF=np.polyfit(x[real],y[real],1)
    else:
        LF=[np.nan,np.nan]
    return LF

def process_day(day,source,TILT,config):
    files=sorted(glob.glob(os.path.join(source,f'*{datetime.strftime(day,"%Y%m%d")}*nc')))
    if len(files)>0:
        output=xr.Dataset()
        filename=f'{os.path.basename(os.path.dirname(source.replace("b0","c0")))}.{datetime.strftime(day,"%Y%m%d")}.000000.nc'
        if os.path.isfile(os.path.join(source.replace("b0","c0"),filename)) and replace==False:
            print(f'{datetime.strftime(day,"%Y%m%d")} skipped')
        else:
            
            for f in files:
                data=xr.open_dataset(f)
            
                #initialize velocities
                u_corr=np.zeros(np.shape(data.u))
                v_corr=np.zeros(np.shape(data.v))
                w_corr=np.zeros(np.shape(data.w))
                
                u_rot=np.zeros(np.shape(data.u))
                v_rot=np.zeros(np.shape(data.v))
                w_rot=np.zeros(np.shape(data.w))
                
                #loop over heights
                for i_h in range(len(data.height_kin)):
                    
                    #extract data
                    h= data.height_kin.values[i_h]
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
                    
                    D=np.array([[np.cos(np.radians(pitch)), 0,np.sin(np.radians(pitch))],
                                [0,                         1,                       0],
                                [-np.sin(np.radians(pitch)),0,np.cos(np.radians(pitch))]])
                   
                    P=D.T@C.T
                    
                    UVWm=np.array([Um,Vm,Wm-bias])
                    UVWp=P@UVWm
                    
                    u_corr[:,i_h]=UVWp[0,:]
                    v_corr[:,i_h]=UVWp[1,:]
                    w_corr[:,i_h]=UVWp[2,:]
                    
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
                    
                data['u_corr']=xr.DataArray(data=u_corr,coords=data.u.coords)
                data['v_corr']=xr.DataArray(data=v_corr,coords=data.v.coords)
                data['w_corr']=xr.DataArray(data=w_corr,coords=data.w.coords)
                
                data['u_rot']=xr.DataArray(data=u_rot,coords=data.u.coords)
                data['v_rot']=xr.DataArray(data=v_rot,coords=data.v.coords)
                data['w_rot']=xr.DataArray(data=w_rot,coords=data.w.coords)
                
                #mean
                data_avg=data.mean(dim='time')
                data_avail=(~np.isnan(data)).sum(dim='time')/len(data.time)*100
                data_avg=data_avg.where(data_avail>config['min_data_avail'])
                
                #wind direction
                data_avg['wd']=(270-np.degrees(np.arctan2(data_avg['v'],data_avg['u']))+config['wd_offset'])%360
                
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
                data_avg['ti']=data_avg['uu_rot']**0.5/data_avg['u_rot']*100
                
                #pressure gradient
                e=vapor_pressure(data_avg['dewp_temp'])
                P_s=data_avg['press'].isel(height_therm=0)*100
                q_s=0.622*e.isel(height_therm=0)/P_s
                Tv_s=(data_avg['air_temp_rec'].isel(height_therm=0)+273.15)*(1+0.61*q_s)
                dP_dz=-config['g']*P_s/(config['R_a']*Tv_s) 
                
                #potential virtual temperature
                data_avg['press']=(P_s+(data.height_therm-data.height_therm[0])*dP_dz)/100
                q=0.622*e/(data_avg['press']*100)
                data_avg['Tv']=(data_avg['air_temp_rec']+273.15)*(1+0.61*q)
                data_avg['theta_v']= data_avg['Tv']*(config['P_ref']/(data_avg['press']*100))**(config['R_a']/config['cp'])
                
                #Obukhov length
                data_avg['u_star']=(data_avg['uw_rot']**2+data_avg['vw_rot']**2)**0.25
                data_avg['Tv_int']=data_avg['Tv'].interp(height_therm=data_avg.height_kin)
                data_avg['L']=-data_avg['u_star']**3/(config['kappa']*config['g'])*data_avg['Tv_int']/data_avg['wT_rot']
                
                #Gradient Richardson
                data_avg['theta_v_int']=data_avg['theta_v'].interp(height_therm=data_avg.height_kin)
                data_avg['du_corr_dz']=lin_fit(data_avg.height_kin,data_avg['u_corr'].values)[0]
                data_avg['dv_corr_dz']=lin_fit(data_avg.height_kin,data_avg['v_corr'].values)[0]
                data_avg['dtheta_v_dz']=lin_fit(data_avg.height_kin,data_avg['theta_v_int'].values)[0]
                data_avg['Ri']=config['g']/data_avg['theta_v'].mean()*data_avg['dtheta_v_dz']/(data_avg['du_corr_dz']**2+data_avg['dv_corr_dz']**2)
                
                #ouput
                data_avg=data_avg.expand_dims(time=[data.time.mean().values]) 
                
                if 'L' in output.data_vars:
                    output=xr.concat([output,data_avg], dim='time')
                else:
                    output=data_avg
                print(f'{f} done')
                
            #save output
            os.makedirs(source.replace("b0","c0"),exist_ok=True)
            output.to_netcdf(os.path.join(source.replace("b0","c0"),filename))
            
            #figure
            plt.figure(figsize=(18,10))
            plt.subplot(5,1,1)
            for h in output.height_kin:
                plt.plot(output.time,output['u_rot'].sel(height_kin=h),label='$z='+str(h.values)+'$ m',color=cmap(int(h)/120))
            plt.ylabel(r'$\overline{u}_w$ [m s$^{-1}$]')
            plt.grid()
            plt.ylim([0,25])
            plt.legend()
            
            plt.subplot(5,1,2)
            for h in output.height_kin:
                plt.plot(output.time,output['wd'].sel(height_kin=h),color=cmap(int(h)/120))
            plt.ylabel(r'$\overline{\theta}_w$ [s$^\circ$]')
            plt.grid()
            plt.ylim([0,360])
            plt.yticks([0,90,180,270,360])
            
            plt.subplot(5,1,3)
            for h in output.height_kin:
                plt.plot(output.time,output['ti'].sel(height_kin=h),color=cmap(int(h)/120))
            plt.ylabel('TI [%]')
            plt.grid()
            plt.ylim([0,100])
            
            plt.subplot(5,1,4)
            for h in output.height_kin:
                plt.plot(output.time,output['L'].sel(height_kin=h),color=cmap(int(h)/120))
            plt.ylabel(r'$L$ [m]')
            plt.gca().set_yscale('symlog')
            plt.grid()
            plt.ylim([-10**5,10**5])
            plt.yticks([-500,-200,0,200,500])
            
            plt.subplot(5,1,5)
            plt.plot(output.time,output['Ri'],label='$z='+str(h.values)+'$ m',color='k')
            plt.ylabel('Ri')
            plt.grid()
            plt.ylim([-10,10])
            plt.yticks([-0.25,-0.01,0,0.01,0.25])
            plt.gca().set_yscale('symlog')
            
            plt.savefig(os.path.join(source.replace("b0","c0"),filename).replace('.nc','.png'))
            plt.close()        
            

#%% Initialization

with open(path_config, 'r') as fid:
    config = yaml.safe_load(fid)

#read tilt
TILT=pd.read_csv(source_tilt)
TILT=TILT.set_index('z [m AGL]')

#days
start = datetime.strptime(sdate, '%Y-%m-%d')
end =   datetime.strptime(edate, '%Y-%m-%d')
days = [start + timedelta(days=i) for i in range((end - start).days + 1)]

#%% Main
#run processing
if mode=="serial":
    for day in days:
        process_day(day,source,TILT,config)
elif mode=="parallel":
    args = [(days[i],source,TILT,config) for i in range(len(days))]
    with Pool() as pool:
        pool.starmap(process_day, args)
else:
    print('Unknown processing mode')
    

    
            
            
       
