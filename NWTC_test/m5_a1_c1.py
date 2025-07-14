'''
calculates 10-min stats of M5 QC'ed data
'''

import os
cd=os.path.dirname(__file__)
import sys
import numpy as np
import xarray as xr
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
matplotlib.rcParams['savefig.dpi'] = 300

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

#qc
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


def process_day(day,source,config):
    files=sorted(glob.glob(os.path.join(source,f'*{datetime.strftime(day,"%Y%m%d")}*nc')))
    if len(files)>0:
        output=xr.Dataset()
        filename=f'{os.path.basename(os.path.dirname(source.replace("a1","c1")))}.{datetime.strftime(day,"%Y%m%d")}.000000.nc'
        if os.path.isfile(os.path.join(source.replace("a1","c1"),filename)) and replace==False:
            print(f'{datetime.strftime(day,"%Y%m%d")} skipped')
        else:
            
            for f in files:
                data=xr.open_dataset(f)
            
                #data availability
                data_avail=(~np.isnan(data)).sum(dim='time')/len(data.time)*100
                
                #mean
                data_avg=data.mean(dim='time')
                data_avg=data_avg.where(data_avail>config['min_data_avail'])
                
                #std
                data_std=data.std(dim='time')
                data_std=data_std.where(data_avail>config['min_data_avail'])
                
                #pressure gradient
                e=vapor_pressure(data_avg['dewp_temp'])
                P_s=data_avg['press'].isel(height=0)*100
                q_s=0.622*e.isel(height=0)/P_s
                Tv_s=(data_avg['air_temp_rec'].isel(height=0)+273.15)*(1+0.61*q_s)
                dP_dz=-config['g']*P_s/(config['R_a']*Tv_s) 
                
                #potential virtual temperature
                data_avg['press']=(P_s+(data.height-data.height[0])*dP_dz)/100
                q=0.622*e/(data_avg['press']*100)
                data_avg['Tv']=(data_avg['air_temp_rec']+273.15)*(1+0.61*q)
                data_avg['theta_v']= data_avg['Tv']*(config['P_ref']/(data_avg['press']*100))**(config['R_a']/config['cp'])
                
                #Richardson number
                data_avg['um']=data_avg['ws']*np.cos(np.radians(270-data_avg['wd']))
                data_avg['vm']=data_avg['ws']*np.sin(np.radians(270-data_avg['wd']))
                height=data_avg.height.values
                for h1 in height:
                    for h2 in height[height>h1]:
                        theta_v_avg=(data_avg['theta_v'].sel(height=h2)+data_avg['theta_v'].sel(height=h1))/2
                        dtheta_v_dz=(data_avg['theta_v'].sel(height=h2)-data_avg['theta_v'].sel(height=h1))/(h2-h1)
                        dum_dz=     (data_avg['um'].sel(height=h2)-     data_avg['um'].sel(height=h1))/(h2-h1)
                        dvm_dz=     (data_avg['vm'].sel(height=h2)-     data_avg['vm'].sel(height=h1))/(h2-h1)
                        data_avg[f'Ri_{h1}_{h2}']=config['g']/theta_v_avg*dtheta_v_dz/(dum_dz**2+dvm_dz**2)

                #ouput
                data_avg=data_avg.expand_dims(time=[data.time.mean().values]) 
                
                if 'Tv' in output.data_vars:
                    output=xr.concat([output,data_avg], dim='time')
                else:
                    output=data_avg
                print(f'{f} done')
                
 
            #save output
            os.makedirs(source.replace("a1","c1"),exist_ok=True)
            output.to_netcdf(os.path.join(source.replace("a1","c1"),filename))
            
            #figure
            height=data_avg.height.values
            plt.figure(figsize=(18,10))
            plt.subplot(4,1,1)
            for h in height:
                plt.plot(output.time,output['ws'].sel(height=h),color=cmap(int(h)/120))
            plt.ylabel(r'$\overline{u}_w$ [m s$^{-1}$]')
            plt.grid()
            plt.ylim([0,25])
            plt.legend()
            plt.title(f'M5 stats on {datetime.strftime(day,"%Y%m%d")}')
            plt.gca().xaxis.set_major_formatter(date_fmt)
            
            plt.subplot(4,1,2)
            for h in height:
                plt.plot(output.time,output['wd'].sel(height=h),color=cmap(int(h)/120))
            plt.ylabel(r'$\overline{\theta}_w$ [s$^\circ$]')
            plt.grid()
            plt.ylim([0,360])
            plt.yticks([0,90,180,270,360])
            plt.gca().xaxis.set_major_formatter(date_fmt)
            
            plt.subplot(4,1,3)
            for h in height:
                plt.plot(output.time,output['air_temp_rec'].sel(height=h),color=cmap(int(h)/120))
            plt.ylabel(r'$T$ [$^\circ$C]')
            plt.grid()
            plt.ylim([0,40])
            plt.gca().xaxis.set_major_formatter(date_fmt)
            
            plt.subplot(4,1,4)
            plt.plot(output.time,output['Ri_38_122'],label='$z='+str(h)+'$ m',color='k')
            plt.ylabel('Ri')
            plt.grid()
            plt.ylim([-100,100])
            plt.yticks([-0.25,-0.01,0,0.01,0.25])
            plt.legend()
            plt.gca().set_yscale('symlog')
            plt.gca().xaxis.set_major_formatter(date_fmt)
            
            plt.savefig(os.path.join(source.replace("a1","c1"),filename).replace('.nc','.png'))
            plt.close()        
            

#%% Initialization

with open(path_config, 'r') as fid:
    config = yaml.safe_load(fid)


#days
start = datetime.strptime(sdate, '%Y-%m-%d')
end =   datetime.strptime(edate, '%Y-%m-%d')
days = [start + timedelta(days=i) for i in range((end - start).days + 1)]

#%% Main
#run processing
if mode=="serial":
    for day in days:
        process_day(day,source,config)
elif mode=="parallel":
    args = [(days[i],source,config) for i in range(len(days))]
    with Pool() as pool:
        pool.starmap(process_day, args)
else:
    print('Unknown processing mode')
    

    
            
            
       
