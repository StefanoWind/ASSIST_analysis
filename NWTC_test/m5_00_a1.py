'''
Format raw M5 data into a1 netCDF
'''

import os
cd=os.path.dirname(__file__)
import scipy.io as spio
import matplotlib.pyplot as plt
import paramiko
from scp import SCPClient
import glob
import xarray as xr
import yaml
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime, timedelta
import os
import warnings
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['savefig.dpi'] = 300
warnings.filterwarnings('ignore')
plt.close('all')

#%% Inputs
path_config=os.path.join(cd,'configs/config.yaml')
source='Y:/Wind-data/Public/Projects/Met135/MetData/M5Twr'
sdate='2022-05-15'#[%Y-%m-%d] start date
edate='2022-08-25'#[%Y-%m-%d] end date
storage=os.path.join(cd,'data/nwtc/nwtc.m5.a1')#where to save
destination='/scratch/sletizia/ASSIST_analysis/NWTC_test/data/nwtc/nwtc.m5.a1'#storage location on Kestrel
replace=False#replace existing files?
send=False#send files to server?
delete=False#delete local files?

zero_datenum=719529#[days] 1970-01-01 in matlab time

#variables
variables={'air_temp':'Air_Temp_{z}m',
           'dewp_temp':'Dewpt_Temp_{z}m',
           'press':'Baro_Presr_{z}m',
           'precip':'Precip_TF',
           'ws':'Cup_WS_{z}m',
           'wd':'Vane_WD_{z}m'}

#units (in mat structures)
units={'air_temp':'C',
       'dewp_temp':'C',
       'press':'kPa',
       'precip':'unitless',
       'ws':'m/s',
       'wd':'degrees'}

#sensor heights a.g.l.
heights=[3,38,87,122]

oversample=20#overample raito in raw data

#graphics
date_fmt = mdates.DateFormatter('%H:%M')
cmap = matplotlib.cm.get_cmap('plasma')

       

#%% Functions
def extract_data(day,source,storage):
    files=sorted(glob.glob(os.path.join(source,day.strftime('%Y/%m/%d')+'/raw_data/*.mat')))
    
    for f in files:
        
        #comopse filename
        time_str = datetime.strptime(os.path.basename(f)[:-4], "%m_%d_%Y_%H_%M_%S_%f").strftime("%Y%m%d.%H%M%S")
        filename=f'nwtc.m5.a1.{time_str}.nc'
        
        if replace==False and os.path.exists(os.path.join(storage,filename))==True:
            print(f'{f} skipped')
        else:
            data=xr.Dataset()
            
            #read mat structure
            mat = spio.loadmat(f, struct_as_record=False, squeeze_me=True)
            
            #matlab datenum to numpy
            tnum=(mat['time_UTC'].val[::oversample]-zero_datenum)*24*3600
            time=np.datetime64('1970-01-01T00:00:00')+np.timedelta64(1,'ms')*np.int64(tnum*1000)
    
            #ingest variable
            for v in variables:
                var=np.zeros((len(time),len(heights)))+np.nan
                i=0
                for h in heights:
                    try:
                        var[:,i]=mat[variables[v].format(z=h)].val[::oversample]
                    except:
                        pass
                    i+=1
                
                data[v]=xr.DataArray(data=var,coords={'time':time,'height':heights},attrs={'units':units[v]})
                                       
            #reconstruct air temperature from delta T
            T_rec=np.zeros((len(time),len(heights)))+np.nan
            T_rec[:,0]=mat['Air_Temp_{z}m'.format(z=heights[0])].val[::oversample]
            for i in range(len(heights)-1):
                h1=heights[i]
                h2=heights[i+1]
                T_rec[:,i+1]=T_rec[:,i]+mat['DeltaT_{z2}_{z1}m'.format(z2=h2,z1=h1)].val[::oversample]
            
            data['air_temp_rec']=xr.DataArray(data=T_rec,coords={'time':time,'height':heights},attrs={'units':units[v]})
            
            
            
            #plots
            date=str(data.time.values[0])[:10]
            plt.figure(figsize=(18,8))
            ctr=1
            for v in data.data_vars:
                ax=plt.subplot(len( data.data_vars),1,ctr)
                ax.set_facecolor([0,0,0,0.1])
                for h in data.height:
                    
                    plt.plot(data.time,data[v].sel(height=h),'.-',color=cmap(int(h)/120),markersize=2,label=r'$z='+str(int(h))+'$ m')
                    plt.ylabel(v)
                    plt.grid(True)
                    if ctr==1:
                        plt.title(f'Raw selected M5 data on {date}')
                    plt.gca().xaxis.set_major_formatter(date_fmt)
                    plt.gca().xaxis.set_major_formatter(date_fmt)
                ctr+=1
            plt.tight_layout()
            plt.legend()
            plt.savefig(os.path.join(storage,filename).replace('nc','png'))
            plt.close()
                
            #output
            data=data.sortby('time').to_netcdf(os.path.join(storage,filename))
            print(f'{filename} created')
            
            #list remote files
            if send:
                stdin, stdout, stderr = ssh.exec_command(f'ls {destination}')
                transfered=np.array(stdout.read().decode().split('\n'))
    
                files=glob.glob(source)
    
                #transfer
                with SCPClient(ssh.get_transport()) as scp:
                   
                    if np.sum(filename==transfered)==0:
                        scp.put(os.path.join(storage,filename), remote_path=destination)  
                        scp.put(os.path.join(storage,filename).replace('nc','png'), remote_path=destination)  
                        print(f"{filename} sent")
                        if delete:
                            os.remove(os.path.join(storage,filename))
                            os.remove(os.path.join(storage,filename).replace('nc','png'))
                            print(f"{filename} deleted locally")
                    else:
                        print(f"{filename} skipped")

            
#%% Initialization
with open(path_config, 'r') as fid:
    config = yaml.safe_load(fid)
    
os.makedirs(storage,exist_ok=True)

start = datetime.strptime(sdate, '%Y-%m-%d')
end = datetime.strptime(edate, '%Y-%m-%d')
days = [start + timedelta(days=i) for i in range((end - start).days + 1)]

#connect to Kestrel
if send:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(config['host'], username=config['username'], password=config['password'])
    ssh.exec_command(f'mkdir -p {destination}')

#%% Main   
args = [(days[i],source,storage,config,send,delete,destination) for i in range(len(days))]

for d in days:
    extract_data(d, source,storage)

if send:
    ssh.close()
            


            
        