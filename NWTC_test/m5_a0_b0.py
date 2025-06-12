import os
cd=os.path.dirname(__file__)
import sys
import numpy as np
from matplotlib import pyplot as plt
import warnings
import matplotlib.dates as mdates
import matplotlib.cm as cm
from datetime import datetime
import glob
import xarray as xr
import re
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 12

warnings.filterwarnings('ignore')
plt.close('all')

#%% Inputs
if len(sys.argv)==1:
    source=os.path.join(cd,'data/nwtc/nwtc.m5.a0')#location of a0 files
    replace=False#replace old files?
    sdate='2022-01-01'#start date
    edate='2024-01-02'#end date
else:
    source=sys.argv[1]
    replace=sys.argv[2]=="True"
    sdate=sys.argv[3]
    edate=sys.argv[4]
    
max_nan=20#[%] maximum number of nan in a series
min_std=0.01#[dimensional] standard deviation of flat signal
N_despike=10#number of iteration of despiking
window=7#window of median filter [Brock et al, 1986]
max_spike=10#[%] maximum percentage of spikes
max_cons_spike=5#maximum number of consecutive spikes
max_max_diff_ratio=5#maximum relative difference of signal
perc_diff=95#[%] percentile to find representative gradient in data
override=['precip']

#graphics
date_fmt = mdates.DateFormatter('%H:%M')
cmap = cm.get_cmap('viridis')

#%% Functions
def consecutive(x,value=True):
    '''
    find maximum number of consecutive values
    '''
    groups=(x!=value).cumsum()
    groups[x!=value]=-1
    count=np.histogram(groups,bins=np.arange(-0.5,np.max(groups)+1))[0]
    
    if count.size>0:
        return np.max(count)
    else:
        return 0
    
def median_filter(data,window,max_MAD=5,p_value=0.16,N_bin=10):
    """
    Despiking based on median filter with adaptive threshold from Brock 1986
    
    """

    from scipy.stats import norm
    #additional inputs
    N_bin_norm=np.array([0.25,0.5,0.75,1])#ratio of max number of bins tested
    
    #initialization
    excl=data==-9999
 
    #median deviation
    data_med=xr.Dataset({
                v: data[v].rolling({'time': window}, center=True)
                             .construct('window')
                             .median('window')
                for v in data.data_vars
            })
    
    MAD=data-data_med
   
    #loop throuhg each time signal
    for v in MAD.data_vars:
        name_height=np.array(MAD[v].coords)[np.array(MAD[v].coords)!='time'][0]#identify name of height coord
        for h in MAD[v][name_height]:
            signal=MAD[v].sel({name_height:h})
            N=len(signal)
            H_min=-max_MAD
            H_max=max_MAD
            data_res=np.nanmedian(np.diff(np.unique(np.round(signal,10))))#data resolution
            if ~np.isnan(data_res)>0:
                for n_norm in N_bin_norm:#loop through increasing bin sizes
                    n=int(2*max_MAD/(data_res*N_bin)*n_norm)#number of bins to cover a multiple of data_res
                    if n/2==int(n/2):
                        n+=1
                    
                    #build histogram
                    H_y,H_x=np.histogram(signal,bins=n,range=[-max_MAD,max_MAD],density=False)
                    p=H_y/N
                    U_y=-norm.ppf(p_value)*(N*p*(1-p))**0.5#standard uncertainty
                    
                    #find island in left quadrant
                    H_x_left= -np.flip(H_x[:int(n/2)+1])
                    H_y_left=  np.flip(H_y[:int(n/2)+1])
                    U_y_left=(np.flip(U_y[1:int(n/2)+1])**2+np.flip(U_y[:int(n/2)])**2)**0.5
                    H_diff=np.diff(H_y_left)    
                    if H_min==-max_MAD:
                        i_min=np.where(H_diff>U_y_left)[0]
                        if len(i_min)>0:
                            H_min=-H_x_left[i_min[0]-1]
                          
                    #find island in right quadrant
                    H_x_right= H_x[int(n/2)+1:]
                    H_y_right= H_y[int(n/2):]
                    U_y_right=(U_y[int(n/2)+1:]**2+U_y[int(n/2):-1]**2)**0.5
                    H_diff=np.diff(H_y_right) 
                    if H_max==max_MAD:
                        i_min=np.where(H_diff>U_y_right)[0]
                        if len(i_min)>0:
                            H_max=H_x_right[i_min[0]-1]
                   
                    if H_min>-max_MAD and H_max<max_MAD:
                        break
                excl[v].loc[{name_height:h}]=(signal<H_min)+(signal>H_max)
                
    return excl

#%% Initialization

#read met data
files_all = np.array(sorted(glob.glob(os.path.join(source, '*.nc'))))
t_file=[]
for f in files_all:
    match = re.search(r'\d{8}\.\d{6}', f)
    t=datetime.strptime(match.group(0),'%Y%m%d.%H%M%S')
    t_file=np.append(t_file,t)

sel_t=(t_file>=datetime.strptime(sdate,'%Y-%m-%d'))*(t_file<datetime.strptime(edate,'%Y-%m-%d'))
files_sel=files_all[sel_t]

os.makedirs(source.replace('a0','b0'),exist_ok=True)

#%% Main
    
# calculate median 10-min std across all heights
data_std=xr.Dataset()
data_std_list=[]
for f in files_all:
    data = xr.open_dataset(f)
    data_std_list.append(data.std(dim='time'))
   
data_std = xr.concat(data_std_list, dim='dataset')
std_z=data_std.median()

#QC files
for f in files_sel:
    
    if replace==False and os.path.exists(f.replace('a0','b0'))==True:
        print(f+' skipped')
    else: 
        
        #load data
        data = xr.open_dataset(f)
        data_qc=data.copy()
    
        #nan ratio
        nans=np.isnan(data_qc).sum(dim='time')
        data_qc=data.where(nans<=max_nan)
    
        #flat signal
        data_qc.where(data_qc.std(dim='time')>min_std)
    
        #despiking
        excl_spike_all=xr.Dataset()
        for i_despike in range(N_despike):
            
            #identify spikes
            data_qc_norm=(data_qc-data_qc.median(dim='time'))/std_z
            excl_spike=median_filter(data_qc_norm, window)   
            
            #exclude and replace
            data_qc=data_qc.where(excl_spike==0)
            data_qc=data_qc.interpolate_na(dim='time', method='linear')
            
            #store spike stats
            if i_despike==0:
                excl_spike_all=excl_spike
            else:
                excl_spike_all+=excl_spike
               
            #check if there are no new spikes
            if int(excl_spike.sum(dim='time').to_array().max().values)==0:
                break
    
        #exclude residual spikes
        data_qc=data_qc.where(excl_spike.sum(dim='time')==0)
        
        #exclude total spikes above threshold 
        data_qc=data_qc.where(excl_spike_all.sum(dim='time')/len(data_qc.time)*100<=max_spike)
        
        #exclude consecutive spikes above threshold
        variables = [v for v in excl_spike_all.data_vars]  # force evaluation now
        cons_spikes = xr.Dataset({
                    v: xr.apply_ufunc(
                        consecutive,
                        excl_spike_all[v],
                        input_core_dims=[['time']],
                        kwargs={"value": True},
                        vectorize=True,
                        dask='parallelized',
                    )
                    for v in variables
                })
        
        data_qc=data_qc.where(cons_spikes<=max_cons_spike)
    
        #find linear trends (null 2nd derivative)
        diff_bw=data_qc-data_qc.shift(time=-1)
        diff_fw=data_qc.shift(time=1)-data_qc
        diff=(np.abs(diff_bw)+np.abs(diff_fw))/2
        diff=diff.where(diff>0)
        diff_diff=np.abs(diff_fw-diff_bw)
        diff_diff=diff_diff.where(diff>0)
        ramp=(diff_diff<10**-5)
        
        #sum difference where linear trend detected
        data_deramp=data_qc.copy()
        data_deramp=data_deramp.where(~ramp).bfill(dim='time')
        diff_merged_bw=data_deramp-data_deramp.shift(time=-1)
        diff_merged_fw=data_deramp.shift(time=1)-data_deramp
        diff_merged=(np.abs(diff_merged_bw)+np.abs(diff_merged_fw))/2
        
        #calculate maximum difference and compare to typical high difference across all heights
        max_diff=diff_merged.max(dim='time')
             
        high_diff = xr.Dataset({
                    v:  np.nanpercentile(diff[v],perc_diff)
                    for v in variables
                })
        
        #filter out channels with excessive jumps
        max_diff_ratio=max_diff/high_diff
        data_qc=data_qc.where(max_diff_ratio<=max_max_diff_ratio)
    
        #store output
        for v in override:
            data_qc[v]=data[v]
        data_qc.to_netcdf(f.replace('a0','b0'))
        
        #plots
        plt.close('all')
        date=str(data_qc.time.values[0])[:10]
        for v in variables:
            plt.figure(figsize=(18,8))
            name_height=np.array(data_qc[v].coords)[np.array(data_qc[v].coords)!='time'][0]#identify name of height coord
            for h in data_qc[name_height]:
                ax1=plt.subplot(2,1,1)
                ax1.set_facecolor([0,0,0,0.1])
                plt.plot(data.time,data[v].sel({name_height:h}),'.-',color=cmap(int(h)/120),markersize=2)
                plt.ylabel(v)
                plt.grid(True)
                plt.title(f'Raw selected M5 data on {date}')
                plt.gca().xaxis.set_major_formatter(date_fmt)
                ax2=plt.subplot(2,1,2)
                ax2.set_facecolor([0,0,0,0.1])
                plt.plot(data_qc.time,data_qc[v].sel({name_height:h}),'.-',label='$z='+str(h.values)+'$ m',color=cmap(int(h)/120),markersize=2)
                plt.ylabel(v)
                plt.grid(True)
                plt.title('Quality checked')
                plt.gca().xaxis.set_major_formatter(date_fmt)
            plt.tight_layout()
            plt.legend()
            
            plt.savefig(f.replace('a0','b0')[:-3]+'.'+v+'.png')
            plt.close()
                
        print(f+' done')

        
    

