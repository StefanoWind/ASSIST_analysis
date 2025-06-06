import os
cd=os.path.dirname(__file__)
    
import pandas as pd
import numpy as np
import utils as utl
from matplotlib import pyplot as plt
import warnings
import matplotlib.dates as mdates
import matplotlib.cm as cm
from datetime import datetime
import glob
import re
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 12

warnings.filterwarnings('ignore')
plt.close('all')

#%% Inputs
source=os.path.join(cd,'data/nwtc/nwtc.m5.a0')
replace=False
sdate='2024-01-01'
edate='2024-01-02'

max_nan=20#[%] maximum number of nan in a series
min_std=0.01#[dimensional] standard deviation of flat signal
N_despike=10#number of iteration of despiking
window=7#window of median filter [Brock et al, 1986]
max_spike=10#[%] maximum percentage of spikes
max_cons_spike=5#maximum number of consecutive spikes
max_max_diff_ratio=10#maximum relative difference of signal
perc_diff=95#[%] percentile to find representative gradient in data

#graphics
date_fmt = mdates.DateFormatter('%H-%M')
cmap = cm.get_cmap('viridis')

#%% Functions
def consecutive(Data,value):
    #09/08/2023: finalized
    max_consecutive=pd.Series()
    for c in Data.columns:
        Data_diff=pd.DataFrame()
        Data_diff['group']= (Data[c]< Data[c].shift()).cumsum()
        Data_diff['group'].iloc[Data[c].values!=value]=np.nan
        max_consecutive[c] = Data_diff.groupby('group').size().max()
    
    return max_consecutive

def med_smooth(Data,window):
    #08/03/2023: created, finalized
    Data_smooth1=Data.rolling(window,center=False).median()#backward window
    Data_smooth2=Data.iloc[::-1].rolling(window,center=False).median().iloc[::-1]#backward window
    Data_smooth=Data.rolling(window,center=True).median()#forward window
    
    Data_smooth.iloc[:int(np.floor(window/2)),:]= Data_smooth2.iloc[:int(np.floor(window/2)),:]
    Data_smooth.iloc[-int(np.floor(window/2)):,:]=Data_smooth1.iloc[-int(np.floor(window/2)):,:]
    
    return Data_smooth

def median_filter_v4(Data,window,max_MAD=5,p_value=0.16,N_bin=10):
    #median despiking [Brock 1986]
    #Data: data frame
    #window: number of point used for rolling normalization
    #max_MAD: maximum deviation from median expected
    
    #11/01/2022: created
    #11/04/2022: finalized
    #11/18/2022: embedded normalization
    #12/07/2022: added minimum number of points, rolling window centered
    #08/03/2023: added edge median, finalized
    #08/29/2023 (v 2): using bin edge instead of center
    #09/11/2023 (v 3): added uncertianty of histogram, finalized
    #09/12/2023 (v 4): added data resolution
    #09/13/2023: finalized
    
    from scipy.stats import norm
    #inputs
    N_bin_norm=np.array([0.25,0.5,0.75,1])#ratio of max number of bins tested
    
    excl=pd.DataFrame()
    H_min=pd.Series()
    H_max=pd.Series()
    
    #median deviation
    MAD=Data-med_smooth(Data, window)
    for c in MAD.columns:
        H_min[c]=-max_MAD
        H_max[c]=max_MAD
        data_res=np.nanmedian(np.diff(np.unique(np.round(MAD[c],10))))
        if ~np.isnan(data_res)>0:
            for n_norm in N_bin_norm:
                n=int(2*max_MAD/(data_res*N_bin)*n_norm)
                if n/2==int(n/2):
                    n+=1
                H_y,H_x=np.histogram(MAD[c],bins=n,range=[-max_MAD,max_MAD],density=False)
                p=H_y/N
                U_y=-norm.ppf(p_value)*(N*p*(1-p))**0.5
                
                #left quadrant
                H_x_left= -np.flip(H_x[:int(n/2)+1])
                H_y_left=  np.flip(H_y[:int(n/2)+1])
                U_y_left=(np.flip(U_y[1:int(n/2)+1])**2+np.flip(U_y[:int(n/2)])**2)**0.5
                H_diff=np.diff(H_y_left)    
                if H_min[c]==-max_MAD:
                    i_min=np.where(H_diff>U_y_left)[0]
                    if len(i_min)>0:
                        H_min[c]=-H_x_left[i_min[0]-1]
                      
                #right quadrant
                H_x_right= H_x[int(n/2)+1:]
                H_y_right= H_y[int(n/2):]
                U_y_right=(U_y[int(n/2)+1:]**2+U_y[int(n/2):-1]**2)**0.5
                H_diff=np.diff(H_y_right) 
                if H_max[c]==max_MAD:
                    i_min=np.where(H_diff>U_y_right)[0]
                    if len(i_min)>0:
                        H_max[c]=H_x_right[i_min[0]-1]
               
                if H_min[c]>-max_MAD and H_max[c]<max_MAD:
                    break
    
    for c in MAD.columns:
        excl[c]=(MAD[c]<H_min[c])+(MAD[c]>H_max[c])

    return excl,H_min,H_max,MAD

#%% Initialization

#read met data
files_all = np.array(sorted(glob.glob(os.path.join(source, '*.csv'))))
t_file=[]
for f in files_all:
    match = re.search(r'\d{8}\.\d{6}', f)
    t=utl.datenum(match.group(0),'%Y%m%d.%H%M%S')
    t_file=np.append(t_file,t)

sel_t=(t_file>=utl.datenum(sdate,'%Y-%m-%d'))*(t_file<utl.datenum(edate,'%Y-%m-%d'))
files_sel=files_all[sel_t]
Output=pd.DataFrame()

utl.mkdir(source.replace('a0','b0'))

# calculate std
Data_std=pd.DataFrame()
for f in files_all:
    D = pd.read_csv(f)
    D=D.set_index('Time (UTC)').drop(columns='Timestamp')
    Data_std=Data_std.append(D.std(),ignore_index=True)

#variable extraction
variables=[]
heights=[]
for c in Data_std.columns:
    variables.append(re.split(r'(\d+)', c)[0])
    try:
        heights=np.append(heights,re.split(r'(\d+)', c)[1])
    except:
        heights=np.append(heights,np.nan)
        
std_z=Data_std.median()*0
for v in np.unique(variables):
    sel=[v==v_i for v_i in variables]
    std_z[sel]=np.nanmedian(Data_std.median()[sel])
   

#%% Main
    
for f in files_sel:
    try:
        if replace==False and os.path.exists(f.replace('a0','b0'))==True:
            print(f+' skipped')
        else: 
            Data = pd.read_csv(f)
            Data=Data.set_index('Time (UTC)')
            N=len(Data)
            Data_qc=Data.drop(columns='Timestamp')
        
            #nan ratio
            nans=np.isnan(Data_qc).sum()/N*100
            excl_nan=np.where(nans>=max_nan)[0]
            Data_qc.iloc[:,excl_nan]=np.nan
        
            #flat signal
            excl_flat=np.where(Data_qc.std()<min_std)[0]
            Data_qc.iloc[:,excl_flat]=np.nan
        
            #despiking
            N_excl_spike=pd.DataFrame()
            excl_spike_all=pd.DataFrame()
            for i_despike in range(N_despike):
                Data_qc_norm=(Data_qc-Data_qc.median())/std_z
                excl_spike,H_min,H_max,MAD=median_filter_v4(Data_qc_norm, window)   
                Data_qc[excl_spike]=np.nan
                Data_qc=Data_qc.interpolate()
                
                N_excl_spike=N_excl_spike.append(excl_spike.sum()/N*100,ignore_index=True)
                
                if i_despike==0:
                    excl_spike_all=excl_spike
                else:           
                    excl_spike_all=excl_spike_all+excl_spike
                if excl_spike.sum().max()==0:
                    break
        
            #exclude residual spikes
            excl_spike2=np.where(excl_spike.sum()>0)[0]
            Data_qc.iloc[:,excl_spike2]=np.nan
            
            #exclude total spikes above threshold 
            excl_spike3=np.where(N_excl_spike.sum()>max_spike)[0]
            Data_qc.iloc[:,excl_spike3]=np.nan
            
            #exclude consecutive spikes above threshold
            cons_spikes=consecutive(excl_spike_all,True)
            excl_spike4=np.where(cons_spikes>max_cons_spike)[0]
            Data_qc.iloc[:,excl_spike4]=np.nan
        
            #ramp filter
            diff=Data_qc.diff().abs()
            diff[diff==0]=np.nan
            
            #find linear trends (null 2nd derivative)
            diff_diff_bw=diff.diff().abs()
            diff_diff_fw=diff_diff_bw.copy()*np.nan
            diff_diff_fw.iloc[0:-1,:]=diff_diff_bw.iloc[1:,:]
            multi_step=(diff_diff_fw<10**-10)
            
            #sum difference where linear trend detected
            Data_merged=Data_qc.copy()
            Data_merged[multi_step==1]=np.nan
            diff_merged=diff.copy()*np.nan
            for c in Data_merged.columns:
                sel=~np.isnan(Data_merged[c].values)
                diff_merged[c][sel]=Data_merged[c][sel].diff().abs()

            max_diff=diff_merged.max()
            high_diff=np.nanpercentile(diff,perc_diff,axis=0)
            max_diff_ratio=max_diff/high_diff
        
            excl_max_diff=np.where(max_diff_ratio>max_max_diff_ratio)[0]
            Data_qc.iloc[:,excl_max_diff]=np.nan
        
            #output
            Data_qc['Timestamp']=Data['Timestamp']
            Data_qc.to_csv(f.replace('a0','b0'))
            
            #plots
            plt.close('all')
            time_plot=[datetime.utcfromtimestamp(t) for t in Data['Timestamp'].values]
            
            for v in np.unique(variables):
                if v!='Timestamp':
                    plt.figure(figsize=(18,8))
                    sel=np.where([v==v_i for v_i in variables])[0]
                    for s,z  in zip(sel,heights[sel]):
                        ax1=plt.subplot(2,1,1)
                        ax1.set_facecolor([0,0,0,0.1])
                        plt.plot(time_plot,Data.iloc[:,s].values,'.-',label='$z='+str(z)+'$ m (AGL)',color=cmap(int(z)/120),markersize=2)
                        plt.ylabel(v[:-1])
                        plt.grid(True)
                        plt.title('Raw selected M5 data on '+utl.datestr(Data['Timestamp'].values[0],'%Y%m%d'))
                        plt.gca().xaxis.set_major_formatter(date_fmt)
                        ax2=plt.subplot(2,1,2)
                        ax2.set_facecolor([0,0,0,0.1])
                        plt.plot(time_plot,Data_qc.iloc[:,s].values,'.-',label='$z='+str(z)+'$ m (AGL)',color=cmap(int(z)/120),markersize=2)
                        plt.ylabel(v[:-1])
        
                        plt.grid(True)
                        plt.title('Quality checked')
                        plt.gca().xaxis.set_major_formatter(date_fmt)
                    plt.tight_layout()
                    plt.legend()
                    
                    plt.savefig(f.replace('a0','b0')[:-4]+'_'+v[:-1]+'_qc.png')
                    
            print(f+' done')

    except Exception as e:
        print('Error at '+f+': '+str(e))
        pass
    
                
        
    

