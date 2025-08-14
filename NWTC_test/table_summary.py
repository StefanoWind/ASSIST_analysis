# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Summary of TROPoe stats
"""
import os
cd=os.path.dirname(__file__)
import numpy as np
import xarray as xr
import matplotlib
import yaml
import glob
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 14

#%% Inputs
source_config=os.path.join(cd,'configs','config.yaml')
var_sel=['sigma_temperature','vres_temperature','cdfs_temperature']
unit='ASSIST11'

heights=[10,100,1000]#[m] selected heights

#%% Functions
def to_latex(arr, align="c"):
    ncols = arr.shape[1]
    header = "\\begin{tabular}{" + ("|".join([align] * ncols)) + "}"
    lines = [header, "\\hline"]
    for row in arr:
        line = " & ".join(map(str, row)) + r" \\ \hline"
        lines.append(line)
    lines.append("\\end{tabular}")
    return "\n".join(lines)

#%% Initialization
#config
with open(source_config, 'r') as fid:
    config = yaml.safe_load(fid)

#%% Main

#load tropoe data
files=glob.glob(config['sources_trp'][unit])
Data_trp=xr.open_mfdataset(files)

#qc tropoe data
Data_trp['cbh'][(Data_trp['lwp']<config['min_lwp']).compute()]=Data_trp['height'].max()+1000#remove clouds with low lwp

qc_gamma=Data_trp['gamma']<=config['max_gamma']
qc_rmsa=Data_trp['rmsa']<=config['max_rmsa']
qc_cbh=Data_trp['height']<Data_trp['cbh']
qc=qc_gamma*qc_rmsa*qc_cbh
Data_trp['qc']=~qc+0
    
print(f'{np.round(np.sum(qc_gamma).values/qc_gamma.size*100,1)}% retained after gamma filter', flush=True)
print(f'{np.round(np.sum(qc_rmsa).values/qc_rmsa.size*100,1)}% retained after rmsa filter', flush=True)
print(f'{np.round(np.sum(qc_cbh).values/qc_cbh.size*100,1)}% retained after cbh filter', flush=True)

#interpolate height
Data_trp=Data_trp.assign_coords(height=Data_trp.height*1000+config['height_assist'])
Data_trp=Data_trp.where(Data_trp['qc']==0)
Data_trp=Data_trp[var_sel].interp(height=heights)

#compile summary table
mat=np.zeros((len(heights),len(var_sel)+1))
mat[:,0]=(~np.isnan(Data_trp.sigma_temperature)).mean(dim='time').values*100
mat[:,1]=Data_trp.sigma_temperature.mean(dim='time').values
mat[:,2]=Data_trp.vres_temperature.mean(dim='time').values*1000
mat[:,3]=Data_trp.cdfs_temperature.mean(dim='time').values

matrix = to_latex(mat)
print(matrix)

