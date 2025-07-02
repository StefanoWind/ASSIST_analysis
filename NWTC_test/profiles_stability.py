# -*- coding: utf-8 -*-
"""
Cluster profiles by atmospheric stability
"""
import os
cd=os.path.dirname(__file__)
import numpy as np
import sys
import xarray as xr
import matplotlib
from matplotlib import pyplot as plt
import yaml
import glob
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 14

#%% Inputs

source_stab=os.path.join(cd,'data/nwtc/nwtc.m5.c0/*nc')#source of met stats
source_waked=os.path.join(cd,'data/turbine_wakes.nc')

#user
unit='ASSIST11'#assist id

var_trp='temperature'
var_met='temperature'#selected temperature variable in M5 data

height_sel=119#[m]

stab_classes_uni=['S','NS','N','NU','U']

stab_classes={'S':[0,200],
            'NS':[200,500],
            'N1':[500,np.inf],
            'N2':[-np.inf,-500],
            'NU':[-500,-200],
            'U':[-200,0]}#stability classes from Obukhov length [Hamilton and Debnath, 2019]

#%% Initialization
Data_trp=xr.open_dataset(os.path.join(cd,'data',f'tropoe.{unit}.nc'))
Data_met=xr.open_dataset(os.path.join(cd,'data',f'met.b0.{unit}.nc'))

#read met data
files=glob.glob(source_stab)
met=xr.open_mfdataset(files)
L=met.L.sel(height_kin=height_sel)

#read wake data
waked=xr.open_dataset(source_waked)

#%% Main

#stability class
stab_class=xr.DataArray(data=['null']*len(L.time),coords={'time':L.time})

for s in stab_classes.keys():
    sel=(L>=stab_classes[s][0])*(L<stab_classes[s][1])
    if s=='N1' or s=='N2':
        s='N'
    stab_class=stab_class.where(~sel,other=s)
    
Data_trp['waked']=waked['Site 3.2'].interp(time=Data_trp.time)
f_trp=Data_trp[var_trp].where(Data_trp['waked'].sum(dim='turbine')==0)
print(f"{int(np.sum(Data_trp['waked'].sum(dim='turbine')>0))} wake events at Site 3.2 excluded")

Data_met['waked']=waked['M5'].interp(time=Data_met.time)
f_met=Data_met[var_met].where(Data_met['waked'].sum(dim='turbine')==0)
print(f"{int(np.sum(Data_met['waked'].sum(dim='turbine')>0))} wake events at M5 excluded")

for s in stab_classes_uni:
    sel=stab_class==s
    
    