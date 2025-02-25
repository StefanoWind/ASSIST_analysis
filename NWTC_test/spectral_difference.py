# -*- coding: utf-8 -*-
'''
Compare spectra 
'''
import os
cd=os.path.dirname(__file__)
import sys
import numpy as np
from matplotlib import pyplot as plt
import yaml
import xr
import glob
import matplotlib
from datetime import datetime

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 18

#%% Inputs
source_config=os.path.join(cd,'config.yaml')
sdate='20220515'
edate='20220801'
download=False
channels=['awaken/nwtc.assist.z02.00',
          'awaken/nwtc.assist.z03.00']

tropoe_bands= np.array([[612.0,618.0],
                        [624.0-660.0],
                        [674.0-713.0],
                        [713.0-722.0],
                        [538.0-588.0],
                        [793.0-804.0],
                        [860.1-864.0],
                        [872.2-877.5],
                        [898.2-905.4]])

#%% Initalization
#config
with open(source_config, 'r') as fid:
    config = yaml.safe_load(fid)
    
#imports
sys.path.append(config['path_utils'])
sys.path.append(config['path_dap'])
import utils as utl
from doe_dap_dl import DAP


#%% Main

if download:
    a2e = DAP('a2e.energy.gov',confirm_downloads=False)
    a2e.setup_basic_auth(username=config['username'], password=config['password'])
    for channel in channels:
        _filter = {
            'Dataset': channel,
            'date_time': {
                'between': [sdate,edate]
            },
            'file_type':'cha',
        }
        
        os.makedirs(os.path.join(cd,'data',channel),exist_ok=True)
        a2e.download_with_order(_filter, path=os.path.join(config['path_data'],channel),replace=False)
        
        


for channel in channels:
    files=glob.glob(os.path.join(cd,'data',channel,'*cdf'))
    
    for f in files:
        Data=xr.open_dataset(f)