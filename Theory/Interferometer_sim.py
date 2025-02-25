# -*- coding: utf-8 -*-
"""
Retrieval example following Rogers 2000 but for a ground-based IRS   
"""
import os
cd=os.path.dirname(__file__)
import sys
sys.path.append('C:/Users/SLETIZIA/OneDrive - NREL/Desktop/PostDoc/utils')
import utils as utl
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from scipy.linalg import sqrtm
import matplotlib.gridspec as gridspec
from scipy.signal import find_peaks
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 18




#%% Inputs

x=np.arange(-360*2,361*2,10)

t=np.arange(5000)
omega=1

#%% Main
ctr=0
for xx in x:
    
    plt.figure()
    
    plt.fill_between(t,t**0*1/2**0.5,t*0,color='k',alpha=0.5)
    plt.plot(t,utl.cosd(omega*t),'k')
    
    
    plt.fill_between(t,t**0*1/2**0.5-3,t*0-3,color='b',alpha=0.5)
    plt.plot(t,utl.cosd(omega*t+xx)-3,'b')
    
    amp=np.max(utl.cosd(omega*t)+utl.cosd(omega*t+xx))
    plt.fill_between(t,t**0*1/2**0.5*amp-6,t*0-6,color='r',alpha=0.5)
    plt.plot(t,utl.cosd(omega*t)+utl.cosd(omega*t+xx)-6,'r')
    plt.title('Phase difference: '+str(int(xx))+r'$^\circ$')
    
    plt.ylim([-8,1])
    plt.xticks([])
    plt.yticks([])
    
    plt.savefig('figures/video-interf/{i:03d}'.format(i=ctr))
    plt.close()
    ctr+=1