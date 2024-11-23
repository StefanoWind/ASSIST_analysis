# -*- coding: utf-8 -*-
"""
Plot Voigt lineshape functions 
"""
import os
cd=os.path.dirname(__file__)
import sys
sys.path.append('C:/Users/SLETIZIA/OneDrive - NREL/Desktop/PostDoc/utils')
import utils as utl
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from scipy.special import wofz
import numpy as np
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 12


#%% Inputs
nu=np.arange(-50,50.1,1)#[cm^{-1}]
nu_cutoff=25#[cm^-1]
sigmas=[0.001,10,5]#[cm^-1]
gammas=[10,0.001,5]#[cm^-1]

colors=['r','b','k']
labels=['Pure collision','Pure Doppler','Voigt']

#%% Functions
def voigt(x, sigma, gamma):
    z = (x + 1j*gamma) / (sigma * np.sqrt(2))
    return wofz(z).real / (sigma * np.sqrt(2*np.pi))


#%% Plots
plt.figure()
for sigma,gamma,c,l in zip(sigmas,gammas,colors,labels):
    f=voigt(nu,sigma,gamma)
    f_cut=f.copy()
    f_cut[np.abs(nu)>nu_cutoff]=np.nan
    plt.plot(nu,f,'--',color=c)
    plt.plot(nu,f_cut,'k',color=c,label=l,linewidth=3)
    print(np.trapz(f,nu))
    
plt.xlabel(r'$\tilde{\nu}-\tilde{\nu}_i$ [cm$^{-1}$]')
plt.ylabel(r'$f(\tilde{\nu}-\tilde{\nu}_i)$')
plt.grid()
plt.tight_layout()
plt.legend()