# -*- coding: utf-8 -*-
"""
Check error on Tb due to instrumental bias
"""
import os
cd=os.getcwd()
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 12


#%% Inputs
k=1.380649*10**-23#[J/Kg] Boltzman's constant
h=6.62607015*10**-34#[J s] Plank's constant
c=299792458.0#[m/s] speed of light

err=0.01#relative error on radiance
T_amb=25#[C] ambient temperature

#%% Initialization
wnum=np.arange(500,1000,5)#wavenumber vector

#%% Main
c1=100*h*c*wnum/k
c2=2*10**11*c**2*h*wnum**3

B=c2/(np.exp(c1/(273.15+T_amb))-1)

Tb=c1/np.log(c2/B+1)-273.15

err_Tb=c1*np.log(c2/B+1)**(-2)*(c2/B+1)**(-1)*c2/B**2*(B*err)

err_Tb2=c1/np.log(c2/(B*(1+err/2))+1)-\
        c1/np.log(c2/(B*(1-err/2))+1)
        
#%% Plots
plt.figure()
plt.plot(wnum,err_Tb,'-k')
plt.plot(wnum,err_Tb2,'.r')
plt.xlabel(r'$\tilde{\nu}$ [cm $^{-1}$]')
plt.ylabel(r'$\sigma T_b$ [K]')
plt.grid()