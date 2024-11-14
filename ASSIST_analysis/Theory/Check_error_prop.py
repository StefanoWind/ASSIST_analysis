# -*- coding: utf-8 -*-
"""
Check math of linear error propagation of radiometric calibration
"""


import os
cd=os.getcwd()
import sys
sys.path.append('C:/Users/SLETIZIA/OneDrive - NREL/Desktop/PostDoc/utils')
import numpy as np
import matplotlib


matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 18

#%% Inputs
BH=10#HBB energy
BC=8#CBB energy
DH=1000#HBB FT of igram
DC=600#CBB FT of igram
D=1000#sky FT of igram

M=10000#laucnes

#%% Initalization
sigmas=np.random.rand(5)

R=((BH-BC)/(DH-DC))**-1
Delta=(D-DC)/(DH-DC)

#%% Main
BH_sim=np.random.normal(BH,sigmas[0],M)
BC_sim=np.random.normal(BC,sigmas[1],M)
DH_sim=np.random.normal(DH,sigmas[2],M)
DC_sim=np.random.normal(DC,sigmas[3],M)
D_sim=np.random.normal(D,sigmas[4],M)

B=(BH-BC)/(DH-DC)*(D-DC)+BC
B_sim=(BH_sim-BC_sim)/(DH_sim-DC_sim)*(D_sim-DC_sim)+BC_sim

err_std=np.std(B_sim)
err_std2=(Delta**2*sigmas[0]**2+(1-Delta)**2*sigmas[1]**2+1/R**2*Delta**2*sigmas[2]**2+1/R**2*(Delta-1)**2*sigmas[3]+1/R**2*sigmas[4]**2)**0.5

