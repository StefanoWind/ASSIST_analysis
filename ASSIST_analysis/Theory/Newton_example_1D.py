# -*- coding: utf-8 -*-
'''
1D Newton algorithm example
'''

import os
cd=os.path.dirname(__file__)
import sys
sys.path.append('C:/Users/SLETIZIA/OneDrive - NREL/Desktop/PostDoc/utils')
import utils as utl

import numpy as np
from matplotlib import pyplot as plt
import warnings
import matplotlib

warnings.filterwarnings('ignore')
plt.close('all')

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm' 
matplotlib.rcParams['font.size'] = 12

#%% Inputs
tol=10**-6#convergence tolerance
N=6
Xa=1
As=[0.5,0.9,1]
y=-20

#%% Functions
def fun(x):
    
    y=-x**2-4*x+10

    return y

def derivative(x):
    df_dx=-2*x-4
    return df_dx

#%% Initialization
X=np.zeros(N)
X[0]=Xa

x=np.arange(-1,10,0.1)

#%% Main
plt.figure(figsize=(18,9))
ctr=0
for A in As:
    for n in range(1,N):
        X[n]=Xa*(1-A)+A*((derivative(X[n-1])**-1)*(y-fun(X[n-1]))+X[n-1])
        
        plt.subplot(len(As),N-1,n+ctr*(N-1))
        plt.plot(x,fun(x),'k')
        plt.plot(x,derivative(X[n-1])*(x-X[n-1])+fun(X[n-1]),'b')
        plt.plot(X[n-1],fun(X[n-1]),'xb')
        plt.plot(X[n],y,'xr')
        plt.plot(Xa,y,'or')
        plt.plot(x,x**0*y,'--r')
        if n==1:
            plt.ylabel(r'$A='+str(A)+'$')
        plt.ylim([-50,10])
        plt.title('Iter#'+str(n)+r': $X_n$='+str(np.round(X[n],1)))
        plt.grid()
    ctr+=1
    
plt.tight_layout()