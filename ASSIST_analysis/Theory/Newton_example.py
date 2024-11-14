# -*- coding: utf-8 -*-

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
L=1000
mu_x=np.array([-1,3])#mean of state vector
sigma_x=np.array([1,1.5])#stdev of state vector
tol=10**-6#convergence tolerance
gamma=1#regularization factor

#%% Functions
def fun(x):
    y=np.zeros(3)
    
    y[0]=x[0]-x[1]**2
    y[1]=np.cos(x[0])+x[1]*3
    y[2]=np.sin(x[0])+x[1]
    return y

def Jacobian(x):
    K=np.zeros((3,2))
    
    K[0,0]=1
    K[0,1]=-2*x[1]
    K[1,0]=-np.sin(x[0])
    K[1,1]=3
    K[2,0]=np.cos(x[0])
    K[2,1]=1
    
    return K

#%% Initialization
x_est=[]
x_tar=[]
A=0
#%% main

for l in range(L):
    x=np.random.normal(mu_x,sigma_x)
    
    y=fun(x)
    X=np.zeros((2,2))
    res=np.zeros(2)+np.nan
    
    Sa=np.zeros((2,2))
    Sa[0,0]=sigma_x[0]
    Sa[1,1]=sigma_x[1]
    
    #%% Main
    for n in range(100):
        J=Jacobian(X[-1,:])
        B=np.linalg.inv(gamma*np.linalg.inv(Sa)+np.matmul(J.T,J))
        M=np.matmul(B,J.T)
        X=utl.vstack(X,mu_x+np.matmul(M,y-fun(X[-1,:])+np.matmul(J,X[-1,:]-mu_x)))
        res=np.append(res,np.sum((y-fun(X[-1,:]))**2))
        
        if np.max(np.abs(X[-1,:]-X[-2,:]))<tol:
            break
    
    A+=np.matmul(M,J)
    X=X[1:,:]
    res=res[1:]
    
    x_tar=utl.vstack(x_tar,x)
    x_est=utl.vstack(x_est,X[-1,:])
    

A=A/L

#%% Plots
plt.figure()
plt.bar(np.arange(2)-0.25,x,color='g',width=0.125,label='State vector')
plt.bar(np.arange(2)+0.25,mu_x,color='r',width=0.125,label='Prior')
plt.bar(np.arange(2),X[-1,:],color='k',width=0.125,label='Retrieval')
plt.xticks([0,1],labels=[r'$x_1$',r'$x_2$'])
plt.legend()
plt.title(r'$\gamma='+str(gamma)+'$')