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
Xa=np.array([-1,3])#mean of state vector
sigma_x=np.array([1,1.5])#stdev of state vector
sigma_y=np.array([1,1,1])*0.01
tol=10**-6#convergence tolerance
gamma=10#regularization factor
N=5


vmin=-50
vmax=50

#%% Functions
def fun(x,omega=0.25):
    y=[[] for i in range(3)]
    
    y[0]=x[0]-x[1]**2
    y[1]=np.cos(x[0]*omega)+x[1]*3
    y[2]=np.sin(x[0]*omega)+x[1]
    return y

def Jacobian(x,omega=0.25):
    K=np.zeros((3,2))
    
    K[0,0]=1
    K[0,1]=-2*x[1]
    K[1,0]=-np.sin(x[0]*omega)*omega
    K[1,1]=3
    K[2,0]=np.cos(x[0]*omega)*omega
    K[2,1]=1
    
    return K

#%% Initialization
x_est=[]
x_tar=[]
A=0

#%% Main
x=np.random.normal(Xa,sigma_x)

y=fun(x)+np.random.normal(sigma_y*0,sigma_y)
X=np.zeros((2,2))
X[0,0]=Xa[0]
X[0,1]=Xa[1]  
      
res=np.zeros(2)+np.nan

Sa=np.zeros((2,2))
Sa[0,0]=sigma_x[0]
Sa[1,1]=sigma_x[1]
Sa_inv=np.linalg.inv(Sa)

Se=np.zeros((3,3))
Se[0,0]=sigma_y[0]
Se[1,1]=sigma_y[1]
Se[2,2]=sigma_y[2]
Se_inv=np.linalg.inv(Se)

x1=np.arange(-5,5)+Xa[0]
x2=np.arange(-5,5)+Xa[1]

X1,X2=np.meshgrid(x1,x2)
X12=[X1,X2]
F_plot=fun(X12)
  
#%% Main
plt.figure(figsize=(18,8))
for n in range(N):
    J=Jacobian(X[-1,:])
    B=gamma*Sa_inv+np.matmul(np.matmul(J.T,Se_inv),J)
    M=np.matmul(np.matmul(np.linalg.inv(B),J.T),Se_inv)
    X=utl.vstack(X,Xa+np.matmul(M,y-fun(X[-1,:])+np.matmul(J,X[-1,:]-Xa)))
    res=np.append(res,np.sum((y-fun(X[-1,:]))**2))
    
    plt.subplot(3,N,n+1)
    plt.contourf(X1,X2,F_plot[0]-y[0],np.linspace(vmin,vmax,100),vmin=vmin,vmax=vmax,cmap='seismic',extend='both')
    plt.plot(x[0],x[1],'ok',markersize=5)
    plt.plot(X[-1,0],X[-1,1],'xk',markersize=10)
    plt.plot(Xa[0],Xa[1],'*k',markersize=10)
    
    plt.subplot(3,N,n+1+(N))
    plt.contourf(X1,X2,F_plot[1]-y[1],np.linspace(vmin,vmax,100),vmin=vmin,vmax=vmax,cmap='seismic',extend='both')
    plt.plot(x[0],x[1],'ok',markersize=5)
    plt.plot(X[-1,0],X[-1,1],'xk',markersize=10)
    plt.plot(Xa[0],Xa[1],'*k',markersize=10)
    

    plt.subplot(3,N,n+1+2*(N))
    plt.contourf(X1,X2,F_plot[2]-y[2],np.linspace(vmin,vmax,100),vmin=vmin,vmax=vmax,cmap='seismic',extend='both')
    plt.plot(x[0],x[1],'ok',markersize=5)
    plt.plot(X[-1,0],X[-1,1],'xk',markersize=10)
    plt.plot(Xa[0],Xa[1],'*k',markersize=10)
    
    if np.max(np.abs(X[-1,:]-X[-2,:]))<tol:
        break



J=np.array([[-1,2],[3,4],[-5,6]])

A_plus=np.matmul(np.linalg.inv(np.matmul(J.T,J)),J.T)
A_plus2=np.matmul(J.T,np.linalg.inv(np.matmul(J,J.T)))

I1=np.matmul(A_plus,J)

I2=np.matmul(J,A_plus2)