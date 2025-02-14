# -*- coding: utf-8 -*-
"""
Checking prior bias based on retrieval example following Rogers 2000 but for a ground-based IRS   
"""
import os
cd=os.path.dirname(__file__)
import sys
sys.path.append('C:/Users/SLETIZIA/OneDrive - NREL/Desktop/Main/utils')
import utils as utl
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from scipy.linalg import sqrtm
import matplotlib.gridspec as gridspec
from scipy.stats import norm
from scipy.signal import find_peaks
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 18


#%% Inputs
m=8#number of channels

z=np.arange(0,80,0.25)# [Km] vertical heights
zmin=1#[Km] minimum centroid height
zmax=20#[Km] maximum centroid height

#empirical profile (US Standard Atmopshere 1976s)
z_prof=[0,11,20,32,47,51,71,86]#[Km]
T0=288.15#[K]
dT_dz_prof=[np.nan,-6.5,0,1,2.8,0,-2.8,-2]#[K]

DT=10#[K] amplitude of temperarature oscillation
max_z_DT=3#[m] maximum height at which the oscillation occurs
Nt=30


sigma_e=0.5#[K] noise stdev
sigma_a0=2.5#[K] prior stdev
beta=np.exp(-0.05)#coefficient of decay of correlation

#stats
L_mc=10000#MC samples

time=np.arange(0,1,1/Nt)
pert=np.sin(time*2*np.pi)*time
pert=(pert-np.mean(pert))/np.ptp(pert)*2

#%% Initalizations

n=len(z)
T_prof=[T0]
for i in range(1,len(z_prof)):
    T_prof=np.append(T_prof,T_prof[i-1]+dT_dz_prof[i]*(z_prof[i]-z_prof[i-1]))

#zeroing
K=np.zeros((m,n))

#%% Initialization

#noise covariance
S_e=np.eye(m)*sigma_e**2

#prior
x_a=np.interp(z,z_prof,T_prof)
x_a_rep=x_a.reshape(n,1)@np.ones(L_mc).reshape(1,L_mc)
I,J=np.meshgrid(np.arange(n),np.arange(n))
S_a0=sigma_a0**2*beta**(2*np.abs(I-J))

S_a1=DT**2*(1-z[I]/max_z_DT)*(1-z[J]/max_z_DT)*np.trapz(pert**2,time)
S_a1[z[I]>max_z_DT]=0
S_a1[z[J]>max_z_DT]=0

S_a=S_a0+S_a1

#contribution matrix
ac=1/np.linspace(zmin,zmax,m)
for i in range(m):
    K[i,:]=ac[i]*np.exp(-ac[i]*z)

x_all=np.zeros((n,L_mc,len(time)))
x_hat_all=np.zeros((n,L_mc,len(time)))
x1_all=np.zeros((n,len(time)))

#retrieval
S_hat=np.linalg.inv(K.T @ np.linalg.inv(S_e) @ K + np.linalg.inv(S_a))
G=S_hat @ K.T @ np.linalg.inv(S_e)
A=G @ K
sigma2=np.diag(S_hat)**0.5

#%% Main
for it in range(len(time)):
    
    #generate temperature
    lambda_a0, L_a0 = np.linalg.eig(S_a0)
    E_a0=L_a0@np.diag(lambda_a0**0.5)
    a=np.random.normal(0,1,(n,L_mc))
    x0=+E_a0@a
    x1=-DT*(1-z/max_z_DT)*pert[it]
    x1[z>max_z_DT]=0
    x1_rep=x1.reshape(n,1)@np.ones(L_mc).reshape(1,L_mc)
    x=x_a_rep+x0+x1_rep
    
    x1_all[:,it]=x1
    x_all[:,:,it]=x

    #generate observations
    err=np.random.normal(0,sigma_e,(m,L_mc))
    y= K @ x+err
    x_hat=x_a_rep+G @ (y-K @ x_a_rep)
    x_hat_all[:,:,it]=x_hat
    print(it)

#check stats
x_all_flat=x_all.reshape(n,len(time)*L_mc)
x_hat_all_flat=x_hat_all.reshape(n,len(time)*L_mc)
x_a2=np.mean(x_all_flat,axis=1)
S_a2=np.cov(x_all_flat)

#time-z mean map
x_dav=np.mean(x_all,axis=1)
x_hat_dav=np.mean(x_hat_all,axis=1)

#daily stats
bias_dav=np.mean(x_hat_all-x_all,1)

#overall error
bias=np.mean(x_hat_all_flat-x_all_flat,axis=1)
sigma=np.std(x_hat_all_flat-x_all_flat,axis=1)

bias2=(A-np.eye(n))@x1_all

sigma3=np.std(bias2,axis=1)

#%% Plots
plt.close('all')
plt.figure(figsize=(16,10))
plt.subplot(2,2,1)
plt.pcolor(time,z,x_dav)
plt.colorbar()

plt.subplot(2,2,2)
plt.pcolor(time,z,x_hat_dav)
plt.colorbar()

plt.subplot(2,2,3)
plt.pcolor(time,z,x_hat_dav-x_dav,cmap='seismic')
plt.colorbar()

plt.subplot(2,2,4)
plt.pcolor(time,z,bias2,cmap='seismic')
plt.colorbar()


plt.figure(figsize=(18,5))
plt.subplot(1,3,1)
plt.pcolor(z,z,S_a)
plt.colorbar()

plt.subplot(1,3,2)
plt.pcolor(z,z,S_a2)
plt.colorbar()

plt.subplot(1,3,3)
plt.pcolor(z,z,S_a2-S_a,cmap='seismic')
plt.colorbar()

plt.figure()
plt.subplot(1,2,1)
plt.plot(bias,z)

plt.subplot(1,2,2)
plt.plot(sigma,z)
plt.plot(sigma2,z)
plt.plot(sigma3,z)

plt.figure()
plt.hist(x_all_flat[0,:],100,density=True,color='k')
plt.plot(np.arange(270,331),norm.pdf(np.arange(270,331),loc=x_a[0],scale=S_a[0,0]**0.5),'r')

plt.figure()
plt.hist(x_all_flat[0,:]-x_hat_all_flat[0,:],100,density=True,color='k')
plt.plot(np.arange(-2,2,0.1),norm.pdf(np.arange(-2,2,0.1),loc=0,scale=sigma[0]),'r')