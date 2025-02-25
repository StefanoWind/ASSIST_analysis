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
from scipy.stats import norm
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 18


#%% Inputs
m=8#number of channels

z=np.arange(0,80)# [Km] vertical heights
zmin=1#[Km] minimum centroid height
zmax=20#[Km] maximum centroid height

#empirical profile (US Standard Atmopshere 1976s)
z_prof=[0,11,20,32,47,51,71,86]#[Km]
T0=288.15#[K]
dT_dz_prof=[np.nan,-6.5,0,1,2.8,0,-2.8,-2]#[K]

#perturbation
DT=10#[K] amplitude of temperarature oscillation
max_z_DT=3#[m] maximum height at which the oscillation occurs
Nt=24#number of time steps
exp=1#exponent of time 

sigma_e=0.5#[K] noise stdev
sigma_a_rand=2.5#[K] prior stdev
beta=np.exp(-0.05)#coefficient of decay of correlation

#stats
L_mc=10000#MC samples


#%% Initalizations
#generate temperature profile
n=len(z)
T_prof=[T0]
for i in range(1,len(z_prof)):
    T_prof=np.append(T_prof,T_prof[i-1]+dT_dz_prof[i]*(z_prof[i]-z_prof[i-1]))

#build perturbation
time=np.arange(0,1,1/Nt)
pert=np.sin(time*2*np.pi)*time**exp
pert=(pert-np.mean(pert))/np.ptp(pert)*2

#zeroing
K=np.zeros((m,n))

#noise covariance
S_e=np.eye(m)*sigma_e**2

#prior
x_a=np.interp(z,z_prof,T_prof)
x_a_rep=x_a.reshape(n,1)@np.ones(L_mc).reshape(1,L_mc)
I,J=np.meshgrid(np.arange(n),np.arange(n))

#perturbation
S_a_pert=DT**2*(1-z[I]/max_z_DT)*(1-z[J]/max_z_DT)*np.trapz(pert**2,time)
S_a_pert[z[I]>max_z_DT]=0
S_a_pert[z[J]>max_z_DT]=0

#random component
S_a_rand=sigma_a_rand**2*beta**(2*np.abs(I-J))
lambda_a_rand, L_a_rand = np.linalg.eig(S_a_rand)
E_a_rand=L_a_rand@np.diag(lambda_a_rand**0.5)

#total covariance
S_a=S_a_rand+S_a_pert

#contribution matrix
ac=1/np.linspace(zmin,zmax,m)
for i in range(m):
    K[i,:]=ac[i]*np.exp(-ac[i]*z)

#retrieval
S_hat=np.linalg.inv(K.T @ np.linalg.inv(S_e) @ K + np.linalg.inv(S_a))
G=S_hat @ K.T @ np.linalg.inv(S_e)
A=G @ K
sigma_hat=np.diag(S_hat)**0.5

#zeroing
x_all=np.zeros((n,L_mc,len(time)))
x_hat_all=np.zeros((n,L_mc,len(time)))
x_pert_all=np.zeros((n,len(time)))

#%% Main
for it in range(len(time)):
    
    #generate perturbation
    x_pert=-DT*(1-z/max_z_DT)*pert[it]
    x_pert[z>max_z_DT]=0
    x_pert_rep=x_pert.reshape(n,1)@np.ones(L_mc).reshape(1,L_mc)
    x_pert_all[:,it]=x_pert
    
    #generate random fluctuations
    a=np.random.normal(0,1,(n,L_mc))
    x_rand=E_a_rand@a
    
    #generate full profile
    x=x_a_rep+x_pert_rep+x_rand
    x_all[:,:,it]=x

    #generate observations
    err=np.random.normal(0,sigma_e,(m,L_mc))
    y= K @ x+err
    
    #retrieval
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
bias_dav_pert=(A-np.eye(n))@x_pert_all

#overall error
bias=np.mean(x_hat_all_flat-x_all_flat,axis=1)
sigma=np.std(x_hat_all_flat-x_all_flat,axis=1)
sigma_pert=np.std(bias_dav_pert,axis=1)

#%% Plots
plt.close('all')
plt.figure()
plt.plot(x_all_flat[0,:200],'k',label='$x$')
plt.plot(x_hat_all_flat[0,:200],'r',label='$\hat{x}$')
plt.xlabel('Time')
plt.ylabel(r'$x$')
plt.grid()
plt.legend()
plt.title('Time series at gate 0')

plt.figure(figsize=(16,10))
plt.subplot(2,2,1)
plt.pcolor(time,z,x_dav)
plt.xlabel('Time')
plt.ylabel(r'$z$')
plt.colorbar(label='$x$')
plt.title('Phase-average of state')

plt.subplot(2,2,2)
plt.pcolor(time,z,x_hat_dav)
plt.xlabel('Time')
plt.ylabel(r'$z$')
plt.colorbar(label='$\hat{x}$')
plt.title('Phase-average of retrieval')

plt.subplot(2,2,3)
plt.pcolor(time,z,bias_dav,cmap='seismic')
plt.xlabel('Time')
plt.ylabel(r'$z$')
plt.colorbar(label='$\hat{x}-x$')
plt.title('Phase-average of bias')

plt.subplot(2,2,4)
plt.pcolor(time,z,bias_dav_pert,cmap='seismic')
plt.xlabel('Time')
plt.ylabel(r'$z$')
plt.colorbar(label=r'$(A-I)\tilde{x}$')
plt.title('Phase-average of predicted bias')
plt.tight_layout()

plt.figure(figsize=(18,5))
plt.subplot(1,3,1)
plt.pcolor(z,z,S_a)
plt.colorbar(label='Covariance')
plt.xlabel(r'$z$')
plt.ylabel(r'$z$')
plt.title('Prior covariance')

plt.subplot(1,3,2)
plt.pcolor(z,z,S_a2)
plt.colorbar(label='Covariance')
plt.xlabel(r'$z$')
plt.ylabel(r'$z$')
plt.title('Prior covariance (MC)')

plt.subplot(1,3,3)
plt.pcolor(z,z,S_a2-S_a,cmap='seismic')
plt.colorbar(label='Covariance difference')
plt.xlabel(r'$z$')
plt.ylabel(r'$z$')
plt.title('Prior covariance difference')
plt.tight_layout()

plt.figure()
plt.subplot(1,2,1)
plt.plot(bias,z,'k')
plt.xlabel('Bias')
plt.ylabel(r'$z$')
plt.grid()

plt.subplot(1,2,2)
plt.plot(sigma,z,'k',label='MC')
plt.plot(sigma_hat,z,'.r',label='OE Theory')
plt.plot(sigma_pert,z,'b',label='Due to perturbation')
plt.xlabel('Error st.dev.')
plt.ylabel(r'$z$')
plt.grid()
plt.legend()
plt.tight_layout()

plt.figure(figsize=(18,8))
plt.subplot(1,2,1)
plt.hist(x_all_flat[0,:],100,density=True,color='k',label='MC')
plt.plot(np.arange(260,321),norm.pdf(np.arange(260,321),loc=x_a[0],scale=S_a[0,0]**0.5),'r',label='Pior')
plt.xlabel('$x$ at first gate')
plt.grid()
plt.legend()

plt.subplot(1,2,2)
plt.hist(x_all_flat[0,:]-x_hat_all_flat[0,:],100,density=True,color='k',label='MC')
plt.plot(np.arange(-2,2,0.1),norm.pdf(np.arange(-2,2,0.1),loc=0,scale=sigma_hat[0]),'r',label='Posterior')
plt.xlabel('$\hat{x}-x$ at first gate')
plt.legend()
plt.grid()
plt.tight_layout()