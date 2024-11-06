# -*- coding: utf-8 -*-
"""
Retrieval example follwing Rogers 2000
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
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 16


#%% Inputs
m=8#number of channels
d_log_p=0.75#vertical spacing of peak of contribution functions
pi0=2#first peak pressure level

z=np.arange(0,10,0.1)#vertical heights

#empirical profile (Rodgers 2000)
z_prof=[0,1.5,3.7,6.7,7.4,10]
T_prof=[277,134,134,265,265,126]#[K]

sigma_e=0.5#[K] noise stdev
sigma_a=10#[K] prior stdev
beta=np.exp(-0.05)#coefficient of decay of correlation

#stats
L_mc=10000#MC samples

#%% Initalization
p=np.exp(-z) #pressure levels
n=len(p)
pi=np.exp(-np.arange(m)*d_log_p)*np.exp(-pi0)#peak of weighting functions

#zeroing
K=np.zeros((m,n))

#%% Initialization

#noise covariance
S_e=np.eye(m)*sigma_e**2

I,J=np.meshgrid(np.arange(len(p)),np.arange(len(p)))
S_a=sigma_a**2*beta**(2*np.abs(I-J))

#contribution matrix
for i in range(m):
    K[i,:]=p*np.exp(-p/pi[i])/pi[i]

#temperature
x_a=np.interp(z,z_prof,T_prof)
x_a_rep=x_a.reshape(n,1)@np.ones(L_mc).reshape(1,L_mc)

#%% Main

#generate temperature
lambda_a, L_a = np.linalg.eig(S_a)
E_a=L_a@np.diag(lambda_a**0.5)
a=np.random.normal(0,1,(n,L_mc))
x=x_a_rep+E_a@a

#check stats
x_a2=np.mean(x,axis=1)
S_a2=np.cov(x)

err=np.random.normal(0,sigma_e,(m,L_mc))
y= K @ x+err

#SVD
U, L, VT = np.linalg.svd(K, full_matrices=False)

#SVD of retrieval
K_tilde=np.matmul(np.matmul(np.linalg.inv(sqrtm(S_e)),K),sqrtm(S_a))
U_tilde, L_tilde, VT_tilde = np.linalg.svd(K_tilde, full_matrices=False)

#retrieval
S_hat=np.linalg.inv(K.T @ np.linalg.inv(S_e) @ K + np.linalg.inv(S_a))
G=S_hat @ K.T @ np.linalg.inv(S_e)
A=G @ K

x_hat=x_a_rep+G @ (y-K @ x_a_rep)

#errors
S_s=(A-np.eye(n))@S_a@(A-np.eye(n)).T#smoothing error
S_n=G@S_e@G.T#noise error

#%% Plots
plt.close('all')

#weighting functions
cmap = plt.get_cmap('viridis')
colors = cmap(np.linspace(0, 1, m))
plt.figure()
for i in range(m):
    plt.plot(K[i,:],-np.log(p),color=colors[i])
plt.ylim([0,10])
plt.xlim([0,0.40])
plt.xlabel(r'$\frac{\partial \tau}{\partial z}$ [m$^{-1}$]')
plt.ylabel(r'$z$ [m]')
plt.grid()
plt.tight_layout()

#SVD
plt.figure(figsize=(12,10))
for i in range(m):
    plt.subplot(2,4,i+1)
    plt.plot(VT[i,:],-np.log(p),'k')
    plt.title(r'$\lambda_'+str(i)+'='+str(np.round(L[i],3))+'$')
    plt.xlim([-0.3,0.3])
    plt.xticks(np.arange(-0.3,0.4,0.3))
    plt.ylim([0,10])
    plt.xlabel(r'$V_{:,'+str(i)+'}$')
    plt.ylabel(r'$z$ [m]')
    plt.grid()
plt.tight_layout()

plt.figure(figsize=(18,6))
for i in range(m):
    ax=plt.subplot(2,4,i+1)
    plt.bar(np.arange(m),U.T[i,:]*L[i],color='k')
    ax.set_yscale('symlog',linthresh=0.01)
    plt.ylabel(r'$U_{:,'+str(i)+'}\cdot \lambda_'+str(i)+'$')
    plt.ylim([-1.5,1.5])
    plt.grid()
plt.tight_layout()

#prior
fig=plt.figure(figsize=(18,8))
gs = gridspec.GridSpec(1, 3, width_ratios=[1,3,0.25])  # 0.1 row for the colorbars
ax=fig.add_subplot(gs[0,0])
plt.plot(x_a,z,'k')
plt.ylim([0,10])
plt.xlabel(r'$x_a$ [K]')
plt.ylabel(r'$z$')
plt.grid()

ax=fig.add_subplot(gs[0,1])
pc=plt.pcolor(S_a,cmap='hot',vmin=0,vmax=sigma_a**2)
plt.xlabel(r'$z$')
plt.grid()

cbar_ax=fig.add_subplot(gs[0,2])
cbar=fig.colorbar(pc, cax=cbar_ax,label=r'$S_a$ [K$^2$]')

#prior check
fig=plt.figure(figsize=(18,8))
gs = gridspec.GridSpec(1, 3, width_ratios=[1,3,0.25])  # 0.1 row for the colorbars
ax=fig.add_subplot(gs[0,0])
plt.plot(x_a2,z,'k')
plt.ylim([0,10])
plt.xlabel(r'$x_a$ [K]')
plt.ylabel(r'$z$')
plt.grid()

ax=fig.add_subplot(gs[0,1])
pc=plt.pcolor(S_a2,cmap='hot',vmin=0,vmax=sigma_a**2)
plt.xlabel(r'$z$')
plt.grid()

cbar_ax=fig.add_subplot(gs[0,2])
cbar=fig.colorbar(pc, cax=cbar_ax,label=r'$S_a$ [K$^2$]')

#posterior stdev
plt.figure()
plt.scatter(np.std(x_hat-x,axis=1),z,s=10,edgecolor='k',facecolor=None,label='Monte Carlo, total')
plt.plot(np.diag(S_hat)**0.5,z,'k',label='Theory, total')
plt.plot(np.diag(S_s)**0.5,z,'--k',label='Theory, smoothing')
plt.plot(np.diag(S_n)**0.5,z,'-.k',label='Theory, noise')
plt.xlabel(r'$\sigma (\hat{x}-x)$ [K]')
plt.ylabel(r'$z$')
plt.grid()
plt.legend()
plt.tight_layout()

#posterior covariance
fig=plt.figure(figsize=(18,8))
gs = gridspec.GridSpec(1, 3, width_ratios=[3,3,0.25])
ax=fig.add_subplot(gs[0,0])
pc=plt.pcolor(np.cov(x_hat-x),cmap='seismic',vmin=-np.max(S_hat),vmax=np.max(S_hat))
plt.xlabel(r'$z$')
plt.ylabel(r'$z$')
ax=fig.add_subplot(gs[0,1])
plt.pcolor(S_hat,cmap='seismic',vmin=-np.max(S_hat),vmax=np.max(S_hat))
plt.xlabel(r'$z$')
cbar_ax=fig.add_subplot(gs[0,2])
cbar=fig.colorbar(pc, cax=cbar_ax,label=r'$S(\hat{x}-x)$ [K$^2$]')

#selected cases
plt.figure(figsize=(18,8))

err_all=np.mean((x_hat-x)**2,axis=0)**0.5
err_index=np.argsort(err_all)

plt.subplot(1,3,1)
i=err_index[0]
plt.plot(x_a,z,'--k',label=r'$x_a$')
plt.plot(x[:,i],z,'k',label=r'$x$')
plt.plot(x_hat[:,i],z,'r',label=r'$\hat{x}$')
plt.xlim([100,350])
plt.ylim([0,10])
plt.xlabel(r'$x_a$ [K]')
plt.ylabel(r'$z$')
plt.grid()
plt.text(110,7,'$\sqrt{\overline{\epsilon_t^2}}='+str(np.round(err_all[i]),2)+'$ K',bbox={'facecolor':'w','alpha':0.5})

plt.subplot(1,3,2)
i=err_index[int(L_mc/2)]
plt.plot(x_a,z,'--k',label=r'$x_a$')
plt.plot(x[:,i],z,'k',label=r'$x$')
plt.plot(x_hat[:,i],z,'r',label=r'$\hat{x}$')
plt.xlim([100,350])
plt.ylim([0,10])
plt.xlabel(r'$x_a$ [K]')
plt.grid()
plt.text(110,7,'$\sqrt{\overline{\epsilon_t^2}}='+str(np.round(err_all[i]),2)+'$ K',bbox={'facecolor':'w','alpha':0.5})

plt.subplot(1,3,3)
i=err_index[-1]
plt.plot(x_a,z,'--k',label=r'$x_a$')
plt.plot(x[:,i],z,'k',label=r'$x$')
plt.plot(x_hat[:,i],z,'r',label=r'$\hat{x}$')
plt.xlim([100,350])
plt.ylim([0,10])
plt.xlabel(r'$x_a$ [K]')
plt.grid()
plt.text(110,7,'$\sqrt{\overline{\epsilon_t^2}}='+str(np.round(err_all[i]),2)+'$ K',bbox={'facecolor':'w','alpha':0.5})
    
plt.legend()
