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
m=8#number of channels

z=np.arange(0,80)# [Km] vertical heights
zmin=1#[Km] minimum centroid height
zmax=20#[Km] maximum centroid height

#empirical profile (US Standard Atmopshere 1976s)
z_prof=[0,11,20,32,47,51,71,86]#[Km]
T0=288.15#[K]
dT_dz_prof=[np.nan,-6.5,0,1,2.8,0,-2.8,-2]#[K]

sigma_e=0.5#[K] noise stdev
sigma_a=10#[K] prior stdev
beta=np.exp(-0.05)#coefficient of decay of correlation

#stats
L_mc=10000#MC samples

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
S_a=sigma_a**2*beta**(2*np.abs(I-J))

#contribution matrix
ac=1/np.linspace(zmin,zmax,m)
for i in range(m):
    K[i,:]=ac[i]*np.exp(-ac[i]*z)

#%% Main

#generate temperature
lambda_a, L_a = np.linalg.eig(S_a)
E_a=L_a@np.diag(lambda_a**0.5)
a=np.random.normal(0,1,(n,L_mc))
x=x_a_rep+E_a@a

#check stats
x_a2=np.mean(x,axis=1)
S_a2=np.cov(x)

#generate observations
err=np.random.normal(0,sigma_e,(m,L_mc))
y= K @ x+err

#SVD
U, L, VT = np.linalg.svd(K, full_matrices=False)

#SVD of retrieval
K_tilde=np.linalg.inv(sqrtm(S_e))@K@sqrtm(S_a)
U_tilde, L_tilde, VT_tilde = np.linalg.svd(K_tilde, full_matrices=False)

#retrieval
S_hat=np.linalg.inv(K.T @ np.linalg.inv(S_e) @ K + np.linalg.inv(S_a))
G=S_hat @ K.T @ np.linalg.inv(S_e)
A=G @ K

x_hat=x_a_rep+G @ (y-K @ x_a_rep)

#errors
S_s=(A-np.eye(n))@S_a@(A-np.eye(n)).T#smoothing error
S_n=G@S_e@G.T#noise error

#resolution
zc1=[]
zc2=[]
for i in range(n):
    
    #maximas
    pos_peaks=np.unique(find_peaks(A[i,:])[0])
    if A[i,0] > A[i,1]:  
        pos_peaks = np.append(0,pos_peaks)
    if A[i,-1] > A[i,-2]: 
        pos_peaks = np.append(pos_peaks, n)
    
    #minima
    neg_peaks=np.unique(np.concatenate(([0],find_peaks(-A[i,:])[0],[n])))
    
    #peak closer to layer
    peak_sel=pos_peaks[np.argmin(np.abs(pos_peaks-i))]
    
    #select curve around selected peak
    if peak_sel==0:#if peak is on first layer
        j1=0
        j2=neg_peaks[np.where(neg_peaks>peak_sel)[0][0]]
    elif peak_sel==n:#if peak is on last layers
        j1=neg_peaks[np.where(neg_peaks<peak_sel)[0][-1]]
        j2=n
    else:#if peak is in the middle
        j1=neg_peaks[np.where(neg_peaks<=peak_sel)[0][-1]]
        j2=neg_peaks[np.where(neg_peaks>peak_sel)[0][0]]
        
    #FWHM
    f=A[i,j1:j2]/np.max(A[i,j1:j2])-0.5
    z_sel=z[j1:j2]
    
    #zero crossing
    zc_ind=np.where(f[:-1]*f[1:]<0)[0]
    
    if len(zc_ind)==2:#if there are two zero crossings
        zc_ind1=zc_ind[0]
        zc_ind2=zc_ind[1]
        
        #interpolate to 0
        zc1=np.append(zc1,(z_sel[zc_ind1+1]-z_sel[zc_ind1])/(f[zc_ind1+1]-f[zc_ind1])*(-f[zc_ind1])+z_sel[zc_ind1])
        zc2=np.append(zc2,(z_sel[zc_ind2+1]-z_sel[zc_ind2])/(f[zc_ind2+1]-f[zc_ind2])*(-f[zc_ind2])+z_sel[zc_ind2])
        
    elif len(zc_ind)==1:#if there is one zero crossing
        if f[zc_ind]>=0:#if negative slope
            zc_ind2=zc_ind[0]
            
            zc1=np.append(zc1,z_sel[0])
            zc2=np.append(zc2,(z_sel[zc_ind2+1]-z_sel[zc_ind2])/(f[zc_ind2+1]-f[zc_ind2])*(-f[zc_ind2])+z_sel[zc_ind2])
        else:#if postive slope
            zc_ind1=zc_ind[0]
            
            zc1=np.append(zc1,(z_sel[zc_ind1+1]-z_sel[zc_ind1])/(f[zc_ind1+1]-f[zc_ind1])*(-f[zc_ind1])+z_sel[zc_ind1])
            zc2=np.append(zc2,np.nan)

#%% Plots
plt.close('all')

#weighting functions
matplotlib.rcParams['font.size'] = 18
plt.figure(figsize=(16,6))
cmap = plt.get_cmap('viridis')
colors = cmap(np.linspace(0, 1, m))
for i in range(m):
    plt.semilogx(K[i,:],z,color=colors[i])
for i in range(m):
    plt.plot(ac[i]*np.exp(-1),1/ac[i],'.k',markersize=15,color=colors[i])
plt.xlim([10**-4,1])
plt.xlabel(r'$a_m e^{-a_m z_n} \Delta z$ [Km$^{-1}$]')
plt.ylabel(r'$z$ [Km]')
plt.grid()
plt.tight_layout()

#SVD
plt.figure(figsize=(12,10))
for i in range(m):
    plt.subplot(2,4,i+1)
    plt.plot(VT[i,:],z,'k')
    plt.title(r'$\lambda_'+str(i)+'='+str(np.round(L[i],3))+'$')
    plt.xlim([-0.3,0.3])
    plt.xticks(np.arange(-0.3,0.4,0.3))
    plt.ylim([0,80])
    plt.xlabel(r'$V_{:,'+str(i)+'}$')
    plt.ylabel(r'$z$ [Km]')
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

#posterior stdev
plt.figure(figsize=(18,6))

plt.plot(np.diag(S_a)**0.5,z,'r',label='Theory, prior')
plt.scatter(np.diag(S_a2)**0.5,z,s=15,edgecolor='k',facecolor='r',label='Monte Carlo, prior')
plt.plot(np.diag(S_hat)**0.5,z,'k',label='Theory, posterior')
plt.scatter(np.std(x_hat-x,axis=1),z,s=15,edgecolor='k',facecolor='k',label='Monte Carlo, posterior',marker='s')
plt.plot(np.diag(S_s)**0.5,z,'--k',label='Theory, posterior (smoothing)')
plt.plot(np.diag(S_n)**0.5,z,'-.k',label='Theory, postrior (noise)')

plt.xlabel(r'Standard deviation [K]')
plt.ylabel(r'$z$ [Km]')
plt.grid()
plt.legend()
plt.tight_layout()

#prior and posterior covariance
cov_mc=np.zeros((n,n))+np.nan
cov_mc[I<J]=S_hat[I<J]
cov_mc[I>J]=S_a[I>J]

cov_mc2=np.zeros((n,n))+np.nan
cov_mc2[I<J]=np.cov(x_hat-x)[I<J]
cov_mc2[I>J]=S_a2[I>J]

fig=plt.figure(figsize=(18,8))
gs = gridspec.GridSpec(1, 3, width_ratios=[3,3,0.25])
ax=fig.add_subplot(gs[0,0])
pc=plt.pcolor(cov_mc,cmap='seismic',vmin=-np.max(S_hat),vmax=np.max(S_hat))
ax.set_facecolor("gray")
plt.xlabel(r'$z$ [Km]')
plt.ylabel(r'$z$ [Km]')
ax=fig.add_subplot(gs[0,1])
plt.pcolor(cov_mc2,cmap='seismic',vmin=-np.max(S_hat),vmax=np.max(S_hat))
ax.set_facecolor("gray")
plt.xlabel(r'$z$ [Km]')
cbar_ax=fig.add_subplot(gs[0,2])
cbar=fig.colorbar(pc, cax=cbar_ax,label=r'Covariance [K$^2$]')

#selected cases
plt.figure(figsize=(18,8))

err_all=np.mean((x_hat-x)**2,axis=0)**0.5
err_index=np.argsort(err_all)

plt.subplot(1,3,1)
i=err_index[0]
plt.plot(x_a,z,'--k',label=r'$x_a$')
plt.plot(x[:,i],z,'k',label=r'$x$')
plt.plot(x_hat[:,i],z,'r',label=r'$\hat{x}$')
plt.xlim([150,300])
plt.ylim([0,80])
plt.xlabel(r'$x_a$ [K]')
plt.ylabel(r'$z$ [Km]')
plt.grid()
plt.text(160,7,'$\sqrt{\overline{\epsilon_t^2}}='+str(np.round(err_all[i],2))+'$ K',bbox={'facecolor':'w','alpha':0.5})

plt.subplot(1,3,2)
i=err_index[int(L_mc/2)]
plt.plot(x_a,z,'--k',label=r'$x_a$')
plt.plot(x[:,i],z,'k',label=r'$x$')
plt.plot(x_hat[:,i],z,'r',label=r'$\hat{x}$')
plt.xlim([150,300])
plt.ylim([0,80])
plt.xlabel(r'$x_a$ [K]')
plt.grid()
plt.text(160,7,'$\sqrt{\overline{\epsilon_t^2}}='+str(np.round(err_all[i],2))+'$ K',bbox={'facecolor':'w','alpha':0.5})

plt.subplot(1,3,3)
i=err_index[-1]
plt.plot(x_a,z,'--k',label=r'$x_a$')
plt.plot(x[:,i],z,'k',label=r'$x$')
plt.plot(x_hat[:,i],z,'r',label=r'$\hat{x}$')
plt.xlim([150,300])
plt.ylim([0,80])
plt.xlabel(r'$x_a$ [K]')
plt.grid()
plt.text(160,7,'$\sqrt{\overline{\epsilon_t^2}}='+str(np.round(err_all[i],2))+'$ K',bbox={'facecolor':'w','alpha':0.5})
    
plt.legend()


#resolution
fig=plt.figure(figsize=(18,8))
gs = gridspec.GridSpec(1, 4, width_ratios=[0.25,3,1,1])

ax=fig.add_subplot(gs[0,1])
pc=plt.pcolor(A,cmap='seismic',vmin=[-0.25,0.25])
plt.plot(zc1,z,'k')
plt.plot(zc2,z,'k')
plt.xlabel(r'$z$ [Km]')
plt.ylabel(r'$z$ [Km]')

cbar_ax=fig.add_subplot(gs[0,0])
cbar=fig.colorbar(pc, cax=cbar_ax,label=r'Magnitude of element A',location='left')

ax=fig.add_subplot(gs[0,2])
plt.plot(zc2-zc1,z,'k')
plt.xlabel(r'Vertical resolution [Km]')
plt.ylabel(r'$z$ [Km]')
plt.xlim([0,50])
plt.ylim([0,80])
plt.grid()

ax=fig.add_subplot(gs[0,3])
plt.plot(np.diag(A),z,'k')
plt.ylim([0,80])
plt.xlabel(r'Local DFS')
plt.ylabel(r'$z$ [Km]')
plt.grid()
plt.tight_layout()
