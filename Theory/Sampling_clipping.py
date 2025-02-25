# -*- coding: utf-8 -*-
"""
Effect on spectrum of cut function
"""

import os
cd=os.getcwd()
import sys
sys.path.append('C:/Users/SLETIZIA/OneDrive - NREL/Desktop/PostDoc/utils')
import numpy as np
import utils as utl
from matplotlib import pyplot as plt

#%% Input
sigma=0.5
xmax=10
dx=0.01
downsampling=10
clipping=0.5

#%% Functions
def phase(c):
    c[np.abs(c)<10**-10]=np.nan
    return np.angle(c)

def circ_conv(x1,x2):
    N=len(x1)
    conv=np.zeros(N)+0+0j
    for k in range(N):
        for l in range(N):
            conv[k]+=x1[l]*x2[(k-l)%N]
    return conv/N

#%% Initialization
x=np.arange(-xmax,xmax+dx/2,dx)
omega=x.copy()

#clipping+downsampling
dxs=dx*downsampling
xs=x[::downsampling]
xs=xs[np.abs(xs)<=xmax*clipping+10**-10]
    
#FT on coarse signal
N=len(xs)
n=np.arange(N)-(N-1)/2
k=np.arange(N)-(N-1)/2
omega_k=2*np.pi/N*k/dxs

#FT on full signal
N2=len(x)
n2=np.arange(N2)-(N2-1)/2
k2=np.arange(N2)-(N2-1)/2
omega_k2=2*np.pi/N2*k2/dx

#analytic functions
f=np.exp(-x**2/(2*sigma**2))
F=(2*np.pi)**0.5*sigma*np.exp(-omega**2*sigma**2/2)

H=2*np.sin(omega*xmax*clipping)/omega

#%% Main

#FT on full signal
NN,KK=np.meshgrid(n,k)
DFM=np.exp(-1j*KK*NN*2*np.pi/N)
fn=np.exp(-xs**2/(2*sigma**2))
Fk=np.matmul(DFM,fn)

#comb function
hn=np.zeros(len(x))
hn[::downsampling]=1
hn[np.abs(x)>=xmax*clipping-10**-10]=0

#FT on full signal
NN2,KK2=np.meshgrid(n2,k2)
DFM2=np.exp(-1j*KK2*NN2*2*np.pi/N2)
fn2=f*hn
Fk2=np.matmul(DFM2,fn2)

Hn=np.matmul(DFM2,hn)

#%% Plots
plt.figure()
plt.plot(x,f)
plt.plot(x,fn2,'.-r')
plt.plot(xs,fn,'.-b')
plt.grid()

plt.figure()
plt.subplot(2,1,1)
plt.plot(omega,np.abs(F),'-k')
plt.plot(omega+2*np.pi/dxs,np.abs(F),'-k',alpha=0.5)
plt.plot(omega-2*np.pi/dxs,np.abs(F),'-k',alpha=0.5)
plt.plot(omega_k2,np.abs(Fk2)*dxs,'.-b')
plt.plot(omega_k,np.abs(Fk)*dxs,'.-r')
# plt.xlim([np.min(omega),np.max(omega)])

plt.grid()
plt.subplot(2,1,2)
plt.plot(omega,phase(F),'-k')
plt.plot(omega_k2,phase(Fk2),'.-b')
plt.plot(omega_k,phase(Fk),'.-r')
plt.yticks(np.array([-1,0,1])*np.pi,labels=[r'$-\pi$',r'$0$',r'$\pi$'])
# plt.xlim([np.min(omega),np.max(omega)])
plt.grid()

plt.figure()
plt.subplot(2,1,1)
plt.plot(x,H,'k')
plt.plot(omega_k2,np.real(Hn)*dxs,'.-b')
plt.grid()
plt.subplot(2,1,2)
plt.plot(omega_k2,phase(Hn),'.-b')
plt.yticks(np.array([-1,0,1])*np.pi,labels=[r'$-\pi$',r'$0$',r'$\pi$'])
# plt.xlim([np.min(omega),np.max(omega)])
plt.grid()


