# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 16:30:25 2024

@author: sletizia
"""

"""
Non-linear 1D retrieval example
"""
import os
cd=os.path.dirname(__file__)
import sys
sys.path.append('C:/Users/SLETIZIA/OneDrive - NREL/Desktop/PostDoc/utils')
import utils as utl
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp2d

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 16


#%% Inputs
xa=10#prior mean
sigma_a=10#prior stdev
sigma_e=5#noise stdev

bs=[0,0.01,0.1]#non-linear factor

L_mc=10000000#MC draws
tol=0.001#convergence tolerance
min_iter=10#minimum iterations
max_iter=300#maximum iterations

#graphics
y_sel=5

#%% Functions
def Gauss(x,mu,sigma):
    return 1/(2*np.pi)**0.5/sigma*np.exp(-(x-mu)**2/sigma**2/2)

#%% Initialization
x=np.arange(-10,30,.5)
y=np.arange(-10,30,.5)

#%% Plots
fig=plt.figure(figsize=(18,7))
gs = gridspec.GridSpec(3, 6, height_ratios=[1,5, 1],width_ratios=[1,5,1,5,1,5])

ctr=0
for b in bs:
    
    #MC
    x_mc=np.random.normal(xa,sigma_a,L_mc)
    eps=np.random.normal(0,sigma_e,L_mc)
    y_mc=x_mc+b*x_mc**2+eps
    p_mc=np.histogram2d(x_mc, y_mc, bins=[utl.rev_mid(x),utl.rev_mid(y)])
    pxy_mc=p_mc[0].T/np.sum(p_mc[0])
    
    #extract posterior from MC
    f = interp2d(x, y, pxy_mc, kind='linear')
    px_y_mc=np.array([f(xx,y_sel) for xx in x]).squeeze()
    
    #Gauss-Newton
    x_hat=[]
    for yy in y:
        xi=[xa]
        for i in range(max_iter):
            ki=1+2*b*xi[-1]
            sigma=(1/sigma_a**2+ki**2/sigma_e**2)**-0.5
            xi=np.append(xi,xa+sigma**2*ki/sigma_e**2*(yy-(xi[-1]+b*xi[-1]**2)+ki*(xi[-1]-xa)))
            if np.abs(xi[-1]-xi[-2])<tol and i>min_iter:
                break
        if i<max_iter-1:
            x_hat=np.append(x_hat,xi[-1])    
        else:
            x_hat=np.append(x_hat,np.nan)    
    
    
    ax = fig.add_subplot(gs[1, ctr+1])
    plt.pcolor(x,y,pxy_mc,cmap='hot')
    plt.plot(x,x+b*x**2,'g',label=r'$kx+bx^2$')
    plt.plot(x_hat,y,'.m',label=r'$\hat{x}$')
    plt.xlim([-10,30])
    plt.ylim([-10,30])
    plt.xlabel(r'$x$')
    if ctr==0:
        plt.ylabel(r'$y$')
    plt.grid()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    if ctr==0:
        plt.legend(draggable=True)
    
    plt.plot(x,x*0+y_sel,'--',color=[0.5,0.5,0.5])

    #P(x|y)
    ax = fig.add_subplot(gs[0, ctr+1])
    plt.plot(x,Gauss(x,x_hat[y==y_sel],sigma)/np.max(Gauss(x,x_hat[y==y_sel],sigma)),'k',label='Theory')
    plt.plot(x,px_y_mc/np.max(px_y_mc),'sk',label='Monte Carlo',markersize=3)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.ylabel(r'$P(x|y)$')

    ctr+=2
plt.legend(draggable=True)
plt.tight_layout()
    
    
    
    