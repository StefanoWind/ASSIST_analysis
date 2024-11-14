# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 16:30:25 2024

@author: sletizia
"""

"""
Retrieval example
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
from scipy.linalg import sqrtm

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 16


#%% Inputs
xa=10#prior mean
sigmas_a=[10,10,5]#prior stdev
sigmas_e=[1,5,10]#noise stdev

#graphics
x_sel=15
y_sel=5

#%% Functions
def Gauss(x,mu,sigma):
    return 1/(2*np.pi)**0.5/sigma*np.exp(-(x-mu)**2/sigma**2/2)

#%% Initialization
x=np.arange(-10,30,.1)
y=np.arange(-10,30,.1)
X,Y=np.meshgrid(x,y)

#%% Plots
fig=plt.figure(figsize=(18,7))
gs = gridspec.GridSpec(3, 6, height_ratios=[1,5, 1],width_ratios=[1,5,1,5,1,5])

ctr=0
for sigma_a,sigma_e in zip(sigmas_a,sigmas_e):
    
    #P(x,y)
    Px=Gauss(X,xa,sigma_a)
    Pyx=Gauss(Y,X,sigma_e)
    Py=Gauss(Y,xa,(sigma_a**2+sigma_e**2)**0.5)
    Pxy=Px*Pyx/Py
    Pxy_bayes=Pyx*Px
    #check P(x,y)
    
    S=np.array([[sigma_a**2,sigma_a**2],[sigma_a**2,sigma_a**2+sigma_e**2]])
    S_sqrtm=np.linalg.det(sqrtm(S))
    Pxy_check=X*0
    for i in range(len(x)):
        for j in range(len(y)):
            xx=np.array([x[i]-xa,y[j]-xa])
            exponent=np.matmul(np.matmul(xx.T,np.linalg.inv(S)),xx)
            Pxy_check[j,i]=1/(2*np.pi)/S_sqrtm*np.exp(-0.5*exponent)
            
    print('Max error on P(x,y) = '+str(np.max(np.abs(Pxy_bayes-Pxy_check))))
    
    #retrieval mean
    y_hat=((sigma_a**2+sigma_e**2)*x-xa*sigma_e**2)/sigma_a**2
    
    ax = fig.add_subplot(gs[1, ctr+1])
    plt.pcolor(X,Y,Pxy_bayes,cmap='hot')
    plt.plot(x,x,'g',label=r'$kx$')
    plt.plot(x,y_hat,'m',label=r'$\hat{x}$')
    plt.xlim([-10,30])
    plt.ylim([-10,30])
    plt.grid()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    if ctr==0:
        plt.legend(draggable=True)
    
    plt.plot(x,x*0+y_sel,'--',color=[0.5,0.5,0.5])
    plt.plot(y*0+x_sel,y,'--',color=[0.5,0.5,0.5])
    
    #P(y|x)
    pyx=Gauss(Y,x_sel,sigma_e)
    ax = fig.add_subplot(gs[1, ctr])
    plt.plot(-pyx,y,'k')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    if ctr==0:
        plt.ylabel(r'$y$')
    plt.xlabel(r'$P(y|x)$')
    
    #P(x|y)
    ax = fig.add_subplot(gs[0, ctr+1])
    pxy=Gauss(x,xa,sigma_a)*Gauss(y_sel,x,sigma_e)/Gauss(y_sel,xa,(sigma_a**2+sigma_e**2)**0.5)
    
    plt.plot(x,pxy,'k')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.ylabel(r'$P(x|y)$')
    
    #P(x)
    ax = fig.add_subplot(gs[2, ctr+1])
    plt.plot(x,Gauss(x,xa,sigma_a),'k')
    plt.xlim([-10,30])
    plt.xlabel(r'$x$')
    plt.ylabel(r'$P(x)$')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ctr+=2
    

    
    
    
    