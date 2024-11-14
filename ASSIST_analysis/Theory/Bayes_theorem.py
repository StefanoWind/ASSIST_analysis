# -*- coding: utf-8 -*-
"""
Retrieval for linear system
"""
import os
cd=os.path.dirname(__file__)
import sys
sys.path.append('C:/Users/SLETIZIA/OneDrive - NREL/Desktop/PostDoc/utils')
import utils as utl
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 14

#%% Inputs
n=3
m=5
L=100000
sigma_ratio=.1#noise max stdev to state stdev

#%%Main

#build state vector
x_a=np.random.rand(n)
random_matrix = (np.random.rand(n, n)-0.5)*2
S_a = np.dot(random_matrix, random_matrix.T)
    
x=np.random.multivariate_normal(x_a, S_a, L)

#build observations
random_matrix = (np.random.rand(m, m)-0.5)*2*sigma_ratio
S_e = np.dot(random_matrix, random_matrix.T)

K=np.random.rand(m,n)
e=np.random.multivariate_normal(np.zeros(m), S_e, L)
y=np.matmul(K,x.T).T+e

#retrieval
S=np.linalg.inv(np.matmul(np.matmul(K.T,np.linalg.inv(S_e)),K)+np.linalg.inv(S_a))
x_r=x_a+np.matmul(np.matmul(np.matmul(S,K.T),np.linalg.inv(S_e)),(y-np.matmul(K,x_a.T).T).T).T

S_a_check = np.cov(x, rowvar=False)
S_e_check = np.cov(e, rowvar=False)
S_check = np.cov(x-x_r, rowvar=False)


print(np.min(S[np.eye(n, dtype=bool)]))
print(np.min(np.linalg.eig(S_e)[0]))
print(np.linalg.det(S_e))

#%% Plots
plt.figure(figsize=(18,6))
for i in range(n):
    plt.subplot(1,n,i+1)
    utl.plot_lin_fit(x[:,i],x_r[:,i])
    plt.gca().grid(True)
    plt.xlabel(r'$x_'+str(i+1)+'$')
    plt.ylabel(r'$\hat{x}_'+str(i+1)+'$')
plt.tight_layout()


plt.figure(figsize=(18,6))
for i in range(n):
    plt.subplot(1,n,i+1)
    hist, bin_edges = np.histogram(x[:,i]-x_r[:,i], bins=30) 
    hist=hist/np.trapz(hist,utl.mid(bin_edges))
    plt.plot(utl.mid(bin_edges),hist,color='k')
    plt.plot(utl.mid(bin_edges),1/((2*np.pi)**0.5*S[i,i]**0.5)*np.exp(-0.5*(utl.mid(bin_edges)/(S[i,i]**0.5))**2),'-r')
plt.show()