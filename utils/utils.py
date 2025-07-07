# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 13:43:10 2025

@author: sletizia
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

def filt_stat(x,func,perc_lim=[5,95]):
    '''
    Statistic with percentile filter
    '''
    x_filt=x.copy()
    lb=np.nanpercentile(x_filt,perc_lim[0])
    ub=np.nanpercentile(x_filt,perc_lim[1])
    x_filt=x_filt[(x_filt>=lb)*(x_filt<=ub)]
       
    return func(x_filt)

def filt_BS_stat(x,func,p_value=5,M_BS=100,min_N=10,perc_lim=[5,95]):
    '''
    Statstics with percentile filter and bootstrap
    '''
    x_filt=x.copy()
    lb=np.nanpercentile(x_filt,perc_lim[0])
    ub=np.nanpercentile(x_filt,perc_lim[1])
    x_filt=x_filt[(x_filt>=lb)*(x_filt<=ub)]
    
    if len(x)>=min_N:
        x_BS=bootstrap(x_filt,M_BS)
        stat=func(x_BS,axis=1)
        BS=np.nanpercentile(stat,p_value)
    else:
        BS=np.nan
    return BS

def bootstrap(x,M):
    '''
    Bootstrap sample drawer
    '''
    i=np.random.randint(0,len(x),size=(M,len(x)))
    x_BS=x[i]
    return x_BS

def datenum(string,format="%Y-%m-%d %H:%M:%S.%f"):
    '''
    Turns string date into unix timestamp
    '''
    from datetime import datetime
    num=(datetime.strptime(string, format)-datetime(1970, 1, 1)).total_seconds()
    return num

def datestr(num,format="%Y-%m-%d %H:%M:%S.%f"):
    '''
    Turns Unix timestamp into string
    '''
    from datetime import datetime
    string=datetime.utcfromtimestamp(num).strftime(format)
    return string



def plot_lin_fit(x, y, bins=50, cmap='Greys',ax=None,cax=None):

    # Remove NaNs
    mask = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[mask], y[mask]

    # Linear regression
    slope, intercept, r_value, _, _ = linregress(x, y)
    y_fit = slope * x + intercept
    rmsd = np.sqrt(np.mean((y - y_fit)**2))
    r_squared = r_value**2

    # Plot setup
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # 2D histogram
    h = ax.hist2d(x, y, bins=bins, cmap=cmap)
    if cax is not None:
        plt.colorbar(h[3], ax=ax,cax=cax, label='Counts')

    # Regression line
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot([np.min(x),np.max(x)],[np.min(x),np.max(x)],'--b')
    ax.plot(x_line, slope * x_line + intercept, color='red', linewidth=2, label='Linear fit')
    
    # Stats textbox
    textstr = '\n'.join((
        f'Intercept: {intercept:.2f}',
        f'Slope: {slope:.2f}',
        f'RMSD: {rmsd:.2f}',
        r'$R^2$: {:.2f}'.format(r_squared)
    ))
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.set_aspect("equal")
    plt.show()

    
    