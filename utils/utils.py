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

    
def RF_feature_selector(X,y,test_size=0.8,n_search=30,n_repeats=10,limits={}):
    '''
    Feature importance selector based on random forest. The optimal set of hyperparameters is optimized through a random search.
    Importance is evaluated through the permutation method, which gives higher scores to fatures whose error metrics drops more after reshuffling.
    '''
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance
    from scipy.stats import randint
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.metrics import mean_absolute_error

    #build train/test datasets
    data = np.hstack((X, y.reshape(-1, 1)))

    data = data[~np.isnan(data).any(axis=1)]
    train_set, test_set = train_test_split(data, random_state=42, test_size=test_size)

    X_train = train_set[:,0:-1]
    y_train = train_set[:,-1]

    X_test = test_set[:,0:-1]
    y_test = test_set[:,-1]
    
    #default grid of hyperparamters (Bodini and Optis, 2020)
    if limits=={}:
        p_grid = {'n_estimators': randint(low=10, high=100), # number of trees
                  'max_features': randint(low=1,high= 6), # number of features to consider when looking for the best split
                  'min_samples_split' : randint(low=2, high=11),
                  'max_depth' : randint(low=4, high=10),
                  'min_samples_leaf' : randint(low=1, high=15)
            }
        
    # Choose cross-validation techniques for the inner and outer loops,
    # independently of the dataset.
    forest_reg = RandomForestRegressor()
    rnd_search = RandomizedSearchCV(forest_reg, param_distributions = p_grid, n_jobs = -1,
                                    n_iter=n_search, cv=5, scoring='neg_mean_squared_error')
    rnd_search.fit(X_train, y_train)
    print('Best set of hyperparameters found:')
    print(rnd_search.best_estimator_)

    predicted_test = rnd_search.best_estimator_.predict(X_test)
    test_mae = mean_absolute_error(y_test, predicted_test)
    print("Average testing MAE:", test_mae)

    predicted_train = rnd_search.best_estimator_.predict(X_train)
    train_mae = mean_absolute_error(y_train, predicted_train)
    print("Average training MAE:", train_mae)

    best_params=rnd_search.best_estimator_.get_params()    
        
    #random forest prediction with optimized hyperparameters
    reals=np.sum(np.isnan(np.hstack((X, y.reshape(-1, 1)))),axis=1)==0
    rnd_search.best_estimator_.fit(X[reals,:], y[reals])
        
    y_pred=y+np.nan
    y_pred[reals] = rnd_search.best_estimator_.predict(X[reals])
       
    reals=~np.isnan(y+y_pred)
    result = permutation_importance(rnd_search.best_estimator_, X[reals], y[reals], n_repeats=n_repeats, random_state=42, n_jobs=2)

    importance=result.importances_mean
    importance_std=result.importances_std
    
    return importance,importance_std,y_pred,test_mae,train_mae,best_params
