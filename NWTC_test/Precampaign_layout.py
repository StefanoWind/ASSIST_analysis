# -*- coding: utf-8 -*-
"""
Plot topogrpahic map for precampaign ASSIST tests

2025-06-06-created
"""

import os
cd=os.getcwd()
import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
import warnings
import matplotlib
import pandas as pd
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import utm
from matplotlib.markers import MarkerStyle

warnings.filterwarnings('ignore')
plt.close('all')

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['savefig.dpi'] = 300  

#%% Inputs
source=os.path.join(cd,'data/FC_topo_v2.nc')
source_nwtc='data/NWTC.xlsx'

#domain limits [m from M5]
xlim=[-1000,500]
ylim=[-1000,500]

#%% Functions
def draw_turbine_top(ax,x,y,d,facecolor='w',edgecolor='k'):
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]  # 6 points (3 outer, 3 inner)
    outer_radius = d/2
    inner_radius = d/20
    coords = []

    for i, angle in enumerate(angles):
        r = outer_radius if i % 2 == 0 else inner_radius
        xi = r * np.cos(angle)+x
        yi = r * np.sin(angle)+y
        coords.append((xi, yi))

    coords.append(coords[0])  # close the shape
    path=Path(coords)
    patch = PathPatch(path, facecolor=facecolor, edgecolor=edgecolor, lw=1)
    ax.add_patch(patch)
    
def three_point_star():
    # Points of a 3-pointed star (scaled and centered)
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]  # 6 points (3 outer, 3 inner)
    outer_radius = 1
    inner_radius = 0.1
    coords = []

    for i, angle in enumerate(angles):
        r = outer_radius if i % 2 == 0 else inner_radius
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        coords.append((x, y))

    coords.append(coords[0])  # close the shape
    return Path(coords)

#%% Initialization
Data=xr.open_dataset(source)
FC=pd.read_excel(source_nwtc).set_index('Site')

#locations
xy=utm.from_latlon(FC['Lat'].values,FC['Lon'].values)
xref=utm.from_latlon(FC['Lat']['M5'],FC['Lon']['M5'])[0]
yref=utm.from_latlon(FC['Lat']['M5'],FC['Lon']['M5'])[1]
FC['x']=xy[0]-xref
FC['y']=xy[1]-yref

#turbine locations
x_turbine=[]
y_turbine=[]
d_turbine=[]
for s in FC.index:
    if FC['Description'][s]=='Wind turbine':
        x_turbine.append(FC['x'][s])
        y_turbine.append(FC['y'][s])
        d_turbine.append(FC['Diameter'][s])
        
#grid
x=Data.x.values-xref
y=Data.y.values-yref
Z=Data.z.values.T

sel_x=(x>xlim[0])*(x<xlim[1])
sel_y=(y>ylim[0])*(y<ylim[1])

#%% Plots
plt.close('all')
star_marker = MarkerStyle(three_point_star())

plt.figure(figsize=(18,8))
cf=plt.contourf(x[sel_x],y[sel_y], Z[:,sel_x][sel_y,:],np.arange(1780,1850,5),cmap='summer',extend='both')
ax=plt.gca()
ct=plt.contour(x[sel_x],y[sel_y], Z[:,sel_x][sel_y,:],np.arange(1780,1850,5),colors='k',linewidths=0.5,alpha=0.5)
plt.plot(FC['x']['M5'],FC['y']['M5'],'^k',markersize=10)
plt.plot(FC['x']['Site 3.2'],FC['y']['Site 3.2'],'sk',markersize=7)
plt.plot(FC['x']['Site 4.0'],FC['y']['Site 4.0'],'sk',markersize=7)
plt.plot(FC['x']['Ceilometer'],FC['y']['Ceilometer'],'vk',markersize=7)
for xt,yt,dt in zip(x_turbine,y_turbine,d_turbine):
    draw_turbine_top(ax,xt,yt,dt)

plt.xlim(xlim)
plt.ylim(ylim)  
plt.colorbar(cf,label='Terrain elevation [m.a.s.l.]')

ax.clabel(ct, inline=True, fontsize=10)
ax.set_aspect('equal')
plt.xlabel('W-E [m]')
plt.ylabel('S-N [m]')

