# -*- coding: utf-8 -*-
"""
Plot topogrpahic map of AWAKEN ASSISTs

"""
import os
cd=os.getcwd()
import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
import warnings
import matplotlib
from matplotlib.path import Path
from matplotlib.markers import MarkerStyle

warnings.filterwarnings('ignore')
plt.close('all')

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['savefig.dpi'] = 300

#%% Inputs
source=os.path.join(cd,'data/20250225_AWAKEN_layout.nc')

farms_sel=['Armadillo Flats','King Plains','Breckinridge']
sites_sel=['B','C1a','G']
site_rad='H'
sites_ceil=['A1','H','C1','E37']
site_ref='C1a'

#graphics
xlim=[-22000,15000]
ylim=[-17000,12000]

#%% Functions
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
Topo=xr.open_dataset(source,group='topography')
Turbines=xr.open_dataset(source,group='turbines').rename({'Wind plant':'wind_plant'})
Sites=xr.open_dataset(source,group='ground_sites')

x_ref=float(Sites.x_utm.sel(site=site_ref))
y_ref=float(Sites.y_utm.sel(site=site_ref))

#%% Plots
plt.close('all')
star_marker = MarkerStyle(three_point_star())

plt.figure(figsize=(18,8))
cf=plt.contourf(Topo.x_utm-x_ref,Topo.y_utm-y_ref, Topo.elevation.T,np.arange(290,390,5),cmap='summer',extend='both')
ax=plt.gca()
ct=plt.contour(Topo.x_utm-x_ref,Topo.y_utm-y_ref, Topo.elevation.T,np.arange(290,390,5),colors='w',linewidths=0.5,alpha=0.5)
for site in sites_sel:
    plt.plot(Sites.x_utm.sel(site=site)-x_ref,Sites.y_utm.sel(site=site)-y_ref,'sr',markersize=7)
    
plt.plot(Sites.x_utm.sel(site=site_rad)-x_ref,Sites.y_utm.sel(site=site_rad)-y_ref,'.c',markersize=20)
for site in sites_ceil:
    if Sites.x_utm.sel(site=site)-x_ref<xlim[0]:
         plt.plot(xlim[0]+200,Sites.y_utm.sel(site=site)-y_ref,'^b',markersize=7,markerfacecolor="none")
    elif Sites.x_utm.sel(site=site)-x_ref>xlim[1]:
         plt.plot(xlim[1]-200,Sites.y_utm.sel(site=site)-y_ref,'^b',markersize=7,markerfacecolor="none")
    elif Sites.y_utm.sel(site=site)-y_ref<ylim[0]:
         plt.plot(Sites.x_utm.sel(site=site)-x_ref,ylim[0]+200,'^b',markersize=7,markerfacecolor="none")
    elif Sites.y_utm.sel(site=site)-y_ref>ylim[1]:
         plt.plot(Sites.x_utm.sel(site=site)-x_ref,ylim[1]-200,'^b',markersize=7,markerfacecolor="none")
    
    else:
        plt.plot(Sites.x_utm.sel(site=site)-x_ref,Sites.y_utm.sel(site=site)-y_ref,'^b',markersize=7)


for wf in farms_sel:
    x_turbine=Turbines.x_utm.where(Turbines.wind_plant==wf).values-x_ref
    y_turbine=Turbines.y_utm.where(Turbines.wind_plant==wf).values-y_ref
    for xt,yt in zip(x_turbine,y_turbine):
        plt.plot(xt,yt,'xk', marker=star_marker, markersize=10, color='k')


plt.xlim(xlim)
plt.ylim(ylim)  
plt.colorbar(cf,label='Terrain elevation [m.a.s.l.]')

ax.clabel(ct, inline=True, fontsize=10)
ax.set_aspect('equal')
plt.xlabel('W-E [m]')
plt.ylabel('S-N [m]')

