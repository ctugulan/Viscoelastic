# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 21:28:46 2021

@author: Claudia
"""

import numpy as np
import numpy.ma as ma
import scipy as sc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import LightSource

def surface_full(N,M,dx,dy,x1,zeta):
    x = dx*sc.r_[:N] + x1
    y = dy*sc.c_[:M] 

    # extend domain via symmetry
    yfull = np.vstack((np.flip(-y[1:,:]),y))
    zfull = np.vstack((np.flip(zeta[1:,:],axis=0),zeta))
    xx,yy = np.meshgrid(x, yfull) 

    fig = plt.figure(dpi=600)
    ax = fig.gca(projection='3d', proj_type='ortho')
    

    ax.set_xlabel(r'$x$')  
    ax.set_ylabel(r'$y$', rotation=0)  
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel(r'$\zeta$',rotation=0)   
    # #ax.grid(True)

    # create light source object.
    ls = LightSource(azdeg=-10, altdeg=45)
    rgb = ls.shade(zfull, cmap=cm.ocean, vert_exag=1.5, blend_mode='soft')
    surf = ax.plot_surface(xx, yy, zfull , rstride=1, cstride=1, facecolors=rgb, alpha=.8)
    
    ''' uncomment to add contour plot and color bar legend '''
    
    cset = ax.contourf(xx, yy, zfull, zdir='z', offset=np.min(zfull), cmap=cm.ocean)
    cbar=fig.colorbar(cset, shrink=0.5)
    cbar.set_label(r"$\zeta$",rotation=0)


    
    #ax.view_init(elev=30., azim=-95)
    plt.tight_layout()
    plt.show()
    
    
def plot_jac(J):
    # replace all zero matrix elements with 10e-8 (default precision of numpy arrays)
    mj = ma.masked_values(np.abs(J),0.).filled(10.**-8)
    data = np.log10(mj)

    plt.figure(figsize=(6, 5), dpi=600)

    plt.imshow(data,cmap='binary',vmin=-8.0,vmax=1.0)
    plt.colorbar()
     
    #plt.savefig(pltname)
    plt.show()
    
def Bifurcation_plot(z,F_start,Fmin,step):
    plt.figure(dpi=600)
    F_end=Fmin+step
    F = np.arange(F_start,F_end,-step)
    plt.plot(F, z,'k--')
    plt.ylabel(r'$\zeta (0,0)$')
    plt.xlabel(r'$f_L$')
    
    plt.tight_layout()
    plt.show()