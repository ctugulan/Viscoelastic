# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 16:20:00 2022

@author: Claudia
"""


import numpy as np
from scipy import optimize
import time
from WavesUnderIce import auxfuncs
import BIFunction_simple
import jacfuncs_simple

from WavesUnderIce import PlotSurface

# Defining parameters
N,M, deltaX, deltaY, x0, H = 40,20,0.8,0.8,-16.,1.
Fh=0.6
Fr = 1./Fh**2.
mu=0.
tau=0.
beta=0.
epsilon=1.
Lx, Ly = 1., 1.
# Initial guess: flat surface
uInit = np.vstack((np.tile(np.vstack(([x0],np.ones((N,1)))),(M,1)),np.zeros((M*(N+1),1))))

fname=auxfuncs.get_filename(N,M,deltaX,deltaY,Fh,Lx,beta,mu,tau,False) #True for long name

Jnum = jacfuncs_simple.NumJacobian(uInit,M,N,deltaX,deltaY,x0,Fr,epsilon,Lx,Ly,H,beta,mu,tau)
np.save('J_infiniteDepth_'+fname,Jnum)
PlotSurface.plot_jac(Jnum)