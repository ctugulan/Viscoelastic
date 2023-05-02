# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 11:30:20 2021
save dense submatrix
"""

import numpy as np
import time
from WavesUnderIce import auxfuncs
import jacfuncs_simple


# Defining parameters
N, M = 160, 80
deltaX,deltaY = 0.3, 0.3
x0 = -24.
H = 1.


# Get parameters as strings
deltaXname=str(deltaX).replace('.','')
deltaYname=str(deltaY).replace('.','')
Hname=str(H).replace('.','')
x0name=str(x0).replace('.','')
# Save dense submatrix
fnameDense = 'dense_n'+str(N)+'m'+str(M)+'dx'+deltaXname+'dy'+deltaYname+'H'+Hname+'x0'+x0name
    
start_time = time.time()
C_dense = jacfuncs_simple.densePhi(M,N,deltaX,deltaY,x0,H)
D_dense = jacfuncs_simple.denseMat(M,N,deltaX,deltaY,x0,H)
#C_dense = np.load('phidense_n60m60dx04dy08H10.npy')
end_time = time.time()
auxfuncs.timer(start_time,end_time,'time to generate dense phi submatrix')
np.save('phi'+fnameDense,C_dense)# save dense submatrix
np.save('zeta'+fnameDense,D_dense)# save dense submatrix

