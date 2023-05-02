
"""
Created on Thu Sep  2 00:51:31 2021

Clean code to compute flexural term using linear bi-Laplacian model for ice
"""

import numpy as np
import scipy as sc

import auxfuncs

def Bilaplacian(zeta,N,M,deltaX,deltaY,x0,Fr,beta,tau):
    '''
    BiLaplacian linear model for the ice sheet
    Parameters
    ----------
    z : zeta
    deltaX, deltaY: mesh spacing
    Fr : Froude number to determine ghost points in U</>cmin regimes

    Returns
    -------
    PflexHalf : PFlex at the half-mesh points
    '''
    fmin = 0.472 # fr for which the minimum phase speed is reached
    if fmin < Fr:
        ghosts = np.pad(zeta,((0,2),(0,2)),'constant',constant_values=0)
    elif fmin > Fr:
        downstream_ghost1 = 2*zeta[:,-1:]-zeta[:,-2:-1]
        downstream_ghost2 = 2*downstream_ghost1-zeta[:,-1:]
        downstream_ghosts = np.hstack((zeta,downstream_ghost1,downstream_ghost2))
        lateral_ghost1 = 2*downstream_ghosts[-1:,:]-downstream_ghosts[-2:-1,:]
        lateral_ghost2 = 2*lateral_ghost1-downstream_ghosts[-1:,:]
        ghosts = np.vstack((downstream_ghosts,lateral_ghost1,lateral_ghost2))
    f = np.pad(ghosts,((2,0),(0,0)),'reflect')
    
    # initialize arrays for derivatives
    fYYYY = np.zeros_like(zeta)
    fXXXX = np.zeros_like(zeta)
    fXXYY = np.zeros_like(zeta)

    # j=0,...,M and i=0,...,N-1
    fYYYY[:,:]=(f[4:,:-2]-4*f[3:-1,:-2]+6*f[2:-2,:-2] -4*f[1:-3,:-2]+f[0:-4,:-2])/deltaY**4

    # j=0,...,M and i=2,...,N-1
    fXXXX[:,2:]=(f[2:-2,4:]-4*f[2:-2,3:-1]+ 6*f[2:-2,2:-2]-4*f[2:-2,1:-3]+f[2:-2,0:-4])/deltaX**4
    # j=0,...,M and i=0
    fXXXX[:,0]=(f[2:-2,4]-4*f[2:-2,3]+ 6*f[2:-2,2]-4*f[2:-2,1]+f[2:-2,0])/deltaX**4
    # j=0,...,M and i=1
    fXXXX[:,1]=fXXXX[:,0]
    
    # j=0,...,M and i=1,...,N-1 
    fXXYY[:,1:] = (f[3:-1,2:-1] + f[3:-1,:-3] + f[1:-3,2:-1] + f[1:-3,:-3] 
         -2*(f[2:-2,2:-1]+f[2:-2,:-3]+f[3:-1,1:-2]+f[1:-3,1:-2]) +4*f[2:-2,1:-2] )/deltaX**2/deltaY**2
    # j=0,...,M and i=0
    fXXYY[:,0] = (f[3:-1,2] + f[3:-1,0] + f[1:-3,2] + f[1:-3,0] 
         -2*(f[2:-2,2]+f[2:-2,0]+f[3:-1,1]+f[1:-3,1]) +4*f[2:-2,1] )/deltaX**2/deltaY**2

    BL = fXXXX+fYYYY+2*fXXYY
    
    # viscoelastic pflex
    BLdiff =  auxfuncs.XXderiv(BL,deltaX)
    visc = BL+ tau*BLdiff#np.gradient(BL,deltaX,axis=1, edge_order=1)
    
    # PFlex for linear model
    D = DFunction(M,N,deltaX,deltaY,x0,beta)
    #Pflex = np.multiply(D,BL)
    Pflex = np.multiply(D,visc)
    PflexHalf = (Pflex[:,1:N] + Pflex[:,0:N-1])/2.
    return PflexHalf


def DFunction(M,N,deltaX,deltaY,x0,beta):
    '''
    Parameters
    ----------
    M,N : integrers - define size of mesh
    deltaX, deltaY: mesh spacing
    beta : nondimensional flexural rigidity parameter

    Returns
    -------
    D0 : variable flexural rigidity function for the ice (heterogeneous)
    '''
    x = deltaX*sc.r_[:N] + x0
    y = deltaY*sc.c_[:M]
    
    D0 = beta
    
    #Dx3 = beta*(1/2*np.sin(20*np.pi/(x[-1]-x[0])*x) + 1)
    #D0y = beta*(1/(1. + np.exp(0.5*(y-np.max(y)/2)))+1)# fat in the middle
    #D0y_channel = beta*(1/(1. + np.exp(-0.5*(y-np.max(y)/2)))+1)# thin in the middle
    #Dy_sinusoidal = 1/4*np.cos(10*np.pi/np.max(y)*y)+3/4
    return D0

'''
Biharmonic from Abramowitz and Stegun 13-point stencil
Domain is padded upstream and downstream with zeros
on latteral y=0 edge, domain is padded with reflection
'''
def biharmonic(f,deltaX,Db):
    func=f
    '''
    Other padding types: edge,linear_ramp, mean,symmetric, wrap
    '''
    pad_width=2 # padding width at each boundary (pad_latteral, pad_upstream, pad_doenstream)
    fpd=np.pad(func,((0,pad_width),(pad_width,pad_width)),'constant',constant_values=0) #f padded at (0,latteral),(upstream,downstream)
    f= np.pad(fpd,((pad_width,0),(0,0)),'reflect')# reflect in y=0 axis

    bhrmnc=np.zeros_like(func)

    start_index, end_index = 0, -1 #numpy [row_index_,column_index], slice [start_index:end_index]
    ind0 = start_index+pad_width 
    indf = end_index-pad_width+1# has to do with how array includes last elempent in slice
        
    bhrmnc[:,:]=20*f[ind0:indf,ind0:indf] -8*(f[ind0+1:indf+1,ind0:indf]+f[ind0:indf,ind0+1:indf+1]+f[ind0-1:indf-1,ind0:indf]+f[ind0:indf,ind0-1:indf-1])\
    +2*(f[ind0+1:indf+1,ind0+1:indf+1]+f[ind0+1:indf+1,ind0-1:indf-1]+f[ind0-1:indf-1,ind0+1:indf+1]+f[ind0-1:indf-1,ind0-1:indf-1])\
    +(f[ind0:indf,ind0+2:]+f[ind0+2:,ind0:indf]+f[ind0-2:indf-2,ind0:indf]+f[ind0:indf,ind0-2:indf-2])
    bhrmnc/=deltaX**4
    
    pflex = Db*bhrmnc
    pflexHalf = (pflex[:,1:] + pflex[:,0:-1])/2.
    return pflexHalf



