"""
This is where I compute the Jacobian of the linear problem
with SIMPLIFIED BOUNDARY CONDITIONS
used to generate the preconditioner
"""
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import numpy as np
import scipy as sc
from scipy.sparse import block_diag
from scipy.linalg import inv
from Shared import ice # for ice preconditioner
import time

from Shared import auxfuncs
import BIFunction_simple # for numerical jacobian


def TridiMatrix(N,M):
    E34 = np.zeros((2,N+1))
    E34[0,[0,1]]=[1,0]
    E34[1,[1,2]]=[1,0]

    dphix = (np.eye(N-1,N,k=0) + np.eye(N-1,N,k=1))/2.
    dphi = np.zeros((N-1,1))
    block = np.vstack((E34, np.hstack((dphi,dphix))))
    A = block_diag((block,)*M).toarray()
    if A.size != ((N+1)*M)**2:
        print(f"shape of matrix: {A.shape}")
    return A

def BlockMatrix(N,M,dx,const):
    T = np.tri(N-1)
    T += 2.*np.tril(T,-1) + np.tril(T,-2)
    v = np.ones((N-1,1))
    v[1:,:] *= 2.
    block = const*np.hstack((np.ones((N-1,1)),dx/4.*np.hstack((v,T))))
    block = np.vstack((np.zeros((2,(N+1))),block))
    return block_diag((block,)*M).toarray()

def integ(x,y,xH):
    # analytical solution 
    F1 = lambda s, t: t*(np.log(s+np.sqrt(s**2+t**2))+np.log(2))
    F2 = lambda s, t: s*(np.log(t+np.sqrt(s**2+t**2))+np.log(2))
    EVL= lambda f, t: f(sN, t) - f(s1, t)
    # xH = (x[1:]+x[:-1])/2.    
    slim = [[x[0]],[x[-1]]] - xH
    tlim = y[[0,-1],:][:,None] - y

    s1, sN = (slim[0], slim[1])
    t1, tM = (tlim[0], tlim[1])

    I = EVL(F2,tM) - EVL(F2,t1)
    # define masks
    mt1 = t1!=0.
    mtM = tM!=0.

    I[np.ravel(mt1),:] -= EVL(F1, t1[mt1][:,None])
    I[np.ravel(mtM),:] += EVL(F1, tM[mtM][:,None])

    I += EVL(F1,tM+2*y) + EVL(F2,tM+2*y) - EVL(F2,t1+2*y)

    mt0 = t1+2*y!=0.
    I[np.ravel(mt0),:] -= EVL(F1, (t1+2*y)[mt0][:,None])
    
    return I

def denseMatInf(M,N,deltaX,deltaY,x0):
    '''
    Returns
    -------
    dense submatrix for infinite depth

    '''
    # Defining the domain
    x = deltaX*sc.r_[:N] + x0
    y = deltaY*sc.c_[:M]

    xH = (x[1:]+x[0:-1])/2.
    I = integ(x,y,xH)
    
    all_K3 = None
    
    for jj in range(M):
        a = np.zeros((2,N*M+M))
        zind = jj*(N+1)
        a[0,[zind,zind+1]]=[-1,0]
        a[1,[zind+1,zind+2]]=[-1,0]
        for ii in range(N-1):
 
            # calculate K3(x,y;x*,y*)
            denom = lambda c: 1./np.sqrt((x-xH[ii])**2+(y-c*y[jj])**2)
            K3 = denom(1.) + denom(-1.)
            #print(f"(k={ii},l={jj})\n {I[jj,ii]}")
            
            # Apply weighting function
            K3[:,::N-1] /= 2. 
            K3[::M-1,:] /= 2.
            K3 *= -deltaX*deltaY
            # Calculate sum
            Sum = -K3.sum()
            
            #print(f"contition satisfied: {k3[jj,ii:ii+2]}\n")
            K3[jj, ii:ii+2] += (Sum - I[jj,ii])/2.
            
            # combine
            K3_new = np.hstack((np.zeros((M,1)),K3)).reshape((1,(N+1)*M))
            if ii % (N-1) == 0:
                K3_new = np.vstack((a,K3_new))
            
            if all_K3 is None:
                all_K3 = K3_new
            else:
                all_K3 = np.vstack((all_K3,K3_new))
                
    return -all_K3


'''
Here's the new dense submatrix for finite depth
'''

def denseMat(M,N,deltaX,deltaY,x0,H):
    '''
    denseMat is the dense zeta sub-matrix for finite depth H
    with kernel functions K5 and K6
    '''
    # Defining the domain
    x = deltaX*sc.r_[:N] + x0
    y = deltaY*sc.c_[:M]

    xH = (x[1:]+x[0:-1])/2.
    I = integ(x,y,xH)
    
    all_E2 = None
    
    for jj in range(M):
        E56 = np.zeros((2,N*M+M)) # simple radiation conditions E5 and E6
        zind = jj*(N+1)
        E56[0,[zind,zind+1]]=[1,0]
        E56[1,[zind+1,zind+2]]=[1,0]
        for ii in range(N-1):
 
            # calculate K5(x,y;x*,y*)
            K5denomYNeg = np.sqrt((x-xH[ii])**2+(y-y[jj])**2)
            K5denomYPos = np.sqrt((x-xH[ii])**2+(y+y[jj])**2)
            K5 = 1./K5denomYNeg + 1./K5denomYPos
            
            # calculate K6(x,y;x*,y*)
            K6denomYNeg = np.sqrt((x-xH[ii])**2+(y-y[jj])**2 + 4.*H**2.)
            K6denomYPos = np.sqrt((x-xH[ii])**2+(y+y[jj])**2 + 4.*H**2.)
            K6 = 1./K6denomYNeg + 1./K6denomYPos
            
            # Apply weighting function to K5
            K5[:,::N-1] /= 2. 
            K5[::M-1,:] /= 2.
            K5 *= -deltaX*deltaY
            
            # Apply weighting function to K5
            K6[:,::N-1] /= 2. 
            K6[::M-1,:] /= 2.
            K6 *= deltaX*deltaY
            
            #print(f"contition satisfied: {k3[jj,ii:ii+2]}\n")
            K5[jj, ii:ii+2] += (-K5.sum()- I[jj,ii])/2.
            
            # Add in the finite depth part
            K56 = -K5 + K6
            
            # combine
            E2_new = np.hstack((np.zeros((M,1)),K56)).reshape((1,(N+1)*M))
            if ii % (N-1) == 0:
                E2_new = np.vstack((E56,E2_new))
            
            if all_E2 is None:
                all_E2 = E2_new
            else:
                all_E2 = np.vstack((all_E2,E2_new))
                
    return all_E2

''' 
Here is where I try to calculate the contribution for the ice preconditioner 
'''
def IceSheet(u,N,M,deltaX,deltaY,x0,beta,Fr,tau,model):
    indx = (N+1)*np.arange(M) # indices of zeta1
    zeta1 = u[indx] # extract values of zeta1
    zetaX = np.delete(u,indx).reshape(M,N) # extract zetax as an MXN matrix
    zeta = auxfuncs.allVals(zeta1,zetaX,deltaX,M,N)
    
    if model is False:
        PFlexHalf = ice.Bilaplacian(zeta,N,M,deltaX,deltaY,x0,Fr,beta,tau)
    elif model is True:
        PFlexHalf = ice.biharmonic(zeta,deltaX,beta)
    
    bc = np.zeros((M,1))
    E1 = np.hstack((bc,bc,PFlexHalf)).reshape(M*(N+1),1)
    return E1[:,0]

def JFlex(u,N,M,dx,dy,x1,beta,Fr,tau,model):
    '''
    Function returns contribution to the Jacobian from the bilaplacian
    Calculate jacobian numerically using centered finite differences
    '''
    shift = 1e-10 # step size for centered finite difference calculation
    neq = M*(N+1) # number of equations for Jacobian submatrix
    J = np.zeros((neq,neq)) # Initialize
    # initial guess for zeta: flat surface
    #uInit= np.zeros((M*(N+1),1))
    uInit = u[M+N*M:] # all values of zeta1 and zetax
    for i in range(neq):
        y1 = uInit.copy()
        y2 = uInit.copy()
        y1[i]+= shift
        y2[i]-= shift
        f1 = IceSheet(y1,N,M,dx,dy,x1,beta,Fr,tau,model)
        f2 = IceSheet(y2,N,M,dx,dy,x1,beta,Fr,tau,model)
        J[:,i]=(f1-f2)/(2*shift)
    return J

'''
Here is where I calculate the contribution from the I3 integral for finite depth
'''
def densePhi(M,N,deltaX,deltaY,x0,H):
    x = deltaX*sc.r_[:N] + x0
    y = deltaY*sc.c_[:M] 
    xHalf = (x[1:]+x[:-1])/2.
    yHalf = y
    
    # calculating often used quantities
    yNegDiff = y - yHalf[:,None]
    yPosDiff = y + yHalf[:,None]
    xDiff = x - xHalf[:,None]
    
    indx = (N+1)*np.arange(M)# indices of phi1 
   
    all_E2 = None
    RadCdn = np.zeros((2,N*M+M)) # simple radiation conditions 
    for l in range(M):
        vec = np.zeros((1,(N+1)*M)) # vector of phi_nm differences
        for k in range(N-1):
        
            KdenomYNegFiniteDepth = np.sqrt(xDiff[k]**2.+yNegDiff[l]**2.+4*H**2.)
            KdenomYPosFiniteDepth = np.sqrt(xDiff[k]**2.+yPosDiff[l]**2.+4*H**2.)
            K6num = 2.*H
            K6 = K6num/KdenomYNegFiniteDepth**3. + K6num/KdenomYPosFiniteDepth**3.
            #print(f"(k={k},l={l})\n {K6}\n\n{K6.sum(axis=1)}\n")
        
            # Apply weighting function to K5
            K6[:,::N-1] /= 2. 
            K6[::M-1,:] /= 2.
            K6 *= deltaX*deltaY
            
            # calculating dE2(l,k)/dphi(1,m)
            vec[0,indx]=K6.sum(axis=1)
            
            # calculating dE2(l,k)/dphiX(m,n) for n=1
            vec[0][indx+1]=(K6[:,1:]*deltaX/2.).sum(axis=1) #for n=1
        
            # calculating dE2(l,k)/dphiX(m,n) for 1<n<N
            K6n = (K6[:,1:N-1])*deltaX/2. # K for 1<n<N
            for i in range(N-2):
                Sum1 = (K6[:,i+2:]).sum(axis=1)
                vec[0,indx+i+2]=deltaX*Sum1 + K6n[:,i]
        
            # calculating dE2(l,k)/dphiX(m,n) for n=N
            vec[0][indx+N]=(K6[:,-1])*deltaX/2. # for n=N
            
            # dE2(l,k)/dphiX(m,n) for current l,k
            E2_new = vec 
            
            # putting everything in the right place E2(1,1)...E2(N-1,1)
            if k % (N-1) == 0: # putting in the radiation conditions
                E2_new = np.vstack((RadCdn,E2_new))
            if all_E2 is None: # Initialize first row for l=1, k=1 
                all_E2 = E2_new
            else: # building up the submatrix stacking dE2(l,k)/dphiX(m,n) from previous l,k
                all_E2 = np.vstack((all_E2,E2_new))
    return all_E2


def GetIcePreconditioner(u,M,N,deltaX,deltaY,x0,H,Fr,beta,mu,tau,model):
    
    start_time = time.time()
    PFlex = JFlex(u,N,M,deltaX,deltaY,x0,beta,Fr,tau,model)
    end_time = time.time()
    auxfuncs.timer(start_time,end_time,'time to compute flexural part numerically')
    
    # comment this out if you already have a saved file
    # Get parameters as strings
    deltaXname=str(deltaX).replace('.','')
    deltaYname=str(deltaY).replace('.','')
    Hname=str(H).replace('.','')
    # Save dense submatrix
    fnameDense = 'dense_n'+str(N)+'m'+str(M)+'dx'+deltaXname+'dy'+deltaYname+'H'+Hname
    
    # time how long it takes to generate dense submatrix
    start_time = time.time()
    # D = denseMat(M,N,deltaX,deltaY,x0,H)
    D = np.load('zetadense_n80m40dx06dy06H10.npy')
    end_time = time.time()
    auxfuncs.timer(start_time,end_time,'time to generate dense zeta submatrix')
    # np.save('zeta'+fnameDense,D)# save dense submatrix
    
    start_time = time.time()
    # C_dense = densePhi(M,N,deltaX,deltaY,x0,H)
    C_dense = np.load('phidense_n80m40dx06dy06H10.npy')
    end_time = time.time()
    auxfuncs.timer(start_time,end_time,'time to generate dense phi submatrix')
    # np.save('phi'+fnameDense,C_dense)# save dense submatrix
    
    
    A = TridiMatrix(N,M) + BlockMatrix(N,M,deltaX,mu)
    B = BlockMatrix(N,M,deltaX,Fr) + PFlex
    C = BlockMatrix(N,M,deltaX,-2.*np.pi) + C_dense
    
    
    J = np.block([[A,B],[C,D]])
    
    start_time = time.time()
    J_inv = inv(J)
    end_time = time.time()
    auxfuncs.timer(start_time,end_time,'time to invert Jacobian')
    
    
    return J_inv


'''
Function to generate the linear Jacobian
'''
def GetJacobian(u,M,N,deltaX,deltaY,x0,H,Fr,beta,mu,model):
    
    fname=auxfuncs.get_filename(N,M,deltaX,deltaY,Fr,1.,beta,mu,True)
    
    start_time = time.time()
    # PFlex = icefuncs.jacobian_flexural(N,M,deltaX,deltaY,x0,beta,Fr,model)
    PFlex = JFlex(u,N,M,deltaX,deltaY,x0,beta,Fr,model)
    end_time = time.time()
    auxfuncs.timer(start_time,end_time,'time to compute flexural part numerically')
    
    A = TridiMatrix(N,M) + BlockMatrix(N,M,deltaX,mu)
    B = BlockMatrix(N,M,deltaX,Fr) + PFlex
    C = BlockMatrix(N,M,deltaX,-2.*np.pi)
    
    start_time = time.time()
    D = denseMat(M,N,deltaX,deltaY,x0,H)
    end_time = time.time()
    auxfuncs.timer(start_time,end_time,'time to generate dense submatrix')
    
    J = np.block([[A,B],[C,D]])
    
    np.save('J_linear_'+fname,J)
    
    return J


'''
Function to generate jacobian numerically
'''
def NumJacobian(uInit,M,N,deltaX,deltaY,x0,Fr,epsilon,Lx,Ly,H,beta,mu,tau):
    shift = 1e-8
    neq = 2*M*(N+1)
    J_nonlin = np.zeros((neq,neq))
    start_time = time.time()
    for i in range(neq):
        y1 = uInit.copy()
        y2 = uInit.copy()
        y1[i]+= shift
        y2[i]-= shift
        f1 = BIFunction_simple.BIFunction(y1,M,N,deltaX,deltaY,x0,Fr,epsilon,Lx,Ly,H,beta,mu,tau,model=False)
        f2 = BIFunction_simple.BIFunction(y2,M,N,deltaX,deltaY,x0,Fr,epsilon,Lx,Ly,H,beta,mu,tau,model=False)
        J_nonlin[:,i]=(f1-f2)/(2*shift)
    end_time = time.time()
    auxfuncs.timer(start_time,end_time,'time to generate numerical Jacobian')
    return J_nonlin




