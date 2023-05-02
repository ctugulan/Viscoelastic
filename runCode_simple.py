"""
runCode.py - main function that runs all the code
with SIMPLIFIED BOUNDARY CONDITIONS
'''
# dimensional truck parameters
Mg=36e3
Lx = 3.33#1.
Ly =1.68#1.
U=2.5
depth = 2.#1.
hi = 0.37

epsilon, beta, Fh, strain=auxfuncs.dimensionalQtys(Mg,Lx,Ly,U,depth,hi)

#Fh=1.1288#0.7
Fr = 1./Fh**2.
H=depth/Ly
#beta = 191.7#1.
#epsilon = 1.275#1.
'''

# dimensional parameters
aleph = 2**4. #aleph=D/(rho*g)
g = 9.8 # gravity
Hdim = 6.8 #depth
tauf = 0.1
U = 5.78 #speed

# nondimensional parameters
H = 1.
beta = aleph*g/(U**2.*Hdim**3.)
Fr = g*Hdim/U**2.
Fh = U/np.sqrt(g*Hdim)# only printing this for comparison
tau = tauf*U/Hdim
Lx, Ly = 1.,1.#0.51, 0.51
epsilon = 0.1 #/U**2.
mu = 0.1 

"""
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import numpy as np
from scipy import optimize
import time


from Shared import auxfuncs
import BIFunction_simple
import jacfuncs_simple
from Shared import PlotSurface

# Defining mesh parameters
N, M = 80, 40
deltaX,deltaY = 0.6, 0.6
x0 = -(deltaX*N)/2.

# Initial guess: flat surface
uInit = np.vstack((np.tile(np.vstack(([x0],np.ones((N,1)))),(M,1)),np.zeros((M*(N+1),1))))

# nondimensional parameters
H = 1.
beta = 0.5 
F = 1/(0.7**2.)
Fh = 1/np.sqrt(F)# only printing this for comparison
tau = 0.
epsilon = 1. 
mu = 0. 
Lx, Ly = 1.,1.#0.51, 0.51

ice_bool = False # true for biharmonic model centered differences

# Printing the parameters:
print(" N = "+str(N)+ "\n M = "+str(M) + "\n \u0394x = " +str(deltaX) + "\n \u0394y = "+str(deltaY) + "\n x\u2080 = " +str(x0) + "\n F = " +str(F) + "\n \u03b5 = " +str(epsilon) + "\n \u03b2 = " +str(beta) + "\n \u03bc = " +str(mu) + "\n \u03c4 = " +str(tau))

pfname=auxfuncs.get_filename(N,M,deltaX,deltaY,F,Lx,beta,mu,tau,True)

'''Get the preconditioner'''
start_time = time.time()
M_ice = jacfuncs_simple.GetIcePreconditioner(uInit, M, N, deltaX, deltaY, x0,H, F, beta, mu,tau,ice_bool)
end_time = time.time()
auxfuncs.timer(start_time,end_time,'time to compute Jacobian analytically')
# np.save('P_'+pfname,M_ice) 

# M_ice = np.load('P_n80m40dx06dy06f07b05mu00tau00.npy')

# fname=auxfuncs.get_filename(N,M,deltaX,deltaY,U,Lx,Hdim,mu,tauf,True) #True for long name


# Jnum = jacfuncs_simple.NumJacobian(uInit,M,N,deltaX,deltaY,x0,Fr,epsilon,Lx,H,beta,mu,tau)
# np.save('J_infiniteDepth_'+fname,Jnum)
# PlotSurface.plot_jac(Jnum)
# J_flex = jacfuncs_simple.GetJacobian(uInit,M,N,deltaX,deltaY,x0,H,Fr,beta,mu,model=False)
# np.save('J_finiteDepth_'+fname,J_flex)
# PlotSurface.plot_jac(J_flex)
minLx, maxLx = 1,1
strainvec=np.zeros((maxLx,1))

for l in range(1,maxLx+1):
    Lx=l/1.
    print("\n L\u2093 = " +str(Lx))
    
    fname=auxfuncs.get_filename(N,M,deltaX,deltaY,F,Lx,beta,mu,tau,False) #True for long name
    print(fname)

    start_time = time.time()
    uNew = optimize.newton_krylov(lambda u: BIFunction_simple.BIFunction(u,M,N,deltaX,deltaY,x0,F,epsilon,Lx,Ly,H,beta,mu,tau,ice_bool), uInit, method='lgmres', inner_M=M_ice, verbose=True)
    end_time = time.time()
    auxfuncs.timer(start_time,end_time,'time for solver')

    [phi1, phix, zeta1, zetax] = auxfuncs.reshapingUnknowns(uNew,M,N)
    zeta = auxfuncs.allVals(zeta1,zetax,deltaX,M,N)
    # np.save('Results/variable_Lx/zeta_'+fname, zeta)
    
    znew=(zeta.T).reshape(M*N,1)
    # np.savetxt('Results/variable_Lx/zeta_'+fname+'.csv', znew, delimiter=',')

    PlotSurface.surface_full(N,M,deltaX,deltaY,x0,zeta)
    np

    
    strains=auxfuncs.strains(zeta,deltaX,deltaY,1.)
    strainvec[l-1,:]=strains
    print(strains)
'''    
np.save('Results/variable_Lx/strains_'+fname, strainvec)
np.savetxt('Results/variable_Lx/strains_'+fname+'.csv', strainvec, delimiter=',')
'''
    
#np.save('conv_'+fname,zeta)





