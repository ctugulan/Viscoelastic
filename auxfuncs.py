"""
Auxiliary funtions:
    getValues - Compute zeta, phi, and their dervivatives from the vector of unknowns

"""
import numpy as np
from scipy.linalg import eigh#for strains calculation
import sys

def reshapingUnknowns(u,M,N):
    # get phi1, phix, zeta1, zetax from vector of unknowns
    allPhi = u[:M*(N+1)] # all values of phi1 and phix
    allZet = u[M+N*M:] # all values of zeta1 and zetax
    indx = (N+1)*np.arange(M) # indices of phi1 and zeta1
    phi1 = allPhi[indx] # extract values of phi1
    zet1 = allZet[indx] # extract values of zeta1
    phix = np.delete(allPhi,indx).reshape(M,N) # extract phix as an MxN matrix
    zetx = np.delete(allZet,indx).reshape(M,N) # extract zetax as an MXN matrix
    return phi1, phix, zet1, zetx

def allVals(u1,ux,dx,M,N):
    v1 = u1.reshape(M,1)
    v  = v1 + np.zeros((N,))
    for ii in range(N-1):
        v[:,ii+1] = v[:,ii] + dx/2 * (ux[:,ii+1] + ux[:,ii])
    return v

# def getZeta(u,M,N,dx):
#     indx = (N+1)*np.arange(M) # indices of zeta1
#     zeta1 = u[indx] # extract values of zeta1
#     zetaX = np.delete(u,indx).reshape(M,N) # extract zetax as an MXN matrix
#     zeta = allVals(zeta1,zetaX,dx,M,N)
#     return zeta

def xDerivs(v,dx):
    # second order forward differentiation
    dv = -(v[:,2] - 4.*v[:,1] + 3.*v[:,0])/(2.*dx)
    return dv

def yDerivs(v,deltaY):
    f = np.pad(v,((1,0),(0,0)),'reflect')
    i=1
    f_y = np.vstack( (((-1*f[i-1:-i-1,:]+1*f[i+1:,:])/(2*deltaY)), ((v[-1,:]-v[-2,:])/(2*deltaY))))
    return f_y

'''
Finite difference schemes for the nonlinear model
'''
def Yderiv(f,deltaY):
    fY=np.zeros_like(f)
    fY[1,:]=(1*f[1,:]-8*f[0,:]+8*f[2,:]-1*f[3,:])/(12*deltaY)
    fY[2:-2,:]=(1*f[0:-4,:]-8*f[1:-3,:]+8*f[3:-1,:]-1*f[4:,:])/(12*deltaY) # interior points
    fY[-2,:]=(-1*f[-5,:]+6*f[-4,:]-18*f[-3,:]+10*f[-2,:]+3*f[-1,:])/(12*deltaY) # j=M-1
    fY[-1,:]=(3*f[-5,:]-16*f[-4,:]+36*f[-3,:]-48*f[-2,:]+25*f[-1,:])/(12*deltaY) # j=M
    return fY

def YYderiv(fY,deltaY):
    fYY=np.zeros_like(fY)
    fYY[0,:] = fY[1,:]/deltaY
    fYY[1:-1,:]=(fY[2:,:]-fY[0:-2,:])/(2*deltaY)
    fYY[-1,:]=(3*fY[-1,:]-4*fY[-2,:]+fY[-3,:])/(2*deltaY)
    return fYY
def XXderiv(fX,deltaX):
    fXX=np.zeros_like(fX)
    fXX[:,1]=fX[:,1]/deltaX
    fXX[:,1:-1]=(fX[:,1:-1]-fX[:,0:-2])/(2*deltaX)
    fXX[:,-1]=(3*fX[:,-1]-4*fX[:,-2]+fX[:,-3])/(2*deltaX)
    return fXX



'''
Here is the pressure function for the forcing term
'''
def pressure(x,y,Lx,Ly):
    '''
    Returns normalized pressure distribution scaled by Lx
    '''
    xInd = np.array((np.abs(x)<Lx),'double')
    yInd = np.array((np.abs(y)<Ly),'double')
    pInd = np.outer(xInd,yInd)
    p = np.exp(Lx**2./(x**2.-Lx**2.)+Ly**2./(y**2.-Ly**2.))*pInd.T 
    #p /= Lx # normalize the pressure distribution
    return p


def timer(start,end,string):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("--- {:0>2}:{:0>2}:{:05.2f} ".format(int(hours),int(minutes),seconds)+string+" ---")
    #original_stdout = sys.stdout # Save a reference to the original standard output
    with open('filename.txt', 'a+') as f:
        #sys.stdout = f # Change the standard output to the file we created.
        f.write("--- {:0>2}:{:0>2}:{:05.2f} ".format(int(hours),int(minutes),seconds)+string+" ---")
        #sys.stdout = original_stdout # Reset the standard output to its original value
    
def get_filename(N,M,deltaX,deltaY,Fr,Lx,beta,mu,tau, nametype):
    # Get parameters as strings
    deltaXname=str(deltaX).replace('.','')
    deltaYname=str(deltaY).replace('.','')
    betaname=str(round(beta,3)).replace('.','')
    muname=str(mu).replace('.','')
    tauname=str(round(tau,2)).replace('.','')
    Frname=str(round(Fr,2)).replace('.','')
    # Lname=str(Lx).replace('.','')
    # Lname = str(int(Lx))
    if nametype is True:
        fname = 'n'+str(N)+'m'+str(M)+'dx'+deltaXname+'dy'+deltaYname+'f'+Frname+'b'+betaname+'mu'+muname+'tau'+tauname #+'L'+Lname
    elif nametype is False:
        fname = 'n'+str(N)+'m'+str(M)+'dx'+deltaXname+'dy'+deltaYname
    return fname   


def dimensionalQtys(Mg,Lx,Ly,U,H,hi):
    gravity=9.81
    #rhoIce=917#kg/m^3
    rhoFluid=1024
    E=4.9e9
    D=E*hi**3/12/(1.-(1/3)**2.)
    P0=Mg/(Lx*Ly)/0.19713050882
    epsilon=P0/rhoFluid/U**2.
    F=U/np.sqrt(gravity*H)
    beta=D/(rhoFluid*U**2.*Ly**3.)
    strain=hi*P0/(2*rhoFluid*gravity*Ly*Lx)#dimensionless
    return epsilon, beta, F,strain

'''
def strains(f,deltaX,deltaY,s):
    # padding domain to compute derivatives
    #ghosts = np.pad(zeta,((0,2),(2,2)),'constant',constant_values=0)
    #f = np.pad(ghosts,((2,0),(0,0)),'reflect')
    i,j = 2,2
    #zetaXX = (1*f[j:-j,i-1:-i-1]-2*f[j:-j,i:-i] + 1*f[j:-j,i+1:-i+1])/(deltaX**2)
    zetaXX = (-1*f[j:-j,i+2:]+16*f[j:-j,i+1:-i+1]-30*f[j:-j,i:-i]+16*f[j:-j,i-1:-i-1]-1*f[j:-j,i-2:-i-2])/(12*deltaX**2)
    zetaYY = (-1*f[j+2:,i:-i]+16*f[j+1:-j+1,i:-i]-30*f[j:-j,i:-i]+16*f[j-1:-j-1,i:-i]-1*f[j-2:-j-2,i:-i])/(12*deltaY**2)
    zetaXY = (1*f[j-1:-j-1,i-1:-i-1] -1*f[j+1:-j+1,i-1:-i-1] -1*f[j-1:-j-1,i+1:-i+1] +1*f[j+1:-j+1,i+1:-i+1])/(4*deltaX*deltaY)
    return np.max(zetaXX)*s, np.max(zetaYY)*s, np.max(zetaXY)*s
'''

def strains(f,deltaX,deltaY,s):
    #computing the strains
    i,j = 2,2
    #zetaXX = (1*f[j:-j,i-1:-i-1]-2*f[j:-j,i:-i] + 1*f[j:-j,i+1:-i+1])/(deltaX**2)
    zetaXX = (-1*f[j:-j,i+2:]+16*f[j:-j,i+1:-i+1]-30*f[j:-j,i:-i]+16*f[j:-j,i-1:-i-1]-1*f[j:-j,i-2:-i-2])/(12*deltaX**2)
    zetaYY = (-1*f[j+2:,i:-i]+16*f[j+1:-j+1,i:-i]-30*f[j:-j,i:-i]+16*f[j-1:-j-1,i:-i]-1*f[j-2:-j-2,i:-i])/(12*deltaY**2)
    zetaXY = (1*f[j-1:-j-1,i-1:-i-1] -1*f[j+1:-j+1,i-1:-i-1] -1*f[j-1:-j-1,i+1:-i+1] +1*f[j+1:-j+1,i+1:-i+1])/(4*deltaX*deltaY)
    
    n=np.size(zetaXX,axis=1)
    m=np.size(zetaXX,axis=0)
    E_eigvals=np.zeros((m,n)) 
    for k in range(n):
        for l in range(m):
            E=np.array([[zetaXX[l,k],zetaXY[l,k]],[zetaXY[l,k],zetaYY[l,k]]])#strain tensor
            w = eigh(E, eigvals_only=True)# compute largest eigenvalue
            E_eigvals[l,k]=w[1]
    return np.max(np.abs(E_eigvals))*s