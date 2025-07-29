# This program solves the 1D Kohn-Sham equation for two electrons in a
# harmonic-oscillator potential, using exchange-only LDA.
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import simpson   # integration with Simpson's rule
from scipy.integrate import quad    # integration with Gaussian quadrature
from scipy.special import kn        # The modified Bessel functions
#
#--------------------------------------------------------------------------
NGRID = 101   # number of grid points (always an odd number)
XMAX = 5.     # the numerical grid goes from -XMAX < x < XMAX
KSPRING = 1.  # spring constant of the harmonic oscillator potential
A = 1.      # Coulomb softening parameter
TOL = 1.e-10  # numerical tolerance (the convergence criterion)
MIX = 0.8    # mixing parameter for the self-consistency iterations
#--------------------------------------------------------------------------
#
DX = 2.*XMAX/(NGRID-1)  # grid spacing
PI = math.pi  # define pi here

x = np.linspace(-XMAX,XMAX,NGRID)  # Define the numerical grid as an array
#
# Now initialize a bunch of arrays
#
n = np.zeros(NGRID)
n_noninteracting = np.zeros(NGRID)
n1 = np.zeros(NGRID)
psi = np.zeros(NGRID)
di = np.zeros(NGRID)
VH = np.zeros(NGRID) 
VX = np.zeros(NGRID)
VEXT = np.zeros(NGRID)
vint = np.zeros(NGRID)
MAT = np.zeros((NGRID,NGRID)) 
#
# Initialize the density of the noninteracting 1D harmonic oscillator.
#
for i in range(NGRID):
    n_noninteracting[i]=2.*math.exp(-x[i]**2)/math.sqrt(PI)
    n[i] = n_noninteracting[i]
    n1[i] = n[i]
#
#  Three-point formula:
#
#for i in range(NGRID):
#    diag[i]=g(x[i]) + 1./DX**2
#for i in range(NGRID-1):
#    MAT[i,i+1] = -0.5/DX**2
#    MAT[i+1,i] = -0.5/DX**2
#
#  Seven-point formula:
#    
for i in range(NGRID):
    di[i]=0.5*KSPRING*x[i]**2 + 490./(360.*DX**2)
for i in range(NGRID-1):
    MAT[i,i+1] = -270./(360.*DX**2)
    MAT[i+1,i] = -270./(360.*DX**2)
for i in range(NGRID-2):
    MAT[i,i+2] = 27./(360.*DX**2)
    MAT[i+2,i] = 27./(360.*DX**2)
for i in range(NGRID-3):
    MAT[i,i+3] = -2./(360.*DX**2)
    MAT[i+3,i] = -2./(360.*DX**2)
#
#------------------------------------------------------------------------
# Here is the start of the self-consistency loop. We initialize the
# energy as that of two noninteacting electrons in a 1D harmonic potential
#------------------------------------------------------------------------
crit = 1.
EKS_previous = math.sqrt(KSPRING) 
counter = 0
while crit > TOL:
    counter += 1
    print(counter)
#
#  mix with the density of the previous iteration
#
    for i in range(NGRID):
        n[i] = MIX*n[i] + (1.-MIX)*n1[i]
        n1[i] = n[i]
#
# Calculate the Hartree potential VH:
#   
    for i in range(NGRID):
        for j in range(NGRID):
            vint[j]=n[j]/math.sqrt(A**2 + (x[i]-x[j])**2)
        result = simpson(vint,x=x)
        VH[i] = result
        MAT[i,i] = di[i] + VH[i]   
#
# Calculate the exchange potential VX:
#    
    for i in range(NGRID):        
        '''
        upper = n[i]*PI*A  
        def f(j):
            return kn(0,j)
        result, _ = quad(f,0.,upper)
        VX[i] = -result/(PI*A)
        '''
        z = A*n[i]
        uu = 0.0194*z + 1.06*z**2 + 0.5*z**3
        vv = 0.704 + 2.23*z + z**2
        VX[i] = -((0.0194 + 2.12*z + 1.5*z**2)*vv - uu*(2.23+2.*z))/(A*vv**2)
        
        MAT[i,i] = MAT[i,i] +VX[i]        
#
# Now find the eigenvalues and eigenvectors of the matrix MAT,
# and sort them to make sure we keep the lowest one only.
#
    vals, vecs = np.linalg.eig(MAT)
    sorted = np.argsort(vals)
    lowest = sorted[0]
    EKS = 2*vals[lowest]
    for i in range(NGRID):
        psi[i]=vecs[i,lowest]   
#
# Now we need to normalize our solution. 
#
    norm = simpson(psi**2,x=x)
    psi = psi/math.sqrt(norm)
    n = 2*psi**2    
#
#  We define the convergence criterion (crit) as the difference between the 
#  Kohn-Sham lowest energy eigenvalue in this iteration step to the one of  
#  the previous step. The criterion needs to be < TOL for the 
#  iteration to end.
#
    crit = abs(EKS_previous - EKS)
    EKS_previous = EKS
# --------------------------------------------------------------------------
# End of the self-consistency loop
# --------------------------------------------------------------------------
print('converged')

plt.plot(x,n_noninteracting)
plt.plot(x,n) 
plt.legend(['Noninteracting','DFT'])
plt.title("Density of two electrons in a parabolic potential")
plt.show()
#
# Calculate the various contributions to the total energy:
#
vint = VH * n
result = simpson(vint,x=x)
EH = -0.5*result
#
vint = VX * n
result = simpson(vint,x=x)
EVX = - result
#
vint = (0.5*x**2+VH+VX) * n
result = simpson(vint,x=x)
ETS = result
#
vint = (0.5*x**2) * n
result = simpson(vint,x=x)
EV = result
#
'''
for i in range(NGRID):
    upper = n[i]*PI*A 
    def f(j):
        def g(l):
            return kn(0,l)
        result, _ = quad(g,0.,j)
        return result
    result, _ = quad(f,0.,upper)
    vint[i] = -result/(PI*A)**2        
'''
for i in range(NGRID):
    z = A*n[i]
    uu = 0.0194*z + 1.06*z**2 + 0.5*z**3
    vv = 0.704 + 2.23*z + z**2
    vint[i] = -uu/(A**2*vv)
    
EX = simpson(vint,x=x)
#
E = EKS + EH + EVX + EX    # total ground-state energy
#
TS = EKS - ETS
#
print('')
print(' Kohn-Sham energy: EKS =', EKS)
print('          V energy: EV =', EV)
print('    Hartree energy: EH =', -EH)
print('        VX energy: EVX =', -EVX)
print('   Exchange energy: EX =', EX)
print('    kinetic energy: TS =', TS)
print('--------------------------------------------')
print('ground-state energy: E = EKS - EH - EVX + EX = TS + EV + EH + EX')
print('                     E = ',E)
#
# Write the density to a file
#
#f = open("nks.txt", "w")
#for i in range(NGRID):
#    f.write(str(x[i]) + "  " + str(n[i]) + '\n')
#f.close()
#    
    
    
    