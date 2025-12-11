# This program solves the 1D Kohn-Sham equation for two electrons in a
# quartic-oscillator potential, v(x) = 0.5*k*x**4. 
# It first calculates the self-consistent
# ground state, and then does a time propagation, assuming a short "kick".
#
# From the time-dependent dipole moment d(t), the Fourier transform d(omega)
# is calculated and the dipole power spectrum |d(omega)}**2 is plotted
# The plot uses a logarighmic scale.

#----------------------------User input starts here----------------------------

# HERE ARE THE PARAMETERS FOR THE GROUND-STATE CALCULATION:

NGRID = 51    # number of grid points (always an odd number)
XMAX = 4.      # the numerical grid goes from -XMAX < x < XMAX
KSPRING = 1.   # spring constant of the quartic oscillator potential
A = 1          # Coulomb softening parameter
TOL = 1.e-10   # numerical tolerance (the convergence criterion)
MIX = 0.75     # mixing parameter for the self-consistency iterations

#  HERE ARE THE PARAMETERS FOR THE TIME-DEPENDENT CALCULATION:

TSTEPS = 20000   # number of time steps
DT = 0.01       # time step
EFIELD = 0.01   # strength of the electric field kick
TKICK = 0.1    # duration of the kick
DYN = 0        # If DYN=2, use both time-dependent Hartree and exchange
               # IF DYN=1 use time-dependent Hartree and ground-state exchange
               # If DYN=0, use both ground-state Hartree and exchange 
               
PC = 0         # Number of predictor-corrector steps.
               # if DYN=0, you can set PC = 0.
               # if DYN=1 or 2, you should set PC = 1.

               # (The "predictor-corrector" algorithm is needed for the
               # self-consistent time propagation of the TDKS equation.
               # The time propagation itself is done using the so-called
               # Crank-Nicolson algorithm. To learn more, see the TDDFT book 
               # by C.A. Ullrich (Oxford, 2012), Section 4.5.1 and 4.5.2.)

#-----------------------------User input ends here-----------------------------

import matplotlib.pyplot as plt # Plotting library
import numpy as np
import scipy
from scipy.integrate import simpson   # integration with Simpson's rule
import sys
import math
import cmath

DX = 2.*XMAX/(NGRID-1)  # grid spacing
PI = 3.141592653589793  # define pi here
IONE = 0. + 1.j         # this is the imaginary unit i
#
# Define the numerical grid as an array
#
x = np.linspace(-XMAX,XMAX,NGRID)
# 
# Initialize the density of the noninteracting 1D harmonic oscillator.
#
def h(i):
    return 2.*math.exp(-i**2)/math.sqrt(PI)
#
# initialize a bunch of arrays
#
n = np.zeros(NGRID)
n0 = np.zeros(NGRID)
n1 = np.zeros(NGRID)
psi = np.zeros(NGRID)
PHI = np.zeros(NGRID, dtype=complex)
PHI1 = np.zeros(NGRID, dtype=complex)
VH = np.zeros(NGRID) 
VX = np.zeros(NGRID)
vint = np.zeros(NGRID)
HMAT = np.zeros((NGRID,NGRID))
TMAT = np.zeros((NGRID,NGRID), dtype=complex)
HKIN = np.zeros((NGRID,NGRID))
ONE = np.zeros((NGRID,NGRID))
VEXT = np.zeros(NGRID)
VT = np.zeros(NGRID)
RHS = np.zeros(NGRID, dtype=complex)
#
for i in range(NGRID):
    n[i]=h(x[i])
n1 = n.copy()
#
# Define the quadratic function of the spring, and use this to 
# define the external potential Hamiltonian:
#
def g(i):
    return 0.5*KSPRING*i**4
for i in range(NGRID):
    VEXT[i]=g(x[i]) 
    ONE[i,i] = 1.
#
#  define the kinetic energy operator:
#    
for i in range(NGRID):
    HKIN[i,i] = 490./(360.*DX**2)
for i in range(NGRID-1):
    HKIN[i,i+1] = -270./(360.*DX**2)
    HKIN[i+1,i] = -270./(360.*DX**2)
for i in range(NGRID-2):
    HKIN[i,i+2] = 27./(360.*DX**2)
    HKIN[i+2,i] = 27./(360.*DX**2)
for i in range(NGRID-3):
    HKIN[i,i+3] = -2./(360.*DX**2)
    HKIN[i+3,i] = -2./(360.*DX**2)   
#------------------------------------------------------------------------
# Here is the start of the self-consistency loop. We initialize the
# energy as that of two noninteacting electrons in a 1D harmonic potential
#------------------------------------------------------------------------
crit = 1.
EKS_previous = math.sqrt(KSPRING) 
counter = 0
while crit > TOL:
    counter += 1
    #print(counter)
    if counter==1000: 
        print('not converged')
        sys.exit()
    for i in range(NGRID):
        for j in range(NGRID):  
            HMAT[i,j] = HKIN[i,j]
#
#  mix with the density of the previous iteration
#
    n = MIX*n + (1.-MIX)*n1
    n1 = n.copy()
#
# Calculate the Hartree potential VH:
#   
    for i in range(NGRID):
        for j in range(NGRID):
            vint[j]=n[j]/math.sqrt(A**2 + (x[i]-x[j])**2)
        result = simpson(vint,x=x)
        VH[i] = result
        HMAT[i,i] += VEXT[i] + VH[i]   
#
# Calculate the exchange potential VX:
#    
    for i in range(NGRID):
        
        # This calculates the LDA exchange potential from eqn (12.10), see
        # exercise (12.1).
        
        z = A*n[i]
        uu = 0.0194*z + 1.06*z**2 + 0.5*z**3
        vv = 0.704 + 2.23*z + z**2
        VX[i] = -((0.0194 + 2.12*z + 1.5*z**2)*vv - uu*(2.23+2.*z))/(A*vv**2)
        
        HMAT[i,i] += VX[i]    
#
# Now find the eigenvalues and eigenvectors of the matrix MAT,
# and sort them to make sure we keep the lowest one only.
#
    vals, vecs = np.linalg.eig(HMAT)
    sorted = np.argsort(vals)
    lowest = sorted[0]
    EKS = vals[lowest]
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
#  the previous step. The criterion needs to be < TOL for the iteration to end.
#
    crit = abs(EKS_previous - EKS)
    EKS_previous = EKS
# --------------------------------------------------------------------------
# End of the self-consistency loop
# --------------------------------------------------------------------------
print('ground state converged')
#
print('lowest four Kohn-Sham eigenvalues:')
print(EKS,vals[sorted[1]],vals[sorted[2]],vals[sorted[3]])
print()
#
n0 = n.copy()
n1 = n.copy()

for i in range(NGRID):
    PHI[i] = psi[i] * cmath.exp(IONE*EFIELD*x[i])

#sys.exit()     # if you want to stop the calculation here

# --------------------------------------------------------------------------
# Now that we have calculated the ground state, we can begin with the
# time propagation. We take psi as the initial wave function (need to 
# convert it into the complex function PHI) and propagate it forward in time.
# --------------------------------------------------------------------------

Time = np.zeros(TSTEPS)
Dipole = np.zeros(TSTEPS)

f = open("dip.txt", "w")    # The data for d(t) will be written here
T=0
counter = -1
while counter < TSTEPS-1:  
    counter += 1
    T = T + DT
    if (counter+1)%100==0: print('time =',round(T,5))
    Time[counter] = T
#    
#   First, define the time-dependent perturbation VT. We assume that it
#   is a short "kick" in the form of a uniform electric field, which lasts
#   for a short time 0 < t < TKICK.
#
    if T <= TKICK:
        VT = EFIELD*x     # You can modify this, for example VT = EFIELD*x**2
                          # Can you think of a reason why you would do that?
    else:
        VT = np.zeros(NGRID)
#
#   this is the beginning of the predictor-corrector loop
#              
    PSTEP = -1
    while PSTEP < PC:
        PSTEP = PSTEP + 1
        n = (n + n1)/2.
        
#   If DYN=1 calculate the time-dependent Hartree and exchange potential

        if DYN == 1 or DYN == 2:
            for i in range(NGRID):
                for j in range(NGRID):
                    vint[j]=n[j]/math.sqrt(A**2 + (x[i]-x[j])**2)
                result = simpson(vint,x=x)
                VH[i] = result
                
        if DYN == 2:
            for i in range(NGRID):
                
                z = A*n[i]
                uu = 0.0194*z + 1.06*z**2 + 0.5*z**3
                vv = 0.704 + 2.23*z + z**2
                VX[i] = -((0.0194 + 2.12*z + 1.5*z**2)*vv - uu*(2.23+2.*z))/(A*vv**2)                
#
#   Now construct the time-dependent Hamiltonian
#    
        HMAT = HKIN.copy()
        
        for i in range(NGRID):

            HMAT[i,i] = HMAT[i,i] + VEXT[i] + VH[i] + VX[i] + VT[i]
    
#   Next, do the time propagation step by solving a linear equation
#   (Crank-Nicolson algorithm)
#        
        TMAT = ONE - 0.5*DT*IONE*HMAT
    
        RHS = TMAT.dot(PHI)
        
        TMAT = ONE + 0.5*DT*IONE*HMAT

        PHI1 = scipy.linalg.solve(TMAT, RHS, assume_a='gen')
        
        n = 2.*abs(PHI1)**2
#
#   end of the predictor-corrector loop
#        
    n1 = n.copy() 
    PHI = PHI1.copy()  
    
#    norm = simpson(n,x=x)
#    print('norm',norm)   # if you want to check whether the norm is conserved.
    
    vint = n*x     # if you modify VT above, then you should also modify this,
                   # such as vint = n*x**2. Why?
                   
    dip = simpson(vint,x=x)
    Dipole[counter] = dip
# --------------------------------------------------------------------------
# Now comes the Fourier transformation
# --------------------------------------------------------------------------

from scipy.fft import fft

T_end = Time[TSTEPS-1]
domega = 2*PI/T_end
OMM = np.zeros(TSTEPS)

for i in range(TSTEPS): OMM[i] = i*domega 

plt.figure(1)
plt.plot(Time,Dipole)
plt.xlabel('time t')
plt.ylabel('dipole moment d(t)')

y = DT*fft(Dipole)/EFIELD
plt.figure(2)
plt.xlim(0.0,7.0)
plt.plot(OMM,abs(y)**2,label='TSTEPS=20000, DT=0.01')
plt.xlabel('omega')
plt.ylabel('|d(omega)|^2')
plt.legend()
plt.yscale('log')
plt.show()
