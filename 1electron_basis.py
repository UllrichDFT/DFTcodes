import matplotlib.pyplot as plt
import numpy as np
import math
#------------------------------------------------------------------------
# This program solves the Schroedinger equation for one electron in using an 
# expansion in particle-in-a-box cosine functions. The user can choose between
# a harmonic oscillator potential or a delta potential.
#------------------------------------------------------------------------
NGRID = 101   # number of grid points (always an odd number)
NBASIS = 10   # number of basis functions (only choose cosines)
XMAX = 5     # The numerical grid goes from -XMAX < x < XMAX.
             # XMAX also defines the box size for the
             # particle-in-a-box basis functions. It may have to be adjusted.

Potential = 1  # Choose potential: 1 (harmonic potential) or 2 (delta potential)

Kspring = 1.  # spring constant of the harmonic oscillator potential
alpha = 1.    # strength of the delta potential
#--------------------------------------------------------------------------
#
L = 2*XMAX
DX = 2.*XMAX/(NGRID-1)  # grid spacing
PI = 3.141592653589793  # define pi here
#
# Define the numerical grid (used for plotting) as an array:
#
x = np.linspace(-XMAX,XMAX,NGRID)
# 
# Define the exact ground-state density in the harmonic oscillator and in
# the delta potential:
#
def h_osc(i):
    return Kspring**0.25*math.exp(-i**2*math.sqrt(Kspring))/math.sqrt(PI)
def h_delta(i):
    return math.exp(-2.*alpha*abs(i))*alpha
#
# initialize a bunch of arrays
#
n = np.zeros(NGRID)
psi = np.zeros(NGRID)
C1 = np.zeros(NBASIS)
MAT = np.zeros((NBASIS,NBASIS))
#
for i in range(NGRID):
    if Potential ==1: n[i]=h_osc(x[i])
    if Potential ==2: n[i]=h_delta(x[i])
#
plt.plot(x,n,label='exact')
#
#  Define the Hamiltonian matrix
#
if Potential==1:
    for i in range(NBASIS):
        for j in range(NBASIS):
            nu = 2*i+1
            mu = 2*j+1
            if i==j:       
                MAT[i,i] = 0.5*(nu*PI/L)**2 + (Kspring*L**2/(4*PI**2))*(PI**2/6 - 1/nu**2)
            else:
                MAT[i,j] = (-1)**((nu-mu)/2)/(nu-mu)**2 + (-1)**((nu+mu)/2)/(nu+mu)**2
                MAT[i,j] = MAT[i,j]*Kspring*L**2/PI**2
                
if Potential==2:
    for i in range(NBASIS):
        for j in range(NBASIS):
            nu = 2*i+1
            mu = 2*j+1
            MAT[i,j] = -2*alpha/L
            if i==j: MAT[i,i] += 0.5*(nu*PI/L)**2
#
# Now find the eigenvalues and eigenvectors of the matrix MAT,
# and sort them to make sure we pick the lowest one.
#
vals, vecs = np.linalg.eig(MAT)
sorted = np.argsort(vals)
lowest = sorted[0]
E = vals[lowest]
for i in range(NBASIS):
    C1[i]=vecs[i,lowest]   
#
if Potential==1: E_exact = math.sqrt(Kspring)/2.
if Potential==2: E_exact = -alpha**2/2.
print('Exact ground-state energy: E =', E_exact)
print('Numerical ground-state energy: E =', E)
print(' ')
print('Error:', E-E_exact)
#
for i in range(NGRID):
    for j in range(NBASIS):
        nu = 2*j+1
        psi[i] = psi[i] + C1[j]*math.sqrt(2/L)*math.cos(nu*PI*x[i]/L)
#
plt.plot(x,psi**2)

plt.legend(['Exact','Numerical'])
plt.show()  
    
    