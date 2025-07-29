# This program solves the Schroedinger equation for one electron in a
# harmonic-oscillator potential, using an expansion in particle-in-a-box states.
#
#------------------------------------------------------------------------
NGRID = 101   # number of grid points (always an odd number)
NBASIS = 5   # number of basis functions (only choose cosines)
XMAX = 5     # the numerical grid goes from -XMAX < x < XMAX
KSPRING = 1.  # spring constant of the harmonic oscillator potential
#--------------------------------------------------------------------------
#
L = 2*XMAX
DX = 2.*XMAX/(NGRID-1)  # grid spacing
PI = 3.141592653589793  # define pi here
#
# Define the numerical grid as an array
#
import numpy as np
x = np.linspace(-XMAX,XMAX,NGRID)
# 
# Initialize the density of the noninteracting 1D harmonic oscillator.
#
import math
def h(i):
    return math.exp(-i**2)/math.sqrt(PI)
#
# initialize a bunch of arrays
#
n = np.zeros(NGRID)
psi = np.zeros(NGRID)
C1 = np.zeros(NBASIS)
MAT = np.zeros((NBASIS,NBASIS))
#
for i in range(NGRID):
    n[i]=h(x[i])
#
from pylab import *
plot(x,n)
#
#  Define the Hamiltonian matrix
#
for i in range(NBASIS):
    for j in range(NBASIS):
        nu = 2*i+1
        mu = 2*j+1
        if i==j:       
            MAT[i,i] = 0.5*(nu*PI/L)**2 + (KSPRING*L**2/(4*PI**2))*(PI**2/6 - 1/nu**2)
        else:
            MAT[i,j] = (-1)**((nu-mu)/2)/(nu-mu)**2 + (-1)**((nu+mu)/2)/(nu+mu)**2
            MAT[i,j] = MAT[i,j]*KSPRING*L**2/PI**2
#
# Now find the eigenvalues and eigenvectors of the matrix MAT,
# and sort them to make sure we pick the lowest one.
#
vals, vecs = np.linalg.eig(MAT)
sorted = np.argsort(vals)
lowest = sorted[1]
E = vals[lowest]
for i in range(NBASIS):
    C1[i]=vecs[i,lowest]   
#
print('Ground-state energy: E =', E)
print(' ')
print('Error:', E-0.5)
#
for i in range(NGRID):
    for j in range(NBASIS):
        nu = 2*j+1
        psi[i] = psi[i] + C1[j]*math.sqrt(2/L)*math.cos(nu*PI*x[i]/L)
#
plot(x,psi**2)

print(NBASIS,E)

    
    
    