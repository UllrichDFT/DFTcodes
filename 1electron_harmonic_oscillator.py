# This program solves the Schroedinger equation for one electron in a
# harmonic-oscillator potential. The purpose is to test different versions
# of the finite-difference method.
import numpy as np
import math
import matplotlib.pyplot as plt
#
#------------------------------------------------------------------------
NGRID = 101    # number of grid points (always an odd number)
XMAX = 5.     # the numerical grid goes from -XMAX < x < XMAX
KSPRING = 1.  # spring constant of the harmonic oscillator potential
NPOINT = 3    # which formula to use: three-point, five-point or seven-point
#--------------------------------------------------------------------------
#
DX = 2.*XMAX/(NGRID-1)  # grid spacing
PI = math.pi  # define pi here
#
# Define the numerical grid as an array
#
x = np.linspace(-XMAX,XMAX,NGRID)
# 
# Initialize the density of the noninteracting 1D harmonic oscillator.
#
def h(i):
    return math.exp(-i**2)/math.sqrt(PI)
#
# initialize a bunch of arrays
#
n = np.zeros(NGRID)
psi = np.zeros(NGRID)
MAT = np.zeros((NGRID,NGRID))
#
for i in range(NGRID):
    n[i]=h(x[i])
#
plt.plot(x,n) 
#
# Define the quadratic function of the spring
#
def g(i):
    return 0.5*KSPRING*i**2
#
#  Three-point formula:
#
if NPOINT == 3:
    for i in range(NGRID):
        MAT[i,i] = g(x[i]) + 1./DX**2
    for i in range(NGRID-1):
        MAT[i,i+1] = -0.5/DX**2
        MAT[i+1,i] = -0.5/DX**2
#
#  Five-point formula:
#  
if NPOINT == 5:
    for i in range(NGRID):
        MAT[i,i] = g(x[i]) + 30./(24.*DX**2)
    for i in range(NGRID-1):
        MAT[i,i+1] = -16./(24.*DX**2)
        MAT[i+1,i] = -16./(24.*DX**2)
    for i in range(NGRID-2):
        MAT[i,i+2] = 1./(24.*DX**2)
        MAT[i+2,i] = 1./(24.*DX**2)
#
#  Seven-point formula:
#  
if NPOINT == 7:
    for i in range(NGRID):
        MAT[i,i] = g(x[i]) + 490./(360.*DX**2)
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
# Now find the eigenvalues and eigenvectors of the matrix MAT,
# and sort them to make sure we pick the lowest one.
#
vals, vecs = np.linalg.eig(MAT)
sorted = np.argsort(vals)
lowest = sorted[0]
E = vals[lowest]
for i in range(NGRID):
    psi[i]=vecs[i,lowest]   
#
# Now we need to normalize our solution. 
#
from scipy.integrate import simpson   # integration with Simpson's rule
norm = simpson(psi**2,x=x)
psi = psi/math.sqrt(norm)
n = psi**2    
#
plt.plot(x,n)
plt.title("Density of one electron in a parabolic potential")
plt.legend(['exact solution','numerical solution'])
plt.show() 
#
print('Ground-state energy: E =', E)
print(' ')
print('Error:', E-0.5)


    
    
    