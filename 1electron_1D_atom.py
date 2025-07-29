# This program solves the Schroedinger equation for one electron in a
# soft-Coulomb external potential. 
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson   # integration with Simpson's rule
from scipy.integrate import quad    # integration with Gaussian quadrature
from scipy.special import kn        # The modified Bessel functions
#
#------------------------------------------------------------------------
NGRID = 601    # number of grid points (always an odd number)
XMAX = 30.     # the numerical grid goes from -XMAX < x < XMAX
A = 2.         # Coulomb softening parameter of the external potential
A_INT = 1.     # Coulomb softening parameter of the electron interaction
#--------------------------------------------------------------------------
#
DX = 2.*XMAX/(NGRID-1)  # grid spacing
PI = math.pi  # define pi here
#
# Define the numerical grid as an array
#
x = np.linspace(-XMAX,XMAX,NGRID)
# 
# initialize a bunch of arrays
#
V = np.zeros(NGRID)
psi = np.zeros(NGRID)
n = np.zeros(NGRID)
MAT = np.zeros((NGRID,NGRID))
VXC = np.zeros(NGRID)
VX_LDA = np.zeros(NGRID)
vint = np.zeros(NGRID)
V_one_over_x = np.zeros(NGRID)
#
# Define the soft Coulomb potential (this is the external potential)
#
for i in range(NGRID):
    V[i]=-1.0/math.sqrt(A**2 + x[i]**2)
#
#  Seven-point formula:
#  
for i in range(NGRID):
    MAT[i,i] = V[i] + 490./(360.*DX**2)
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
norm = simpson(psi**2,x=x)
psi = psi/math.sqrt(norm)
n = psi**2    
#
print('Ground-state energy: E =', E)
print(' ')
#
plt.axis([-XMAX,XMAX,-1.0,0.5])  #set plotting range: x runs from -XMAX to XMAX
#                                                   y runs from -1.5 to 0.5
#
#p.plot(x,V)    #plot the external potential
plt.plot(x,n,label='density')      #plot the density
#
# Now calculate the exact XC potential:
#
for i in range(NGRID):
    
    if abs(x[i]) > 1.e-6:
        V_one_over_x[i] = -abs(1./x[i])
    else:
        V_one_over_x[i] = -1.e8
        
    for j in range(NGRID):
        vint[j]=n[j]/math.sqrt(A_INT**2 + (x[i]-x[j])**2)
    result = simpson(vint,x=x)
    VXC[i] = -result
#
#  Now plot the exact XC potential and compare it with -1/|x|
#
plt.plot(x,VXC,label='exact VXC')
plt.plot(x,V_one_over_x, label='-1/x')
#
# Calculate the LDA exchange potential VX_LDA and plot it:
#    
for i in range(NGRID):
    upper = n[i]*PI*A_INT  
    def f(j):
        return kn(0,j)
    result, _ = quad(f,0.,upper)
    VX_LDA[i] = -result/(PI*A_INT)
#
plt.plot(x,VX_LDA, label='VX LDA')
plt.xlabel('x')
plt.legend()

for i in range(NGRID):
    if x[i]>=0:
        print('x=',round(x[i],6),'  VXC=',round(VXC[i],10),'  -1/|x|=',\
          round(V_one_over_x[i],10),'  n=',round(n[i],10))

#
# save the picture as a png file
#
plt.savefig('plot.png', dpi=300)




    
    
    