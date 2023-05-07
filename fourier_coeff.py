import scipy as sp
from scipy import integrate
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

# Step 1: build the periodic function f

# Step 1.1: Get coefficients for g = f-tilde and build g

max_range = 500
num_points = 250 #number of points for integration over each probability bin


x_l = 0

x_u = max_range

P = max_range + 10 #periodicity

x_y = P + x_l # x_u_tilde from paper

#matrix for LGS Av = b, where v is the array of coefficients
A = np.array([[x_y**3,x_y**2,x_y,1],[3*x_y**2,2*x_y,1,0],[x_u**3,x_u**2,x_u,1],[3*x_u**2,2*x_u,1,0]])


# cost function 
F = lambda x : max(x-100,0)
b = np.array([F(x_l), 0, F(x_u), 1])


#solve for coefficients
v = np.linalg.solve(A,b)
g = lambda s: v[0]*s**3 + v[1]*s**2 + v[2]*s + v[3]
print(b-A.dot(v))


# Step 1.2: Build the periodic function f
def f(s):
    
    x = (s - x_l)%(P) + x_l
    
    if x <= x_u:
        val = F(x)
    else:
        val = g(x)
    
    return val



# Non-periodic sawtooth function defined for a range [-l,l]
def sawtooth(x):
    return x
 
 
# Non-periodic square wave function defined for a range [-l,l]
def square(x):
    if x>0:
        return np.pi
    else:
        return -np.pi
 
 
 
# Non-periodic triangle wave function defined for a range [-l,l]
def triangle(x):
    if x>0:
        return x
    else:
        return -x
 
# Non-periodic cycloid wave function defined for a range [-l,l]
def cycloid(x):
    return np.sqrt(np.pi**2-x**2)




def fourier(li, lf, n, f):
    l = (lf-li)/2
    # Constant term
    a0=1/l*integrate.quad(lambda x: f(x), li, lf)[0]
    # Cosine coefficents
    A = np.zeros((n))
    # Sine coefficents
    B = np.zeros((n))
     
    for i in range(1,n+1):
        A[i-1]=1/l*integrate.quad(lambda x: f(x)*np.cos(i*np.pi*x/l), li, lf)[0]
        B[i-1]=1/l*integrate.quad(lambda x: f(x)*np.sin(i*np.pi*x/l), li, lf)[0]
 
    return [a0/2.0, A, B]
 
 
# Limits for the functions
li = -np.pi
lf = np.pi
 
# Number of harmonic terms
n =100
 
# Fourier coeffficients for various functions
coeffs = fourier(li,lf,n,sawtooth)
#print('Fourier coefficients for the Sawtooth wave\n')
#print('a0 ='+str(coeffs[0]))
#print('an ='+str(coeffs[1]))
#print('bn ='+str(coeffs[2]))
#print('-----------------------\n\n')
 
square_coeffs = fourier(li,lf,n,square)
# print('Fourier coefficients for the Square wave\n')
# print('a0 ='+str(square_coeffs[0]))
# print('an ='+str(square_coeffs[1]))
# print('bn ='+str(square_coeffs[2]))
# print('-----------------------\n\n')
 
# triangle_coeffs = fourier(li,lf,n,triangle)
# print('Fourier coefficients for the Triangular wave\n')
# print('a0 ='+str(triangle_coeffs[0]))
# print('an ='+str(triangle_coeffs[1]))
# print('bn ='+str(triangle_coeffs[2]))
# print('-----------------------\n\n')
 
coeffs = fourier(li,lf,n,cycloid)
# print('Fourier coefficients for the Cycloid wave\n')
# print('a0 ='+str(coeffs[0]))
# print('an ='+str(coeffs[1]))
# print('bn ='+str(coeffs[2]))
# print('-----------------------\n\n')


petias_coeffs = fourier(0,P,n,f)
# print('Fourier coefficients for the Petias function wave\n')
# print('a0 ='+str(petias_coeffs[0]))
# print('an ='+str(petias_coeffs[1]))
# print('bn ='+str(petias_coeffs[2]))
# print('-----------------------\n\n')

L=P/2
def new_function(x, coeff):
    out = coeff[0]
    for i in range(1, len(coeff[1])+1):
        out += coeff[1][i-1]*np.cos(i*np.pi*x/L)
        out += coeff[2][i-1]*np.sin(i*np.pi*x/L)
    return out


x = np.linspace(-2*P,2*P,1000)
fx = np.zeros(len(x))
for i in range(len(x)):
    fx[i] = new_function(x[i],petias_coeffs)

plt.style.use('dark_background')
# Create the figure and axes objects
fig, ax = plt.subplots()
# Plot the data with a white line
ax.plot(x, np.vectorize(f)(x), color='white', label="original function")
ax.plot(x,fx, color='darkturquoise')
#plt.legend(loc=2, fontsize=14)
#ax.set_title('Sinusoidal function', fontsize=20)
        
# Adjust tick label font size
plt.xticks([])
plt.yticks([])
# Show the plot
plt.show()
    
