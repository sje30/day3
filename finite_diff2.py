## finite difference approach

import math
import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return (x*x + (3/x) + 4)

def g(x):
    return (2*x - (3/(x*x)))

f(0.1)

def error_fs(h):
    x = 0.1
    approx = ( f(x+h) - f(x)) / h
    return abs( g(x) - approx)



hs = [10**x for x in range(-1, -15, -1)]
e1 = [error_fs(h) for h in hs]
plt.ion()
plt.loglog(hs, e1)
plt.title('finite difference')


def error_cs(h):
    x = 0.1
    xc = complex(x, h)
    fc = f(xc)
    fderiv = fc.imag/h
    return abs( g(x) - fderiv)
    
error_cs(1e-13)    
e2 = [error_cs(h) for h in hs]
plt.loglog(hs, e2)
plt.title('complex step')


## Dual numbers

class Dual():
    def __init__(self, value, grad=0):
        self.value = value
        self.grad = grad

    def __repr__(self):
        return f'Dual: {self.value} + {self.grad}ε'

    def __add__(self, other):
        out = Dual(self.value+ other.value,
                   self.grad + other.grad)
        return out

    def __mul__(self, other):
        out = Dual(self.value * other.value,
                   self.grad * other.value + self.value*other.grad)
        return out

    def __truediv__(self, other):
        out = Dual(self.value / other.value,
                   (self.grad / other.value) -
                   (self.value * other.grad / other.value**2))
        return out

    def sin(self):
        out = Dual(np.sin(self.value), np.cos(self.value)*self.grad)
        return out

    def cos(self):
        out = Dual(np.cos(self.value), -np.sin(self.value)*self.grad)
        return out

    def exp(self):
        out = Dual(np.exp(self.value), np.exp(self.value)*self.grad)
        return out
    


def dual_f(x):
    x1 = Dual(x, 1)
    a = x1 * x1
    b = Dual(3.0, 0) / x1
    c = a + b
    d = c + Dual(4.0, 0)
    return d


res = dual_f(0.1)

## This is a 'difficult' check for equality.
g(0.1) == res.grad




## Moler version


def F(x):
    return ( np.exp(x)  / ( (np.sin(x)**3) + (np.cos(x)**3) ) )

x = math.pi / 4
F(x)

def G(x):
    return (np.exp(x)*(np.cos(3*x) + np.sin(3*x)/2 + (3*np.sin(x))/2))/(np.cos(x)**3 + np.sin(x)**3)**2


G(x)

def error_fs(h):
    x = math.pi/4
    approx = ( F(x+h) - F(x)) / h
    return abs( G(x) - approx)



hs = [10**x for x in range(-1, -15, -1)]
e1 = [error_fs(h) for h in hs]
plt.ion()
plt.loglog(hs, e1)
plt.title('finite difference')




def error_cs(h):
    x = math.pi/4
    xc = complex(x, h)
    fc = F(xc)
    fderiv = fc.imag/h
    return (abs(G(x) - fderiv))

    
error_cs(1e-13)    
e2 = [error_cs(h) for h in hs]
plt.loglog(hs, e2)
plt.title('complex step')


plt.clf()
plt.title("Moler's test function")
plt.xlabel("step size, h")
plt.ylabel("abs error")
plt.loglog(hs, e1,  label="finite step")
plt.loglog(hs, e2,  label="complex step")
plt.legend()
plt.show()


def dual_f2(x):
    x1 = Dual(x, 1)
    n = x1.exp()
    s = x1.sin()
    c = x1.cos()
    r = n / ( (s*s*s) + (c*c*c) )
    return r


## answer is very close.
res_f2 = dual_f2(math.pi/4)
G(math.pi/4) - res_f2.grad
