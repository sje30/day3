import math
import numpy as np
import matplotlib.pyplot as plt

def g(x):
    return ( 1.0 /  (1.0 + math.exp(-x) ))

def gprime(x):
    y = g(x)
    return ( y * (1-y) )


## check these are the right shape.

xs = np.linspace(-3, 3, 100)
gs = [g(x) for x in xs]
gps = [gprime(x) for x in xs]

plt.ion()
plt.clf()
plt.xlabel("x")
plt.ylabel("g or gprime")
plt.plot(xs, gs,  label="g, activation")
plt.plot(xs, gps,  label="gprime")
plt.legend()
plt.show()

bias = -1                       # value of bias unit

epsilon = 0.5

# data = np.array([[0, 0, bias, 0],
#                  [0, 1, bias, 1],
#                  [1, 0, bias, 1],
#                  [1, 1, bias, 0],
#                  ]
#                 )

data = np.loadtxt('moons1.dat', delimiter=',')


targets = data[:,2]
inputs = data[:,0:2]
ninputs = inputs.shape[0]
inputs = np.c_[inputs, np.ones(ninputs)*bias]


I=2                             # number of input units, excluding bias
J=4                             # number of hidden units, excluding bias
K=1                             # only one output unit

## Weight matrices

W1 = np.random.rand(J,I+1)
W2 = np.random.rand(K,J+1)


y_j = np.zeros(J+1)             # outputs of hidden units
delta_j = np.zeros(J)           # delta for hidden units

nepoch = 2000
errors = np.zeros(nepoch)




for epoch in range(nepoch):

    ## accumulate errors for weight matrices
    DW1 = np.zeros(W1.shape)
    DW2 = np.zeros(W2.shape)
    epoch_err = 0.0

    ally = np.zeros(ninputs)
    for i in range(ninputs):

        ## Step 1. Forward propagation activity, adding
        ## bias activity along the way.


        ## 1a - input to hidden
        y_i = inputs[i,:]
        a_j = np.matmul(W1, y_i)

        for q in range(J):
            y_j[q] = g( a_j[q] )

        y_j[J] = bias

        ## 1b - hidden to output
        a_k = np.matmul(W2, y_j)
        y_k = g(a_k)

        ## 1c - compare output to target
        t_k  = targets[i]
        error = np.sum(0.5 * (t_k - y_k)**2 )
        epoch_err += error

        ally[i] = y_k           #  keep a record of output.
        
        ## Step 2.  Back propagate activity, calculating
        ## errors and dw along the way.


        ## 2a - output to hidden
        delta_k = gprime(a_k) * (t_k - y_k)
        for q in range(J+1):
            ##for r in range(K):
            r=0
            DW2[r,q] += y_j[q] * delta_k
                
            
        ## 2b - calculate delta for hidden layer

        for q in range(J):
            delta_j[q] = gprime(a_j[q]) * delta_k * W2[0,q]

        ## 2c - calculate error for input to hidden weights
        for p in range(I+1):
            for q in range(J):
                DW1[q,p] += y_i[p] * delta_j[q] 


    ## end of an epoch - now update weights
    errors[epoch] = epoch_err
    if ( epoch % 50)== 0:
        print(f'Epoch {epoch} error {epoch_err:.4f}')

    W1 = W1 + (epsilon*DW1)
    W2 = W2 + (epsilon*DW2)
        
## how has it worked?
np.c_[targets, np.around(ally, 3)]


# mesh won't work, as we don't have a model() function to just create forward pass.

## my model seems to work much easier than 2-layer moons in micrograd.  is it because of RELU?
## one layer is enough in micrograd too, but I think much more than 4 hidden units.


# X = inputs
# h = 0.25
# x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
# y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#                      np.arange(y_min, y_max, h))
# Xmesh = np.c_[xx.ravel(), yy.ravel()]
# inputs = [list(map(Value, xrow)) for xrow in Xmesh] #  this doesn't work...
# scores = list(map(model, inputs))
# Z = np.array([s.data > 0 for s in scores])
# Z = Z.reshape(xx.shape)

# fig = plt.figure()
# plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
# plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
# plt.xlim(xx.min(), xx.max())
# plt.ylim(yy.min(), yy.max())

