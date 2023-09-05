import math
import numpy as np
import matplotlib.pyplot as plt

def g(x):
    return ( 1.0 /  (1.0 + math.exp(-x) ))

def gprime(x):
    return ( x * (1-x) )


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

bias = 0                       # value of bias unit

epsilon = 0.5

data = np.array([[0, 0, bias, 0],
                 [0, 1, bias, 1],
                 [1, 0, bias, 1],
                 [1, 1, bias, 0],
                 ]
                )

targets = data[:,3]
inputs = data[:,0:3]

ninputs = inputs.shape[0]

I=2                             # number of input units, excluding bias
J=2                             # number of hidden units, excluding bias
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
        y_k = a_k

        ## 1c - compare output to target
        t_k  = targets[i]
        error = np.sum(0.5 * (t_k - y_k)**2 )
        epoch_err += error

        ## Step 2.  Back propagate activity, calculating
        ## errors and dw along the way.


        ## 2a - output to hidden
        delta_k = gprime(a_k) 
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
                DW1[q,p] += y_i[p] 


    ## end of an epoch - now update weights
    errors[epoch] = epoch_err
    if ( epoch % 50)== 0:
        print(f'Epoch {epoch} error {epoch_err:.4f}')

    W1 = W1 + (epsilon*DW1)
    W2 = W2 + (epsilon*DW2)
        
## how has it worked?
