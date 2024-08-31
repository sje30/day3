## This file is broken and has one bug in it.

import numpy as np
import matplotlib.pyplot as plt

## read in data

data = np.loadtxt("eg2d.dat", delimiter=",",skiprows=1)

ninputs = data.shape[0]
wts = np.array([1, 1, 1.5])

def show_points(data, wts, plt, title):
    plt.clf()
    colors=np.array(["red", "blue"])
    plt.scatter(data[:,0], data[:,1], c=colors[data[:,2].astype(int)])
    plt.axis('equal')
    intercept = wts[2]/wts[1] # a
    slope = -wts[0]/wts[1]    # b
    plt.axline( (0, intercept), slope=slope)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.8, 1.5])
    plt.title(title)
    plt.show()

plt.ion()    
show_points(data, wts, plt, 'start')

epsilon = 0.03
nepochs = 100


x = np.array([0.0, 0.0, -1])
for epoch in range(nepochs):
    error = 0.0
    order = np.random.choice(ninputs, ninputs,replace=False)

    for iteration in range(ninputs):
        i = order[iteration]
        x[0] = data[i,0]
        x[1] = data[i,1]
        t    = data[i,2]
        a = np.dot(x, wts)
        y = a > 0
        error = error + (0.5 *(t-y)**2)
        dw = epsilon * (y-t) * x
        wts = wts + dw
    title=f"Epoch {epoch} error {error}"
    print(title)
    show_points(data, wts, plt, title)
    plt.pause(0.05)

## Questions, what happens if you use i=iteration?
## What if you use np.heaviside to calculate output y?  (much quicker)

