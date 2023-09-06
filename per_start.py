import numpy as np
import matplotlib.pyplot as plt
import torch

## reasd in data

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
