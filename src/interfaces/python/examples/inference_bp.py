import opengm
import numpy

from time import time

shape=[20, 20]
nl = 100
unaries=numpy.random.rand(*shape+[nl])
potts=opengm.PottsFunction([nl]*2,0.0,0.4)
gm=opengm.grid2d2Order(unaries=unaries,regularizer=potts)



inf=opengm.inference.BeliefPropagation(gm,parameter=opengm.InfParam(steps=10,damping=0.5,convergenceBound=0.001))
# start inference (in this case unverbose infernce)

t0=time()
inf.infer()
t1=time()

print t1-t0

# get the result states
argmin=inf.arg()
# print the argmin (on the grid)
#print argmin.reshape(*shape)