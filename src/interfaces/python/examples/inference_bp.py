import opengm
import numpy

unaries=numpy.random.rand(10, 10,2)
potts=opengm.PottsFunction([2,2],0.0,0.4)
gm=opengm.grid2d2Order(unaries=unaries,regularizer=potts)



inf=opengm.inference.BeliefPropagation(gm,parameter=opengm.InfParam(steps=10,damping=0.5,convergenceBound=0.001))
# start inference (in this case unverbose infernce)
inf.infer()
# get the result states
argmin=inf.arg()
# print the argmin (on the grid)
print argmin.reshape(10,10)