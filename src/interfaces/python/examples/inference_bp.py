import opengm
import numpy
#---------------------------------------------------------------
# MinSum and MaxSum with Bp
#---------------------------------------------------------------
unaries=numpy.random.rand(10, 10,2).astype(numpy.float32)
potts=opengm.PottsFunction([2,2],0.0,0.4)
gm=opengm.grid2d2Order(unaries=unaries,regularizer=potts)
#---------------------------------------------------------------
# Minimize NEW
#---------------------------------------------------------------
# get an instance of the inference algorithm's parameter object
#param=opengm.inferenceParameter(gm,alg='bp',accumulator='minimizer')
# set up the paraemters
#param.set(steps=100,damping=0.5,convergenceBound=0.00000001)
#get an instance of the optimizer / inference-algorithm

#FIXE


"""
inf=opengm.inference.BeliefPropagation(gm,parameter=opengm.InfParam(steps=10,damping=0.5,convergenceBound=0.001))
# start inference (in this case unverbose infernce)
inf.infer()
# get the result states
argmin=inf.arg()
# print the argmin (on the grid)
print argmin.reshape(10,10)

"""