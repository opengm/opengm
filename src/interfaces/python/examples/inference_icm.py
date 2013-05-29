import opengm
import numpy
#---------------------------------------------------------------
# MinSum  with ICM
#---------------------------------------------------------------
unaries=numpy.random.rand(5 , 5,2)
potts=opengm.PottsFunction([2,2],0.0,0.4)
gm=opengm.grid2d2Order(unaries=unaries,regularizer=potts)
#---------------------------------------------------------------
# Minimize
#---------------------------------------------------------------
#get an instance of the optimizer / inference-algorithm
inf=opengm.inference.Icm(gm)
# start inference (in this case verbose infernce)
visitor=inf.verboseVisitor(printNth=1000,multiline=False)
inf.infer(visitor)
# get the result states
argmin=inf.arg()
# print the argmin (on the grid)
print argmin.reshape(5,5)