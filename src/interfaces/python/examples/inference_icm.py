import opengm
import numpy
#---------------------------------------------------------------
# MinSum  with ICM
#---------------------------------------------------------------

n=1000
nl=10
unaries=numpy.random.rand(n , n, nl)
potts=opengm.PottsFunction([nl,nl],0.0,0.05)
gm=opengm.grid2d2Order(unaries=unaries,regularizer=potts)
#---------------------------------------------------------------
# Minimize
#---------------------------------------------------------------
#get an instance of the optimizer / inference-algorithm


inf=opengm.inference.Icm(gm)
# start inference (in this case verbose infernce)
visitor=inf.verboseVisitor(printNth=10000,multiline=True)
inf.infer(visitor)
# get the result states
argmin=inf.arg()
# print the argmin (on the grid)
print argmin.reshape(n,n)