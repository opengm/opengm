import opengm
import numpy
#---------------------------------------------------------------
# MinSum  with SelfFusion
#---------------------------------------------------------------
numpy.random.seed(42)

n=100
nl=100
unaries=numpy.random.rand(n , n, nl)
potts=opengm.PottsFunction([nl,nl],0.0,0.5)
gm=opengm.grid2d2Order(unaries=unaries,regularizer=potts)
#---------------------------------------------------------------
# Minimize
#---------------------------------------------------------------
#get an instance of the optimizer / inference-algorithm


infParam = opengm.InfParam(
    generator='trws'
)

inf=opengm.inference.SelfFusion(gm,parameter=infParam)
# start inference (in this case verbose infernce)
visitor=inf.verboseVisitor(printNth=1,multiline=True)
inf.infer(visitor)
# get the result states
argmin=inf.arg()
# print the argmin (on the grid)
print argmin.reshape(n,n)
