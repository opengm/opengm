import opengm
import numpy



unaries=numpy.random.rand(3 , 3,2)
potts=opengm.PottsFunction([2,2],0.0,0.2)
gm=opengm.grid2d2Order(unaries=unaries,regularizer=potts)

subGm,subGmVis = gm.fixVariables([0,1,2],[0,0,1])

print subGmVis,subGm
