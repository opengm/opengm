import opengm
import numpy



unaries=numpy.random.rand(10, 10,2)
potts=opengm.PottsFunction([2,2],0.0,0.4)
gm=opengm.grid2d2Order(unaries=unaries,regularizer=potts)


inf=opengm.inference.GraphCut(gm)
inf.infer()
arg=inf.arg()

