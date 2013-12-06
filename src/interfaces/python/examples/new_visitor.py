import opengm
import numpy

unaries=numpy.random.rand(1000, 1000,4)
potts=opengm.PottsFunction([4,4],0.0,0.2)
gm=opengm.grid2d2Order(unaries=unaries,regularizer=potts)





inf=opengm.inference.Icm(gm,parameter=opengm.InfParam())

v=inf.timingVisitor(visitNth=1000,timeLimit=4)
inf.infer(v)

#print v.getTimes()
