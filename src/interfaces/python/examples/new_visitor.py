import opengm
import numpy

unaries=numpy.random.rand(1000, 1000,7)
potts=opengm.PottsFunction([7,7],0.0,0.2)
gm=opengm.grid2d2Order(unaries=unaries,regularizer=potts)





inf=opengm.inference.Icm(gm,parameter=opengm.InfParam())

v=inf.timingVisitor(visitNth=1000,timeLimit=7)
inf.infer(v)

#print v.getTimes()
