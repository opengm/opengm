import opengm
import numpy



unaries=numpy.random.rand(15, 15,3)
potts=opengm.PottsFunction([3,3],0.0,0.15)
gm=opengm.grid2d2Order(unaries=unaries,regularizer=potts)



inf=opengm.inference.Icm(gm)
inf.infer(inf.verboseVisitor(),False)
print inf.arg().reshape(15,15)






