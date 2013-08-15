import opengm
import numpy

chainLength=50
numLabels=1024
numberOfStates=numpy.ones(chainLength,dtype=opengm.label_type)*numLabels
gm=opengm.gm(numberOfStates,operator='adder')
#add some random unaries
for vi in range(chainLength):
   unaryFuction=numpy.random.random(numLabels)
   gm.addFactor(gm.addFunction(unaryFuction),[vi])
#add one 2.order function


f=opengm.differenceFunction(shape=[1024,1024],weight=0.1)
print type(f),f
fid=gm.addFunction(f)
#add factors on a chain of the length 10
for vi in range(chainLength-1):
   gm.addFactor(fid,[vi,vi+1])    



inf = opengm.inference.BeliefPropagation(gm,parameter=opengm.InfParam(steps=40,convergenceBound=0 ,damping=0.9))
inf.infer(inf.verboseVisitor())


print inf.arg()

