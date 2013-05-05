import opengm
import numpy

chainLength=5
numLabels=4
numberOfStates=numpy.ones(chainLength,dtype=opegm.label_type)*numLabels
gm=opengm.graphicalModel(numberOfStates,operator='adder')
#add some random unaries
for vi in range(chainLength):
   unaryFuction=numpy.random.random(numLabels).astype(numpy.float32)
   gm.addFactor(gm.addFunction(unaryFuction),[vi])
#add one 2.order function
f=numpy.ones(numLabels*numLabels,dtype=numpy.float32).reshape(numLabels,numLabels)
#fill function with values...
#.......
#add function
fid=gm.addFunction(f)
#add factors on a chain of the length 10
n=10
for vi in range(chainLength-1):
   gm.addFactor(fid,[vi,vi+1])    