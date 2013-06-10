import opengm
import numpy

#------------------------------------------------------------------------------------
# This example shows how multiple  unaries functions and functions / factors add once
#------------------------------------------------------------------------------------
# add unaries from a for a 2d grid / image
width=10
height=20
numVar=width*height
numLabels=2
# construct gm
gm=opengm.gm(numpy.ones(numVar,dtype=opengm.index_type)*numLabels)
# construct an array with all unaries (random in this example)
unaries=numpy.random.rand(width,height,numLabels)
# reshape unaries is such way, that the first axis is for the different functions
unaries2d=unaries.reshape([numVar,numLabels])
# add all unary functions at once (#numVar unaries)
fids=gm.addFunctions(unaries2d)
# numpy array with the variable indices for all factors
vis=numpy.arange(0,numVar,dtype=numpy.uint64)
# add all unary factors at once
gm.addFactors(fids,vis)
