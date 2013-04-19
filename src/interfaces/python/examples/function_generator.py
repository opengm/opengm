import opengm
import numpy

# (so far the function generator is only implemented for potts functions)

# assuming one needs to add N potts functions 
# function (all for variables with 2 states)
# the only thing differs is the weigt BETA
#   0    BETA
#   BETA  0
# one could  use a function generator which 
# should be realy fast and cheap in memory

# the different arguments can have different length
# (but all must be numpy arrays even if the length is 1 (will be changed soon))
numLabels=numpy.ones(1,dtype=numpy.uint64)*2
valueLabelsEqual=numpy.zeros(1,dtype=numpy.float32)
betas=numpy.array([0.1,0.2,0.3,0.4,0.5],dtype=numpy.float32)

# some gm with 10 variales with 2 states (to match the example)
gm=opengm.gm([2]*10)
fgen=opengm.pottsFunctions(numLabels,numLabels,valueLabelsEqual,betas)
fids=gm.addFunctions(fgen)

print gm