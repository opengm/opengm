import opengm
import numpy
from opengm import learning
np = numpy


numLabels = 3
numVar = 6




#################################################################
# add a unary function 
##################################################################

print opengm.learning.DatasetWithHammingLoss
print opengm.learning.HammingLoss

# make the gm
space = numpy.ones(numVar)*numLabels
gm = opengm.gm(space)



weightVals = numpy.ones(100)*1.0
weights = opengm.learning.Weights(weightVals)




##################################################################
# add a unary function 
##################################################################
features  = numpy.ones([numLabels, 2], dtype=opengm.value_type)
weightIds = numpy.ones([numLabels, 2], dtype=opengm.index_type)

# set up weight ids for each label
weightIds[0,:] = [0, 1]
weightIds[1,:] = [2, 3]
weightIds[2,:] = [4, 5]

print "add f"
f = opengm.LUnaryFunction(weights=weights, numberOfLabels=numLabels, 
                          weightIds=weightIds, features=features)
print "add factor"
fid = gm.addFunction(f)
gm.addFactor(fid, [0])

print "features",features
print "unary",np.array(gm[0])

weights[4] = 0.5
print "unary",np.array(gm[0])


##################################################################
# add a unary function                                           
##################################################################
features  = [
    numpy.array([1.0, 1.0],             dtype=opengm.value_type),
    numpy.array([1.0, 1.0, 1.0],        dtype=opengm.value_type),
    numpy.array([1.0, 1.0, 1.0, 1.0],   dtype=opengm.value_type)
]

weightIds  = [
    numpy.array([6, 7],             dtype=opengm.index_type),
    numpy.array([8, 9, 10],        dtype=opengm.index_type),
    numpy.array([11, 12, 13, 14],   dtype=opengm.index_type)
]


print "add f"
f = opengm.LUnaryFunction(weights=weights, numberOfLabels=numLabels, 
                          weightIds=weightIds, features=features)
print "add factor"
fid = gm.addFunction(f)
gm.addFactor(fid, [0])

print "features",features
print "unary",np.array(gm[1])


print "unary",np.array(gm[1])





##################################################################
# add a potts function
##################################################################
features = numpy.array([1.0, 5.0]).astype(opengm.value_type)
weightIds = numpy.array([6,7]).astype(opengm.index_type)
f = opengm.LPottsFunction(weights=weights, numberOfLabels=numLabels, 
                          weightIds=weightIds, features=features)

# add factor
fid = gm.addFunction(f)
gm.addFactor(fid, [0,1])


