import opengm
import numpy


numLabels = 4
numVar = 6


# make the gm
space = numpy.ones(numVar)*4
gm = opengm.gm(space)



weightVals = numpy.ones(4)*4.0
weights = opengm.Weights(weightVals)






features = numpy.array([1.0, 5.0]).astype(opengm.value_type)
weightIds = numpy.array([0,1]).astype(opengm.index_type)

f = opengm.LPottsFunction(weights=weights, numberOfLabels=numLabels, 
                          weightIds=weightIds, features=features)



fid = gm.addFunction(f)
gm.addFactor(fid, [0,1])

print numpy.array(gm[0])
weights[0] = 0.5

print numpy.array(gm[0])


