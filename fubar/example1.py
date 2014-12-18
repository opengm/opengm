import opengm
import opengm.learning as learning
from opengm import numpy



# weight vector
nWeights = 100
weightVals = numpy.ones(nWeights)*0.5
weights = opengm.learning.Weights(weightVals)



dataset =learning.createDataset(loss='h')

print "type of dataset", dataset




# for grid search learner
lowerBounds = numpy.zeros(nWeights)
upperBounds = numpy.ones(nWeights)
nTestPoints  =numpy.ones(nWeights).astype('uint64')*10


learner = learning.gridSearchLearner(dataset=dataset,lowerBounds=lowerBounds, upperBounds=upperBounds,nTestPoints=nTestPoints)

learner.learn(infCls=opengm.inference.BeliefPropagation, 
              parameter=opengm.InfParam(damping=0.5))
