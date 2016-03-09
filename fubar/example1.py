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

# for struct max margin learner
smm_learnerParam = learning.StructMaxMargin_Bundle_HammingLossParameter(1.0, 0.01, 0)
smm_learner = learning.StructMaxMargin_Bundle_HammingLoss(dataset, smm_learnerParam)
smm_learner.learn(infCls=opengm.inference.Icm)
smm_learner2 = learning.structMaxMarginLearner(dataset, 1.0, 0.001, 0)
smm_learner2.learn(infCls=opengm.inference.BeliefPropagation, parameter=opengm.InfParam(damping=0.5))
