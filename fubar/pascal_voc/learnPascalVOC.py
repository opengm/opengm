import opengm
from opengm import learning
import numpy as np

out_dir = './'
out_prefix = 'pascal_voc_train_'

dataset = learning.createDataset(0, loss='gh')
#dataset = learning.DatasetWithGeneralizedHammingLoss(0)
dataset.load(out_dir, out_prefix)

nWeights = dataset.getNumberOfWeights()
print 'nWeights', nWeights
print 'nModels', dataset.getNumberOfModels()

# for grid search learner
lowerBounds = np.ones(nWeights)*-1.0
upperBounds = np.ones(nWeights)*1.0
nTestPoints  =np.ones(nWeights).astype('uint64')*3

#learner = learning.gridSearchLearner(dataset=dataset,lowerBounds=lowerBounds, upperBounds=upperBounds,nTestPoints=nTestPoints)
learner = learning.structMaxMarginLearner(dataset, 1.0, 0.001, 0)

learner.learn(infCls=opengm.inference.Icm,
              parameter=opengm.InfParam())

weights = dataset.getWeights()

for w in range(nWeights):
    print weights[w]

for i in range(dataset.getNumberOfModels()):
    print 'loss of', i, '=', dataset.getLoss(i,infCls=opengm.inference.Icm,parameter=opengm.InfParam())

print 'total loss =', dataset.getLoss(i,infCls=opengm.inference.Icm,parameter=opengm.InfParam())
