import opengm
import opengm.learning as learning
from opengm import numpy

# create a simple model with exactly one variable with two labels
numWeights = 2
nLabels = 2
nVars = 1

# set weight ids and features for both labels
weightIds = numpy.array([[0, 1],       [0, 1]])
features = numpy.array( [[0.5, -0.25], [-0.5, -1.25]])

# create dataset with 2 weights and get the 2 weights
dataset = learning.createDataset(numWeights)
weights = dataset.getWeights()

# set up graphical model
gm = opengm.gm(numpy.ones(nVars)*nLabels)
fid = gm.addFunction(learning.lUnaryFunction(weights, 2, features, weightIds))
gm.addFactor(fid, [0])

# add graphical model to dataset with ground truth
ground_truth = numpy.array([0]).astype(opengm.label_type)
dataset.pushBackInstance(gm, ground_truth)

# set up learner and run
learner = learning.maxLikelihoodLearner(dataset)
learner.learn(infCls=opengm.inference.TrwsExternal,  parameter=opengm.InfParam())