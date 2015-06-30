import opengm
import opengm.learning as learning
from opengm import numpy

# create a simple model with exactly one variable with two labels
numWeights = 4
nLabels = 2
nVars = 1

# set weight ids and features for all labels
weightIds = numpy.array([[0, 1],       [2,3]])
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
#learner = learning.structMaxMarginLearner(dataset, 0.1, 0.001, 0)
#learner =  learning.subgradientSSVM(dataset, learningRate=1.0, C=100, learningMode='batch')
#learner.learn(infCls=opengm.inference.TrwsExternal,  parameter=opengm.InfParam())


learner = learning.maxLikelihoodLearner(
    dataset,
    maximumNumberOfIterations =1500,gradientStepSize = 0.9,weightStoppingCriteria =   0.001,gradientStoppingCriteria = 0.00000000001,infoFlag = True,infoEveryStep = False,weightRegularizer = 1.0, 
    beliefPropagationMaximumNumberOfIterations = 20,beliefPropagationConvergenceBound = 0.0000000000001,beliefPropagationDamping = 0.5,beliefPropagationTemperature = 1,beliefPropagationIsAcyclic=opengm.Tribool(True))
learner.learn()

for w in range(numWeights):
    print weights[w]
