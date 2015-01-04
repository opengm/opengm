from _learning import *
from _learning import _lunarySharedFeatFunctionsGen,_lpottsFunctionsGen
import numpy
import struct
from opengm import index_type,value_type, label_type
from opengm import configuration as opengmConfig, LUnaryFunction
DatasetWithHammingLoss.lossType = 'hamming'
DatasetWithGeneralizedHammingLoss.lossType = 'generalized-hamming'




def _extendedLearn(self, infCls, parameter = None):
    if parameter is None:
        import opengm
        parameter = opengm.InfParam()
    cppParam  =  infCls.get_cpp_parameter(operator='adder',accumulator='minimizer',parameter=parameter)
    self._learn(cppParam)

def _extendedGetLoss(self, model_idx, infCls, parameter = None):
    if parameter is None:
        import opengm
        parameter = opengm.InfParam()
    cppParam  =  infCls.get_cpp_parameter(operator='adder',accumulator='minimizer',parameter=parameter)
    return self._getLoss(cppParam, model_idx)

def _extendedGetTotalLoss(self, infCls, parameter = None):
    if parameter is None:
        import opengm
        parameter = opengm.InfParam()
    cppParam  =  infCls.get_cpp_parameter(operator='adder',accumulator='minimizer',parameter=parameter)
    return self._getTotalLoss(cppParam)

GridSearch_HammingLoss.learn  =_extendedLearn
GridSearch_GeneralizedHammingLoss.learn  =_extendedLearn

MaxLikelihood_HammingLoss.learn  =_extendedLearn
MaxLikelihood_GeneralizedHammingLoss.learn  =_extendedLearn

StructPerceptron_HammingLoss.learn  =_extendedLearn
StructPerceptron_GeneralizedHammingLoss.learn  =_extendedLearn


if opengmConfig.withCplex or opengmConfig.withGurobi :
    StructMaxMargin_Bundle_HammingLoss.learn = _extendedLearn
    StructMaxMargin_Bundle_GeneralizedHammingLoss.learn = _extendedLearn

DatasetWithHammingLoss.getLoss = _extendedGetLoss
DatasetWithHammingLoss.getTotalLoss = _extendedGetTotalLoss
DatasetWithGeneralizedHammingLoss.getLoss = _extendedGetLoss
DatasetWithGeneralizedHammingLoss.getTotalLoss = _extendedGetTotalLoss

def createDataset(numWeights, loss='hamming', numInstances=0):
    weightVals = numpy.ones(numWeights)
    weights = Weights(weightVals)

    if loss not in ['hamming','h','gh','generalized-hamming']:
        raise RuntimeError("loss must be 'hamming' /'h' or 'generalized-hamming'/'gh' ")    

    if loss in ['hamming','h']:
        dataset = DatasetWithHammingLoss(int(numInstances))
    elif loss in ['generalized-hamming','gh']:
        dataset = DatasetWithGeneralizedHammingLoss(int(numInstances))
    else:
        raise RuntimeError("loss must be 'hamming' /'h' or 'generalized-hamming'/'gh' ")   

    dataset.setWeights(weights)
    return dataset




def gridSearchLearner(dataset, lowerBounds, upperBounds, nTestPoints):

    if dataset.__class__.lossType == 'hamming':
        learnerCls = GridSearch_HammingLoss
        learnerParamCls = GridSearch_HammingLossParameter
    elif dataset.__class__.lossType == 'generalized-hamming':
        learnerCls = GridSearch_GeneralizedHammingLoss
        learnerParamCls = GridSearch_GeneralizedHammingLossParameter

    nr = numpy.require 
    sizeT_type = 'uint64'

    if struct.calcsize("P") * 8 == 32:
        sizeT_type = 'uint32'

    param = learnerParamCls(nr(lowerBounds,dtype='float64'), nr(upperBounds,dtype='float64'), 
                           nr(nTestPoints,dtype=sizeT_type))

    learner = learnerCls(dataset, param)
    return learner


def structPerceptron(dataset, learningMode='online',eps=1e-5, maxIterations=10000, stopLoss=0.0, decayExponent=0.0, decayT0=0.0):


    if dataset.__class__.lossType == 'hamming':
        learnerCls = StructPerceptron_HammingLoss
        learnerParamCls = StructPerceptron_HammingLossParameter
        learningModeEnum = StructPerceptron_HammingLossParameter_LearningMode
    elif dataset.__class__.lossType == 'generalized-hamming':
        learnerCls = StructPerceptron_GeneralizedHammingLossParameter
        learnerParamCls = StructPerceptron_GeneralizedHammingLoss
        learningModeEnum = StructPerceptron_GeneralizedHammingLossParameter_LearningMode

    lm = None
    if learningMode not in ['online','batch']:
        raise RuntimeError("wrong learning mode, must be 'online' or 'batch' ")

    if learningMode == 'online':
        lm = learningModeEnum.online
    if learningMode == 'batch':
        lm = learningModeEnum.batch

    param = learnerParamCls()
    param.eps = float(eps)
    param.maxIterations = int(maxIterations)
    param.stopLoss = float(stopLoss)
    param.decayExponent = float(decayExponent)
    param.decayT0 = float(decayT0)
    param.learningMode = lm
    learner = learnerCls(dataset, param)
    return learner

def structMaxMarginLearner(dataset, regularizerWeight=1.0, minEps=1e-5, nSteps=0, epsStrategy='change', optimizer='bundle'):

    if opengmConfig.withCplex or opengmConfig.withGurobi :
        if optimizer != 'bundle':
            raise RuntimeError("Optimizer type must be 'bundle' for now!")

        if dataset.__class__.lossType == 'hamming':
            learnerCls = StructMaxMargin_Bundle_HammingLoss
            learnerParamCls = StructMaxMargin_Bundle_HammingLossParameter
        elif dataset.__class__.lossType == 'generalized-hamming':
            learnerCls = StructMaxMargin_Bundle_GeneralizedHammingLoss
            learnerParamCls = StructMaxMargin_Bundle_GeneralizedHammingLossParameter

        epsFromGap = False
        if epsStrategy == 'gap':
            epsFromGap = True
        elif epsStrategy == 'change':
            epsFromGap = False

        param = learnerParamCls(regularizerWeight, minEps, nSteps, epsFromGap)
        learner = learnerCls(dataset, param)
        
        return learner
    else:
        raise RuntimeError("this learner needs widthCplex or withGurobi")

def maxLikelihoodLearner(dataset):
    if dataset.__class__.lossType == 'hamming':
        learnerCls = MaxLikelihood_HammingLoss
        learnerParamCls = MaxLikelihood_HammingLossParameter
    elif dataset.__class__.lossType == 'generalized-hamming':
        learnerCls = MaxLikelihood_GeneralizedHammingLoss
        learnerParamCls = MaxLikelihood_GeneralizedHammingLossParameter

    param = learnerParamCls()
    learner = learnerCls(dataset, param)
        
    return learner





def lUnaryFunction(weights, numberOfLabels, features, weightIds):

    assert numberOfLabels >= 2
    features = numpy.require(features, dtype=value_type)
    weightIds = numpy.require(weightIds, dtype=index_type)

    assert features.ndim == weightIds.ndim
    if features.ndim == 1 or weightIds.ndim == 1:
        assert numberOfLabels == 2
        assert features.shape[0]  == weightIds.shape[0]
        features  = features.reshape(1,-1)
        weightIds = weightIds.reshape(1,-1)

    assert features.shape[0] in [numberOfLabels, numberOfLabels-1]
    assert weightIds.shape[0] in [numberOfLabels, numberOfLabels-1]
    assert features.shape[1]  == weightIds.shape[1]


    return LUnaryFunction(weights=weights, numberOfLabels=int(numberOfLabels), 
                          features=features, weightIds=weightIds)




class FeaturePolicy(object):
    sharedBetweenLabels = 0

def lUnaryFunctions(weights,numberOfLabels, features, weightIds,
                    featurePolicy = FeaturePolicy.sharedBetweenLabels, 
                    **kwargs):

    if (featurePolicy == FeaturePolicy.sharedBetweenLabels ):

        makeFirstEntryConst = kwargs.get('makeFirstEntryConst',False)
        addConstFeature = kwargs.get('addConstFeature',False)


        ff = numpy.require(features, dtype=value_type)
        wid = numpy.require(weightIds, dtype=index_type)

        assert features.ndim == 2
        assert weightIds.ndim == 2


        res = _lunarySharedFeatFunctionsGen(
            weights = weights,
            numFunctions = int(ff.shape[0]),
            numLabels = int(numberOfLabels),
            features = ff,
            weightIds = wid,
            makeFirstEntryConst = bool(makeFirstEntryConst),
            addConstFeature = bool(addConstFeature)
        )

        res.__dict__['_features_'] = weights
        res.__dict__['_weights_'] = features
        return res

def lPottsFunctions(weights, numberOfLabels, features, weightIds,
                    addConstFeature = False):

    # check that features has the correct shape
    if features.ndim != 2:
        raise RuntimeError("feature must be two-dimensional")

    # check that weights has the correct shape
    if weightIds.ndim != 1:
        raise RuntimeError("weightIds must be one-dimensional")
    if weightIds.shape[0] != features.shape[1] + int(addConstFeature) :
        raise RuntimeError("weightIds.shape[0]  must be equal to features.shape[1]")



    # do the c++ call here
    # which generates a function generator


    ff = numpy.require(features, dtype=value_type)
    wid = numpy.require(weightIds, dtype=index_type)
    res =  _lpottsFunctionsGen(
        weights=weights,
        numFunctions=long(features.shape[0]),
        numLabels=long(numberOfLabels),
        features=ff,
        weightIds=wid,
        addConstFeature=bool(addConstFeature)
    )

    res.__dict__['_features_'] = wid
    res.__dict__['_weights_'] = ff
    return res
