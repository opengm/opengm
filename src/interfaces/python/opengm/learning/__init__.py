from _learning import *
from _learning import _lunarySharedFeatFunctionsGen,_lpottsFunctionsGen
import numpy
import struct
from opengm import index_type,value_type, label_type
from opengm import configuration as opengmConfig, LUnaryFunction
from opengm import to_native_boost_python_enum_converter





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






DatasetWithFlexibleLoss.lossType = 'flexible'


class LossParameter(FlexibleLossParameter):
    def __init__(self, lossType, labelMult=None, nodeMult=None, factorMult=None):
        super(LossParameter, self).__init__()

        self.lossType = to_native_boost_python_enum_converter(lossType,self.lossType.__class__)

        if labelMult is not None:
            assert self.lossType == LossType.hamming
            self.setLabelLossMultiplier(labelMult)
        if nodeMult is not None:
            assert self.lossType != LossType.partition
            self.setNodeLossMultiplier(nodeMult)
        if factorMult is not None:
            assert self.lossType == LossType.partition
            self.setFactorLossMultiplier(factorMult)



def extend_learn():
    
    def learner_learn_normal(self, infCls, parameter = None):
        if parameter is None:
            import opengm
            parameter = opengm.InfParam()
        cppParam  =  infCls.get_cpp_parameter(operator='adder',accumulator='minimizer',parameter=parameter)
        try:
          self._learn(cppParam)
        except Exception, e:
            #print "an error ",e,"\n\n"
            if (str(e).find("did not match C++ signature")):
                raise RuntimeError("infCls : '%s' is not (yet) exported from c++ to python for learning"%str(infCls))


    def learner_learn_reduced_inf(self, infCls, parameter = None, persistency=True, tentacles=False, connectedComponents=False):
        if parameter is None:
            import opengm
            parameter = opengm.InfParam()
        cppParam  =  infCls.get_cpp_parameter(operator='adder',accumulator='minimizer',parameter=parameter)
        try:
          self._learnReducedInf(cppParam, bool(persistency), bool(tentacles),bool(connectedComponents))
        except Exception, e:
            #print "an error ",e,"\n\n"
            if (str(e).find("did not match C++ signature")):
                raise RuntimeError("infCls : '%s' is not (yet) exported from c++ to python for learning with reduced inference"%str(infCls))

    def learner_learn_reduced_inf_self_fusion(self, infCls, parameter = None, persistency=True, tentacles=False, connectedComponents=False):
        if parameter is None:
            import opengm
            parameter = opengm.InfParam()
        cppParam  =  infCls.get_cpp_parameter(operator='adder',accumulator='minimizer',parameter=parameter)
        try:
          self._learnReducedInf(cppParam, bool(persistency), bool(tentacles),bool(connectedComponents))
        except Exception, e:
            #print "an error ",e,"\n\n"
            if (str(e).find("did not match C++ signature")):
                raise RuntimeError("infCls : '%s' is not (yet) exported from c++ to python for learning with reduced inference"%str(infCls))

    def learner_learn_self_fusion(self, infCls, parameter = None, fuseNth=1, fusionSolver="qpbo",maxSubgraphSize=2,
                                  redInf=True, connectedComponents=False, fusionTimeLimit=100.9, numStopIt=10):
        if parameter is None:
            import opengm
            parameter = opengm.InfParam()
        cppParam  =  infCls.get_cpp_parameter(operator='adder',accumulator='minimizer',parameter=parameter)
        try:
          self._learnSelfFusion(cppParam, int(fuseNth),str(fusionSolver),int(maxSubgraphSize),bool(redInf),
                                bool(connectedComponents),float(fusionTimeLimit),int(numStopIt))
        except Exception, e:
            #print "an error ",e,"\n\n"
            if (str(e).find("did not match C++ signature")):
                raise RuntimeError("infCls : '%s' is not (yet) exported from c++ to python for learning with self fusion inference"%str(infCls))

    def learner_learn(self, infCls, parameter=None, infMode='normal',**kwargs):
        assert infMode in ['normal','n','selfFusion','sf','reducedInference','ri','reducedInferenceSelfFusion','risf']

        if infMode in ['normal','n']:
            self.learnNormal(infCls=infCls, parameter=parameter)
        elif infMode in ['selfFusion','sf']:
            self.learnSelfFusion(infCls=infCls, parameter=parameter,**kwargs)
        elif infMode in ['reducedInference','ri']:
            self.learnReducedInf(infCls=infCls, parameter=parameter,**kwargs)
        elif infMode in ['reducedInferenceSelfFusion','risf']:
            self.learnReducedInfSelfFusion(infCls=infCls, parameter=parameter,**kwargs)

    # all learner classes
    learnerClss = [GridSearch_FlexibleLoss, StructPerceptron_FlexibleLoss,  SubgradientSSVM_FlexibleLoss] 
    if opengmConfig.withCplex or opengmConfig.withGurobi :
        learnerClss.append(StructMaxMargin_Bundle_FlexibleLoss)

    for learnerCls in learnerClss:
        learnerCls.learn = learner_learn
        learnerCls.learnNormal = learner_learn_normal
        learnerCls.learnReducedInf = learner_learn_reduced_inf
        learnerCls.learnSelfFusion = learner_learn_self_fusion
        learnerCls.learnReducedInfSelfFusion = learner_learn_reduced_inf_self_fusion

extend_learn()
del extend_learn





DatasetWithFlexibleLoss.getLoss = _extendedGetLoss
DatasetWithFlexibleLoss.getTotalLoss = _extendedGetTotalLoss


def createDataset(numWeights,  numInstances=0):
    w  = Weights(numWeights)

    # if loss not in ['hamming','h','gh','generalized-hamming']:
    #     raise RuntimeError("loss must be 'hamming' /'h' or 'generalized-hamming'/'gh' ")    
    # if loss in ['hamming','h']:
    #     dataset = DatasetWithHammingLoss(int(numInstances))
    # elif loss in ['generalized-hamming','gh']:
    #     dataset = DatasetWithGeneralizedHammingLoss(int(numInstances))
    # else:
    #     raise RuntimeError("loss must be 'hamming' /'h' or 'generalized-hamming'/'gh' ")   
    dataset = DatasetWithFlexibleLoss(numInstances)
    dataset.setWeights(w)
    weights = dataset.getWeights()
    for wi in range(numWeights):
        weights[wi] = 0.0
    return dataset




def gridSearchLearner(dataset, lowerBounds, upperBounds, nTestPoints):
    assert dataset.__class__.lossType == 'flexible'
    learnerCls = GridSearch_FlexibleLoss
    learnerParamCls = GridSearch_FlexibleLossParameter

    nr = numpy.require 
    sizeT_type = 'uint64'

    if struct.calcsize("P") * 8 == 32:
        sizeT_type = 'uint32'

    param = learnerParamCls(nr(lowerBounds,dtype='float64'), nr(upperBounds,dtype='float64'), 
                           nr(nTestPoints,dtype=sizeT_type))

    learner = learnerCls(dataset, param)
    return learner


def structPerceptron(dataset, learningMode='online',eps=1e-5, maxIterations=10000, stopLoss=0.0, decayExponent=0.0, decayT0=0.0):

    assert dataset.__class__.lossType == 'flexible'
    learnerCls = StructPerceptron_FlexibleLoss
    learnerParamCls = StructPerceptron_FlexibleLossParameter
    learningModeEnum = StructPerceptron_FlexibleLossParameter_LearningMode

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


def subgradientSSVM(dataset, learningMode='batch',eps=1e-5, maxIterations=10000, stopLoss=0.0, learningRate=1.0, C=100.0, averaging=-1):

    assert dataset.__class__.lossType == 'flexible'
    learnerCls = SubgradientSSVM_FlexibleLoss
    learnerParamCls = SubgradientSSVM_FlexibleLossParameter
    learningModeEnum = SubgradientSSVM_FlexibleLossParameter_LearningMode

    lm = None
    if learningMode not in ['online','batch']:
        raise RuntimeError("wrong learning mode, must be 'online', 'batch' or 'workingSets' ")

    if learningMode == 'online':
        lm = learningModeEnum.online
    if learningMode == 'batch':
        lm = learningModeEnum.batch
    param = learnerParamCls()
    param.eps = float(eps)
    param.maxIterations = int(maxIterations)
    param.stopLoss = float(stopLoss)
    param.learningRate = float(learningRate)
    param.C = float(C)
    param.learningMode = lm
    param.averaging = int(averaging)
    #param.nConf = int(nConf)
    learner = learnerCls(dataset, param)
    return learner

def structMaxMarginLearner(dataset, regularizerWeight=1.0, minEps=1e-5, nSteps=0, epsStrategy='change', optimizer='bundle'):

    if opengmConfig.withCplex or opengmConfig.withGurobi :
        if optimizer != 'bundle':
            raise RuntimeError("Optimizer type must be 'bundle' for now!")


        assert dataset.__class__.lossType == 'flexible'
        learnerCls = StructMaxMargin_Bundle_FlexibleLoss
        learnerParamCls = StructMaxMargin_Bundle_FlexibleLossParameter

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
    #raise RuntimeError("not yet implemented / wrapped fully")
    learnerCls = MaxLikelihood_FlexibleLoss
    learnerParamCls = MaxLikelihood_FlexibleLossParameter

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

        res.__dict__['_features_'] =features
        res.__dict__['_ff_'] = ff
        res.__dict__['_weights_'] =  weights

        return res
    else :
        raise RuntimeError("noy yet implemented")

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
