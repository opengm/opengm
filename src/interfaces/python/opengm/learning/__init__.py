from _learning import *
from _learning import _lunarySharedFeatFunctionsGen,_lpottsFunctionsGen
import numpy
import struct
from opengm import index_type,value_type, label_type, graphicalModel,gridVis
from opengm import configuration as opengmConfig, LUnaryFunction
from opengm import to_native_boost_python_enum_converter
from opengm import Tribool
from progressbar import *
from functools import partial


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
    learnerClss = [GridSearch_FlexibleLoss, StructPerceptron_FlexibleLoss,  
                  SubgradientSSVM_FlexibleLoss, Rws_FlexibleLoss] 
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


def rws(dataset,eps=1e-5, maxIterations=10000, stopLoss=0.0, learningRate=1.0, C=100.0, sigma=1.0, p=10):

    assert dataset.__class__.lossType == 'flexible'
    learnerCls = Rws_FlexibleLoss
    learnerParamCls = Rws_FlexibleLossParameter


    param = learnerParamCls()
    param.eps = float(eps)
    param.maxIterations = int(maxIterations)
    param.stopLoss = float(stopLoss)
    param.learningRate = float(learningRate)
    param.C = float(C)
    param.p = int(p)
    param.sigma = float(sigma)
    learner = learnerCls(dataset, param)
    return learner



def subgradientSSVM(dataset, learningMode='batch',eps=1e-5, maxIterations=10000, stopLoss=0.0, learningRate=1.0, C=100.0, averaging=-1, nConf=0):

    assert dataset.__class__.lossType == 'flexible'
    learnerCls = SubgradientSSVM_FlexibleLoss
    learnerParamCls = SubgradientSSVM_FlexibleLossParameter
    learningModeEnum = SubgradientSSVM_FlexibleLossParameter_LearningMode

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
    param.learningRate = float(learningRate)
    param.C = float(C)
    param.learningMode = lm
    param.averaging = int(averaging)
    param.nConf = int(nConf)
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
        raise RuntimeError("this learner needs withCplex or withGurobi")


def maxLikelihoodLearner(
        dataset, 
        maximumNumberOfIterations = 100,
        gradientStepSize = 0.1,
        weightStoppingCriteria = 0.00000001,
        gradientStoppingCriteria = 0.00000000001,
        infoFlag = True,
        infoEveryStep = False,
        weightRegularizer = 1.0,
        beliefPropagationMaximumNumberOfIterations = 40,
        beliefPropagationConvergenceBound = 0.0001,
        beliefPropagationDamping = 0.5,
        beliefPropagationReg = 1.0,
        beliefPropagationTemperature = 1.0,
        beliefPropagationIsAcyclic = Tribool(0)
):

    learnerCls = MaxLikelihood_FlexibleLoss
    learnerParamCls = MaxLikelihood_FlexibleLossParameter

    param = learnerParamCls(
        maximumNumberOfIterations,
        gradientStepSize,
        weightStoppingCriteria,
        gradientStoppingCriteria,
        infoFlag,
        infoEveryStep,
        weightRegularizer,
        beliefPropagationMaximumNumberOfIterations,
        beliefPropagationConvergenceBound,
        beliefPropagationDamping,
        beliefPropagationTemperature,
        beliefPropagationIsAcyclic
    )
    #param.maxIterations = int(maxIterations)
    #param.reg = float(reg)
    #param.temperature = float(temp)

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







def getPbar(size, name):
    widgets = ['%s: '%name, Percentage(), ' ', Bar(marker='0',left='[',right=']'),
               ' ', ETA(), ' ', FileTransferSpeed()] #see docs for other options
    pbar = ProgressBar(widgets=widgets, maxval=size)
    return pbar

def secondOrderImageDataset(imgs, gts, numberOfLabels, fUnary, fBinary, addConstFeature, trainFraction=0.75):
    #try:
    #    import vigra
    #    from progressbar import *
    #except:
    #    pass

    # train test
    nImg = len(imgs)
    nTrain = int(float(nImg)*trainFraction+0.5)
    nTest = (nImg-nTrain)
    
    def getFeat(fComp, im):
        res = []
        for f in fComp:
            r = f(im)
            if r.ndim == 2:
                r = r[:,:, None]
            res.append(r)
        return res

    # compute features for a single image
    tImg = imgs[0]
    unaryFeat = getFeat(fUnary, tImg)
    unaryFeat = numpy.nan_to_num(numpy.concatenate(unaryFeat,axis=2).view(numpy.ndarray))
    nUnaryFeat = unaryFeat.shape[-1] + int(addConstFeature)
    nUnaryFeat *= numberOfLabels - int(numberOfLabels==2)

    if len(fBinary)>0:
        binaryFeat = getFeat(fBinary, tImg)
        binaryFeat = numpy.nan_to_num(numpy.concatenate(binaryFeat,axis=2).view(numpy.ndarray))
        nBinaryFeat = binaryFeat.shape[-1] + int(addConstFeature)
        nWeights  = nUnaryFeat + nBinaryFeat
    else:
        nBinaryFeat = 0
    print "------------------------------------------------"
    print "nTrain",nTrain,"nTest",nTest
    print "nWeights",nWeights,"(",nUnaryFeat,nBinaryFeat,")"
    print "------------------------------------------------"

    train_set = []
    tentative_test_set = []

    for i,(img,gt) in enumerate(zip(imgs,gts)):
        if(i<nTrain):
            train_set.append((img,gt))
        else:
            tentative_test_set.append((img,gt))


    dataset = createDataset(numWeights=nWeights)
    weights = dataset.getWeights()
    uWeightIds = numpy.arange(nUnaryFeat ,dtype='uint64')
    if numberOfLabels != 2:
        uWeightIds = uWeightIds.reshape([numberOfLabels,-1])
    else:
        uWeightIds = uWeightIds.reshape([1,-1])
    bWeightIds = numpy.arange(start=nUnaryFeat,stop=nWeights,dtype='uint64')

    def makeModel(img,gt):
        shape = gt.shape[0:2]
        numVar = shape[0] * shape[1]

        # make model
        gm = graphicalModel(numpy.ones(numVar)*numberOfLabels)




        # compute features
        unaryFeat = getFeat(fUnary, img)
        unaryFeat = numpy.nan_to_num(numpy.concatenate(unaryFeat,axis=2).view(numpy.ndarray))
        unaryFeat  = unaryFeat.reshape([numVar,-1])
        



        # add unaries
        lUnaries = lUnaryFunctions(weights =weights,numberOfLabels = numberOfLabels, 
                                    features=unaryFeat, weightIds = uWeightIds,
                                    featurePolicy= FeaturePolicy.sharedBetweenLabels,
                                    makeFirstEntryConst=numberOfLabels==2, addConstFeature=addConstFeature)
        fids = gm.addFunctions(lUnaries)
        gm.addFactors(fids, numpy.arange(numVar))


        if len(fBinary)>0:
            binaryFeat = getFeat(fBinary, img)
            binaryFeat = numpy.nan_to_num(numpy.concatenate(binaryFeat,axis=2).view(numpy.ndarray))
            binaryFeat  = binaryFeat.reshape([numVar,-1])

            # add second order
            vis2Order=gridVis(shape[0:2],True)

            fU = binaryFeat[vis2Order[:,0],:]
            fV = binaryFeat[vis2Order[:,1],:]
            fB  = (fU +  fV / 2.0)
            
            lp = lPottsFunctions(weights=weights, numberOfLabels=numberOfLabels,
                                          features=fB, weightIds=bWeightIds,
                                          addConstFeature=addConstFeature)
            gm.addFactors(gm.addFunctions(lp), vis2Order) 

        return gm

    # make training models
    pbar = getPbar(nTrain,"Training Models")
    pbar.start()
    for i,(img,gt) in enumerate(train_set):
        gm = makeModel(img, gt)
        dataset.pushBackInstance(gm,gt.reshape(-1).astype(label_type))
        pbar.update(i)
    pbar.finish()


    # make test models
    test_set = []
    pbar = getPbar(nTest,"Test Models")
    pbar.start()
    for i,(img,gt) in enumerate(tentative_test_set):
        gm = makeModel(img, gt)
        test_set.append((img, gt, gm))
        pbar.update(i)
    pbar.finish()

    return dataset, test_set



def superpixelDataset(imgs,sps, gts, numberOfLabels, fUnary, fBinary, addConstFeature, trainFraction=0.75):
    try:
        import vigra
    except:
        raise ImportError("cannot import vigra which is needed for superpixelDataset")

    # train test
    nImg = len(imgs)
    nTrain = int(float(nImg)*trainFraction+0.5)
    nTest = (nImg-nTrain)
    
    def getFeat(fComp, im, topoShape=False):
        res = []
        if(topoShape):
            shape = im.shape[0:2]
            tshape = [2*s-1 for s in shape]
            iiimg = vigra.sampling.resize(im, tshape)
        else:
            iiimg = im
        for f in fComp:
            r = f(iiimg)
            if r.ndim == 2:
                r = r[:,:, None]
            res.append(r)
        return res

    # compute features for a single image
    tImg = imgs[0]
    unaryFeat = getFeat(fUnary, tImg)
    unaryFeat = numpy.nan_to_num(numpy.concatenate(unaryFeat,axis=2).view(numpy.ndarray))
    nUnaryFeat = unaryFeat.shape[-1] + int(addConstFeature)
    nUnaryFeat *= numberOfLabels - int(numberOfLabels==2)
    if len(fBinary)>0:
        binaryFeat = getFeat(fBinary, tImg)
        binaryFeat = numpy.nan_to_num(numpy.concatenate(binaryFeat,axis=2).view(numpy.ndarray))
        nBinaryFeat = binaryFeat.shape[-1] + int(addConstFeature)
    else:
        nBinaryFeat =0

    nWeights  = nUnaryFeat + nBinaryFeat

    print "------------------------------------------------"
    print "nTrain",nTrain,"nTest",nTest
    print "nWeights",nWeights,"(",nUnaryFeat,nBinaryFeat,")"
    print "------------------------------------------------"

    train_set = []
    tentative_test_set = []

    for i,(img,sp,gt) in enumerate(zip(imgs,sps,gts)):
        if(i<nTrain):
            train_set.append((img,sp,gt))
        else:
            tentative_test_set.append((img,sp,gt))


    dataset = createDataset(numWeights=nWeights)
    weights = dataset.getWeights()
    uWeightIds = numpy.arange(nUnaryFeat ,dtype='uint64')
    if numberOfLabels != 2:
        uWeightIds = uWeightIds.reshape([numberOfLabels,-1])
    else:
        uWeightIds = uWeightIds.reshape([1,-1])

    if len(fBinary)>0:
        bWeightIds = numpy.arange(start=nUnaryFeat,stop=nWeights,dtype='uint64')





    def makeModel(img,sp,gt):
        assert sp.min() == 0
        shape = img.shape[0:2]
        gg = vigra.graphs.gridGraph(shape)
        rag = vigra.graphs.regionAdjacencyGraph(gg,sp)
        numVar = rag.nodeNum
        assert rag.nodeNum == rag.maxNodeId +1

        # make model
        gm = graphicalModel(numpy.ones(numVar)*numberOfLabels)

        assert gm.numberOfVariables == rag.nodeNum 
        assert gm.numberOfVariables == rag.maxNodeId +1

        # compute features
        unaryFeat = getFeat(fUnary, img)
        unaryFeat = numpy.nan_to_num(numpy.concatenate(unaryFeat,axis=2).view(numpy.ndarray)).astype('float32')
        unaryFeat = vigra.taggedView(unaryFeat,'xyc')
        accList = []

        #for c in range(unaryFeat.shape[-1]):
        #    cUnaryFeat = unaryFeat[:,:,c]
        #    cAccFeat = rag.accumulateNodeFeatures(cUnaryFeat)[:,None]
        #    accList.append(cAccFeat)
        #accUnaryFeat = numpy.concatenate(accList,axis=1)
        accUnaryFeat = rag.accumulateNodeFeatures(unaryFeat)#[:,None]


        #print accUnaryFeat.shape

        #accUnaryFeat = rag.accumulateNodeFeatures(unaryFeat[:,:,:])
        #accUnaryFeat = vigra.taggedView(accUnaryFeat,'nc')
        #accUnaryFeat = accUnaryFeat[1:accUnaryFeat.shape[0],:]

      



        #binaryFeat  = binaryFeat.reshape([numVar,-1])



        # add unaries
        lUnaries = lUnaryFunctions(weights =weights,numberOfLabels = numberOfLabels, 
                                            features=accUnaryFeat, weightIds = uWeightIds,
                                            featurePolicy= FeaturePolicy.sharedBetweenLabels,
                                            makeFirstEntryConst=numberOfLabels==2, addConstFeature=addConstFeature)
        fids = gm.addFunctions(lUnaries)
        gm.addFactors(fids, numpy.arange(numVar))

        
        if len(fBinary)>0:
            binaryFeat = getFeat(fBinary, img, topoShape=False)
            binaryFeat = numpy.nan_to_num(numpy.concatenate(binaryFeat,axis=2).view(numpy.ndarray)).astype('float32')
            edgeFeat = vigra.graphs.edgeFeaturesFromImage(gg, binaryFeat)
            accBinaryFeat = rag.accumulateEdgeFeatures(edgeFeat)

            uvIds =  numpy.sort(rag.uvIds(), axis=1)
            assert uvIds.min()==0
            assert uvIds.max()==gm.numberOfVariables-1



        
            lp = lPottsFunctions(weights=weights, numberOfLabels=numberOfLabels,
                                          features=accBinaryFeat, weightIds=bWeightIds,
                                          addConstFeature=addConstFeature)
            fids = gm.addFunctions(lp)
            gm.addFactors(fids, uvIds) 

        return gm

    # make training models
    pbar = getPbar(nTrain,"Training Models")
    pbar.start()
    for i,(img,sp,gt) in enumerate(train_set):
        gm = makeModel(img,sp, gt)
        dataset.pushBackInstance(gm,gt.astype(label_type))
        pbar.update(i)
    pbar.finish()


    # make test models
    test_set = []
    pbar = getPbar(nTest,"Test Models")
    pbar.start()
    for i,(img,sp,gt) in enumerate(tentative_test_set):
        gm = makeModel(img,sp, gt)
        test_set.append((img, sp, gm))
        pbar.update(i)
    pbar.finish()

    return dataset, test_set
