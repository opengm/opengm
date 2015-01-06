import numpy
import opengm
from opengm import learning
import vigra
from progressbar import *
from functools import partial



def getPbar(size, name):
    widgets = ['%s: '%name, Percentage(), ' ', Bar(marker='0',left='[',right=']'),
               ' ', ETA(), ' ', FileTransferSpeed()] #see docs for other options
    pbar = ProgressBar(widgets=widgets, maxval=size)
    return pbar

def secondOrderImageDataset(imgs, gts, numberOfLabels, fUnary, fBinary, addConstFeature, trainFraction=0.75):
    assert numberOfLabels == 2

    # train test
    nImg = len(imgs)
    nTrain = int(float(nImg)*trainFraction+0.5)
    nTest = (nImg-nTrain)
    
    

    # compute features for a single image
    tImg = imgs[0]
    unaryFeat = [f(tImg) for f in fUnary]
    unaryFeat = numpy.nan_to_num(numpy.concatenate(unaryFeat,axis=2).view(numpy.ndarray))
    nUnaryFeat = unaryFeat.shape[-1] + int(addConstFeature)

    binaryFeat = [f(tImg) for f in fBinary]
    binaryFeat = numpy.nan_to_num(numpy.concatenate(binaryFeat,axis=2).view(numpy.ndarray))
    nBinaryFeat = binaryFeat.shape[-1] + int(addConstFeature)
    nWeights  = nUnaryFeat + nBinaryFeat
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


    dataset = learning.createDataset(numWeights=nWeights, loss='h')
    weights = dataset.getWeights()
    uWeightIds = numpy.arange(nUnaryFeat ,dtype='uint64')
    bWeightIds = numpy.arange(start=nUnaryFeat,stop=nWeights,dtype='uint64')

    def makeModel(img,gt):
        shape = gt.shape[0:2]
        numVar = shape[0] * shape[1]

        # make model
        gm = opengm.gm(numpy.ones(numVar)*2)

        # compute features
        unaryFeat = [f(img) for f in fUnary]
        unaryFeat = numpy.nan_to_num(numpy.concatenate(unaryFeat,axis=2).view(numpy.ndarray))
        unaryFeat  = unaryFeat.reshape([numVar,-1])
        binaryFeat = [f(img) for f in fBinary]
        binaryFeat = numpy.nan_to_num(numpy.concatenate(binaryFeat,axis=2).view(numpy.ndarray))
        binaryFeat  = binaryFeat.reshape([numVar,-1])



        # add unaries
        lUnaries = learning.lUnaryFunctions(weights =weights,numberOfLabels = numberOfLabels, 
                                            features=unaryFeat, weightIds = uWeightIds.reshape([1,-1]).copy(),
                                            featurePolicy= learning.FeaturePolicy.sharedBetweenLabels,
                                            makeFirstEntryConst=numberOfLabels==2, addConstFeature=addConstFeature)
        fids = gm.addFunctions(lUnaries)
        gm.addFactors(fids, numpy.arange(numVar))

        # add second order
        vis2Order=opengm.gridVis(shape[0:2],True)

        fU = binaryFeat[vis2Order[:,0],:]
        fV = binaryFeat[vis2Order[:,1],:]
        fB  = (fU +  fV / 2.0)
        lp = learning.lPottsFunctions(weights=weights, numberOfLabels=numberOfLabels,
                                      features=fB, weightIds=bWeightIds,
                                      addConstFeature=addConstFeature)
        gm.addFactors(gm.addFunctions(lp), vis2Order) 

        return gm

    # make training models
    pbar = getPbar(nTrain,"Training Models")
    pbar.start()
    for i,(img,gt) in enumerate(train_set):
        gm = makeModel(img, gt)
        dataset.pushBackInstance(gm,gt.reshape(-1).astype(opengm.label_type))
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


