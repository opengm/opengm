from _learning import *
import numpy
import struct

DatasetWithHammingLoss.lossType = 'hamming'
DatasetWithGeneralizedHammingLoss.lossType = 'generalized-hamming'


def createDataset(loss='hamming', numInstances=0):
    
    if loss not in ['hamming','h','gh','generalized-hamming']:
        raise RuntimeError("loss must be 'hamming' /'h' or 'generalized-hamming'/'gh' ")    

    if loss in ['hamming','h']:
        return DatasetWithHammingLoss(int(numInstances))
    elif loss in ['generalized-hamming','gh']:
        return DatasetWithGeneralizedHammingLoss(int(numInstances))
    else:
        raise RuntimeError("loss must be 'hamming' /'h' or 'generalized-hamming'/'gh' ")   




def gridSearchLearner(dataset, lowerBounds, upperBounds, nTestPoints):

    if dataset.__class__.lossType == 'hamming':
        learnerCls = GridSearch_HammingLoss
        leanerParamCls = GridSearch_HammingLossParameter
    elif dataset.__class__.lossType == 'generalized-hamming':
        learnerCls = GridSearch_GeneralizedHammingLoss
        leanerParamCls = GridSearch_GeneralizedHammingLossParameter

    nr = numpy.require 

    sizeT_type = 'uint64'

    if struct.calcsize("P") * 8 == 32:
        sizeT_type = 'uint32'

    param = leanerParamCls(nr(lowerBounds,dtype='float64'), nr(lowerBounds,dtype='float64'), 
                           nr(lowerBounds,dtype=sizeT_type))

    learner = learnerCls(dataset, param)

    return learner
