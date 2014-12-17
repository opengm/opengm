from _learning import *








def createDataset(loss='hamming', numInstances=0):
    
    if loss not in ['hamming','h','gh','generalized-hamming']:
        raise RuntimeError("loss must be 'hamming' /'h' or 'generalized-hamming'/'gh' ")    

    if loss in ['hamming','h']:
        return DatasetWithHammingLoss(int(numInstances))
    elif loss in ['generalized-hamming','gh']:
        return DatasetWithGeneralizedHammingLoss(int(numInstances))
    else:
        raise RuntimeError("loss must be 'hamming' /'h' or 'generalized-hamming'/'gh' ")   

