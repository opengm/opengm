import numpy
import opengm
from opengm import learning
import vigra
from progressbar import *
import glob
import os
from functools import partial
from make_grid_potts_dset import secondOrderImageDataset, getPbar



nImages = 15 
shape = [30, 30]
noise = 1
imgs = []
gts = []


for i in range(nImages):

    gtImg = numpy.zeros(shape)
    gtImg[0:shape[0]/2,:] = 1

    gtImg[shape[0]/4: 3*shape[0]/4, shape[0]/4: 3*shape[0]/4]  = 2


    img = gtImg + numpy.random.random(shape)*float(noise)

    if i == 1000 :
        vigra.imshow(img)
        vigra.show()
    imgs.append(img.astype('float32'))
    gts.append(gtImg)







def getSelf(img):
    return img


fUnary = [
    getSelf,
    partial(vigra.filters.gaussianSmoothing, sigma=1.0),
    partial(vigra.filters.gaussianSmoothing, sigma=1.5),
    partial(vigra.filters.gaussianSmoothing, sigma=2.0),
    partial(vigra.filters.gaussianSmoothing, sigma=3.0)
]

fBinary = [
    partial(vigra.filters.gaussianGradientMagnitude, sigma=1.0),
    partial(vigra.filters.gaussianGradientMagnitude, sigma=1.5),
    partial(vigra.filters.gaussianGradientMagnitude, sigma=2.0),
    partial(vigra.filters.gaussianGradientMagnitude, sigma=3.0),
]


dataset,test_set = secondOrderImageDataset(imgs=imgs, gts=gts, numberOfLabels=3, 
                                          fUnary=fUnary, fBinary=fBinary, 
                                          addConstFeature=False)




learner =  learning.subgradientSSVM(dataset, learningRate=0.05, C=100, 
                                    learningMode='batch',maxIterations=10000)

#learner = learning.structMaxMarginLearner(dataset, 0.1, 0.001, 0)


learner.learn(infCls=opengm.inference.TrwsExternal, 
              redInf=True,
              parameter=opengm.InfParam())



# predict on test test
for (rgbImg, gtImg, gm) in test_set :
    # infer for test image
    inf = opengm.inference.TrwsExternal(gm)
    inf.infer()
    arg = inf.arg()
    arg = arg.reshape( numpy.squeeze(gtImg.shape))

    vigra.imshow(arg+2)
    vigra.show()

