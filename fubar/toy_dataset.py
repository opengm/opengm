import numpy
import opengm
from opengm import learning
import vigra
from progressbar import *
import glob
import os
from functools import partial
from opengm.learning import secondOrderImageDataset, getPbar

numpy.random.seed(42)

nImages = 8 
shape = [30, 30]
noise = 2.0
imgs = []
gts = []


for i in range(nImages):

    gtImg = numpy.zeros(shape)
    gtImg[0:shape[0]/2,:] = 1

    gtImg[shape[0]/4: 3*shape[0]/4, shape[0]/4: 3*shape[0]/4]  = 2

    ra = numpy.random.randint(180)
    #print ra 

    gtImg = vigra.sampling.rotateImageDegree(gtImg.astype(numpy.float32),int(ra),splineOrder=0)

    if i<2 :
        vigra.imshow(gtImg)
        vigra.show()

    img = gtImg + numpy.random.random(shape)*float(noise)
    if i<2 :
        vigra.imshow(img)
        vigra.show()

    imgs.append(img.astype('float32'))
    gts.append(gtImg)







def getSelf(img):
    return img


def getSpecial(img, sigma):
    simg = vigra.filters.gaussianSmoothing(img, sigma=sigma)

    img0  = simg**2
    img1  = (simg - 1.0)**2
    img2  = (simg - 2.0)**2

    img0=img0[:,:,None]
    img1=img1[:,:,None]
    img2=img2[:,:,None]


    return numpy.concatenate([img0,img1,img2],axis=2)


fUnary = [
    partial(getSpecial, sigma=0.5),
    partial(getSpecial, sigma=1.0)
]

fBinary = [
    partial(vigra.filters.gaussianGradientMagnitude, sigma=0.5),
    partial(vigra.filters.gaussianGradientMagnitude, sigma=1.0),
    partial(vigra.filters.gaussianGradientMagnitude, sigma=1.5),
    partial(vigra.filters.gaussianGradientMagnitude, sigma=2.0),
    partial(vigra.filters.gaussianGradientMagnitude, sigma=3.0),
]


dataset,test_set = secondOrderImageDataset(imgs=imgs, gts=gts, numberOfLabels=3, 
                                          fUnary=fUnary, fBinary=fBinary, 
                                          addConstFeature=True)







learningModi = ['normal','reducedinference','selfFusion','reducedinferenceSelfFusion']
lm = 0


infCls = opengm.inference.TrwsExternal
param = opengm.InfParam()

if False:
    print "construct learner"
    learner = learning.maxLikelihoodLearner(dataset)
    print "start to learn"
    learner.learn()
    print "exit"

else:
   learner =  learning.subgradientSSVM(dataset, learningRate=0.5, C=100, learningMode='batch',maxIterations=500,averaging=-1,nConf=0)
   learner.learn(infCls=infCls,parameter=param,connectedComponents=True,infMode='n')


# predict on test test
for (rgbImg, gtImg, gm) in test_set :
    # infer for test image
    inf = opengm.inference.TrwsExternal(gm)
    inf.infer()
    arg = inf.arg()
    arg = arg.reshape( numpy.squeeze(gtImg.shape))

    vigra.imshow(rgbImg)
    vigra.show()

    vigra.imshow(arg+2)
    vigra.show()

