import numpy
import opengm
from opengm import learning
import vigra
from progressbar import *
import glob
import os
from functools import partial
from opengm.learning import secondOrderImageDataset, getPbar,superpixelDataset




nImages = 20 
shape = [100, 100]
noise = 8
imgs = []
gts = []
sps = []



pbar = getPbar((nImages), 'Load Image')
pbar.start()

for i in range(nImages):

    gtImg = numpy.zeros(shape)
    gtImg[0:shape[0]/2,:] = 1

    gtImg[shape[0]/4: 3*shape[0]/4, shape[0]/4: 3*shape[0]/4]  = 2

    ra = numpy.random.randint(180)
    #print ra 

    gtImg = vigra.sampling.rotateImageDegree(gtImg.astype(numpy.float32),int(ra),splineOrder=0)

    if i<1 :
        vigra.imshow(gtImg)
        vigra.show()

    img = gtImg + numpy.random.random(shape)*float(noise)
    if i<1 :
        vigra.imshow(img)
        vigra.show()



    sp,nSeg  = vigra.analysis.slicSuperpixels(gtImg, intensityScaling=0.2, seedDistance=5)
    sp = vigra.analysis.labelImage(sp)-1


    if i<1:
        vigra.segShow(img, sp+1,edgeColor=(1,0,0))
        vigra.show()


    gg  = vigra.graphs.gridGraph(gtImg.shape[0:2])
    rag = vigra.graphs.regionAdjacencyGraph(gg,sp)

    gt,qtq = rag.projectBaseGraphGt(gtImg)

    #rag.show(img, gt)
    #vigra.show()


    imgs.append(img.astype('float32'))
    gts.append(gt)
    sps.append(sp)

    pbar.update(i)


pbar.finish()

def getSelf(img):
    return img


def labHessianOfGaussian(img, sigma):
    l = vigra.colors.transform_RGB2Lab(img)[:,:,0]
    l = vigra.taggedView(l,'xy')
    return vigra.filters.hessianOfGaussianEigenvalues(l, sigma)

def labStructTensorEv(img, sigma):
    l = vigra.colors.transform_RGB2Lab(img)[:,:,0]
    l = vigra.taggedView(l,'xy')
    return vigra.filters.structureTensorEigenvalues(l, sigma, 2*sigma)


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


dataset,test_set = superpixelDataset(imgs=imgs,sps=sps, gts=gts, numberOfLabels=3, 
                                          fUnary=fUnary, fBinary=fBinary, 
                                          addConstFeature=True)
if True :
    dataset.save("simple_dataset", 'simple_')
if True :
    dataset = learning.createDataset(0,  numInstances=0)
    dataset.load("simple_dataset", 'simple_')
if True:

    learner =  learning.subgradientSSVM(dataset, learningRate=0.1, C=100, 
                                        learningMode='batch',maxIterations=1000, averaging=-1)
    learner.learn(infCls=opengm.inference.TrwsExternal, 
                  parameter=opengm.InfParam())

else:
    learner = learning.maxLikelihoodLearner(dataset, temp=0.0000001)
    learner.learn()
# predict on test test
for (rgbImg, sp, gm) in test_set :
    # infer for test image
    inf = opengm.inference.TrwsExternal(gm)
    inf.infer()
    arg = inf.arg()+1



    assert sp.min() == 0
    assert sp.max() == arg.shape[0] -1

    gg  = vigra.graphs.gridGraph(rgbImg.shape[0:2])
    rag = vigra.graphs.regionAdjacencyGraph(gg,sp)

    seg = rag.projectLabelsToBaseGraph(arg.astype('uint32'))

    vigra.imshow(rgbImg)
    vigra.show()


    vigra.imshow(seg)
    vigra.show()

