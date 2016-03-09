import numpy
import opengm
from opengm import learning
import vigra
from progressbar import *
import glob
import os
from functools import partial
from opengm.learning import secondOrderImageDataset, getPbar



def posiFeatures(img):
    shape = img.shape[0:2]
    x = numpy.linspace(0, 1, shape[0])
    y = numpy.linspace(0, 1, shape[1])
    xv, yv = numpy.meshgrid(y, x)
    xv -=0.5
    yv -=0.5

    rad = numpy.sqrt(xv**2 + yv**2)[:,:,None]
    erad = numpy.exp(1.0 - rad)
    xva = (xv**2)[:,:,None]
    yva = (yv**2)[:,:,None]

    res = numpy.concatenate([erad, rad,xva,yva,xv[:,:,None],yv[:,:,None]],axis=2)
    assert res.shape[0:2] == img.shape[0:2]
    return res

#i = numpy.ones([7, 5])
#
#print posiFeatures(i).shape
#
# where is the dataset stored
dsetRoot = '/home/tbeier/datasets/weizmann_horse_db/'
imgPath = dsetRoot + 'brown_horse/'
gtBasePath = dsetRoot + 'figure_ground/'

imgFiles = glob.glob(imgPath+'*.jpg')
takeNth = 2
imgs = []
gts = []
pbar = getPbar(len(imgFiles), 'Load Image')
pbar.start()
for i,path in enumerate(imgFiles):
    gtPath =  gtBasePath + os.path.basename(path)
    rgbImg  = vigra.impex.readImage(path)
    gtImg  = vigra.impex.readImage(gtPath).astype('uint32')[::takeNth,::takeNth]
    gtImg[gtImg<125] = 0
    gtImg[gtImg>=125] = 1
    cEdgeImg = vigra.analysis.regionImageToCrackEdgeImage(gtImg+1)
    cEdgeImg[cEdgeImg>0] = 1
    cEdgeImg = vigra.filters.discErosion(cEdgeImg.astype('uint8'),2)
    gtImg = cEdgeImg.astype(numpy.uint64)

    if i ==0:
        vigra.imshow(cEdgeImg)
        vigra.show()
    rgbImg = vigra.resize(rgbImg, [gtImg.shape[0],gtImg.shape[1]])
    imgs.append(rgbImg)
    gts.append(gtImg)
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

fUnary = [
    #posiFeatures,
    #getSelf,
    vigra.colors.transform_RGB2XYZ,
    vigra.colors.transform_RGB2Lab,
    vigra.colors.transform_RGB2Luv,
    partial(labHessianOfGaussian, sigma=1.0),
    partial(labHessianOfGaussian, sigma=2.0),
    partial(labStructTensorEv, sigma=1.0),
    partial(labStructTensorEv, sigma=2.0),
    partial(vigra.filters.gaussianGradientMagnitude, sigma=1.0),
    partial(vigra.filters.gaussianGradientMagnitude, sigma=2.0),
]

fBinary = [
    #posiFeatures,
    vigra.colors.transform_RGB2XYZ,
    vigra.colors.transform_RGB2Lab,
    vigra.colors.transform_RGB2Luv,
    partial(labHessianOfGaussian, sigma=1.0),
    partial(labHessianOfGaussian, sigma=2.0),
    partial(labStructTensorEv, sigma=1.0),
    partial(labStructTensorEv, sigma=2.0),
    partial(vigra.filters.gaussianGradientMagnitude, sigma=1.0),
    partial(vigra.filters.gaussianGradientMagnitude, sigma=2.0),
]


dataset,test_set = secondOrderImageDataset(imgs=imgs, gts=gts, numberOfLabels=2, 
                                          fUnary=fUnary, fBinary=fBinary, 
                                          addConstFeature=False)




learner =  learning.subgradientSSVM(dataset, learningRate=0.1, C=1000, 
                                    learningMode='batch',maxIterations=1000)

#learner = learning.structMaxMarginLearner(dataset, 0.1, 0.001, 0)


learner.learn(infCls=opengm.inference.QpboExternal, 
              parameter=opengm.InfParam())



# predict on test test
for (rgbImg, gtImg, gm) in test_set :
    # infer for test image
    inf = opengm.inference.QpboExternal(gm)
    inf.infer()
    arg = inf.arg()
    arg = arg.reshape( numpy.squeeze(gtImg).shape)

    vigra.imshow(arg)
    #vigra.segShow(rgbImg, arg+2)
    vigra.show()

