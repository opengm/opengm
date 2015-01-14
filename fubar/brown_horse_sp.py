import numpy
import opengm
from opengm import learning
import vigra
from progressbar import *
import glob
import os
from functools import partial
from opengm.learning import secondOrderImageDataset, getPbar,superpixelDataset



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
takeNth = 1

imgs = []
sps = []
gts = []


pbar = getPbar(len(imgFiles), 'Load Image')
pbar.start()
for i,path in enumerate(imgFiles):
    gtPath =  gtBasePath + os.path.basename(path)
    rgbImg  = vigra.impex.readImage(path)
    gtImg  = vigra.impex.readImage(gtPath).astype('uint32')[::takeNth,::takeNth]
    gtImg[gtImg<125] = 0
    gtImg[gtImg>=125] = 1
    rgbImg = vigra.resize(rgbImg, [gtImg.shape[0],gtImg.shape[1]])
    

    #vigra.imshow(gtImg.astype('float32'))
    #vigra.show()

    labImg = vigra.colors.transform_RGB2Lab(rgbImg.astype('float32'))
    sp,nSeg  = vigra.analysis.slicSuperpixels(labImg, intensityScaling=20.0, seedDistance=5)
    sp = vigra.analysis.labelImage(sp)-1

    #vigra.segShow(rgbImg, sp)
    #vigra.show()
    gg  = vigra.graphs.gridGraph(rgbImg.shape[0:2])
    rag = vigra.graphs.regionAdjacencyGraph(gg,sp)

    gt,qtq = rag.projectBaseGraphGt(gtImg)

    #rag.show(rgbImg, gt)
    #vigra.show()


    imgs.append(rgbImg)
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

fUnary = [
    posiFeatures,
    getSelf,
    vigra.colors.transform_RGB2XYZ,
    vigra.colors.transform_RGB2Lab,
    vigra.colors.transform_RGB2Luv,
    partial(labHessianOfGaussian, sigma=1.0),
    partial(labHessianOfGaussian, sigma=2.0),
    partial(vigra.filters.gaussianGradientMagnitude, sigma=1.0),
    partial(vigra.filters.gaussianGradientMagnitude, sigma=2.0),
]

fBinary = [
    posiFeatures,
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


dataset,test_set = superpixelDataset(imgs=imgs,sps=sps, gts=gts, numberOfLabels=2, 
                                          fUnary=fUnary, fBinary=fBinary, 
                                          addConstFeature=True)





learner =  learning.subgradientSSVM(dataset, learningRate=0.1, C=100, 
                                    learningMode='batch',maxIterations=1000, averaging=-1)


#learner = learning.structMaxMarginLearner(dataset, 0.1, 0.001, 0)


learner.learn(infCls=opengm.inference.QpboExternal, 
              parameter=opengm.InfParam())



# predict on test test
for (rgbImg, sp, gm) in test_set :
    # infer for test image
    inf = opengm.inference.QpboExternal(gm)
    inf.infer()
    arg = inf.arg()+1

    arg  = numpy.concatenate([[0],arg])

    gg  = vigra.graphs.gridGraph(rgbImg.shape[0:2])
    rag = vigra.graphs.regionAdjacencyGraph(gg,sp)

    seg = rag.projectLabelsToBaseGraph(arg.astype('uint32'))

    vigra.segShow(rgbImg, seg+2)
    vigra.show()

