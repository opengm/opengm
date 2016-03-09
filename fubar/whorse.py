import numpy
import opengm
from opengm import learning
import vigra
from progressbar import *

from functools import partial
from make_grid_potts_dset import secondOrderImageDataset

# where is the dataset stored
dsetRoot = '/home/tbeier/datasets/weizmann_horse_db/'
imgPath = dsetRoot + 'rgb/'
gtPath = dsetRoot + 'figure_ground/'
    
# how many should be loaded
# (all if None)
loadN = 20
takeNth  = 3
if loadN is None:
    loadN = 0

imgs = []
gt = []

for i in range(1,loadN+1):

    hName = "horse%03d.jpg" % (i,)
    rgbImg  = vigra.impex.readImage(imgPath+hName)
    gtImg  = vigra.impex.readImage(gtPath+hName).astype('uint32')[::takeNth,::takeNth]
    gtImg[gtImg<125] = 0
    gtImg[gtImg>=125] = 1
    rgbImg = vigra.resize(rgbImg, [gtImg.shape[0],gtImg.shape[1]])
    imgs.append(rgbImg)
    gt.append(gtImg)


fUnary = [
    vigra.colors.transform_RGB2Lab,
    vigra.colors.transform_RGB2Luv,
    partial(vigra.filters.gaussianGradientMagnitude, sigma=1.0),
    partial(vigra.filters.gaussianGradientMagnitude, sigma=2.0),
]

fBinary = [
    vigra.colors.transform_RGB2Lab,
    vigra.colors.transform_RGB2Luv,
    partial(vigra.filters.gaussianGradientMagnitude, sigma=1.0),
    partial(vigra.filters.gaussianGradientMagnitude, sigma=2.0),
]


dataset,test_set = secondOrderImageDataset(imgs=imgs, gts=gt, numberOfLabels=2, 
                                          fUnary=fUnary, fBinary=fBinary, 
                                          addConstFeature=False)




learner =  learning.subgradientSSVM(dataset, learningRate=0.1, C=100, 
                                    learningMode='batch',maxIterations=10)



learner.learn(infCls=opengm.inference.QpboExternal, 
              parameter=opengm.InfParam())



# predict on test test
for (rgbImg, gtImg, gm) in test_set :
    # infer for test image
    inf = opengm.inference.QpboExternal(gm)
    inf.infer()
    arg = inf.arg()
    arg = arg.reshape( numpy.squeeze(gtImg.shape))

    vigra.segShow(rgbImg, arg+2)
    vigra.show()

