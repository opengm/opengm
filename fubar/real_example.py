import opengm
import opengm.learning as learning
from opengm import numpy
import vigra

nModels = 4
nLables = 2 
shape = [10, 10]
numVar = shape[0]*shape[1]
nWeights = 12

def makeGt(shape):
    gt=numpy.ones(shape,dtype='uint8')
    gt[0:shape[0]/2,:] = 0
    return gt



weightVals = numpy.ones(nWeights)
weights = opengm.learning.Weights(weightVals)

uWeightIds = numpy.arange(8,dtype='uint64').reshape(2,4)


print uWeightIds

bWeightIds = numpy.array([8,9,10,11],dtype='uint64')


dataset = learning.createDataset(loss='h')



def makeFeatures(gt):
    random  = numpy.random.rand(*gt.shape)-0.5
    randGt = random + gt
    feat = []
    for sigma in [1.0, 1.5, 2.0]:
        feat.append(vigra.filters.gaussianSmoothing(randGt.astype('float32'),sigma) )

    featB = []
    for sigma in [1.0, 1.5, 2.0]:
        featB.append(vigra.filters.gaussianGradientMagnitude(randGt.astype('float32'),sigma) )



    a =  numpy.rollaxis(numpy.array(feat), axis=0, start=3)
    b =  numpy.rollaxis(numpy.array(featB), axis=0, start=3)
    return a,b

for mi in range(nModels):


    gm = opengm.gm(numpy.ones(numVar)*nLables)
    gt = makeGt(shape) 
    gtFlat = gt.reshape([-1])

    unaries,binaries = makeFeatures(gt)

    print unaries, binaries


    for x in range(shape[0]):
        for y in range(shape[1]):
            uFeat = numpy.append(unaries[x,y,:], [1]).astype(opengm.value_type)
            uFeat = numpy.repeat(uFeat[:,numpy.newaxis],2,axis=1).T

            lu = opengm.LUnaryFunction(weights=weights,numberOfLabels=nLables, features=uFeat, weightIds=uWeightIds)


            fid= gm.addFunction(lu)
            gm.addFactor(fid, y+x*shape[1])



    for x in range(shape[0]):
        for y in range(shape[1]):

            if x+1 < shape[0]:
                bFeat = numpy.append(binaries[x,y,:], [1]).astype(opengm.value_type) +  numpy.append(binaries[x+1,y,:], [1]).astype(opengm.value_type)
                pf = opengm.LPottsFunction(weights=weights,numberOfLabels=nLables, features=bFeat, weightIds=bWeightIds)
                fid= gm.addFunction(pf)
                gm.addFactor(fid, [y+x*shape[1], y+(x+1)*shape[1]])
            if y+1 < shape[1]:
                bFeat = numpy.append(binaries[x,y,:], [1]).astype(opengm.value_type) + numpy.append(binaries[x,y+1,:], [1]).astype(opengm.value_type)
                pf = opengm.LPottsFunction(weights=weights,numberOfLabels=nLables, features=bFeat, weightIds=bWeightIds)
                fid= gm.addFunction(pf)
                gm.addFactor(fid, [y+x*shape[1], y+1+x*shape[1]])



    dataset.pushBackInstance(gm,gtFlat.astype(opengm.label_type))



# for grid search learner
lowerBounds = numpy.ones(nWeights)*-2.0
upperBounds = numpy.ones(nWeights)*2.0
nTestPoints  =numpy.ones(nWeights).astype('uint64')*10


learner = learning.gridSearchLearner(dataset=dataset,lowerBounds=lowerBounds, upperBounds=upperBounds,nTestPoints=nTestPoints)

learner.learn(infCls=opengm.inference.BeliefPropagation, 
              parameter=opengm.InfParam(damping=0.5))
