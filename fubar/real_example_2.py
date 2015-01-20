import opengm
import opengm.learning as learning
from opengm import numpy
import vigra
import pylab as plt
import pylab

#nModels = 20
nModels = 2
nLables = 2 
#shape = [50, 50]
shape = [16, 16]
numVar = shape[0]*shape[1]

sSmooth = [1.0,1.2, 1.5, 2.0, 3.0, 4.0]
sGrad = [1.0, 1.5, 2.0, 4.0]

nUWeights = len(sSmooth) + 1
nBWeights = len(sGrad) + 1
nWeights = nUWeights + nBWeights

def makeGt(shape):
    gt=numpy.ones(shape,dtype='uint8')
    gt[0:shape[0]/2,:] = 0
    return gt
 


weightVals = numpy.ones(nWeights)
weights = opengm.learning.Weights(weightVals)

uWeightIds = numpy.arange(nUWeights ,dtype='uint64')
bWeightIds = numpy.arange(start=nUWeights,stop=nWeights,dtype='uint64')


dataset = learning.createDataset(numWeights=nWeights)
weights = dataset.getWeights()

def makeFeatures(gt):
    random  = (numpy.random.rand(*gt.shape)-0.5)*5.0
    randGt = random + gt

    # vigra.imshow(randGt)
    # plt.colorbar()
    # vigra.show()

    #f = pylab.figure()
    #for n, a in enumerate([gt, randGt]):
    #    f.add_subplot(2, 1, n)  # this line outputs images on top of each other
    #    # f.add_subplot(1, 2, n)  # this line outputs images side-by-side
    #    pylab.imshow(a,cmap='gray')
    #pylab.title('Double image')
    #pylab.show()



    feat = []
    for sigma in sSmooth:
        feat.append(vigra.filters.gaussianSmoothing(randGt.astype('float32'),sigma) )

        #vigra.imshow(feat[-1])
        #plt.colorbar()
        #vigra.show()


    featB = []
    for sigma in sGrad:
        featB.append(vigra.filters.gaussianGradientMagnitude(randGt.astype('float32'),sigma) )

    a=None
    b=None
    if len(feat)>0:    
        a =  numpy.rollaxis(numpy.array(feat), axis=0, start=3)
    if len(featB)>0:
        b =  numpy.rollaxis(numpy.array(featB), axis=0, start=3)
    return a,b

for mi in range(nModels):
    #print mi

    gm = opengm.gm(numpy.ones(numVar)*nLables)
    gt = makeGt(shape) 
    gtFlat = gt.reshape([-1])

    unaries, binaries = makeFeatures(gt)

    # print unaries, binaries


    for x in range(shape[0]):
        for y in range(shape[1]):
            uFeat = numpy.append(unaries[x,y,:],[1])

            #print uFeat
            #print uWeightIds
            #print(unaries[x,y,:])
            #print(unaries.shape)
            #print(uFeat)
            #print(uFeat.shape)

            lu = learning.lUnaryFunction(weights=weights,numberOfLabels=nLables, 
                                         features=uFeat, weightIds=uWeightIds)
            fid = gm.addFunction(lu)
            facIndex = gm.addFactor(fid, y+x*shape[1])
            #facIndex = gm.addFactor(fid, x+y*shape[0])

    if True:
        for x in range(shape[0]):
            for y in range(shape[1]):

                if x+1 < shape[0]:
                    bFeat = numpy.append(binaries[x,y,:], [1])+numpy.append(binaries[x+1,y,:], [1])
                    pf = opengm.LPottsFunction(weights=weights,numberOfLabels=nLables, features=bFeat, weightIds=bWeightIds)
                    fid= gm.addFunction(pf)
                    gm.addFactor(fid, [y+x*shape[1], y+(x+1)*shape[1]])
                if y+1 < shape[1]:
                    bFeat = numpy.append(binaries[x,y,:], [1]).astype(opengm.value_type) + numpy.append(binaries[x,y+1,:], [1]).astype(opengm.value_type)
                    pf = opengm.LPottsFunction(weights=weights,numberOfLabels=nLables, features=bFeat, weightIds=bWeightIds)
                    fid= gm.addFunction(pf)
                    gm.addFactor(fid, [y+x*shape[1], y+1+x*shape[1]])

    dataset.pushBackInstance(gm,gtFlat.astype(opengm.label_type))
    backGt = dataset.getGT(0)

    #print "back",backGt
    #sys.exit()

# for grid search learner
lowerBounds = numpy.ones(nWeights)*-2.0
upperBounds = numpy.ones(nWeights)*2.0
nTestPoints  =numpy.ones(nWeights).astype('uint64')*5

#learner = learning.gridSearchLearner(dataset=dataset,lowerBounds=lowerBounds, upperBounds=upperBounds,nTestPoints=nTestPoints)
#learner =  learning.structPerceptron(dataset, decayExponent=-0.5, learningMode='batch')
#learner = learning.structMaxMarginLearner(dataset, 0.1, 0.001, 0)
#learner =  learning.subgradientSSVM(dataset, learningRate=1.0, C=100, learningMode='batch')


#learner.learn(infCls=opengm.inference.TrwsExternal,
#              parameter=opengm.InfParam())


learner = learning.maxLikelihoodLearner(
    dataset,
    maximumNumberOfIterations =1000,gradientStepSize=0.9,weightStoppingCriteria=0.001,gradientStoppingCriteria=0.01,infoFlag=True,infoEveryStep=False,weightRegularizer=1.0,
    beliefPropagationMaximumNumberOfIterations=5,beliefPropagationConvergenceBound=0.0001,beliefPropagationDamping=0.5,beliefPropagationTemperature=0.3,beliefPropagationIsAcyclic=opengm.Tribool(False))
learner.learn()

for w in range(nWeights):
    print weights[w]
