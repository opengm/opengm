# FIXMEEEEEEEEEEE

import opengm
import vigra    # only to read images 
import numpy 
#import sys

# to animate the current labeling matplotlib is used
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import animation


class PyCallback(object):
    """
    callback functor which will be passed to an inference
    visitor.
    In that way, pure python code can be injected into the c++ inference.
    This functor visualizes the labeling as an image during inference.

    Args :
        shape : shape of the image 
        numLabels : number of labels
    """
    def __init__(self,shape,numLabels):
        self.shape=shape
        self.numLabels=numLabels
        matplotlib.interactive(True)
    def begin(self,inference):
        """
        this function is called from c++ when inference is started

        Args : 
            inference : python wrapped c++ solver which is passed from c++
        """
        print "begin"
    def end(self,inference):
        """
        this function is called from c++ when inference ends

        Args : 
            inference : python wrapped c++ solver which is passed from c++
        """
        print "end"
    def visit(self,inference):
        """
        this function is called from c++ each time the visitor is called

        Args : 
            inference : python wrapped c++ solver which is passed from c++
        """
        
        arg = inference.arg()
        gm  = inference.gm()
        print "energy ",gm.evaluate(arg)

        arg=arg.reshape(self.shape)*255
        plt.imshow(arg.T, cmap='gray',interpolation="nearest") 
        plt.draw()
        

def denoiseModel(
    img,
    norm                    = 2,
    weight                  = 1.0,
    truncate                = None,
    numLabels               = 256,
    neighbourhood           = 4,
    inpaintPixels           = None,
    randInpaitStartingPoint = False
):
    """
    this function is used to set up a graphical model similar to 
    **Denoising and inpainting problems:** from `Mrf- Benchmark <http://vision.middlebury.edu/MRF/results/ >`_
    
    Args : 
        img           : a grayscale image in the range [0,256)
        norm          : used norm for unaries and 2-order functions (default : 2)
        weight        : weight of 2-order functions (default : 1.0)
        truncate      : Truncate second order function at an given value (defaut : None)
        numLabels     : number of labels for each variable in the graphical model, 
                        set this to a lower number to speed up inference  (default : 255)
        neighbourhood : neighbourhood for the second order functions, so far only 4 is allowed (default : 4)
        inpaintPixels : a tuple of x and y coordinates where no unaries are added
        randInpaitStartingPoint : use a random starting point for all pixels without unaries (default : False)
    """
    shape = img.shape
    if(img.ndim!=2):
        raise RuntimeError("image must be gray")
    if neighbourhood != 4 :
        raise RuntimeError("A neighbourhood other than 4 is not yet implemented")

    # normalize and flatten image
    iMin    = numpy.min(img)
    iMax    = numpy.max(img)
    imgNorm = ((img[:,:]-iMin)/(iMax-iMin))*float(numLabels)
    imgFlat = imgNorm.reshape(-1).astype(numpy.uint64)

    # Set up Grapical Model:
    numVar = int(img.size)
    gm = opengm.gm([numLabels]*numVar,operator='adder')
    gm.reserveFunctions(numLabels,'explicit')
    numberOfPairwiseFactors=shape[0]*(shape[1]-1) + shape[1]*(shape[0]-1)
    gm.reserveFactors(numVar-len(inpaintPixels[0]) + numberOfPairwiseFactors )

    # Set up unaries:
    # - create a range of all possible labels
    allPossiblePixelValues=numpy.arange(numLabels)
    pixelValueRep    = numpy.repeat(allPossiblePixelValues[:,numpy.newaxis],numLabels,1)
    # - repeat [0,1,2,3,...,253,254,255] numVar times
    labelRange = numpy.arange(numLabels,dtype=opengm.value_type)
    labelRange = numpy.repeat(labelRange[numpy.newaxis,:], numLabels, 0)
    unaries = numpy.abs(pixelValueRep - labelRange)**norm
    # - add unaries to the graphical model
    fids=gm.addFunctions(unaries.astype(opengm.value_type))
    # add unary factors to graphical model
    if(inpaintPixels is None):
        for l in xrange(numLabels):
            whereL=numpy.where(imgFlat==l)
            gm.addFactors(fids[l],whereL[0].astype(opengm.index_type))
    else:
        # get vis of inpaint pixels
        ipX  = inpaintPixels[0]
        ipY  = inpaintPixels[1]
        ipVi = ipX*shape[1] + ipY

        for l in xrange(numLabels):
            whereL=numpy.where(imgFlat==l)
            notInInpaint=numpy.setdiff1d(whereL[0],ipVi)
            gm.addFactors(fids[l],notInInpaint.astype(opengm.index_type))

    # add ONE second order function
    f=opengm.differenceFunction(shape=[numLabels,numLabels],norm=2,weight=weight)
    fid=gm.addFunction(f)
    vis2Order=opengm.secondOrderGridVis(shape[0],shape[1],True)
    # add all second order factors
    gm.addFactors(fid,vis2Order)

    # create a starting point
    startingPoint = imgFlat.copy()
    if randInpaitStartingPoint :
        startingPointRandom = numpy.random.randint(0,numLabels,size=numVar).astype(opengm.index_type)

        ipVi = inpaintPixels[0]*shape[1] + inpaintPixels[1]
        for x in ipVi:
            startingPoint[x]=startingPointRandom[x]

    startingPoint[startingPoint==numLabels]=numLabels-1            
    return gm,startingPoint.astype(opengm.index_type)


if __name__ == "__main__":

    # setup
    imgPath   = 'houseM-input.png'
    norm      = 2
    weight    = 5.0
    numLabels = 50   # use 256 for full-model (slow)

    # Read image
    img   = numpy.array(numpy.squeeze(vigra.impex.readImage(imgPath)),dtype=opengm.value_type)#[0:100,0:40]
    shape = img.shape

    # get graphical model an starting point 
    gm,startingPoint=denoiseModel(img,norm=norm,weight=weight,inpaintPixels=numpy.where(img==0),
                                  numLabels=numLabels,randInpaitStartingPoint=True)

    inf=opengm.inference.BeliefPropagation(gm,parameter=opengm.InfParam())


    print "inf"
    inf.setStartingPoint(inf.arg())
        # set up visitor
    callback=PyCallback(shape,numLabels)
    visitor=inf.pythonVisitor(callback,visitNth=1)
    inf.infer(visitor) 
    # get the result
    arg=inf.arg()
    arg=arg.reshape(shape)

    # plot final result
    matplotlib.interactive(False)
    # Two subplots, the axes array is 1-d
    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(img.T, cmap = cm.Greys_r)
    axarr[0].set_title('Input Image')
    axarr[1].imshow(arg.T, cmap = cm.Greys_r)
    axarr[1].set_title('Solution')
    plt.show()