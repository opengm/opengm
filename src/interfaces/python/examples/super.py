import opengm,numpy,vigra
import pylab
import matplotlib.cm as cm
import Image
import sys
import os
import glob
     
    
        
def buildGM(rag,dataImage,numLabels,boundaryPixels,verbose=False):
   if verbose==True : print "get boundary evidence" 
   boundaryEvidence=numpy.ones(rag.numberOfBoundaries(),dtype=numpy.float64)
   energy=numpy.ones([rag.numberOfBoundaries(),2],dtype=numpy.float64)
   for b in range(rag.numberOfBoundaries()):
      boundaryEvidence[b]=numpy.mean(dataImage[boundaryPixels[b][:,0],boundaryPixels[b][:,1]])
      r=rag.adjacentRegions(b)

   boundaryEvidence=(boundaryEvidence-boundaryEvidence.min())/(boundaryEvidence.max()-boundaryEvidence.min())*(1.0-2.0*epsilon) + epsilon
   energy [:,0]= (-1.0*numpy.log( (1)*(1.0-beta)  ) ) +boundaryEvidence[:]
   energy [:,1]= (-1.0*numpy.log( (1)*(beta)  ) )    +(1.0-boundaryEvidence[:])
   if verbose==True : print "build gm" 
   gm=opengm.graphicalModel(numpy.ones(rag.numberOfRegions(),dtype=numpy.uint64)*numLabels)
   shapePotts=[numLabels,numLabels]
   for b in range(rag.numberOfBoundaries()):
      f=opengm.pottsFunction(shapePotts, energy[b,0] ,energy[b,1])
      vis=rag.adjacentRegions(b)
      gm.addFactor(gm.addFunction(f),vis)
   return gm
   
def writeResult(img,arg,rag,filename,boundaryPixels):
   imRes=numpy.ones(img.shape)
   imRes[:,:,:]=img[:,:,:]
   for b in range(rag.numberOfBoundaries()):
      vis=rag.adjacentRegions(b)
      if arg[vis[0]]!=arg[vis[1]]:
         imRes[boundaryPixels[b][:,0],boundaryPixels[b][:,1]]=[0.8*255.0,0.0,0.0] + 0.2*imRes[boundaryPixels[b][:,0],boundaryPixels[b][:,1]].astype(numpy.float)
      else:
         imRes[boundaryPixels[b][:,0],boundaryPixels[b][:,1]]=0.7*imRes[boundaryPixels[b][:,0],boundaryPixels[b][:,1]]   
   vigra.impex.writeImage(imRes,filename)   

def infer(gm,alg,verbose=True,multiline=False,printNth=1,param=None,startingPoint=None):
   inf=opengm.inferenceAlgorithm(gm=gm,alg=alg,parameter=param)
   if startingPoint!=None:
      inf.setStartingPoint(startingPoint)
   inf.infer(verbose=verbose,multiline=multiline,printNth=printNth)
   arg=inf.arg()
   print "Energy ",alg," = ",inf.graphicalModel().evaluate(arg)
   return arg

def inferAndWriteResult(img,rag,prefix,filename,boundaryPixels,alg,param,verbose=False,multiline=False,printNth=1,startingPoint=None):
   arg=infer(gm=gm,alg=alg,param=param,verbose=verbose,multiline=multiline,printNth=printNth,startingPoint=startingPoint)
   name=os.path.splitext(filename)
   filename=prefix+name[0]+"-"+alg+".png"
   writeResult(img=img,arg=arg,rag=rag,filename=filename,boundaryPixels=boundaryPixels)
   return arg

verbose=True
verboseInf=False   
inprefix= '/home/tbeier/Desktop/BSDS300/images/train'
outprefix= '/home/tbeier/Desktop/output/'
filetype="*.jpg"


for filename in glob.glob( os.path.join(inprefix, filetype) ):
   print "current file is: " + filename
   #filename=filenames[files]
   img=vigra.readImage(filename)
   basename=os.path.basename(filename)
   numLabels=5
   beta=0.55
   epsilon=0.001
   sigmaSeed=3
   sigmaEnergy=2

   seedMap=vigra.filters.gaussianGradientMagnitude(img,sigma=sigmaSeed)
   gaussMag=vigra.filters.gaussianGradientMagnitude(img,sigma=sigmaEnergy)
   gaussMag=(gaussMag-gaussMag.min())/(gaussMag.max()-gaussMag.min())
   seg,numberOfRegions=vigra.analysis.watersheds(seedMap)
   seg=seg-1
   seg=numpy.squeeze(seg.astype(numpy.uint64))
   rag=opengm.RegionGraph(seg,numberOfRegions,False)

   boundaryPixels=[]
   for b in range(rag.numberOfBoundaries()):
      bPixel=rag.boundaryPixels(b)
      boundaryPixels.append(rag.boundaryPixels(b))

   gm=buildGM(rag=rag,dataImage=gaussMag,numLabels=numLabels,boundaryPixels=boundaryPixels,verbose=False)
   
   alg='trbp'
   param=opengm.inferenceParameter(gm=gm,alg=alg)
   param.set(steps=6,damping=0.1)
   arg=inferAndWriteResult(printNth=1,alg=alg,param=param,img=img,rag=rag,prefix=outprefix,filename=basename,boundaryPixels=boundaryPixels,verbose=verboseInf)
   
   alg='bp'
   param=opengm.inferenceParameter(gm=gm,alg=alg)
   param.set(steps=6,damping=0.5)
   arg=inferAndWriteResult(printNth=1,alg=alg,param=param,img=img,rag=rag,prefix=outprefix,filename=basename,boundaryPixels=boundaryPixels,verbose=verboseInf)
   
   alg='icm'
   param=opengm.inferenceParameter(gm=gm,alg=alg)
   param.set()
   arg=inferAndWriteResult(printNth=1000,alg=alg,param=param,img=img,rag=rag,prefix=outprefix,filename=basename,boundaryPixels=boundaryPixels,verbose=verboseInf,startingPoint=arg)
   
   alg='gibbs'
   param=opengm.inferenceParameter(gm=gm,alg=alg)
   param.set(steps=1000000,tempMin=0.0000005,tempMax=0.4,useTemp=True,periodes=6*3)
   arg=inferAndWriteResult(printNth=1000,alg=alg,param=param,img=img,rag=rag,prefix=outprefix,filename=basename,boundaryPixels=boundaryPixels,verbose=verboseInf,startingPoint=arg)
   
   alg='ab-swap'
   param=opengm.inferenceParameter(gm=gm,alg=alg)
   param.set(steps=500000)
   arg=inferAndWriteResult(printNth=1000,alg=alg,param=param,img=img,rag=rag,prefix=outprefix,filename=basename,boundaryPixels=boundaryPixels,verbose=verboseInf,startingPoint=arg)
   
   
   alg='ae'
   param=opengm.inferenceParameter(gm=gm,alg=alg)
   param.set(steps=500000)
   arg=inferAndWriteResult(printNth=1000,alg=alg,param=param,img=img,rag=rag,prefix=outprefix,filename=basename,boundaryPixels=boundaryPixels,verbose=verboseInf,startingPoint=arg)
   
   alg='lf'
   param=opengm.inferenceParameter(gm=gm,alg=alg)
   param.set(maxSubgraphSize=3)
   arg=inferAndWriteResult(printNth=1000,alg=alg,param=param,img=img,rag=rag,prefix=outprefix,filename=basename,boundaryPixels=boundaryPixels,verbose=verboseInf,startingPoint=arg)

