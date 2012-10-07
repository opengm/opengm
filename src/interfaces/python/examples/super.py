import opengm,numpy,vigra,scipy,scipy.cluster
import pylab
import matplotlib.cm as cm
import Image
import sys
import os
import glob



def doClustering(features,k,steps,thresh=1e-05):
   codebook,distortion=scipy.cluster.vq.kmeans( obs=features, k_or_guess=k, iter=20,  thresh=thresh)
   code ,dist = scipy.cluster.vq.vq(features,codebook)
   dists=numpy.ones([features.shape[0],k],dtype=numpy.float32)
   for i in range(k):
      #print codebook.shape
      codebook_k=codebook[i,:].reshape(1,codebook.shape[1])
      code_k ,dist_k = scipy.cluster.vq.vq(features,codebook_k)
      dists[:,i]=dist_k[:]
   return (code,dists)   
    
        
def buildGM(rag,dataImage,numLabels,boundaryPixels,regionPixels,beta,sigma,verbose=False):
   
   print "get region clustering"   
   regionFeatures=numpy.ones([rag.numberOfRegions(),3],dtype=numpy.float64)
   for r in range(rag.numberOfRegions()):
      for c in range(3):
         regionFeatures[r,c]=numpy.mean(img[regionPixels[r][:,0],regionPixels[r][:,1]][c])
         
   print "do clustering"   
   code,dists=doClustering(regionFeatures,k=numLabels,steps=100)
   dists=(dists-dists.min())/(dists.max()-dists.min())  
   print dists      
   
   if verbose==True : print "get boundary evidence" 
   boundaryEvidence=numpy.ones(rag.numberOfBoundaries(),dtype=numpy.float64)
   be=numpy.ones([rag.numberOfBoundaries(),2],dtype=numpy.float64)
   energy=numpy.ones([rag.numberOfBoundaries(),2],dtype=numpy.float64)
   for b in range(rag.numberOfBoundaries()):
      boundaryEvidence[b]=numpy.mean(dataImage[boundaryPixels[b][:,0],boundaryPixels[b][:,1]])
      r=rag.adjacentRegions(b)

   boundaryEvidence=(boundaryEvidence-boundaryEvidence.min())/(boundaryEvidence.max()-boundaryEvidence.min())*(1.0-2.0*epsilon) + epsilon
   be[:,1]=numpy.exp(-1.0*boundaryEvidence[:]*sigma)
   be[:,0]=1.0-be[:,1]
   energy [:,0]= (-1.0*numpy.log( (1)*(1.0-beta)  ) ) +be[:,0]
   energy [:,1]= (-1.0*numpy.log( (1)*(beta)  ) )    +be[:,1]
   if verbose==True : print "build gm" 
   gm=opengm.graphicalModel(numpy.ones(rag.numberOfRegions(),dtype=numpy.uint64)*numLabels)
   shapePotts=[numLabels,numLabels]
   
   
   print "add unaries"
   for r in range(rag.numberOfRegions()):
      f=dists[r,:]*gamma
      vis=[r]
      gm.addFactor(gm.addFunction(f),vis)
   print "add 2.order"
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
inprefix= '/home/tbeier/Desktop/BSDS300/images/train/nice'
outprefix= '/home/tbeier/Desktop/output/'
filetype="*.jpg"

numLabels=5
beta=0.55
gamma=0.1
sigma=2
epsilon=0.001
sigmaSeed=1.5
sigmaEnergy=2
smoothSeedScale=4
resize=2

for filename in glob.glob( os.path.join(inprefix, filetype) ):
   print "current file is: " + filename
   #filename=filenames[files]
   imgSmall=vigra.readImage(filename)
   #resize image
   
   shape2=(imgSmall.shape[0]*resize,imgSmall.shape[1]*resize)
   img=vigra.sampling.resize(imgSmall,shape2)
   
   basename=os.path.basename(filename)


   
   gaussMag=vigra.filters.gaussianGradientMagnitude(img,sigma=sigmaEnergy)
   gaussMag=(gaussMag-gaussMag.min())/(gaussMag.max()-gaussMag.min())
   
   
   seedMapTemp=vigra.filters.gaussianGradientMagnitude(img,sigma=sigmaSeed)
   seedMapTemp=seedMapTemp**6
   seedMap = vigra.gaussianSmoothing(seedMapTemp, smoothSeedScale)
   seg,numberOfRegions=vigra.analysis.watersheds(seedMap)
   seg=seg-1
   seg=numpy.squeeze(seg.astype(numpy.uint64))
   rag=opengm.RegionGraph(seg,numberOfRegions,False)

   boundaryPixels=[]
   for b in range(rag.numberOfBoundaries()):
      boundaryPixels.append(rag.boundaryPixels(b))
   
   print "get regions pixels"   
   regionPixels=[ ]
   regionFeatures=numpy.ones([rag.numberOfRegions(),3],dtype=numpy.float64)
   for r in range(rag.numberOfRegions()):
      regionPixels.append(rag.regionPixels(r))

   

   
   print "build model"
   gm=buildGM(rag=rag,dataImage=gaussMag,numLabels=numLabels,boundaryPixels=boundaryPixels,regionPixels=regionPixels,beta=beta,sigma=sigma,verbose=False)
   
   print "start inference"
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
   #arg=inferAndWriteResult(printNth=1000,alg=alg,param=param,img=img,rag=rag,prefix=outprefix,filename=basename,boundaryPixels=boundaryPixels,verbose=verboseInf,startingPoint=arg)
   
   
   alg='ae'
   param=opengm.inferenceParameter(gm=gm,alg=alg)
   param.set(steps=500000)
   #arg=inferAndWriteResult(printNth=1000,alg=alg,param=param,img=img,rag=rag,prefix=outprefix,filename=basename,boundaryPixels=boundaryPixels,verbose=verboseInf,startingPoint=arg)
   
   alg='lf'
   param=opengm.inferenceParameter(gm=gm,alg=alg)
   param.set(maxSubgraphSize=3)
   arg=inferAndWriteResult(printNth=1000,alg=alg,param=param,img=img,rag=rag,prefix=outprefix,filename=basename,boundaryPixels=boundaryPixels,verbose=verboseInf,startingPoint=arg)

