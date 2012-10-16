import opengm,numpy,vigra,scipy,scipy.cluster
import pylab
import matplotlib.cm as cm
import Image
import sys
import os
import glob



def doClustering(features,k,steps,thresh=1e-05):
   features=scipy.cluster.vq.whiten(features)
   codebook,distortion=scipy.cluster.vq.kmeans( obs=features, k_or_guess=k, iter=steps,  thresh=thresh)
   code ,dist = scipy.cluster.vq.vq(features,codebook)
   dists=numpy.ones([features.shape[0],k],dtype=numpy.float32)
   for i in range(k):
      #print codebook.shape
      codebook_k=codebook[i,:].reshape(1,codebook.shape[1])
      code_k ,dist_k = scipy.cluster.vq.vq(features,codebook_k)
      dists[:,i]=dist_k[:]
   return (code,dists)   
    
        
def buildGM(img,rag,dataImage,numLabels,boundaryPixels,regionPixels,beta,sigma,verbose=False):
   
   print "get region clustering"   
   regionFeatures=numpy.ones([rag.numberOfRegions(),3],dtype=numpy.float64)
   print "lab type in gm" ,type(img)
   print "lab type in gm shape" ,img.shape
   npimg=numpy.ones(img.shape)
   npimg[:,:,:]=img[:,:,:]
   print "npimg type in gm shape" ,npimg.shape
   for r in range(rag.numberOfRegions()):
      for c in range(3):
         regionFeatures[r,c]=numpy.mean(npimg[regionPixels[r][:,0],regionPixels[r][:,1]][c])
         
   print "do clustering"   
   code,dists=doClustering(regionFeatures,k=numLabels,steps=100)
   dists=(dists-dists.min())/(dists.max()-dists.min())      
   
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
   
def writeResult(img,arg,rag,filename,boundaryPixels,regionPixels):
   imRes=numpy.ones(img.shape)
   imRes[:,:,:]=img[:,:,:]
   for b in range(rag.numberOfBoundaries()):
      vis=rag.adjacentRegions(b)
      if arg[vis[0]]!=arg[vis[1]]:
         imRes[boundaryPixels[b][:,0],boundaryPixels[b][:,1]]=[0.8*255.0,0.0,0.0] + 0.2*imRes[boundaryPixels[b][:,0],boundaryPixels[b][:,1]].astype(numpy.float)
      else:
         imRes[boundaryPixels[b][:,0],boundaryPixels[b][:,1]]=0.7*imRes[boundaryPixels[b][:,0],boundaryPixels[b][:,1]]
   imgArg=numpy.ones(img.shape[0:2],dtype=numpy.uint8)
   vigra.impex.writeImage(imRes,filename)
   for r in range(rag.numberOfRegions()):
      imgArg[regionPixels[r][:,0],regionPixels[r][:,1]]=arg[r]
   imReg=vigra.analysis.labelImage(imgArg)   

   imReg=imReg-imReg.min()
   nReg2=imReg.max()+1
   rag2=opengm.RegionGraph( imReg.astype(numpy.uint64) ,int(nReg2),True)
   
   regionPixels2=[ ]
   for r in range(rag2.numberOfRegions()):
      regionPixels2.append(rag2.regionPixels(r))
   imMean=numpy.ones(img.shape)
   imMean[:,:,:]=img[:,:,:]    
   for r in range(rag2.numberOfRegions()):
      for c in range(3):
         m=numpy.mean(img[regionPixels2[r][:,0],regionPixels2[r][:,1],c])
         imMean[regionPixels2[r][:,0],regionPixels2[r][:,1],c]=m
   vigra.impex.writeImage(imMean,filename+"mean.jpg")
      

               


def writeSP(img,rag,filename,boundaryPixels):
   imRes=numpy.ones(img.shape)
   imRes[:,:,:]=img[:,:,:]
   for b in range(rag.numberOfBoundaries()):
      imRes[boundaryPixels[b][:,0],boundaryPixels[b][:,1]]=0.2*imRes[boundaryPixels[b][:,0],boundaryPixels[b][:,1]]   
   vigra.impex.writeImage(imRes,filename)   

def infer(gm,alg,verbose=True,multiline=False,printNth=1,param=None,startingPoint=None):
   inf=opengm.inferenceAlgorithm(gm=gm,alg=alg,parameter=param)
   if startingPoint!=None:
      inf.setStartingPoint(startingPoint)
   inf.infer(verbose=verbose,multiline=multiline,printNth=printNth)
   arg=inf.arg()
   print "Energy ",alg," = ",inf.graphicalModel().evaluate(arg)
   return arg

def inferAndWriteResult(img,rag,prefix,filename,boundaryPixels,regionPixels,alg,param,verbose=False,multiline=False,printNth=1,startingPoint=None):
   arg=infer(gm=gm,alg=alg,param=param,verbose=verbose,multiline=multiline,printNth=printNth,startingPoint=startingPoint)
   name=os.path.splitext(filename)
   filename=prefix+name[0]+"-"+alg+".png"
   writeResult(img=img,arg=arg,rag=rag,filename=filename,boundaryPixels=boundaryPixels,regionPixels=regionPixels)
   return arg

verbose=True
verboseInf=False   
inprefix= '/home/tbeier/Desktop/BSDS300/images/train'
outprefix= '/home/tbeier/Desktop/output2/'
filetype="*.jpg"

resize=2
numLabels=5


beta=0.5
gamma=15 #0.1
sigma=4     #2
epsilon=0.001
sigmaSeed=1.5*resize
smoothSeedScale=0.75*resize


for filename in glob.glob( os.path.join(inprefix, filetype) ):
   print "current file is: " + filename
   #filename=filenames[files]
   imgSmall=vigra.readImage(filename)
   #resize image
   
   shape2=(imgSmall.shape[0]*resize,imgSmall.shape[1]*resize)
   imgRGB=vigra.sampling.resize(imgSmall,shape2)
   
   
   
   
   if False :
      imgLAB=vigra.colors.transform_RGB2Lab(imgRGB)
      for x in range(img.shape[0]):
         for y in range(img.shape[1]):
            #print "rgb=",imgRGB[x,y,:]
            #print "lab=",imgLAB[x,y,:]
            img[x,y,0]=imgLAB[x,y,0]
            img[x,y,1]=imgLAB[x,y,1]
            img[x,y,2]=imgLAB[x,y,2]
      print "rgb type" ,type(imgRGB)
      print "rgb type in gm shape" ,imgRGB.shape
      print "lab type" ,type(img)
      print "lab type in gm shape" ,img.shape
   basename=os.path.basename(filename)
   img=imgRGB



   
   
   seedMapTemp=vigra.filters.gaussianGradientMagnitude(img,sigma=sigmaSeed)
   seedMapTemp2=seedMapTemp.copy()
   
   integralSmooth=5
   threshold=2
   
   
   seedMapIntegral = vigra.gaussianSmoothing(seedMapTemp,integralSmooth)
   
   vigra.impex.writeImage(seedMapTemp,outprefix+basename+".gausmag.jpg")   
   
   seedMapTemp=seedMapTemp**2
   seedMapTemp[numpy.where(seedMapIntegral<threshold)]=0 
   seedMap = vigra.gaussianSmoothing(seedMapTemp, smoothSeedScale)
   
   
   vigra.impex.writeImage((seedMap-seedMap.min())/(seedMap.max()-seedMap.min())*255,outprefix+basename+".smoothgausmag.jpg")   
   seg,numberOfRegions=vigra.analysis.watersheds(seedMap)
   seg=seg-1
   seg=numpy.squeeze(seg.astype(numpy.uint64))
   rag=opengm.RegionGraph(seg,numberOfRegions,False)
   
   boundaryPixels=[]
   for b in range(rag.numberOfBoundaries()):
      boundaryPixels.append(rag.boundaryPixels(b))
   

   
   
   writeSP(img,rag,outprefix+basename+".sp.jpg",boundaryPixels)
   
   

   
   print "get regions pixels"   
   regionPixels=[ ]
   regionFeatures=numpy.ones([rag.numberOfRegions(),3],dtype=numpy.float64)
   for r in range(rag.numberOfRegions()):
      regionPixels.append(rag.regionPixels(r))

   

   
   print "build model"
   gm=buildGM(img=img,rag=rag,dataImage=seedMapTemp2,numLabels=numLabels,boundaryPixels=boundaryPixels,regionPixels=regionPixels,beta=beta,sigma=sigma,verbose=False)
   
   print "start inference"
   alg='trbp'
   param=opengm.inferenceParameter(gm=gm,alg=alg)
   param.set(steps=6,damping=0.1)
   #arg=inferAndWriteResult(printNth=1,alg=alg,param=param,img=imgRGB,rag=rag,prefix=outprefix,filename=basename,boundaryPixels=boundaryPixels,regionPixels=regionPixels,verbose=verboseInf)
   
   alg='bp'
   param=opengm.inferenceParameter(gm=gm,alg=alg)
   param.set(steps=1,damping=0.6)
   arg=inferAndWriteResult(printNth=1,alg=alg,param=param,img=imgRGB,rag=rag,prefix=outprefix,filename=basename,boundaryPixels=boundaryPixels,regionPixels=regionPixels,verbose=verboseInf)
   
   alg='icm'
   param=opengm.inferenceParameter(gm=gm,alg=alg)
   param.set()
   #arg=inferAndWriteResult(printNth=1000,alg=alg,param=param,img=imgRGB,rag=rag,prefix=outprefix,filename=basename,boundaryPixels=boundaryPixels,regionPixels=regionPixels,verbose=verboseInf,startingPoint=arg)
   
   alg='gibbs'
   param=opengm.inferenceParameter(gm=gm,alg=alg)
   param.set(steps=1000000,tempMin=0.0000005,tempMax=0.4,useTemp=True,periodes=6*3)
    #arg=inferAndWriteResult(printNth=1000,alg=alg,param=param,img=imgRGB,rag=rag,prefix=outprefix,filename=basename,boundaryPixels=boundaryPixels,regionPixels=regionPixels,verbose=verboseInf,startingPoint=arg)
      
  
   alg='lf'
   param=opengm.inferenceParameter(gm=gm,alg=alg)
   param.set(maxSubgraphSize=2)
   arg=inferAndWriteResult(printNth=1000,alg=alg,param=param,img=imgRGB,rag=rag,prefix=outprefix,filename=basename,boundaryPixels=boundaryPixels,regionPixels=regionPixels,verbose=verboseInf,startingPoint=arg)

   alg='ae'
   param=opengm.inferenceParameter(gm=gm,alg=alg)
   param.set(steps=500000)
   arg=inferAndWriteResult(printNth=1000,alg=alg,param=param,img=imgRGB,rag=rag,prefix=outprefix,filename=basename,boundaryPixels=boundaryPixels,regionPixels=regionPixels,verbose=verboseInf,startingPoint=arg)
   
   
   alg='ab-swap'
   param=opengm.inferenceParameter(gm=gm,alg=alg)
   param.set(steps=500000)
   arg=inferAndWriteResult(printNth=1000,alg=alg,param=param,img=imgRGB,rag=rag,prefix=outprefix,filename=basename,boundaryPixels=boundaryPixels,regionPixels=regionPixels,verbose=verboseInf,startingPoint=arg)
