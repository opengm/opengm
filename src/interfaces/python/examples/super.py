import opengm
import numpy
import vigra

import pylab
import matplotlib.cm as cm
import Image
import sys
import os


img=vigra.readImage('lena.jpg')
img2=numpy.ones(img.shape[0:2])
imgt=numpy.ones(img.shape[0:2])





gaussMag=vigra.filters.gaussianGradientMagnitude(img,sigma=7)
seg,numberOfRegions=vigra.analysis.watersheds(gaussMag)
seg=seg-1
seg=numpy.squeeze(seg.astype(numpy.uint64))

segl=[0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,
      1,1,1,2,2,2,1,1,0,0,
      1,1,1,2,2,2,1,1,0,0,
      1,1,1,2,2,2,1,1,0,0,
      0,0,0,0,0,0,0,0,1,0,
      0,0,0,0,0,0,0,0,0,1]


#seg=numpy.array(segl,dtype=numpy.uint64).reshape(-1,10)
img2=numpy.ones(seg.shape[0:2])
imgt=numpy.ones(seg.shape[0:2])

print seg
rag=opengm.RegionGraph(seg,numberOfRegions,True)





for b in range(rag.numberOfBoundaries()):
   print "bindex= ",b
   #print "numReg= ",rag.numberOfAdjacentRegions(b)
   bPixel=rag.boundaryPixels(b)
   #imgt[:]=img2[:]
   for bb in range(len(bPixel)):
      print bPixel[bb]
      imgt[  bPixel[bb][0]  ,bPixel[bb][1] ]=0
pylab.imshow(imgt,cmap=cm.Greys_r,interpolation=None)
pylab.show()        
pylab.imshow(seg,cmap=cm.Greys_r,interpolation=None)
pylab.show()     



vigra.impex.writeImage(imgt,'out.png')   





