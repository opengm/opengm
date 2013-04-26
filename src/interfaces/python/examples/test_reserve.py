import numpy
import opengm
import time 

dimx=100
dimy=100
numVar=dimx*dimy
numLabels=20
beta=0.8

# ---------------------------------
# reserve factors and functions can
# save a lot of time
# ---------------------------------

t=time.time()

numberOfStates=numpy.ones(numVar,dtype=numpy.uint64)*numLabels
gm=opengm.graphicalModel(numberOfStates,operator='adder')
#Adding unary function and factors
for y in range(dimy):
   for x in range(dimx):
      f1=numpy.random.random(numLabels).astype(numpy.float32)
      fid=gm.addFunction( f1)
      gm.addFactor(fid,(x+dimx*y,))
#Adding binary function and factors"
vis=numpy.ones(5,dtype=numpy.uint64)
#add one binary function (potts fuction)
f=numpy.ones(pow(numLabels,2),dtype=numpy.float32).reshape(numLabels,numLabels)*beta
for l in range(numLabels):
   f[l,l]=0  
fid=gm.addFunction(f)
#add binary factors
for y in range(dimy):   
   for x in range(dimx):
      if(x+1<dimx):
         #vi as tuple (list and numpy array can also be used as vi's)
         gm.addFactor(fid,numpy.array([x+dimx*y,x+1+dimx*y],dtype=numpy.uint64))
      if(y+1<dimy):
         #vi as list (tuple and numpy array can also be used as vi's)
         gm.addFactor(fid,[x+dimx*y,x+dimx*(y+1)])

e=time.time()-t
print e
del gm
t=time.time()

numberOfStates=numpy.ones(numVar,dtype=numpy.uint64)*numLabels
gm=opengm.graphicalModel(numberOfStates,operator='adder')
gm.reserveFunctions(dimx*dimy+1,'explicit')
numFactors=dimx+dimy + (dimx-1)*dimy + (dimy-1)*dimx
gm.reserveFactors(numFactors)
#Adding unary function and factors
for y in range(dimy):
   for x in range(dimx):
      f1=numpy.random.random(numLabels).astype(numpy.float32)
      fid=gm.addFunction( f1)
      gm.addFactor(fid,(x+dimx*y,))
#Adding binary function and factors"
vis=numpy.ones(5,dtype=numpy.uint64)
#add one binary function (potts fuction)
f=numpy.ones(pow(numLabels,2),dtype=numpy.float32).reshape(numLabels,numLabels)*beta
for l in range(numLabels):
   f[l,l]=0  
fid=gm.addFunction(f)
#add binary factors
for y in range(dimy):   
   for x in range(dimx):
      if(x+1<dimx):
         #vi as tuple (list and numpy array can also be used as vi's)
         gm.addFactor(fid,numpy.array([x+dimx*y,x+1+dimx*y],dtype=numpy.uint64))
      if(y+1<dimy):
         #vi as list (tuple and numpy array can also be used as vi's)
         gm.addFactor(fid,[x+dimx*y,x+dimx*(y+1)])

e=time.time()-t
print e