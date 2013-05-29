import numpy
import opengm


img=numpy.random.rand(4,4)
dimx=img.shape[0]
dimy=img.shape[1]

numVar=dimx*dimy
numLabels=2
beta=0.3

numberOfStates=numpy.ones(numVar,dtype=opengm.index_type)*numLabels
gm=opengm.graphicalModel(numberOfStates,operator='adder')



#Adding unary function and factors
for y in range(dimy):
   for x in range(dimx):
      f=numpy.ones(2,dtype=numpy.float32)
      f[0]=img[x,y]
      f[1]=1.0-img[x,y]
      fid=gm.addFunction(f)
      gm.addFactor(fid,(x*dimy+y,))



#Adding binary function and factors"
vis=numpy.ones(5,dtype=opengm.index_type)
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
         gm.addFactor(fid,numpy.array([x*dimy+y,(x+1)*dimy+y],dtype=opengm.index_type))
      if(y+1<dimy):
         #vi as list (tuple and numpy array can also be used as vi's)
         gm.addFactor(fid,[x*dimy+y,x*dimy+(y+1)])

icm=opengm.inference.Icm(gm)
icm.infer()
argmin=icm.arg()        


res=argmin.reshape(dimx,dimy)




