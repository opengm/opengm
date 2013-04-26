import opengm
import numpy

# setup
shape=[3,3]
numVar=shape[0]*shape[1]
img=numpy.random.rand(shape[0],shape[1])*255.0

# unaries
img1d=img.reshape(numVar)
lrange=numpy.arange(0,256,1,dtype=numpy.float32)
unaries=numpy.repeat(lrange[:,numpy.newaxis], numVar, 1).T
for l in xrange(256):
   unaries[:,l]=numpy.abs(img1d-l)
unaries=unaries.reshape(shape+[256])

# higher order function
def regularizer(labels):
   val=abs(float(labels[0])-float(labels[1]))
   return val*0.2

print "generate 2d grid gm"
regularizer=opengm.PythonFunction(function=regularizer,shape=[256,256])

print "unaries shape ",unaries.shape
print "reg shape ",regularizer.shape
gm=opengm.grid2d2Order(unaries,regularizer=regularizer)


icm=opengm.inference.Icm(gm)
#icm.infer()
arg=icm.arg()
arg=arg.reshape(shape)

print arg