import opengm
import numpy

# setup
shape=[5,5]
numVar=shape[0]*shape[1]
img=numpy.random.rand(shape[0],shape[1])*255.0

# unaries
img1d=img.reshape(numVar)
lrange=numpy.arange(0,256,1)
unaries=numpy.repeat(lrange[:,numpy.newaxis], numVar, 1).T

for l in xrange(256):
   unaries[:,l]-=img1d

unaries=numpy.abs(unaries)
unaries=unaries.reshape(shape+[256])

# higher order function
def regularizer(labels):
   val=abs(float(labels[0])-float(labels[1]))
   return val*0.4

print "generate 2d grid gm"
regularizer=opengm.PythonFunction(function=regularizer,shape=[256,256])
gm=opengm.grid2d2Order(unaries,regularizer=regularizer)


icm=opengm.inference.Icm(gm)
icm.infer()
arg=icm.arg()
arg=arg.reshape(shape)

print numpy.round(img)
print arg
