import numpy
import opengm
import vigra
import matplotlib.pyplot as plt
import matplotlib.cm as cm


gradScale       = 0.1
energyNotEqual  = 0.2
sigma=0.2
resizeFactor=2

img=vigra.impex.readImage('lena.bmp')
shape=img.shape
imgLab=vigra.colors.transform_RGB2Lab(img)
shape=(shape[0]*resizeFactor,shape[1]*resizeFactor)
imgLab=vigra.sampling.resize(imgLab, shape,order=3)


gradMag=vigra.filters.gaussianGradientMagnitude(imgLab,gradScale)

unaries=numpy.zeros([shape[0],shape[1],2])
unaries[:,:,1]=numpy.exp(-1.0*gradMag[:,:,0]*sigma)
unaries[:,:,0]=1.0-unaries[:,:,1]
regularizer=opengm.PottsFunction([2,2],0.0,energyNotEqual)

gm=opengm.grid2d2Order(unaries=unaries,regularizer=regularizer,order='numpy',operator='adder')
inf=opengm.inference.GraphCut(gm)
inf.infer()
argmin=inf.arg().reshape(shape[0:2])


plt.figure(1)

ax=plt.subplot(2,1,1)
plt.imshow(unaries[:,:,1].T, interpolation="nearest")
plt.set_cmap(cm.copper)
plt.colorbar()
ax.set_title('costs / unaries label=1')

ax=plt.subplot(2,1,2)
plt.imshow(argmin.T, interpolation="nearest")
plt.colorbar()
ax.set_title('argmin')

plt.show()

