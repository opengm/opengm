import opengm
import numpy
import matplotlib
import time
from matplotlib import pyplot as plt
from matplotlib import animation


shape=[100,100]
numLabels=10
unaries=numpy.random.rand(shape[0], shape[1],numLabels).astype(numpy.float32)
potts=opengm.PottsFunction([numLabels,numLabels],0.0,0.4)
gm=opengm.grid2d2Order(unaries=unaries,regularizer=potts)

# alpha beta swap as solver
inf=opengm.inference.AlphaBetaSwap(gm,parameter=opengm.InfParam(steps=20))
inf=opengm.inference.AlphaExpansion(gm,parameter=opengm.InfParam(steps=20))
inf=opengm.inference.BeliefPropagation(gm,parameter=opengm.InfParam())

inf=opengm.inference.Icm(gm,parameter=opengm.InfParam())
class PyCallback(object):
    def __init__(self,shape,numLabels):
        self.shape=shape
        self.numLabels=numLabels
    def begin(self,inference):
        print "begin"
        self.visitNr=1
        self.gm=inference.gm()
        self.labelVector=opengm.LabelVector()
        self.labelVector.resize(self.gm.numberOfVariables)
        matplotlib.interactive(True)
        self.fig = plt.figure()
        self.cmap = matplotlib.colors.ListedColormap ( numpy.random.rand ( self.numLabels,3))

        win = self.fig.canvas.manager.window
        #self.fig.canvas.manager.window.after(100, plt.animate_frames, frames)
    def end(self,inference):
        print "end"
    def visit(self,inference):
        self.labelVector=inference.arg(out=self.labelVector,returnAsVector=True)
        print "energy ",self.gm.evaluate(inference.arg(self.labelVector))
        self.visitNr+=1
        
        asNumpy=self.labelVector.asNumpy()
        asNumpy=asNumpy.reshape(self.shape)
        self.tempCS1 = plt.imshow(asNumpy*255.0, cmap=self.cmap,interpolation="nearest") 
        self.fig.canvas.draw()
        #time.sleep(1) #unnecessary, but useful
        self.fig.clf()
        plt.show()

callback=PyCallback(shape,numLabels)
visitor=inf.pythonVisitor(callback,visitNth=1)



inf.infer(visitor)
# get the result states
argmin=inf.arg()
print "argminEnergy",gm.evaluate(argmin)
# print the argmin (on the grid)
#print argmin.reshape(shape)
print "the end"
