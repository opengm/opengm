"""
Usage: python_visitor_gui.py   

This script shows how one can implement visitors
in pure python and inject them into OpenGM solver.
( not all OpenGM solvers support this kind of 
    code injection )
"""

import opengm
import numpy
import matplotlib
from matplotlib import pyplot as plt


shape=[100,100]
numLabels=10
unaries=numpy.random.rand(shape[0], shape[1],numLabels)
potts=opengm.PottsFunction([numLabels,numLabels],0.0,0.4)
gm=opengm.grid2d2Order(unaries=unaries,regularizer=potts)
inf=opengm.inference.BeliefPropagation(gm,parameter=opengm.InfParam(damping=0.5))

class PyCallback(object):
    def __init__(self,shape,numLabels):
        self.shape=shape
        self.numLabels=numLabels
        self.cmap = matplotlib.colors.ListedColormap ( numpy.random.rand ( self.numLabels,3))
        matplotlib.interactive(True)
    def begin(self,inference):
        print "begin of inference"
    def end(self,inference):
        print "end of inference"
    def visit(self,inference):
        gm=inference.gm()
        labelVector=inference.arg()
        print "energy  ",gm.evaluate(labelVector)
        labelVector=labelVector.reshape(self.shape)
        plt.imshow(labelVector*255.0, cmap=self.cmap,interpolation="nearest") 
        plt.draw()


callback=PyCallback(shape,numLabels)
visitor=inf.pythonVisitor(callback,visitNth=1)

inf.infer(visitor)
argmin=inf.arg()
