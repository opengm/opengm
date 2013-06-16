import numpy 
import opengm
import matplotlib.pyplot as plt

f1=numpy.ones([2])
f2=numpy.ones([2,2])
f3=numpy.ones([2,2,2])
"""
Triangle (non-shared) :
    - 3 variables
    - 3 unaries
    - 2 second order functions
    - 1 third order factor
    - functions are *non* - shared
"""
gm=opengm.gm([2,2,2])
gm.addFactor(gm.addFunction(f1),[0])
gm.addFactor(gm.addFunction(f1),[1])
gm.addFactor(gm.addFunction(f1),[2])
gm.addFactor(gm.addFunction(f2),[0,1])
gm.addFactor(gm.addFunction(f2),[1,2])
gm.addFactor(gm.addFunction(f2),[0,2])
gm.addFactor(gm.addFunction(f3),[0,1,2])

opengm.visualizeGm( gm,show=False,plotFunctions=True,
                    plotNonShared=True)
plt.savefig("triangle.png",bbox_inches='tight',
             dpi=300)  
plt.close()