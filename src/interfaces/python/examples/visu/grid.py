import numpy 
import opengm
import matplotlib.pyplot as plt


f1=numpy.ones([2])
f2=numpy.ones([2,2])

"""
Grid:
    - 4x4=16 variables
    - second order factors in 4-neigbourhood
      all connected to the same function
    - higher order functions are shared
"""

size=3
gm=opengm.gm([2]*size*size)

fid=gm.addFunction(f2)
for y in range(size):   
    for x in range(size):
        gm.addFactor(gm.addFunction(f1),x*size+y)
        if(x+1<size):
            gm.addFactor(fid,[x*size+y,(x+1)*size+y])
        if(y+1<size):
            gm.addFactor(fid,[x*size+y,x*size+(y+1)])


opengm.visualizeGm( gm,layout='spring',iterations=3000,
                    show=False,plotFunctions=True,
                    plotNonShared=True,relNodeSize=0.4)
plt.savefig("grid.png",bbox_inches='tight',dpi=300) 
plt.close()
