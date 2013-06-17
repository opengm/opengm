import numpy 
import opengm
import matplotlib.pyplot as plt

f1=numpy.ones([2])
f2=numpy.ones([2,2])

#Chain (non-shared functions):
numVar=5
gm=opengm.gm([2]*numVar)
for vi in xrange(numVar):
    gm.addFactor(gm.addFunction(f1),vi)
    if(vi+1<numVar):
        gm.addFactor(gm.addFunction(f2),[vi,vi+1])

# visualize gm        
opengm.visualizeGm( gm,show=False,layout='spring',plotFunctions=True,
                    plotNonShared=True,relNodeSize=0.4)
plt.savefig("chain_non_shared.png",bbox_inches='tight',dpi=300)  
plt.close()

#Chain (shared high order functions):
numVar=5
gm=opengm.gm([2]*numVar)
fid2=gm.addFunction(f2)
for vi in xrange(numVar):
    gm.addFactor(gm.addFunction(f1),vi)
    if(vi+1<numVar):
        gm.addFactor(fid2,[vi,vi+1])

# visualize gm  
opengm.visualizeGm( gm,show=False,layout='spring',plotFunctions=True,
                    plotNonShared=True,relNodeSize=0.4)
plt.savefig("chain_shared.png",bbox_inches='tight',dpi=300)  
plt.close()

