import numpy 
import opengm
import matplotlib.pyplot as plt


f1=numpy.ones([2])
f2=numpy.ones([2,2])

"""
Full Connected (non-shared):
    - all possible pairwise connections
    - functions are *non* - shared
"""
numVar=4
gm=opengm.gm([2]*numVar)
for vi0 in xrange(numVar):
    for vi1 in xrange(vi0+1,numVar):
        gm.addFactor(gm.addFunction(f2),[vi0,vi1])
opengm.visualizeGm( gm,show=False,layout='neato',
                    iterations=1000,plotFunctions=True,
                    plotNonShared=True,relNodeSize=0.4)
plt.savefig("full_non_shared.png",bbox_inches='tight',dpi=300)  
plt.close()


"""
Full Connected (shared):
    - 5 variables
    - 10 second order factors
      (all possible pairwise connections)
    - functions are *non* - shared
"""
numVar=4
gm=opengm.gm([2]*numVar)
fid2=gm.addFunction(f2)
for vi0 in xrange(numVar):
    for vi1 in xrange(vi0+1,numVar):
        gm.addFactor(fid2,[vi0,vi1])

opengm.visualizeGm( gm,show=False,layout='neato',
                    iterations=1000,plotFunctions=True,
                    plotNonShared=True,relNodeSize=0.4)
plt.savefig("full_shared.png",bbox_inches='tight',dpi=300)  
plt.close()