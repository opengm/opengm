import numpy
import opengm
import time 


dimx=1000
dimy=1000
numVar=dimx*dimy
numLabels=2



numberOfStates=numpy.ones(numVar,dtype=opengm.index_type)*numLabels
vis2Order=opengm.secondOrderGridVis(dimx,dimy)
numFac = len(vis2Order)
randf = numpy.random.rand(numFac,numLabels,numLabels).astype(numpy.float64)

print randf.shape
print "numVar",numVar,"numFac",numFac


print "# METHOD A"
with opengm.Timer():
   gm=opengm.graphicalModel(numberOfStates,operator='adder',reserveNumFactorsPerVariable=4)
   gm.reserveFunctions(numFac,'explicit')
   fids=gm.addFunctions(randf)
   gm.addFactors(fids,vis2Order)




print "# METHOD B"
with opengm.Timer():
   # (reserve reserveNumFactorsPerVariable does not make sense if we not "finalize" factors directely)
   gm=opengm.graphicalModel(numberOfStates,operator='adder')
   gm.reserveFactors(numFac)
   gm.reserveFunctions(numFac,'explicit')
   fids=gm.addFunctions(randf)
   gm.addFactors(fids,vis2Order,finalize=False)
   gm.finalize()


print "# METHOD C (NAIVE)"
with opengm.Timer():
   # (reserve reserveNumFactorsPerVariable does not make sense if we not "finalize" factors directely)
   gm=opengm.graphicalModel(numberOfStates,operator='adder')
   fids=gm.addFunctions(randf)
   gm.addFactors(fids,vis2Order)
   gm.finalize()



"""
numVar 1000000 numFac 1998000
# METHOD A
   Elapsed: 2.48288202286
# METHOD B
   Elapsed: 3.13179111481
# METHOD C (NAIVE)
   Elapsed: 4.0311460495


"""