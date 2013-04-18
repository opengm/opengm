import opengm
import numpy

unaries=numpy.random.rand(3 , 3,2).astype(numpy.float32)
potts=opengm.PottsFunction([2,2],0.0,0.4)
gm=opengm.grid2d2Order(unaries=unaries,regularizer=potts)


print "gm :",gm            
print "number of Variables : ",gm.numberOfVariables
print "number of Factors :", gm.numberOfFactors
print "is Acyclic :" , gm.isAcyclic()

for v in gm.variables():
    print "\n\n vi=",v, " number of labels : ",gm.numberOfLabels(v)
    print " factors depeding on vi=",v," : ",gm.numberOfFactorsOfVariable(v)

for f in gm.factorIds():
    print "\n\ngm[",f,"]",  gm[f]
    print "gm[",f,"].shape",  gm[f].shape
    #convert shape to tuple,list or numpy ndarray
    shape = gm[f].shape.asTuple()
    shape = gm[f].shape.asList()
    shape = gm[f].shape.asNumpy()
    print "gm[",f,"].variableIndices",  gm[f].variableIndices
    #convert variableIndices to tuple,list or numpy ndarray
    shape= gm[f].variableIndices.asTuple()
    shape = gm[f].variableIndices.asList()
    shape = gm[f].variableIndices.asNumpy()
    #convert the factor to a numpy ndarray (a new numpy ndarray is allocated)
    print "factors values:\n",gm[f].asNumpy()
    #factors min ,max ,sum and product values
    print "min : ",gm[f].min()
    print "max : ",gm[f].max()
    print "sum : ",gm[f].sum()
    print "product : ",gm[f].product()
    #factors properties
    print "isPotts : ",gm[f].isPotts()
    print "isGeneralizedPotts : ",gm[f].isGeneralizedPotts()
    print "isSubmodular : ",gm[f].isSubmodular()
    print "isSquaredDifference : ",gm[f].isSquaredDifference()
    print "isTruncatedSquaredDifference : ",gm[f].isTruncatedSquaredDifference()
    print "isAbsoluteDifference : ",gm[f].isAbsoluteDifference()
    print "isTruncatedAbsoluteDifference : ",gm[f].isTruncatedAbsoluteDifference()