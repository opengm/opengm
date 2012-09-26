Inspect a Graphical Model
-------------------------

The following code shows a lot of usefull functions to inspect a gm and the factors of the gm::

   print "gm :",gm            
   print "number of Variables : ",gm.numberOfVariables
   print "number of Factors :", gm.numberOfFactors
   print "is Acyclic :" , gm.isAcyclic()

   for v in range(gm.numberOfVariables):
       print " vi=",v, " number of labels : ",gm.numberOfLabels(v)
       print " factors depeding on vi=",v," : ",gm.numberOfFactorsForVariable(v)

   for f in range(gm.numberOfFactors):
       print "gm[",f,"]",  gm[f]
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
       

Iterate over a Factors Values
-----------------------------

A direct access via coordinate.
The type of the coordinate can be a tuple ( ``c=(1,3,2)`` ) , a list  ( ``c=[1,1,3]`` )  or a numpy array ( ``c=numpy.array([2,1,3],dtype=numpy.uint64)`` ).
Assuming a third order factor::
   
   factor=gm[someFactorIndex]    
   for l2 in range(factor.numberOfLabels(2)):
      for l1 in range(factor.numberOfLabels(1)):
         for l0 in range(factor.numberOfLabels(0)):
            #coordinate / label sequence as tuple
            print factor[ (l0,l1,l2) ]
 
A factors value table can also be copied to a new allocated numpy ndarray::

   factor=gm[someFactorIndex]    
   valueTable=factor.asNumpy()
   
-------------------------   
.. note::
   TODO  
