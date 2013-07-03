import numpy 
from dtypes import index_type,value_type,label_type 

class FactorSubset(object):
    """ Holds a subset of factor indices of a graphical model.
        This class is used to compute queries for a a subset of a gm.
        This queries are very efficient since allmost all members
        are implemented in pure C++.
        The members are a vectorized subset of the regular 
        factor api of a graphical model. Therefore allmost all factor queries 
        can be vectorized with this class.

    Args :

        gm : the graphical model to which the factors belong

        factorIndices : the factor indices w.r.t. the gm which are in the subset . 
            If factorIndices is not given, the indices of all factors will be used

    Example: ::

        >>> import opengm
        >>> import numpy
        >>> unaries=numpy.random.rand(3,2,2)
        >>> gm=opengm.grid2d2Order(unaries,opengm.PottsFunction([2,2],0.0,0.4))
        >>> factorSubset=opengm.FactorSubset(gm)
        >>> len(factorSubset)==gm.numberOfFactors
        True
        >>> numberOfVariables=factorSubset.numberOfVariables()
        >>> len(numberOfVariables)==gm.numberOfFactors
        True
        >>> unaryFactorIndices=factorSubset.factorsWithOrder(1)
        >>> unaryFactorSubset=opengm.FactorSubset(gm,unaryFactorIndices)
        >>> len(unaryFactorSubset)
        6
        >>> secondOrderFactorIndices=factorSubset.factorsWithOrder(2)
        >>> secondOrderFactorSubset=opengm.FactorSubset(gm,secondOrderFactorIndices)
        >>> len(secondOrderFactorSubset)
        7

    """
    def __init__(self,gm,factorIndices=None):
        self.gm=gm
        if factorIndices is None:
            self.factorIndices=numpy.arange(gm.numberOfFactors,dtype=index_type)
        else :
            self.factorIndices=factorIndices

    def __len__(self):
        """ get the number of factors within the factorSubset """
        return len(self.factorIndices)


    def numberOfVariables(self):
        """ get the number variables for each factors within the factorSubset """
        return self.gm._factor_numberOfVariables(self.factorIndices)

    def gmLabelsToFactorLabels(self,labels):
        numpyLabels=numpy.require(labels,dtype=label_type)
        return self.gm._factor_gmLablingToFactorLabeling(self.factorIndices,labels)

    def variableIndices(self):
        return self.gm._factor_variableIndices(self.factorIndices)

    def numberOfLabels(self):
        return self.gm._factor_numberOfLabels(self.factorIndices)

    def isSubmodular(self):
        return self.gm._factor_isSubmodular(self.factorIndices)

    def mapScalarReturning(self,function,dtype):
        if(dtype==numpy.float32):
            return self.gm._factor_scalarRetFunction_float32(function,self.factorIndices)
        elif(dtype==numpy.float64):
            return self.gm._factor_scalarRetFunction_float64(function,self.factorIndices)
        elif(dtype==numpy.uint64):
            return self.gm._factor_scalarRetFunction_uint64(function,self.factorIndices)
        elif(dtype==numpy.int64):
            return self.gm._factor_scalarRetFunction_int64(function,self.factorIndices)
        elif(dtype==numpy.bool):
            return self.gm._factor_scalarRetFunction_bool(function,self.factorIndices)
        else:
            raise RuntimeError("dtype %s is not supported, so far only float32, float64, int64, uint64 and bool are supported")% (str(dtype),)

    def fullIncluedFactors(self,vis):
        visNumpy=numpy.require(vis,dtype=index_type)
        return self.gm._factor_fullIncluedFactors(self.factorIndices,visNumpy)

    def evaluate(self,labels):
        labelsNumpy=numpy.require(labels,dtype=label_type)
        if(labelsNumpy.ndim==1 and labelsNumpy.shape[0] == self.gm.numberOfLabels):
            return self.gm._factor_evaluateGmLabeling(self.factorIndices,labelsNumpy)
        else :
            if labelsNumpy.ndim==1:
                labelsNumpy=labelsNumpy.reshape([1,-1])
            return self.gm._factor_evaluateFactorLabeling(self.factorIndices,labelsNumpy)

    def factorsWithOrder(self,order):
        return self.gm._factor_withOrder(self.factorIndices,int(order))


if __name__ == "__main__":
  import doctest
  import opengm
  doctest.testmod()