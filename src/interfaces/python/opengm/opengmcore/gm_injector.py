from _opengmcore import adder,multiplier,IndexVector,FunctionIdentifier,FidVector,IndexVectorVector
from factorSubset import FactorSubset
from dtypes import index_type,label_type,value_type
import numpy

from function_injector import isNativeFunctionType,isNativeFunctionVectorType

LabelVector = IndexVector

def _extend_gm_classes():

  def variables(gm,labels=None,minLabels=None,maxLabels=None):
     getNumLabels=gm.numberOfLabels
     if labels is  None and maxLabels is None and minLabels is None:
        for v in xrange(gm.numberOfVariables):
           yield v
     elif labels is not None and maxLabels is None and minLabels is None:     
        for vi in xrange(gm.numberOfVariables):
           if getNumLabels(vi)==labels:
              yield vi
     elif maxLabels is not None and labels is None and minLabels is None:     
        for vi in xrange(gm.numberOfVariables):
           if getNumLabels(vi) <= maxLabels:
              yield vi
     elif minLabels is not None and labels is None and maxLabels is None:      
        for vi in xrange(gm.numberOfVariables):
           if getNumLabels(vi) >= minLabels:
              yield vi
     elif minLabels is not None and labels is None and maxLabels is not None:
        for vi in xrange(gm.numberOfVariables):
           numLabels=getNumLabels(vi)
           if numLabels>= minLabels and numLabels <= maxLabels:
              yield vi            

  def factors(gm,order=None,minOrder=None,maxOrder=None):
     if order is None and maxOrder is None and minOrder is None:
        for i in xrange(gm.numberOfFactors):
           yield gm[i]
     elif order is not None and maxOrder is None and minOrder is None:     
        for i in xrange(gm.numberOfFactors):
           factor = gm[i]
           if factor.numberOfVariables==order:
              yield factor
     elif maxOrder is not None and order is None and minOrder is None:     
        for i in xrange(gm.numberOfFactors):
           factor = gm[i]
           if factor.numberOfVariables <= maxOrder:
              yield factor
     elif minOrder is not None and order is None and maxOrder is None:      
        for i in xrange(gm.numberOfFactors):
           factor = gm[i]
           if factor.numberOfVariables >= minOrder:
              yield factor
     elif minOrder is not None and order is None and maxOrder is not None:
        for i in xrange(gm.numberOfFactors):
           factor = gm[i]
           if factor.numberOfVariables >= minOrder and factor.numberOfVariables <= maxOrder:
              yield factor            

  def factorIds(gm,order=None,minOrder=None,maxOrder=None):
     if order is None and maxOrder is None and minOrder is None:
        for i in xrange(gm.numberOfFactors):
           yield i
     elif order is not None and maxOrder is None and minOrder is None:     
        for i in xrange(gm.numberOfFactors):
           factor = gm[i]
           if factor.numberOfVariables==order:
              yield i
     elif maxOrder is not None and order is None and minOrder is None:     
        for i in xrange(gm.numberOfFactors):
           factor = gm[i]
           if factor.numberOfVariables <= maxOrder:
              yield i
     elif minOrder is not None and order is None and maxOrder is None:      
        for i in xrange(gm.numberOfFactors):
           factor = gm[i]
           if factor.numberOfVariables >= minOrder:
              yield i
     elif minOrder is not None and order is None and maxOrder is not None:
        for i in xrange(gm.numberOfFactors):
           factor = gm[i]
           if factor.numberOfVariables >= minOrder and factor.numberOfVariables <= maxOrder:
              yield i            

  def factorsAndIds(gm,order=None,minOrder=None,maxOrder=None):
     if order is None and maxOrder is None and minOrder is None:
        for i in xrange(gm.numberOfFactors):
           factor = gm[i]
           yield factor, i
     elif order is not None and maxOrder is None and minOrder is None:     
        for i in xrange(gm.numberOfFactors):
           factor = gm[i]
           if factor.numberOfVariables==order:
              yield factor,i
     elif maxOrder is not None and order is None and minOrder is None:     
        for i in xrange(gm.numberOfFactors):
           factor = gm[i]
           if factor.numberOfVariables <= maxOrder:
              yield factor, i
     elif minOrder is not None and order is None and maxOrder is None:      
        for i in xrange(gm.numberOfFactors):
           factor = gm[i]
           if factor.numberOfVariables >= minOrder:
              yield factor, i
     elif minOrder is not None and order is None and maxOrder is not None:
        for i in xrange(gm.numberOfFactors):
           factor = gm[i]
           if factor.numberOfVariables >= minOrder and factor.numberOfVariables <= maxOrder:
              yield factor,i            

  _gmClasses = [adder.GraphicalModel,multiplier.GraphicalModel ]

  for gmClass in _gmClasses:
    #gmClass._init_impl_=gmClass.__init__
    class _InjectorGm(object):
        class __metaclass__(gmClass.__class__):
            def __init__(self, name, bases, dict):

                for b in bases:
                    if type(b) not in (self, type):
                        for k,v in dict.items():
                            setattr(b,k,v)
                return type.__init__(self, name, bases, dict)

    class _more_gm(_InjectorGm, gmClass):
      #def __init__(self,*args,**kwargs):
        #return self._init_impl_(*args,**kwargs)


      def testf(self):
        return 0
      def testf2(self):
        return 0
      
      @property 
      def factorClass(self):
        """
        Get the class of the factor of this gm

        Example :
            >>> import opengm
            >>> gm=opengm.gm([2]*10)
            >>> # fill gm with factors...
            >>> result=gm.vectorizedFactorFunction(gm.factorClass.isSubmodular,range(gm.numberOfFactors))
        """
        if self.operator=='adder':
          return adder.Factor
        elif self.operator=='multiplier':
          return multiplier.Factor
        else:
          raise RuntimeError("wrong operator")
      def connectedComponentsFromLabels(self,labels):
        numpyLabels=numpy.require(labels,dtype=label_type)
        return self._getCCFromLabes(numpyLabels)
      def factorSubset(self,factorIndices=None,order=None):

        if factorIndices is None:
          fIndices=numpy.arange(self.numberOfFactors,dtype=index_type)
        else:
          fIndices=numpy.require(factorIndices,dtype=index_type)
        if order is None:
          return FactorSubset(self,fIndices) 
        else :
          factorIndicesWithOrder=self._factor_withOrder(fIndices,int(order))
          return FactorSubset(self,factorIndicesWithOrder) 

      def variableIndices(self,factorIndices):
        """ get the factor indices of all factors connected to variables within ``variableIndices`` 

        Args:

          factorIndices : factor indices w.r.t. the graphical model 

        Examples :

          >>> import opengm
          >>> import numpy
          >>> unaries=numpy.random.rand(2, 2,4).astype(numpy.float64)
          >>> gm=opengm.grid2d2Order(unaries=unaries,regularizer=opengm.pottsFunction([4,4],0.0,0.4))
          >>> variableIndicesNumpyArray=gm.variableIndices(factorIndices=[3,4])
          >>> [vi for vi in variableIndicesNumpyArray]
          [0, 2, 3]


        Returns :

          a sorted numpy.ndarray of all variable indices

        Notes :

          This function will be fastest if ``variableIndices`` is a numpy.ndarray.
          Otherwise the a numpy array will be allocated and the elements
          of variableIndices are copied.
        """
        if isinstance(object, numpy.ndarray):
          return self._variableIndices(numpy.require(factorIndices,dtype=index_type))
        else:
          return self._variableIndices(numpy.array(factorIndices,dtype=index_type)) 

      def factorIndices(self,variableIndices):
        """ get the factor indices of all factors connected to variables within ``variableIndices`` 

        Args:

          variableIndices : variable indices w.r.t. the graphical model 

        Examples :

          >>> import opengm
          >>> import numpy
          >>> unaries=numpy.random.rand(2, 2,4)
          >>> gm=opengm.grid2d2Order(unaries=unaries,regularizer=opengm.pottsFunction([4,4],0.0,0.4))
          >>> factorIndicesNumpyArray=gm.factorIndices(variableIndices=[0,1])
          >>> [fi for fi in factorIndicesNumpyArray]
          [0, 1, 4, 5, 6]


        Returns :

          a sorted numpy.ndarray of all factor indices

        Notes :

          This function will be fastest if ``variableIndices`` is a numpy.ndarray.
          Otherwise the a numpy array will be allocated and the elements
          of variableIndices are copied.
        """
        if isinstance(object, numpy.ndarray):
          return self._factorIndices(numpy.require(variableIndices,dtype=index_type))
        else:
          return self._factorIndices(numpy.array(variableIndices,dtype=index_type))

      def variables(self,labels=None,minLabels=None,maxLabels=None):
        """ generator object to iterate over all variable indices

        Args:

          labels : iterate only over variables with  ``labels`` number of Labels if this argument is set (default: None) 

          minLabels : iterate only over variabe which have at least ``minLabels`` if this argument is set (default: None)

          minLabels : iterate only over variabe which have at maximum ``maxLabels`` if this argument is set (default: None)

        Examples: ::

          >>> import opengm
          >>> # a graphical model with 6 variables, some variables with 2, 3 and 4 labels
          >>> gm=opengm.gm([2,2,3,3,4,4])
          >>> [vi for vi in gm.variables()]
          [0, 1, 2, 3, 4, 5]
          >>> [vi for vi in gm.variables(labels=3)]
          [2, 3]
          >>> [vi for vi in gm.variables(minLabels=3)]
          [2, 3, 4, 5]
          >>> [vi for vi in gm.variables(minLabels=2,maxLabels=3)]
          [0, 1, 2, 3]

        """
        return variables(self,labels,minLabels,maxLabels)
      def factors(self,order=None,minOrder=None,maxOrder=None):
        return factors(self,order,minOrder,maxOrder)
      def factorsAndIds(self,order=None,minOrder=None,maxOrder=None):
        return factorsAndIds(self,order,minOrder,maxOrder)
      def factorIds(self,order=None,minOrder=None,maxOrder=None):
        return factorIds(self,order,minOrder,maxOrder)     

      def evaluate(self,labels):
        """ evaluate a labeling to get the energy / probability of that given labeling

        Args:

          labels : a labeling for all variables of the graphical model

        Examples: ::

          >>> import opengm
          >>> import numpy
          >>> unaries=numpy.random.rand(2, 2,4)
          >>> gm=opengm.grid2d2Order(unaries=unaries,regularizer=opengm.pottsFunction([4,4],0.0,0.4))
          >>> energy=gm.evaluate([0,2,2,1])
        """
        if len(labels)!=self.numberOfVariables :
          nVar=self.numberOfVariables
          nGiven=len(labels)
          raise RuntimeError('number of given labels (%d) does not match gm.numberOfVariables (%d)'%(nGiven,nVar))
        if isinstance(labels, numpy.ndarray):
          return self._evaluate_numpy(numpy.require(labels,dtype=label_type))
        elif isinstance(labels, list):
          return self._evaluate_list(labels)
        elif isinstance(labels, LabelVector):
          return self._evaluate_vector(labels)
        else:
          raise RuntimeError( "%s is not an supperted type for arument ``labels`` in ``evaluate``" %(str(type(labels)) ,) ) 

      def addFactor(self,fid,variableIndices,finalze=True):
        """ add a factor to the graphical model

        Args:

          fid : function identifier 

          variableIndices : indices of the fator w.r.t. the graphical model.
            The variable indices have to be sorted.

        Examples: ::

          >>> import opengm
          >>> # a graphical model with 6 variables, some variables with 2, 3 and 4 labels
          >>> gm=opengm.gm([2,2,3,3,4,4])
          >>> # Add unary function and factor ( factor which is connect to 1 variable )
          >>> # - add function ( a random function with 2 enties in the value table)
          >>> fid =   gm.addFunction(opengm.randomFunction(shape=[2]))
          >>> # - connect function and variables to factor 
          >>> int(gm.addFactor(fid=fid,variableIndices=0))
          0


        """
        if isinstance(variableIndices, (int,long)):
          return self._addFactor(fid,[variableIndices],finalze)
        elif isinstance(variableIndices,numpy.ndarray):
          return self._addFactor(fid,numpy.require(variableIndices,dtype=index_type),finalze)
        else:
          return self._addFactor(fid,variableIndices,finalze)

      def addFactors(self,fids,variableIndices,finalize=True):
        if isinstance(fids, FunctionIdentifier):
          fidVec=FidVector()
          fidVec.append(fids)
          fids=fidVec
        elif isinstance(fids,list):
          fidVec=FidVector(fids)
          fids=fidVec
        if (isinstance(variableIndices,numpy.ndarray)):
          ndim=variableIndices.ndim
          if(ndim==1):
            return self._addUnaryFactors_vector_numpy(fids,numpy.require(variableIndices,dtype=index_type),finalize)
          elif(ndim==2):
            return self._addFactors_vector_numpy(fids,numpy.require(variableIndices,dtype=index_type),finalize)
        elif (isinstance(variableIndices,IndexVectorVector)):
          return self._addFactors_vector_vectorvector(fids,variableIndices,finalize)
        else :
          try :
            return self._addFactors_vector_numpy(fids,numpy.array(variableIndices,dtype=index_type),finalize)
          except:
            raise RuntimeError( "%s is not an supperted type for arument ``variableIndices`` in ``addFactors``" %(str(type(variableIndices)) ,)  ) 
     
      def fixVariables(self,variableIndices,labels):
        """ return a new graphical model where some variables are fixed to a given label.

        Args:
          variableIndices : variable indices to fix
          labels          : labels of the variables to fix

        Returns:
          new graphical model where variables are fixed.
        """
        if(self.operator=='adder'):
          manip = adder.GraphicalModelManipulator(self)
        elif(self.operator=='multiplier'):
          manip = multiplier.GraphicalModelManipulator(self)
        else:
          raise RuntimeError("uknown operator %s"%self.operator)

        v=numpy.require(variableIndices,dtype=index_type)
        l=numpy.require(labels,dtype=label_type)

        # fix vars
        manip.fixVariables(v,l)
        # build submodel
        manip.buildModifiedModel()
        # get submodel
        subGm = manip.getModifiedModel()
        # get submodel variable indices
        subGmVis=manip.getModifiedModelVariableIndices()
        return subGm,subGmVis
      
        #pass

      def addFunction(self,function):
        """
        Adds a function to the graphical model.

        Args:
          function: a function/ value table
        Returns:
           A function identifier (fid) .

           This fid is used to connect a factor to this function

        Examples : 
          Explicit functions added via numpy ndarrays: ::

            >>> import opengm
            >>> #Add 1th-order function with the shape [3]::
            >>> gm=opengm.graphicalModel([3,3,3,4,4,4,5,5,2,2])
            >>> f=numpy.array([0.8,1.4,0.1])
            >>> fid=gm.addFunction(f)
            >>> print fid.functionIndex
            0
            >>> print fid.functionType
            0
            >>> # Add 2th-order function with  the shape [4,4]::
            >>> f=numpy.ones([4,4])
            >>> #fill the function with values
            >>> #..........
            >>> fid=gm.addFunction(f)
            >>> int(fid.functionIndex),int(fid.functionType)
            (1, 0)
            >>> # Adding 3th-order function with the shape [4,5,2]::
            >>> f=numpy.ones([4,5,2])
            >>> #fill the function with values
            >>> #..........
            >>> fid=gm.addFunction(f)
            >>> print fid.functionIndex
            2
            >>> print fid.functionType
            0

          Potts functions: ::
            
            >>> import opengm
            >>> gm=opengm.gm([2,2,3,3,3,4,4,4])
            >>> # 2-order potts function 
            >>> f=opengm.pottsFunction(shape=[2,2],valueEqual=0.0,valueNotEqual=1.0)
            >>> f[0,0],f[1,0],f[0,1],f[1,1]
            (0.0, 1.0, 1.0, 0.0)
            >>> fid=gm.addFunction(f)
            >>> int(fid.functionIndex),int(fid.functionType)
            (0, 1)
            >>> # connect a second order factor to variable 0 and 1 and the potts function
            >>> int(gm.addFactor(fid,[0,1]))
            0
            >>> # higher order potts function
            >>> f=opengm.pottsFunction(shape=[2,3,4],valueEqual=0.0,valueNotEqual=2.0)
            >>> f[0,0,0],f[1,0,0],f[1,1,1],f[1,1,3]
            (0.0, 2.0, 0.0, 2.0)
            >>> fid=gm.addFunction(f)
            >>> int(fid.functionIndex),int(fid.functionType)
            (0, 2)
            >>> # connect a third order factor to variable 0,2 and 5 and the potts function
            >>> int(gm.addFactor(fid,(0,2,5)))
            1


        Notes:
          .. seealso::
            :class:`opengm.ExplicitFunction`,
            :class:`opengm.SparseFunction`,
            :class:`opengm.TruncatedAbsoluteDifferenceFunction`, 
            :class:`opengm.TruncatedSquaredDifferenceFunction`, 
            :class:`opengm.PottsNFunction`, 
            :class:`opengm.PottsGFunction`
            :func:`opengm.pottsFunction`
            :func:`opengm.differenceFunction` 
            :func:`opengm.modelViewFunction` 
            :func:`opengm.randomFunction`
        """
        if isinstance(function, numpy.ndarray):
          return self._addFunction(numpy.require(function,dtype=value_type))
        else:
          return self._addFunction(function)


      def addFunctions(self,functions):
        if isinstance(functions,numpy.ndarray):
          if functions.ndim==2:
            return self._addUnaryFunctions_numpy(numpy.require(functions,dtype=value_type))
          else:
            return self._addFunctions_numpy(numpy.require(functions,dtype=value_type))
        elif isinstance(self,list):
          return self._addFunctions_list(functions)
        else:
          try:
            return self._addFunctions_vector(functions)
          except:
            try:
              return self._addFunctions_generator(functions)
            except:
              raise RuntimeError( "%s is an not a supported type for addFunctions "%(str(type(functions)),) )
      



#_extend_gm_classes()


if __name__ == "__main__":
  import doctest
  import opengm
  doctest.testmod()
