from _opengmcore import *
import numpy

configuration=OpengmConfiguration()
LabelVector=IndexVector
index_type = numpy.uint64
label_type = numpy.uint64
value_type = numpy.float32


def graphicalModel(numberOfLabels,operator='adder',reserveNumFactorsPerVariable=0):
   """
   Factory function to construct a graphical model.

   Args:
   
   numberOfLabels : number of label sequence (can be a list or  a 1d numpy.ndarray)
   
   operator : operator of the graphical model. Can be 'adder' or 'multiplier' (default: 'adder')
   

   Construct a gm with ``\'adder\'`` as operator::
      >>> import opengm
      >>> gm=opengm.graphicalModel([2,2,2,2,2],operator='adder')
      >>> # or just
      >>> gm=opengm.graphicalModel([2,2,2,2,2])
      
   Construct a gm with ``\'multiplier\'`` as operator::  
   
      gm=opengm.graphicalModel([2,2,2,2,2],operator='multiplier')
      
   """
   if operator=='adder' :
      return adder.GraphicalModel(numberOfLabels,reserveNumFactorsPerVariable)
   elif operator=='multiplier' :
      return multiplier.GraphicalModel(numberOfLabels,reserveNumFactorsPerVariable)
   else:
      raise NameError('operator must be \'adder\' or \'multiplier\'') 

gm = graphicalModel

def movemaker(gm,labels=None):
   if gm.operator=='adder':
      if labels is None:
         return adder.Movemaker(gm)
      else:
         return adder.Movemaker(gm,labels)
   elif gm.operator=='multiplier':
      if labels is None:
         return multiplier.Movemaker(gm)
      else:
         return multiplier.Movemaker(gm,labels)
   else:
      assert false              



def shapeWalker(shape):
  """
  generator obect to iterate over a multi-dimensional factor / value table 

  Args:
    shape : shape of the factor / value table

  Yields:
    coordinate as list of integers

  Example: ::

    >>> import opengm
    >>> import numpy
    >>> # some graphical model 
    >>> # -with 2 variables with 2 labels.
    >>> # -with 1  2-order functions
    >>> # -connected to 1 factor
    >>> gm=opengm.gm([2]*2)
    >>> f=opengm.PottsFunction(shape=[2,2],valueEqual=0.0,valueNotEqual=1.0)
    >>> int(gm.addFactor(gm.addFunction(f),[0,1]))
    0
    >>> # iterate over all factors  of the graphical model 
    >>> # (= 1 factor in this example)
    >>> for factor in gm.factors():
    ...   # iterate over all labelings with a "shape walker"
    ...   for coord in opengm.shapeWalker(f.shape):
    ...      pass
    ...      print "f[%s]=%.1f" %(str(coord),factor[coord])
    f[[0, 0]]=0.0
    f[[1, 0]]=1.0
    f[[0, 1]]=1.0
    f[[1, 1]]=0.0

  Note :

    Only implemented for dimension<=10
  """
  dim=len(shape)
  c=[int(0)]*dim

  if(dim==1):
    for c[0] in xrange(shape[0]):
      yield c
  elif (dim==2):
    for x1 in xrange(shape[1]):
      for x0 in xrange(shape[0]):
        yield [x0,x1]
  elif (dim==3):
    for x2 in xrange(shape[2]):
      for x1 in xrange(shape[1]):
        for x0 in xrange(shape[0]):
          yield [x0,x1,x2]
  elif (dim==4):
    for c[3] in xrange(shape[3]):
      for c[2] in xrange(shape[2]):
        for c[1] in xrange(shape[1]):
          for c[0] in xrange(shape[0]):
            yield c
  elif (dim==5):
    for c[4] in xrange(shape[4]):
      for c[3] in xrange(shape[3]):
        for c[2] in xrange(shape[2]):
          for c[1] in xrange(shape[1]):
            for c[0] in xrange(shape[0]):
              yield c

  elif (dim==6):
    for c[5] in xrange(shape[5]):
      for c[4] in xrange(shape[4]):
        for c[3] in xrange(shape[3]):
          for c[2] in xrange(shape[2]):
            for c[1] in xrange(shape[1]):
              for c[0] in xrange(shape[0]):
                yield c
  elif (dim==7):
    for c[6] in xrange(shape[6]):
      for c[5] in xrange(shape[5]):
        for c[4] in xrange(shape[4]):
          for c[3] in xrange(shape[3]):
            for c[2] in xrange(shape[2]):
              for c[1] in xrange(shape[1]):
                for c[0] in xrange(shape[0]):
                  yield c              
  elif (dim==8):
    for c[7] in xrange(shape[7]):
      for c[6] in xrange(shape[6]):
        for c[5] in xrange(shape[5]):
          for c[4] in xrange(shape[4]):
            for c[3] in xrange(shape[3]):
              for c[2] in xrange(shape[2]):
                for c[1] in xrange(shape[1]):
                  for c[0] in xrange(shape[0]):
                    yield c
  elif (dim==9):
    for c[8] in xrange(shape[8]):
      for c[7] in xrange(shape[7]):
        for c[6] in xrange(shape[6]):
          for c[5] in xrange(shape[5]):
            for c[4] in xrange(shape[4]):
              for c[3] in xrange(shape[3]):
                for c[2] in xrange(shape[2]):
                  for c[1] in xrange(shape[1]):
                    for c[0] in xrange(shape[0]):
                      yield c
  elif (dim==10):
    for c[9] in xrange(shape[9]):
      for c[8] in xrange(shape[8]):
        for c[7] in xrange(shape[7]):
          for c[6] in xrange(shape[6]):
            for c[5] in xrange(shape[5]):
              for c[4] in xrange(shape[4]):
                for c[3] in xrange(shape[3]):
                  for c[2] in xrange(shape[2]):
                    for c[1] in xrange(shape[1]):
                      for c[0] in xrange(shape[0]):
                        yield c
  else :
    raise TypeError("shapeWalker is only implemented for len(shape)<=10 ")



class Adder:
   def neutral(self):
      return float(0.0)
  

class Multiplier:
   def neutral(self):
      return float(1.0)

 


def modelViewFunction(factor):
  class _ModelViewFunction:
    def __init__(self,factor):
      self.factor=factor
    def __call__(self,labeling):
      return self.factor[labeling]
  return PythonFunction( _ModelViewFunction(factor) ,factor.shape.asTuple())


#Model generators
def grid2d2Order(unaries,regularizer,order='numpy',operator='adder'):
   """ 
   returns a 2d-order model on a 2d grid (image).
   The regularizer is the same for all 2.-order functions.

   Keyword arguments:
   unaries -- unaries as 3d numy array where the last dimension iterates over the labels
   regularizer -- second order regularizer
   order -- order how to compute a scalar index from (x,y) (default: 'numpy')
   operator -- operator of the graphical model (default: 'adder')

   Example : :: 

      >>> import opengm
      >>> import numpy
      >>> unaries=numpy.random.rand(10, 10,2).astype(numpy.float32)
      >>> gm=opengm.grid2d2Order(unaries=unaries,regularizer=opengm.pottsFunction([2,2],0.0,0.4))
      >>> int(gm.numberOfVariables)
      100

   """
   shape=unaries.shape
   assert(len(shape)==3)
   numLabels=shape[2]
   numVar=shape[0]*shape[1]
   numFactors=(shape[0]-1)*shape[1] + (shape[1]-1)*shape[0] +numVar
   numberOfLabels=numpy.ones(numVar,dtype=numpy.uint64)*numLabels
   gm=graphicalModel(numberOfLabels,reserveNumFactorsPerVariable=5,operator=operator)
   gm.reserveFunctions(numVar+1,'explicit')
   gm.reserveFactors(numFactors)
   # add unaries
   unaries2d=unaries.reshape([numVar,numLabels])
   #fids=
   
   #vis=
   gm.addFactors(gm.addFunctions(unaries2d),numpy.arange(0,numVar,dtype=numpy.uint64))

   # add 2-order function
   vis2Order=secondOrderGridVis(shape[0],shape[1],bool(order=='numpy'))
   fid2Order=gm.addFunction(regularizer)
   fids=FidVector()
   fids.append(fid2Order)
   gm.addFactors(fids,vis2Order)
   return gm


_GmAdder      = adder.GraphicalModel
_GmMultiplier = multiplier.GraphicalModel

_FactorAdder      = adder.Factor
_FactorMultiplier = multiplier.Factor


_ExplicitFunction                    = ExplicitFunction
_ExplicitFunction.__module__="opengm.opengmcore"
_SparseFunction                      = SparseFunction
_AbsoluteDifferenceFunction          = AbsoluteDifferenceFunction
_SquaredDifferenceFunction           = SquaredDifferenceFunction
_TruncatedAbsoluteDifferenceFunction = TruncatedAbsoluteDifferenceFunction
_TruncatedSquaredDifferenceFunction  = TruncatedSquaredDifferenceFunction
_PottsFunction                       = PottsFunction
_PottsNFunction                      = PottsNFunction
_PottsGFunction                      = PottsGFunction
_PythonFunction                      = PythonFunction

def _extend_classes():

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



  _factorClasses=[adder.Factor,multiplier.Factor]

  for factorClass in _factorClasses:

    class _FactorInjector(object):
        class __metaclass__(factorClass.__class__):
            def __init__(self, name, bases, dict):

                for b in bases:
                    if type(b) not in (self, type):
                        for k,v in dict.items():
                            setattr(b,k,v)
                return type.__init__(self, name, bases, dict)
                
    class _more_factor(_FactorInjector, factorClass):
      def __getitem__(self,labeling):
        """ get the value of factor for a given labeling:

        Example: ::
          >>> import opengm
          >>> gm=opengm.gm([3]*10)
          >>> f=opengm.pottsFunction(shape=[3,3,3],valueEqual=0.0,valueNotEqual=1.0)
          >>> int(gm.addFactor( gm.addFunction(f) ,  [0,1,2] ))
          0
          >>> factor=gm[0]
          >>> # explicit labeling
          >>> factor[0,0,0]
          0.0
          >>> # list with labels
          >>> factor[[1,1,1]]
          0.0
          >>> # tuple with labels
          >>> factor[(0,1,2)]
          1.0
          >>> # numpy array with labels
          >>> factor[numpy.array([1,1,1] ,dtype=opengm.label_type)]
          0.0

        """
        return self._getitem(labeling)
      def asNumpy(self):
        """ 
        get a copy of the factors value table as an numpy array 

        Example : :: 

          >>> import opengm
          >>> import numpy
          >>> unaries=numpy.random.rand(10, 10,2).astype(numpy.float32)
          >>> gm=opengm.grid2d2Order(unaries=unaries,regularizer=opengm.pottsFunction([2,2],0.0,0.4))
          >>> aFactor=gm[100]
          >>> valueTable=aFactor.asNumpy()
          >>> valueTable.shape
          (2, 2)

        """
        return self.copyValuesSwitchedOrder().reshape(self.shape)

      def subFactor(self,fixedVars,fixedVarsLabels):
        """
        get the value table of of a sub-factor where some variables of
        the factor have been fixed to a given label

        Args:

          fixedVars : a 1d-sequence of variable indices to fix w.r.t. the factor

          fixedVarsLabels : a 1d-sequence of labels for the given indices in ``fixedVars``

        Example : :: 

          >>> import opengm
          >>> import numpy
          >>> unaries=numpy.random.rand(10, 10,4).astype(numpy.float32)
          >>> gm=opengm.grid2d2Order(unaries=unaries,regularizer=opengm.pottsFunction([4,4],0.0,0.4))
          >>> factor2Order=gm[100]
          >>> int(factor2Order.numberOfVariables)
          2
          >>> print factor2Order.shape
          [4, 4, ]
          >>> # fix the second variable index w.r.t. the factor to the label 3
          >>> subValueTable = factor2Order.subFactor(fixedVars=[1],fixedVarsLabels=[3])
          >>> subValueTable.shape
          (4,)
          >>> for x in range(4):
          ...     print factor2Order[x,3]==subValueTable[x]
          True
          True
          True
          True
        """
        f = self.asNumpy()
        shape   = f.shape
        dim     = len(shape)
        slicing = [ slice (x) for x in shape ]

        for fixedVar,fixedVarLabel in zip(fixedVars,fixedVarsLabels):
          slicing[fixedVar]=slice(fixedVarLabel,fixedVarLabel+1)
        return numpy.squeeze(f[slicing])




  _factorClasses=[IndependentFactor]

  for factorClass in _factorClasses:

    class _FactorInjector(object):
        class __metaclass__(factorClass.__class__):
            def __init__(self, name, bases, dict):

                for b in bases:
                    if type(b) not in (self, type):
                        for k,v in dict.items():
                            setattr(b,k,v)
                return type.__init__(self, name, bases, dict)
                
    class _more_factor(_FactorInjector, factorClass):
      def __getitem__(self,labeling):
        """ get the value of factor for a given labeling:

        Example: ::

          >>> import opengm
          >>> gm=opengm.gm([3]*10)
          >>> f=opengm.pottsFunction(shape=[3,3,3],valueEqual=0.0,valueNotEqual=1.0)
          >>> fIndex=gm.addFactor( gm.addFunction(f) ,  [0,1,2] )
          >>> factor=gm[fIndex].asIndependentFactor()
          >>> # explicit labeling
          >>> factor[0,0,0]
          0.0
          >>> # list with labels
          >>> factor[[1,1,1]]
          0.0
          >>> # tuple with labels
          >>> factor[(0,1,2)]
          1.0
          >>> # numpy array with labels
          >>> factor[numpy.array([0,1,1] ,dtype=opengm.label_type)]
          1.0

        """
        if(isinstance(labeling, tuple)):
          return self._getitem(list(labeling))
        else:
          return self._getitem(labeling)
      def asNumpy(self):
        """ 
        get a copy of the factors value table as an numpy array 

        Example : :: 

          >>> import opengm
          >>> import numpy
          >>> unaries=numpy.random.rand(10, 10,2).astype(numpy.float32)
          >>> gm=opengm.grid2d2Order(unaries=unaries,regularizer=opengm.pottsFunction([2,2],0.0,0.4))
          >>> aFactor=gm[100].asIndependentFactor()
          >>> valueTable=aFactor.asNumpy()
          >>> valueTable.shape
          (2, 2)

        """
        return self.copyValuesSwitchedOrder().reshape(self.shape)

      def subFactor(self,fixedVars,fixedVarsLabels):
        """
        get the value table of of a sub-factor where some variables of
        the factor have been fixed to a given label

        Args:

          fixedVars : a 1d-sequence of variable indices to fix w.r.t. the factor

          fixedVarsLabels : a 1d-sequence of labels for the given indices in ``fixedVars``

        Example : :: 

          >>> import opengm
          >>> import numpy
          >>> unaries=numpy.random.rand(10, 10,4).astype(numpy.float32)
          >>> gm=opengm.grid2d2Order(unaries=unaries,regularizer=opengm.pottsFunction([4,4],0.0,0.4))
          >>> factor2Order=gm[100].asIndependentFactor()
          >>> int(factor2Order.numberOfVariables)
          2
          >>> print factor2Order.shape
          [4, 4, ]
          >>> # fix the second variable index w.r.t. the factor to the label 3
          >>> subValueTable = factor2Order.subFactor(fixedVars=[1],fixedVarsLabels=[3])
          >>> subValueTable.shape
          (4,)
          >>> for x in range(4):
          ...     print factor2Order[x,3]==subValueTable[x]
          True
          True
          True
          True
        """
        f = self.asNumpy()
        shape   = f.shape
        dim     = len(shape)
        slicing = [ slice (x) for x in shape ]

        for fixedVar,fixedVarLabel in zip(fixedVars,fixedVarsLabels):
          slicing[fixedVar]=slice(fixedVarLabel,fixedVarLabel+1)
        return numpy.squeeze(f[slicing])
     
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
      

      class FactorSubset(object):
        def __init__(self,gm,factorIndices):
          self.gm=gm
          self.factorIndices=factorIndices

        def numberOfVariables(self):
          return self.gm._factor_numberOfVariables(self.factorIndices)

        def evaluateLabeling(self,labels):
          if(labels.ndim==1):
            return self.gm._factor_evaluateGmLabeling(self.factorIndices,labels)
          elif(labels.ndim==2):
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
            raise RuntimeError(" dtype is not supported, so far only float32, float64, int64, uint64 and bool are supported")

        def withOrder(self,order):
          self._factor_withOrder(factorIndices,int(order))


      def factorSubset(self,factorIndices=None,order=None):
        if factorIndices is None:
          factorIndices=numpy.arange(self.numberOfFactors,dtype=index_type)

        if order is None:
          return gmClass.FactorSubset(self,factorIndices) 
        else :
          factorIndicesWithOrder=self._factor_withOrder(factorIndices,int(order))
          return gmClass.FactorSubset(self,factorIndicesWithOrder) 




      """
       .def _factor_withOrder


       .def _factor_scalarRetFunction_bool
       .def _factor_scalarRetFunction_uint64
       .def _factor_scalarRetFunction_int64
       .def _factor_scalarRetFunction_float32
       .def _factor_scalarRetFunction_float64
      """






      def vectorizedFactorFunction(self,function,factorIndices=None):
        """
        call a function for a sequence of factor 

        Args:

          function : a function which takes a factor as input

          factorIndices : a sequence of factor indices w.r.t. the graphial model
          
        Returns :

          a list with the results of each function call

        Example :
            >>> import opengm
            >>> gm=opengm.gm([2]*10)
            >>> # fill gm with factors...
            >>> result=gm.vectorizedFactorFunction(gm.factorClass.isSubmodular,range(gm.numberOfFactors))
        """
        if factorIndices is None :
          factorIndices = range(self.numberOfFactors)
        assert function is not None
        return map(lambda findex: function(self[findex]),factorIndices)

      def vectorizedFactorFunction2(self,function,factorIndices=None):
        """
        call a function for a sequence of factor 

        Args:

          function : a function which takes a factor as input

          factorIndices : a sequence of factor indices w.r.t. the graphial model
          
        Returns :

          a list with the results of each function call

        Example :
            >>> import opengm
            >>> gm=opengm.gm([2]*10)
            >>> # fill gm with factors...
            >>> result=gm.vectorizedFactorFunction(gm.factorClass.isSubmodular,range(gm.numberOfFactors))
        """
        if factorIndices is None :
          factorIndices = range(self.numberOfFactors)
        assert function is not None

        def f(findex):
          return function(self[findex])
        return map(f,factorIndices)


      """
      def isSubmodular(self,factors):
        def f(index):
          return self[index].isSubmodular()
        return map(lambda findex: self[findex].isSubmodular(),factors)
      """
      """
      def _map_factors(self,function,factors):
        pass
      def _caller_(self,index):
        return self[index]
      """

      def variableIndices(self,factorIndices):
        """ get the factor indices of all factors connected to variables within ``variableIndices`` 

        Args:

          factorIndices : factor indices w.r.t. the graphical model 

        Examples :

          >>> import opengm
          >>> import numpy
          >>> unaries=numpy.random.rand(2, 2,4).astype(numpy.float32)
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
          return self._variableIndices(factorIndices)
        else:
          return self._variableIndices(numpy.array(factorIndices,dtype=index_type)) 

      def factorIndices(self,variableIndices):
        """ get the factor indices of all factors connected to variables within ``variableIndices`` 

        Args:

          variableIndices : variable indices w.r.t. the graphical model 

        Examples :

          >>> import opengm
          >>> import numpy
          >>> unaries=numpy.random.rand(2, 2,4).astype(numpy.float32)
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
          return self._factorIndices(variableIndices)
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
          >>> unaries=numpy.random.rand(2, 2,4).astype(numpy.float32)
          >>> gm=opengm.grid2d2Order(unaries=unaries,regularizer=opengm.pottsFunction([4,4],0.0,0.4))
          >>> energy=gm.evaluate([0,2,2,1])
        """
        if isinstance(labels, numpy.ndarray):
          return self._evaluate_numpy(labels)
        elif isinstance(labels, list):
          return self._evaluate_list(labels)
        elif isinstance(labels, LabelVector):
          return self._evaluate_vector(labels)
        else:
          raise RuntimeError( "%s is not an supperted type for arument ``labels`` in ``evaluate``" %(str(type(labels)) ,) ) 

      def addFactor(self,fid,variableIndices):
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
        if isinstance(variableIndices, int):
          return self._addFactor(fid,[variableIndices])
        else:
          return self._addFactor(fid,variableIndices)

      def addFactors(self,fids,variableIndices):
        if isinstance(fids, FunctionIdentifier):
          fidVec=FidVector()
          fidVec.append(fids)
          fids=fidVec
        if isinstance(fids,list):
          fidVec=FidVector(fids)
          fids=fidVec
        if (isinstance(variableIndices,numpy.ndarray)):
          ndim=variableIndices.ndim
          if(ndim==1):
            return self._addUnaryFactors_vector_numpy(fids,variableIndices)
          elif(ndim==2):
            return self._addFactors_vector_numpy(fids,variableIndices)
        elif (isinstance(variableIndices,IndexVectorVector)):
          return self._addFactors_vector_vectorvector(fids,variableIndices)
        else :
          try :
            return self._addFactors_vector_numpy(fids,numpy.array(variableIndices,dtype=index_type))
          except:
            raise RuntimeError( "%s is not an supperted type for arument ``variableIndices`` in ``addFactors``" %(str(type(variableIndices)) ,)  ) 



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
            >>> f=numpy.array([0.8,1.4,0.1],dtype=numpy.float32)
            >>> fid=gm.addFunction(f)
            >>> print fid.functionIndex
            0
            >>> print fid.functionType
            0
            >>> # Add 2th-order function with  the shape [4,4]::
            >>> f=numpy.ones([4,4],dtype=numpy.float32)
            >>> #fill the function with values
            >>> #..........
            >>> fid=gm.addFunction(f)
            >>> int(fid.functionIndex),int(fid.functionType)
            (1, 0)
            >>> # Adding 3th-order function with the shape [4,5,2]::
            >>> f=numpy.ones([4,5,2],dtype=numpy.float32)
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
            :class:`opengm.AbsoluteDifferenceFunction`,
            :class:`opengm.SquaredDifferenceFunction`,
            :class:`opengm.TruncatedAbsoluteDifferenceFunction`, 
            :class:`opengm.TruncatedSquaredDifferenceFunction`, 
            :class:`opengm.PottsNFunction`, 
            :class:`opengm.PottsGFunction`
            :func:`opengm.pottsFunction`
            :func:`opengm.differenceFunction` 
            :func:`opengm.modelViewFunction` 
            :func:`opengm.randomFunction`
        """
        return self._addFunction(function)


      def addFunctions(self,functions):
        if isinstance(functions,numpy.ndarray):
          if functions.ndim==2:
            return self._addUnaryFunctions_numpy(functions)
          else:
            return self._addFunctions_numpy(functions)
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

   
  function_classes=[ExplicitFunction,SparseFunction,AbsoluteDifferenceFunction,SquaredDifferenceFunction,TruncatedAbsoluteDifferenceFunction,TruncatedSquaredDifferenceFunction,PottsFunction,PottsNFunction,PottsGFunction,PythonFunction]   
     
  for function_class in function_classes:

    #if hasattr(function_class, "__init__"):
    #  pass
    #if hasattr(function_class, "_raw_init_")==False:
    # function_class._raw_init_=function_class.__init__
    #assert hasattr(function_class, "_raw_init_")
    
    class InjectorGenericFunction(object):
        class __metaclass__(function_class.__class__):
            def __init__(self, name, bases, dict):
                for b in bases:
                    if type(b) not in (self, type):
                        for k,v in dict.items():
                            setattr(b,k,v)
                return type.__init__(self, name, bases, dict)


    class PyAddon_GenerricFunction(InjectorGenericFunction,function_class):
        def __copy__(self):
          return self.__class__(self)
        def __deepcopy__(self):
          return self.__class__(self)
        def __str__(self):
            " get a function as string "
            return self.asNumpyArray().__str__()
        def __repr__(self):
            " get a function representation as s string "
            return self.asNumpyArray().__repr__()

        def __getitem__(self,labels):
          """ get the values of a function for a given labeling

          Arg:

            labels : labeling has to be as long as the dimension of the function
          """ 
          if(isinstance(labels, tuple)):
            return self._getitem_tuple(labels)
          elif(isinstance(labels, list)):
            return self._getitem_list(labels)
          elif(isinstance(labels, numpy.ndarray)):
            return self._getitem_numpy(labels)
          else:
            return self._getitem_numpy(  numpy.array(labels,dtype=label_type))



    if function_class == SparseFunction :

      class PyAddon_SparseFunction(InjectorGenericFunction,function_class):
        def __setitem__(self,index,value):
          return self._setitem(index,value)

        @property 
        def defaultValue(self):
          """ Default value of the sparse function 

          Example:
              >>> import opengm
              >>> f=opengm.SparseFunction([2,2],1.0)
              >>> f.defaultValue
              1.0
          """
          return self._defaultValue()

        @property 
        def container(self):
          """ storage container of the sparse function which is a c++ std::map exported to python.
              The Interface of the container is very similar to the interface of a python dictonary.

          Example:
              >>> import opengm
              >>> f=opengm.SparseFunction([2,2],0.0)
              >>> c=f.container
              >>> len(c)
              0
              >>> f[1,0]=1.0
              >>> 0 in c
              False
              >>> 1 in c
              True
              >>> c[1]
              1.0
          """
          return self._container()

        def assignDense(self,ndarray,defaultValue):
          self.__init__(ndarray.shape,defaultValue)
          dimension=self.dimension
          nonDefaultCoords=numpy.where(ndarray!=defaultValue)
          nonDefaultValues=ndarray[nonDefaultCoords]
          if dimension == 1 :
             nonDefaultCoords=[nonDefaultCoords]
          numCoords=len(nonDefaultCoords[0])
          allCoords=numpy.ones([numCoords,dimension],dtype=numpy.uint64)
          for d in xrange(dimension):
             allCoords[:,d]=nonDefaultCoords[d]
          for c in xrange(numCoords):
             self[allCoords[c,:]]=nonDefaultValues[c]

        def keyToCoordinate(self,key,out=None):
          if out is None:
             out = numpy.ones(self.dimension,dtype=numpy.uint64)
          self._keyToCoordinateCpp(key,out)
          return out
          
        def coordinateToKey(self,coordinate):
          if isinstance(coordinate, numpy.ndarray):
            return self._coordinateToKey( list(coordinate) )
          else:
            return self._coordinateToKey( coordinate )

_extend_classes()


if __name__ == "__main__":
  import doctest
  import opengm
  doctest.testmod()
  #raise RuntimeError(" error")
  #doctest.run_docstring_examples(opengm.adder.GraphicalModel.addFactor, globals())