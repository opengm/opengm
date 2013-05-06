from _opengmcore import  adder,multiplier ,IndependentFactor
import numpy

def _extend_factor_classes():

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
      def __array__(self):
        """ 
        get a copy of the factors value table as an numpy array 

        Example : :: 

          >>> import opengm
          >>> import numpy
          >>> unaries=numpy.random.rand(10, 10,2)
          >>> gm=opengm.grid2d2Order(unaries=unaries,regularizer=opengm.pottsFunction([2,2],0.0,0.4))
          >>> aFactor=gm[100]
          >>> valueTable=numpy.array(aFactor)
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
          >>> unaries=numpy.random.rand(10, 10,4)
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
        f = self.__array__()
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
      def __array__(self):
        """ 
        get a copy of the factors value table as an numpy array 

        Example : :: 

          >>> import opengm
          >>> import numpy
          >>> unaries=numpy.random.rand(10, 10,2)
          >>> gm=opengm.grid2d2Order(unaries=unaries,regularizer=opengm.pottsFunction([2,2],0.0,0.4))
          >>> aFactor=gm[100].asIndependentFactor()
          >>> valueTable=numpy.array(aFactor)
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
          >>> unaries=numpy.random.rand(10, 10,4)
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
        f = self.__array__()
        shape   = f.shape
        dim     = len(shape)
        slicing = [ slice (x) for x in shape ]

        for fixedVar,fixedVarLabel in zip(fixedVars,fixedVarsLabels):
          slicing[fixedVar]=slice(fixedVarLabel,fixedVarLabel+1)
        return numpy.squeeze(f[slicing])
     
   




if __name__ == "__main__":
  import doctest
  import opengm
  doctest.testmod()
  #raise RuntimeError(" error")
  #doctest.run_docstring_examples(opengm.adder.GraphicalModel.addFactor, globals())