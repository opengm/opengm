from _opengmcore import *
from factorSubset import FactorSubset
from gm_injector import _extend_gm_classes
from factor_injector import _extend_factor_classes
from function_injector import _extend_function_type_classes,\
                              _extend_function_vector_classes,\
                              isNativeFunctionType,\
                              isNativeFunctionVectorType
from dtypes import index_type,value_type,label_type
from printing import prettyValueTable
import numpy

configuration=OpengmConfiguration()
LabelVector=IndexVector


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
   if isinstance(numberOfLabels,numpy.ndarray):
      numL=numpy.require(numberOfLabels,dtype=label_type)
   else:
      numL=numberOfLabels
   if operator=='adder' :
      return adder.GraphicalModel(numL,reserveNumFactorsPerVariable)
   elif operator=='multiplier' :
      return multiplier.GraphicalModel(numL,reserveNumFactorsPerVariable)
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
  return PythonFunction( _ModelViewFunction(factor) ,factor.shape.__tuple__())

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
      >>> unaries=numpy.random.rand(10, 10,2)
      >>> gridGm=opengm.grid2d2Order(unaries=unaries,regularizer=opengm.pottsFunction([2,2],0.0,0.4))
      >>> int(gridGm.numberOfVariables)
      100

   """
   shape=unaries.shape
   assert(len(shape)==3)
   numLabels=shape[2]
   numVar=shape[0]*shape[1]
   numFactors=(shape[0]-1)*shape[1] + (shape[1]-1)*shape[0] +numVar
   numberOfLabels=numpy.ones(numVar,dtype=numpy.uint64)*numLabels
   gm=graphicalModel(numberOfLabels,operator=operator)
   gm.reserveFunctions(numVar+1,'explicit')
   gm.reserveFactors(numFactors)
   # add unaries
   unaries2d=unaries.reshape([numVar,numLabels])
   #fids=
   
   #vis=
   gm.addFactors( gm.addFunctions(unaries2d),numpy.arange(0,numVar,dtype=numpy.uint64),finalize=False)

   # add 2-order function
   vis2Order=secondOrderGridVis(shape[0],shape[1],bool(order=='numpy'))
   fid2Order=gm.addFunction(regularizer)
   fids=FidVector()
   fids.append(fid2Order)
   gm.addFactors(fids,vis2Order,finalize=False)
   gm.finalize()
   return gm


# the following is to enable doctests of pure boost::python classes
# if there is a smarter way, let me know
_GmAdder                             = adder.GraphicalModel
_GmMultiplier                        = multiplier.GraphicalModel
_FactorAdder                         = adder.Factor
_FactorMultiplier                    = multiplier.Factor
_ExplicitFunction                    = ExplicitFunction
_SparseFunction                      = SparseFunction
_TruncatedAbsoluteDifferenceFunction = TruncatedAbsoluteDifferenceFunction
_TruncatedSquaredDifferenceFunction  = TruncatedSquaredDifferenceFunction
_PottsFunction                       = PottsFunction
_PottsNFunction                      = PottsNFunction
_PottsGFunction                      = PottsGFunction
_PythonFunction                      = PythonFunction
_FactorSubset                        = FactorSubset


_extend_gm_classes()
_extend_factor_classes()
_extend_function_type_classes()
_extend_function_vector_classes()

if __name__ == "__main__":
  import doctest
  import opengm
  doctest.testmod()
  #raise RuntimeError(" error")
  #doctest.run_docstring_examples(opengm.adder.GraphicalModel.addFactor, globals())