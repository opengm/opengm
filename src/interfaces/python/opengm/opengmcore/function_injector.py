from _opengmcore import ExplicitFunction,SparseFunction, \
                        TruncatedAbsoluteDifferenceFunction, \
                        TruncatedSquaredDifferenceFunction,PottsFunction,PottsNFunction, \
                        PottsGFunction,PythonFunction,\
                        ExplicitFunctionVector,SparseFunctionVector, \
                        TruncatedAbsoluteDifferenceFunctionVector, \
                        TruncatedSquaredDifferenceFunctionVector,PottsFunctionVector,PottsNFunctionVector, \
                        PottsGFunctionVector,PythonFunctionVector
import numpy



class FunctionType(object):
    pass


def isNativeFunctionType(f):
    return hasattr(f,'_opengm_native_function_vector_type')

def isNativeFunctionVectorType(f):
    return hasattr(f,'_opengm_native_function_type')


def _extend_function_vector_classes():
    function_vector_classes=[   ExplicitFunctionVector,SparseFunctionVector,
                                TruncatedAbsoluteDifferenceFunctionVector,
                                TruncatedSquaredDifferenceFunctionVector,PottsFunctionVector,
                                PottsNFunctionVector,PottsGFunctionVector,
                                PythonFunctionVector ]  

    for function_vector in function_vector_classes:
        class InjectorGenericFunctionVector(object):
            class __metaclass__(function_vector.__class__):
                def __init__(self, name, bases, dict):
                    for b in bases:
                        if type(b) not in (self, type):
                            for k,v in dict.items():
                                setattr(b,k,v)
                    return type.__init__(self, name, bases, dict)


        class PyAddon_GenerricFunctionVector(InjectorGenericFunctionVector,function_vector):
            @staticmethod
            def _opengm_native_function_vector_type(cls):
                pass



def _extend_function_type_classes():
  function_classes=[ExplicitFunction,SparseFunction,
                    TruncatedAbsoluteDifferenceFunction,
                    TruncatedSquaredDifferenceFunction,PottsFunction,
                    PottsNFunction,PottsGFunction,
                    PythonFunction]



     
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
        @staticmethod
        def _opengm_native_function_type(cls):
            pass


        def __copy__(self):
          return self.__class__(self)
        def __deepcopy__(self):
          return self.__class__(self)
        def __str__(self):
            " get a function as string "
            return numpy.array(self).__str__()
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



if __name__ == "__main__":
  import doctest
  import opengm
  doctest.testmod()
