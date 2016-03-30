import numpy

from opengmcore._opengmcore import (SparseFunction, 
                                    TruncatedAbsoluteDifferenceFunction,
                                    TruncatedSquaredDifferenceFunction,
                                    PottsFunction, PottsNFunction,
                                    PottsGFunction , PythonFunction,
                                    SparseFunctionVector, 
                                    TruncatedAbsoluteDifferenceFunctionVector,
                                    TruncatedSquaredDifferenceFunctionVector,
                                    PottsFunctionVector, PottsNFunctionVector,
                                    PottsGFunctionVector , PythonFunctionVector,
                                    
                                    )

from opengmcore import value_type,index_type,label_type



def pottsFunctions(shape,valueEqual,valueNotEqual):
  order = len(shape)
  numL0 = numpy.array([int(shape[0])],dtype=label_type)
  numL1 = numpy.array([int(shape[1])],dtype=label_type)

  if order == 2:
    return PottsFunctionVector(numL0,numL1,numpy.require(valueEqual,dtype=value_type),
                                      numpy.require(valueNotEqual,dtype=value_type) )
  elif order > 2:
    raise RuntimeError("not yet implemented")
  elif order < 2:
    raise RuntimeError("len(shape)>=2 is violated")


def relabeledPottsFunctions(shape, relabelings, valueEqual,valueNotEqual, dtype=value_type):
  order = len(shape)
  numL0 = int(shape[0])
  numL1 = int(shape[1])

def differenceFunctions(shape, norm, weight, truncate=None,dtype=value_type):
  order  = len(shape)
  numL0  = int(shape[0])
  numL1  = int(shape[1])
  lMax   = max(numL0,numL1)
  weight = numpy.require(weight,dtype=value_type)
  if truncate is None:
    truncate = numpy.ones(1,dtype=value_typ)*(abs(lMax)**norm)
  else :
    truncate = numpy.require(trunacte=dtype,valueType=value_type)

  if norm == 1 :
    return opengm.TruncatedAbsoluteDifferenceFunctionVector(shape=shape,weigh=weight,truncate=truncate)
  elif norm == 2 :
    return opengm.SquaredAbsoluteDifferenceFunctionVector(shape=shape,weigh=weight,truncate=truncate)
  else :
    return opengm.explicitDifferenceFunction(shape=shape,weigh=weight,truncate=truncate)


def relabeledDifferenceFunctions(shape, relabelings, norm=2, weight=1.0,truncate=None, dtype=value_type):
  order = len(shape)
  numL0 = int(shape[0])
  numL1 = int(shape[1])

def randomFunctions(shape):
  order = len(shape)
  numL0 = int(shape[0])
  numL1 = int(shape[1])

def sparseFunctions(*args,**kwargs):
  order = len(shape)
  numL0 = int(shape[0])
  numL1 = int(shape[1])


def sparseFunction(*args,**kwargs):
  pass

def randomFunction(shape):
    tshape = tuple(x for x in shape)
    return numpy.random.rand(*tshape).astype(value_type)


def pottsFunction(shape, valueEqual=0.0, valueNotEqual=1.0):
    """
    factory function to generate a potts-function

    Args:
      shape : shape of the potts-functions

      valueEqual : value if all labels are valueEqual

      valueNotEqual : value if not all labels are valueEqual

    Returns:

      :class:`opengm.PottsFunction` if ``len(shape) == 2``

      :class:`opengm.PottsNFunction` if ``len(shape) > 2``

    Example: ::

      >>> import opengm
      >>> f = opengm.pottsFunction(shape=[2,2],valueEqual=0.0,valueNotEqual=1.0)
      >>> print "f[0,0]=%.1f" % (f[0,0],)
      f[0,0]=0.0
      >>> print "f[1,0]=%.1f" % (f[1,0],)
      f[1,0]=1.0
      >>> print "f[0,1]=%.1f" % (f[0,1],)
      f[0,1]=1.0
      >>> print "f[1,1]=%.1f" % (f[1,1],)
      f[1,1]=0.0
      >>> f = opengm.pottsFunction(shape=[3,3,3],valueEqual=0.0,valueNotEqual=1.0)
      >>> print "f[0,0,0]=%.1f" % (f[0,0,0],)
      f[0,0,0]=0.0
      >>> print "f[1,0,0]=%.1f" % (f[1,0,0],)
      f[1,0,0]=1.0
      >>> print "f[0,1,0]=%.1f" % (f[0,1,0],)
      f[0,1,0]=1.0
      >>> print "f[1,1,2]=%.1f" % (f[1,1,2],)
      f[1,1,2]=1.0
      >>> print "f[2,2,2]=%.1f" % (f[2,2,2],)
      f[2,2,2]=0.0

    .. seealso::
      :class:`opengm.PottsFunction` ,:class:`opengm.PottsNFunction`
    """
    order = len(shape)
    if(order == 2):
        return PottsFunction(shape, valueEqual, valueNotEqual)
    elif(order > 2):
        return PottsNFunction(shape, valueEqual, valueNotEqual)


def relabeledPottsFunction(shape, relabelings, valueEqual=0.0,
                           valueNotEqual=1.0, dtype=value_type):
    """Factory function to construct a numpy array which encodes a
    potts-function. The labelings on which the potts function is computed are
    given by relabelings

    Keyword arguments:

       shape : shape / number of of labels of the potts-function

       relabelings : a list of relabelings for the 2 variables

       valueEqual  : value if labels are equal (default : 0.0)

       valueNotEqual : value if labels are not valueEqual (default : 1.0)

       dtype : data type of the numpy array (default : value_type)

    get a potts-function ::

       >>> import opengm
       >>> f=opengm.relabeledPottsFunction(shape=[4,3],relabelings=[[4,2,3,5],[2,4,5]],valueEqual=0.0,valueNotEqual=1.0)
       >>> f[0,0] # relabling => 4,2
       1.0
       >>> f[0,1] # relabling => 4,1
       0.0

    Returns:
       a numpy array with ``dtype`==value_type``

    """
    assert len(shape) == 2
    assert len(relabelings) == 2
    assert len(relabelings[0]) == shape[0]
    assert len(relabelings[1]) == shape[1]
    f = numpy.empty(shape, dtype=dtype)
    f[:] = valueNotEqual
    rl1 = relabelings[0]
    rl2 = relabelings[1]
    for x in range(shape[0]):
        for y in range(shape[1]):
            if(rl1[x] == rl2[y]):
                f[x, y] = valueEqual

    return f


def differenceFunction(shape, norm=2, weight=1.0, truncate=None,
                       dtype=value_type):
    """Factory function to construct a numpy array which encodes a
    difference-function.  The difference can be of any norm (1,2,...) and can
    be truncated or untruncated.

    Keyword arguments:

    shape -- shape / number of of labels of the potts-function

    weight  -- weight which is multiplied to the norm

    truncate -- truncate all values where the norm is bigger than truncate

    dtype -- data type of the numpy array

    Example: ::
       >>> import opengm
       >>> f=opengm.differenceFunction([2,4],weight=0.5,truncate=5)

    """
    assert len(shape) == 2
    if norm == 1:
        if truncate is None:
            # BUG: undefined function
            return TruncatedAbsoluteDifferenceFunction(shape,truncate=float(max(shape[0],shape[1])), weight=float(weight))
        else:
            return TruncatedAbsoluteDifferenceFunction(shape, truncate=float(truncate), weight=float(weight))
    elif norm == 2:
        if truncate is None:
            return TruncatedSquaredDifferenceFunction(shape,truncate=float( max(shape[0],shape[1])**2), weight=float(weight))
        else:
            return TruncatedSquaredDifferenceFunction(shape, truncate=float(truncate), weight=float(weight))
    else:
        f = numpy.empty(shape, dtype=dtype)
        if shape[0] < shape[1]:
            yVal = numpy.arange(0, shape[1])
            for x in range(shape[0]):
                f[x, :] = (numpy.abs(x - yVal) ** norm)
        else:
            xVal = numpy.arange(0, shape[0])
            for y in range(shape[1]):
                f[:, y] = (numpy.abs(xVal - y) ** norm)
        if truncate is not None:
            f[numpy.where(f > truncate)] = truncate
        f *= weight
        return f


def relabeledDifferenceFunction(shape, relabelings, norm=2, weight=1.0,
                                truncate=None, dtype=value_type):
    """Factory function to construct a numpy array which encodes a
    difference-function.  The difference can be of any norm (1,2,...) and can
    be truncated or untruncated.  The labelings on which the potts function is
    computed are given by relabelings

    Keyword arguments:

    shape -- shape / number of of labels of the potts-function

    weight  -- weight which is multiplied to the norm

    truncate -- truncate all values where the norm is bigger than truncate

    dtype -- data type of the numpy array

    get a truncated squared difference function ::
       >>> import opengm
       >>> f=opengm.relabeledDifferenceFunction([2,4],[[1,2],[2,3,4,5]],weight=0.5,truncate=5)
    """
    assert len(shape) == 2
    f = numpy.empty(shape, dtype=dtype)
    if shape[0] < shape[1]:
        rl1 = relabelings[0]
        yVal = numpy.array(relabelings[1])
        for x in range(shape[0]):
            f[x, :] = (numpy.abs(rl1[x] - yVal) ** norm)
    else:
        rl2 = relabelings[1]
        xVal = numpy.array(relabelings[2])
        for y in range(shape[1]):
            f[:, y] = (numpy.abs(xVal - rl2[y]) ** norm)
    if truncate is not None:
        f[numpy.where(f > truncate)] = truncate
    f *= weight
    return f




def labelSumFunction(shape,allowedLabelSum,valueForbidden=1.0,valueAllowed=0.0):
    ranges=[]
    for s in shape:
        ranges.append(numpy.arange(s))

    mgRes = numpy.meshgrid(*ranges)
    f     = numpy.zeros(shape=shape,dtype=value_type)

    for r in mgRes:
        f+=r

    whereAllowed = numpy.where(f==allowedLabelSum)
    f[:]            = valueForbidden
    f[whereAllowed] = valueAllowed

    return f










if __name__ == "__main__":
    import doctest
    doctest.testmod()
