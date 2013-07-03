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
