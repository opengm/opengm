#from opengmcore import _opengmcore.adder as adder
from opengmcore   import *
from __version__                    import version
from functionhelper                 import pottsFunction, relabeledPottsFunction, differenceFunction, relabeledDifferenceFunction,randomFunction
from _inf_param                     import _MetaInfParam , InfParam
from _visu                          import visualizeGm
from _misc                          import defaultAccumulator
#import version 
from __version__ import version
#from functionhelper import *
import time

from _inference_interface_generator import _inject_interface , InferenceBase

import inference
import hdf5

"""
import sys
import types
import numpy
import inspect
from cStringIO import StringIO
"""
         

# initialize solver/ inference dictionaries
_minSum  = inference.adder.minimizer.solver.__dict__ 
_maxSum  = inference.adder.maximizer.solver.__dict__ 
_minProd = inference.multiplier.minimizer.solver.__dict__ 
_maxProd = inference.multiplier.maximizer.solver.__dict__ 


_solverDicts=[
   (_minSum, 'adder',       'minimizer'),
   (_maxSum, 'adder',       'maximizer'),
   (_minProd,'multiplier',  'minimizer'),
   (_maxProd,'multiplier',  'maximizer')
]


_result=_inject_interface(_solverDicts)

for infClass,infName in _result: 
  inference.__dict__[infName]=infClass




class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        if self.name:
            print '[%s]' % self.name
        self.tstart = time.time()


    def __exit__(self, type, value, traceback):
        #if self.name:
        #    print '[%s]' % self.name,
        print '   Elapsed: %s' % (time.time() - self.tstart)



"""
with opengm.Timer("compute variable indices for higher order factors"):
    # Compute a numpy array which holds the variable indices for
    # all second order variable indices

    # arrays filles with x and y coordinates
    xv, yv = numpy.meshgrid(numpy.arange(0,shape[0]), numpy.arange(0,shape[1]))
    # for horizontal factors (remove last column since it has no right neighbour)
    xh = xv[:,0:-1].reshape(-1)
    yh = yv[:,0:-1].reshape(-1)
    # for vertical factors (remove last row since it has no lower neighbour)
    xv = xv[0:-1,:].reshape(-1)
    yv = yv[0:-1,:].reshape(-1)

    # increment coordinates of right and lower neighbours
    xhN = xh +1
    yvN = yv +1

    # compute variable indices from coordinates:
    # -viH for pixel itself (for horizontal factors)
    # -viH for pixel itself (for vertical factors)
    # -viHN right neighbour pixel vi (horizontal)
    # -viVN lower neighbour pixel vi (vertical)
    viH  = xh  * shape[1] + yh
    viV  = xv  * shape[1] + yv
    viHN = xhN * shape[1] + yh
    viVN = xv  * shape[1] + yvN

    # combine pixel vi with neighbour pixel vi
    visH=numpy.array([viH,viHN])
    visV=numpy.array([viV,viVN])
    # combine horizontal and vertical vis into one array
    vis = numpy.hstack([visH,visV]).T
    vis = numpy.require(vis,dtype=opengm.index_type)
"""

if __name__ == "__main__":
    import doctest
    doctest.testmod()
