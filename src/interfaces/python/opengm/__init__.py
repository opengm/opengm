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


_result=_inject_interface(solverDicts)

for infClass,infName in _result: 
  inference.__dict__[infName]=infClass








if __name__ == "__main__":
    import doctest
    doctest.testmod()
