#from opengmcore import _opengmcore.adder as adder
from opengmcore   import *
from __version__                    import version
from functionhelper                 import *
from _inf_param                     import _MetaInfParam , InfParam
from _visu                          import visualizeGm
from _misc                          import defaultAccumulator

from __version__ import version
import time

from _inference_interface_generator import _inject_interface , InferenceBase

import inference
import hdf5


# initialize solver/ inference dictionaries
_solverDicts=[
   (inference.adder.minimizer.solver.__dict__ ,     'adder',       'minimizer' ),
   (inference.adder.maximizer.solver.__dict__,      'adder',       'maximizer' ),
   (inference.multiplier.integrator.solver.__dict__,'adder',       'integrator'),
   (inference.multiplier.minimizer.solver.__dict__, 'multiplier',  'minimizer' ),
   (inference.multiplier.maximizer.solver.__dict__, 'multiplier',  'maximizer' ),
   (inference.multiplier.integrator.solver.__dict__,'multiplier',  'integrator')
]
for infClass,infName in _inject_interface(_solverDicts): 
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




def saveGm(gm,f,d='gm'):
  """ save a graphical model to a hdf5 file:
  Args:
    gm : graphical model to save
    f  : filepath 
    g  : dataset (defaut : 'gm')
  """
  hdf5.saveGraphicalModel(f,d)

def loadGm(f,d='gm',operator='adder'):
  """ save a graphical model to a hdf5 file:
  Args:
    f  : filepath 
    g  : dataset (defaut : 'gm')
    operator : operator of the graphical model ('adder' / 'multiplier')
  """
  if(operator=='adder'):
    gm=adder.GraphicalModel()
  elif(operator=='multiplier'):
    gm=multiplier.GraphicalModel()
  else:
    raise RuntimeError("unknown operator: "+ operator)
  hdf5.loadGraphicalModel(gm,f,d)
  return gm




class TestModels(object):
  @staticmethod
  def thirdOrderChain(nVar,nLabels,operator='adder'):
    model=adder.GraphicalModel([nLabels]*nVar)
    unaries = numpy.random.rand(nVar,nLabels)
    model.addFactors(model.addFunctions(unaries),numpy.arange(nVar))

    numpy.random.seed(42)
    for x0 in range(nVar-2):
      f=numpy.random.rand(nLabels,nLabels,nLabels)
      model.addFactor(model.addFunction(f),[x0,x0+1,x0+2])
    return model






if __name__ == "__main__":
    pass