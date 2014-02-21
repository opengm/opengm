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
import benchmark

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



class GenericTimingVisitor(object):
    def __init__(self,visitNth=1,reserve=0,verbose=True,multiline=True):
        self.visitNth=visitNth
        self.reserve=reserve
        self.verbose=verbose
        self.multiline=multiline

        self.values_     = None
        self.runtimes_   = None
        self.bounds_     = None
        self.iterations_ = None

        self.t0          = None
        self.t1          = None
        self.iterNr      = 0
    def getValues(self):
        return numpy.require(self.values_,dtype=value_type)
    def getTimes(self):
        return numpy.require(self.runtimes_,dtype=value_type)
    def getBounds(self):
        return numpy.require(self.bounds_,dtype=value_type)
    def getIterations(self):
        return numpy.require(self.iterations_,dtype=value_type)

    def begin(self,inf):
      v = inf.value()
      b = inf.bound()


      self.values_    =[v]
      self.bounds_    =[b]
      self.runtimes_  =[0.0]
      self.iterations_=[self.iterNr]
      if self.verbose :
        print 'Begin :        %d  Value : %f  Bound : %f '%(self.iterNr,v,b)




      # start the timing
      self.t0         =time.time()
      self.t1         =time.time()
    def visit(self,inf):

      if(self.iterNr==0 or self.iterNr%self.visitNth==0):

        # "stop the timing"
        self.t1=time.time()

        # get the runtime of the run
        rt=self.t1-self.t0
        v = inf.value()
        b = inf.bound()
        if self.verbose :
          print 'Step  :        %d  Value : %f  Bound : %f '%(self.iterNr,v,b)


        # store results
        self.values_.append(v)
        self.bounds_.append(b)
        self.runtimes_.append(rt)
        self.iterations_.append(self.iterNr)

        # increment iteration number
        self.iterNr+=1

        # restart the timing
        self.t0=time.time()

      else:
        # increment iteration number
        self.iterNr+=1





    def end(self,inf):
        # "stop the timing"
        self.t1=time.time()

        # get the runtime of the run
        rt=self.t1-self.t0
        v = inf.value()
        b = inf.bound()
        if self.verbose :
          print 'End :        %d  Value : %f  Bound : %f '%(self.iterNr,v,b)
        # store results
        self.values_.append(v)
        self.bounds_.append(b)
        self.runtimes_.append(rt)
        self.iterations_.append(self.iterNr)



class __RandomFusion__(object):
    def __init__(self,gm,accumulator=None,parameter=InfParam()):

        if accumulator is None:
            self.accumulator=defaultAccumulator(gm=gm)
        else:
            self.accumulator=accumulator
        kwargs=parameter.kwargs
        self.gm_=gm


        self.steps = kwargs.get('steps', 100)
        self.fusionSolver = kwargs.get('fuisionSolver', 'lf2')

        self.arg_  = None
        self.value_ = None

        self.fusionMover=inference.adder.minimizer.FusionMover(self.gm_)

        self.nLabels = self.gm_.numberOfLabels(0)
        self.nVar    = self.gm_.numberOfVariables

    def timingVisitor(self,visitNth=1,reserve=0,verbose=True,multiline=True):
        return GenericTimingVisitor(visitNth,reserve,verbose,multiline)

    def setStartingPoint(self,arg):
      self.arg_=arg
      self.value_=gm.evaluate(self.arg_)

    def infer(self,visitor=None):
        if(self.arg_ is None):
          self.arg_ = numpy.zeros(self.gm_.numberOfVariables,dtype=label_type)
          self.value_ = self.value_=self.gm_.evaluate(self.arg_)

        # start inference
        if visitor is not None:
          visitor.begin(self)

        # start fusion moves
        for x in range(self.steps):
          randState=numpy.random.randint(low=0, high=self.nLabels, size=self.nVar).astype(label_type)
          r = self.fusionMover.fuse(self.arg_,randState,self.fusionSolver)
          self.arg_=r[0]
          self.value_=r[1]
          visitor.visit(self)


        # end inference
        if visitor is not None:
          visitor.end(self)

    def name(self):
      return "RandomFusion"

    def bound(self):
      return -1.0*float('inf')

    def arg(self):
        return self.arg_

    def value(self):
        return self.value_



class __CheapInitialization__(object):
    def __init__(self,gm,accumulator=None,parameter=InfParam()):

        if accumulator is None:
            self.accumulator=defaultAccumulator(gm=gm)
        else:
            self.accumulator=accumulator
        kwargs=parameter.kwargs
        self.gm_=gm

        self.arg_  = None
        self.value_ = None
        self.initType = kwargs.get('initType', 'localOpt')


    def timingVisitor(self,visitNth=1,reserve=0,verbose=True,multiline=True):
        return GenericTimingVisitor(visitNth,reserve,verbose,multiline)

    def setStartingPoint(self,arg):
      self.arg_=arg
      self.value_=gm.evaluate(self.arg_)

    def infer(self,visitor=None):
        if(self.arg_ is None):
          self.arg_ = numpy.zeros(self.gm_.numberOfVariables,dtype=label_type)
          self.value_ = self.value_=self.gm_.evaluate(self.arg_)

        # start inference
        if visitor is not None:
          visitor.begin(self)
        
        if(self.initType=='localOpt'):
          print "move local opt"
          self.arg_ = self.gm_.moveLocalOpt('minimizer')
          print "done"
          visitor.visit(self)

        # end inference
        if visitor is not None:
          visitor.end(self)

    def name(self):
      return "CheapInitialization"

    def bound(self):
      return -1.0*float('inf')

    def arg(self):
        return self.arg_

    def value(self):
        return self.value_






inference.__dict__['RandomFusion']=__RandomFusion__


if __name__ == "__main__":
    pass