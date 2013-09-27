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




class TestModels(object):
  @staticmethod
  def chain3(nVar,nLabels):
    model=adder.GraphicalModel([nLabels]*nVar)
    unaries = numpy.random.rand(nVar,nLabels)
    model.addFactors(model.addFunctions(unaries),numpy.arange(nVar))

    numpy.random.seed(42)
    for x0 in range(nVar-2):
      f=numpy.random.rand(nLabels,nLabels,nLabels)
      model.addFactor(model.addFunction(f),[x0,x0+1,x0+2])
    return model

  @staticmethod
  def chain4(nVar,nLabels):
    model=adder.GraphicalModel([nLabels]*nVar)
    unaries = numpy.random.rand(nVar,nLabels)
    model.addFactors(model.addFunctions(unaries),numpy.arange(nVar))

    numpy.random.seed(42)
    for x0 in range(nVar-3):
      f=numpy.random.rand(nLabels,nLabels,nLabels,nLabels)
      model.addFactor(model.addFunction(f),[x0,x0+1,x0+2,x0+3])
    return model

  @staticmethod
  def chainN(nVar,nLabels,order,nSpecialUnaries=0,beta=1.0):
    model=adder.GraphicalModel([nLabels]*nVar)
    unaries = numpy.random.rand(nVar,nLabels)

    for sn in range(nSpecialUnaries):
      r=int(numpy.random.rand(1)*nVar-1)
      rl=int(numpy.random.rand(1)*nLabels-1)

      unaries[r,rl]=0.0  

    model.addFactors(model.addFunctions(unaries),numpy.arange(nVar))

    numpy.random.seed(42)
    for x0 in range(nVar-(order-1)):
      f=numpy.random.rand( *([nLabels]*order))
      f*=beta
      vis=numpy.arange(order)
      vis+=x0

      model.addFactor(model.addFunction(f),vis)
    return model


  @staticmethod
  def secondOrderGrid(dx,dy,nLabels):
    nVar=dx*dy
    model=adder.GraphicalModel([nLabels]*nVar)
    unaries = numpy.random.rand(nVar,nLabels)
    model.addFactors(model.addFunctions(unaries),numpy.arange(nVar))

    vis2Order=secondOrderGridVis(dx,dy,True)

    nF2=len(vis2Order)#.shape[0]
    f2s=numpy.random.rand(nF2,nLabels)

    model.addFactors(model.addFunctions(f2s),vis2Order)

    return model

    











class __Crusher__(object):
    def __init__(self,gm,accumulator=None,parameter=InfParam()):
        if accumulator is None:
            self.accumulator=defaultAccumulator(gm=gm)
        else:
            self.accumulator=accumulator
        kwargs=parameter.kwargs
        self.gm_=gm
        self.accumulator=accumulator
        self.phase1Inf  = kwargs.pop('phase1Inf', inference.SelfFusion)
        self.phase1InfP = kwargs.pop('phase1InfP', InfParam(
            maxSubgraphSize=3,toFuseInf='gibbs',
            infParam=InfParam(steps=100)
            )
        )

        self.phase2Inf  = kwargs.pop('phase2Inf', inference.LazyFlipper)
        self.phase2InfP = kwargs.pop('phase2InfP', InfParam(maxSubgraphSize=10))


        print self.phase1Inf

    def infer(self,visitor=None):

        arg=numpy.ones(self.gm_.numberOfVariables)
        print self.gm_.evaluate(arg)

        print "phase 1 "
        phase1Inf=self.phase1Inf(gm=self.gm_,accumulator=self.accumulator,parameter=self.phase1InfP)
        phase1Inf.infer()
        arg=phase1Inf.arg()
        print self.gm_.evaluate(arg)


        print "phase 2 "
        phase2Inf=self.phase2Inf(gm=self.gm_,accumulator=self.accumulator,parameter=self.phase2InfP)
        phase2Inf.setStartingPoint(arg)
        phase2Inf.infer()
        arg=phase2Inf.arg()
        print self.gm_.evaluate(arg)





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

    def getValues(self):
        return self.values_
    def getTimes(self):
        assert self.runtimes_ is not None
        return self.runtimes_
    def getBounds(self):
        return self.bounds_
    def getIterations(self):
        return self.iterations_




class __ChainedInf__(object):
    def __init__(self,gm,accumulator=None,parameter=InfParam()):
        print "fresh constructor "
        if accumulator is None:
            self.accumulator=defaultAccumulator(gm=gm)
        else:
            self.accumulator=accumulator
        kwargs=parameter.kwargs
        self.gm_=gm


        self.solverList    = kwargs.get('solvers', [])
        self.parameterList = kwargs.get('parameters', [])

        self.arg_ = numpy.zeros(gm.numberOfVariables,dtype=numpy.uint64)

    def timingVisitor(self,visitNth=1,reserve=0,verbose=True,multiline=True):
        return GenericTimingVisitor(visitNth,reserve,verbose,multiline)


    def infer(self,visitor=None):
        
        print "CINNNNF"
        for index,(cls,infParm) in enumerate(zip(self.solverList,self.parameterList)):

            print  "construct solver"
            solver=cls(gm=self.gm_,accumulator=self.accumulator,parameter=infParm)
            print "inference"
            solverTv=solver.timingVisitor(verbose=False)

            if(index>0):
                solver.setStartingPoint(self.arg_)

            solver.infer(solverTv)
            self.arg_=solver.arg()

            if(index==0):
                print "first solver"
                visitor.values_     =solverTv.getValues()
                visitor.runtimes_   =solverTv.getTimes()
                visitor.bounds_     =solverTv.getBounds()
                visitor.iterations_ =solverTv.getIterations()
            else:
                print "NOOOOOT first solver"
                assert visitor.runtimes_ is not None
                visitor.values_     =numpy.append(visitor.values_,     solverTv.getValues())
                visitor.runtimes_   =numpy.append(visitor.runtimes_,   solverTv.getTimes())
                visitor.bounds_     =numpy.append(visitor.bounds_,     solverTv.getBounds())
                visitor.iterations_ =numpy.append(visitor.iterations_, solverTv.getIterations())
            assert visitor.runtimes_ is not None
        print "CINNNNF DOOOOONE"
        print "da rt",visitor.runtimes_[0]
    def arg(self):
        return self.arg_

    def value(self):
        return self.gm_.evaluate(self.arg_)







inference.__dict__['Crusher']=__Crusher__
inference.__dict__['ChainedInf']=__ChainedInf__

if __name__ == "__main__":
    pass