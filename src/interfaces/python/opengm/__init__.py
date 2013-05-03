#from opengmcore import _opengmcore.adder as adder
from opengmcore   import *
from __version__                    import version
from functionhelper                 import pottsFunction, relabeledPottsFunction, differenceFunction, relabeledDifferenceFunction,randomFunction
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





class InfAndFlip(InferenceBase):

    def __init__(self,gm,accumualtor,parameter):
        self.gm=gm
        self.accumualtor=accumualtor
        self._value=None
        self._bound=None
        self._arg=None


        kwargs = parameter.kwargs
        self._mainSolverClass     = kwargs.get('mainSolver',opengm.inference.BeliefPropagation)
        self._mainSolverParam= kwargs.get('mainSolverParameter',None)

        # construct main solver
        self.mainSolver=self._mainSolverClass(gm,accumualtor,self._mainSolverParam)

        # construct lazy flipper
        maxSubgraphSize =kwargs.get('maxSubgraphSize',2)
        self.lf=opengm.inference.LazyFlipper(gm,accumualtor,
                                             opengm.InfParam(maxSubgraphSize=maxSubgraphSize))

        # construct gibbs
        maxSubgraphSize =kwargs.get('steps',2)
        self.gibbsParam=opengm.InfParam(
            steps= kwargs.get("steps",1e8),
            useTemp= kwargs.get("useTemp",True),
            tempMin= kwargs.get("tempMin",0.0000001),
            tempMax= kwargs.get("tempMax",0.001),
            periodeLength= kwargs.get("periodeLength",1e7)
        )
        self.gibbs=opengm.inference.Gibbs(gm,accumualtor,self.gibbsParam)

    def infer(self,visitor=None):

        print "main inference ..."
        inf=self.mainSolver
        inf.infer()
        print "done ..."
        self._arg=inf.arg()
        self.__updateValue(inf.value())
        self.__updateBound(inf.value())
        self._bound=inf.bound()
        self.callVisitor()


        counter=0
        while(True):
            print counter
            print "lf inference ..."
            inf=self.lf
            if counter == 0:
                inf.reset()
            inf.setStartingPoint(self._arg)
            inf.infer()
            print "done ..."
            self._arg=inf.arg()
            self.__updateValue(inf.value())
            self.__updateBound(inf.value())
            self.callVisitor()


            print "gibbs inference ..."
            inf=self.gibbs
            if counter == 0:
                inf.reset()
            inf.setStartingPoint(self._arg)
            inf.infer(inf.verboseVisitor(10000,False))
            print "done ..."

            gibbsValue=inf.value()

            print "gibbs value ",gibbsValue

            self.callVisitor()
            if self.__betterValue(gibbsValue):

                self._arg=inf.arg()
                self.__updateValue(inf.value())
                self.__updateBound(inf.value())
                
                counter+=1
            else:
                break



    def __betterValue(self,newValue):
        if self.accumualtor=='minimizer':
            return newValue < self._value
        elif self.accumualtor =='maximizer':
            return newValue > self._value


    def __updateValue(self,newValue):
        if self._value is None:
            self._value=newValue
        elif self.accumualtor == 'minimizer':
            assert newValue <= self._value
            self._value=newValue
        elif self.accumualtor == 'maximizer':
            assert newValue >= self._value
            self._value=newValue

    def __updateBound(self,newBound):
        if self._bound is None:
            self._bound=newBound
        elif self.accumualtor == 'minimizer':
            if  newBound >= self._value:
                self._value=newBound
        elif self.accumualtor == 'maximizer':
            if newBound <= self._value :
                self._value=newBound

    def callVisitor(self,msg=""):
        print "value: ",self._value," bound: ",self._bound

    def arg(self):
        return self._arg



if __name__ == "__main__":




    import doctest
    doctest.testmod()


    import opengm 

    numLabels=10
    unaries=numpy.random.rand(200 , 200,numLabels)
    potts=opengm.PottsFunction([numLabels]*2,0.0,0.3)
    gm=opengm.grid2d2Order(unaries=unaries,regularizer=potts)


    mainInfParam  = opengm.InfParam(steps=10,damping=0.5)
    infParam      = opengm.InfParam(
                        mainSolver=opengm.inference.BeliefPropagation,
                        mainSolverParameter=mainInfParam,
                        maxSubgraphSize=2
                    )


    inf      = InfAndFlip(gm=gm,accumualtor='minimizer',parameter=infParam)
    inf.infer()