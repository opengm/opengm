from _opengmcore import *
import numpy

class Adder:
   def neutral(self):
      return float(0.0)
  

class Multiplier:
   def neutral(self):
      return float(1.0)

adder.Operator = Adder
adder.Operator.neutral = Adder.neutral
multiplier.Operator = Multiplier


BoostPythonMetaclassIFactor = IndependentFactor.__class__
BoostPythonMetaclassAdderFactor = adder.Factor.__class__
BoostPythonMetaclassMultiplierFactor = multiplier.Factor.__class__
MetaGmAdder = adder.GraphicalModel.__class__
MetaGmMult = multiplier.GraphicalModel.__class__

class IFactorInjector(object):
    class __metaclass__(BoostPythonMetaclassIFactor):
        def __init__(self, name, bases, dict):

            for b in bases:
                if type(b) not in (self, type):
                    for k,v in dict.items():
                        setattr(b,k,v)
            return type.__init__(self, name, bases, dict)

class FactorInjectorAdder(object):
    class __metaclass__(BoostPythonMetaclassAdderFactor):
        def __init__(self, name, bases, dict):

            for b in bases:
                if type(b) not in (self, type):
                    for k,v in dict.items():
                        setattr(b,k,v)
            return type.__init__(self, name, bases, dict)
            
class FactorInjectorMultiplier(object):
    class __metaclass__(BoostPythonMetaclassMultiplierFactor):
        def __init__(self, name, bases, dict):

            for b in bases:
                if type(b) not in (self, type):
                    for k,v in dict.items():
                        setattr(b,k,v)
            return type.__init__(self, name, bases, dict)

class InjectorGmAdder(object):
    class __metaclass__(MetaGmAdder):
        def __init__(self, name, bases, dict):

            for b in bases:
                if type(b) not in (self, type):
                    for k,v in dict.items():
                        setattr(b,k,v)
            return type.__init__(self, name, bases, dict)
            
class InjectorGmMult(object):
    class __metaclass__(MetaGmMult):
        def __init__(self, name, bases, dict):

            for b in bases:
                if type(b) not in (self, type):
                    for k,v in dict.items():
                        setattr(b,k,v)
            return type.__init__(self, name, bases, dict)
            
class ifactor(IFactorInjector,IndependentFactor):
    def __init__(self):
        assert(False)
    def asNumpy(self):
        return self.copyValuesSwitchedOrder().reshape(self.shape)            
class adder_factor(FactorInjectorAdder, adder.Factor):
    def __init__(self):
        assert(False)
    def asNumpy(self):
        return self.copyValuesSwitchedOrder().reshape(self.shape)

class multiplier_factor(FactorInjectorMultiplier, multiplier.Factor):
    def __init__(self):
        assert(False)
    def asNumpy(self):
        return self.copyValuesSwitchedOrder().reshape(self.shape)
            
# adder_gm(InjectorGmAdder, adder.GraphicalModel):
#    def addFunction(self,ndarray):
#        return self.addFunctionRaw(ndarray)

#class multiplier_gm(InjectorGmMult, multiplier.GraphicalModel):
#    def addFunction(self,ndarray):
#        return self.addFunctionRaw(ndarray)

        
        

   
   
   
