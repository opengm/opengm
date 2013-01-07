from _opengmcore import *
import numpy



def variables(gm,labels=None,maxLabels=None,minLabels=None):
   if labels is  None and maxLabels is None and minLabels is None:
      for v in xrange(gm.numberOfVariables):
         yield v


def factors(gm,order=None,maxOrder=None,minOrder=None):
   if order is None and maxOrder is None and minOrder is None:
      for i in xrange(gm.numberOfFactors):
         yield gm[i]
   elif order is not None and maxOrder is None and minOrder is None:     
      for i in xrange(gm.numberOfFactors):
         factor = gm[i]
         if factor.numberOfVariables==order:
            yield factor
   elif maxOrder is not None and order is None and minOrder is None:     
      for i in xrange(gm.numberOfFactors):
         factor = gm[i]
         if factor.numberOfVariables <= maxOrder:
            yield factor
   elif minOrder is not None and order is None and maxOrder is None:      
      for i in xrange(gm.numberOfFactors):
         factor = gm[i]
         if factor.numberOfVariables >= minOrder:
            yield factor
   elif minOrder is not None and order is None and maxOrder is not None:
      for i in xrange(gm.numberOfFactors):
         factor = gm[i]
         if factor.numberOfVariables >= minOrder and factor.numberOfVariables <= maxOrder:
            yield factor            

def factorIds(gm,order=None,maxOrder=None,minOrder=None):
   if order is None and maxOrder is None and minOrder is None:
      for i in xrange(gm.numberOfFactors):
         yield i
   elif order is not None and maxOrder is None and minOrder is None:     
      for i in xrange(gm.numberOfFactors):
         factor = gm[i]
         if factor.numberOfVariables==order:
            yield i
   elif maxOrder is not None and order is None and minOrder is None:     
      for i in xrange(gm.numberOfFactors):
         factor = gm[i]
         if factor.numberOfVariables <= maxOrder:
            yield i
   elif minOrder is not None and order is None and maxOrder is None:      
      for i in xrange(gm.numberOfFactors):
         factor = gm[i]
         if factor.numberOfVariables >= minOrder:
            yield i
   elif minOrder is not None and order is None and maxOrder is not None:
      for i in xrange(gm.numberOfFactors):
         factor = gm[i]
         if factor.numberOfVariables >= minOrder and factor.numberOfVariables <= maxOrder:
            yield i            

def factorsAndIds(gm,order=None,maxOrder=None,minOrder=None):
   if order is None and maxOrder is None and minOrder is None:
      for i in xrange(gm.numberOfFactors):
         factor = gm[i]
         yield factor, i
   elif order is not None and maxOrder is None and minOrder is None:     
      for i in xrange(gm.numberOfFactors):
         factor = gm[i]
         if factor.numberOfVariables==order:
            yield factor,i
   elif maxOrder is not None and order is None and minOrder is None:     
      for i in xrange(gm.numberOfFactors):
         factor = gm[i]
         if factor.numberOfVariables <= maxOrder:
            yield factor, i
   elif minOrder is not None and order is None and maxOrder is None:      
      for i in xrange(gm.numberOfFactors):
         factor = gm[i]
         if factor.numberOfVariables >= minOrder:
            yield factor, i
   elif minOrder is not None and order is None and maxOrder is not None:
      for i in xrange(gm.numberOfFactors):
         factor = gm[i]
         if factor.numberOfVariables >= minOrder and factor.numberOfVariables <= maxOrder:
            yield factor,i            


class Adder:
   def neutral(self):
      return float(0.0)
  

class Multiplier:
   def neutral(self):
      return float(1.0)

adder.Operator = Adder
adder.Operator.neutral = Adder.neutral
multiplier.Operator = Multiplier


BoostPythonMetaclassIFactor          = IndependentFactor.__class__
BoostPythonMetaclassAdderFactor      = adder.Factor.__class__
BoostPythonMetaclassMultiplierFactor = multiplier.Factor.__class__
MetaGmAdder                          = adder.GraphicalModel.__class__
MetaGmMult                           = multiplier.GraphicalModel.__class__

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
            
class adder_gm(InjectorGmAdder, adder.GraphicalModel):
    def factors(self,order=None,maxOrder=None,minOrder=None):
        return factors(self,order,maxOrder,minOrder)
    def factorsAndIds(self,order=None,maxOrder=None,minOrder=None):
        return factorsAndIds(self,order,maxOrder,minOrder)
    def factorIds(self,order=None,maxOrder=None,minOrder=None):
        return factorIds(self,order,maxOrder,minOrder)     

class multiplier_gm(InjectorGmMult, multiplier.GraphicalModel):
    def factors(self,order=None,maxOrder=None,minOrder=None):
        return factors(self,order,maxOrder,minOrder)
    def factorsAndIds(self,order=None,maxOrder=None,minOrder=None):
        return factorsAndIds(self,order,maxOrder,minOrder)
    def factorIds(self,order=None,maxOrder=None,minOrder=None):
        return factorIds(self,order,maxOrder,minOrder)     

        
        

   
   
   
