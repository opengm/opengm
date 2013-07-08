import opengm
import numpy

unaries=numpy.random.rand(100 , 100,2)
potts=opengm.PottsFunction([2,2],0.0,0.2)
gm=opengm.grid2d2Order(unaries=unaries,regularizer=potts)

class IcmPurePython():
   def __init__(self,gm):
      self.gm=gm
      self.numVar=gm.numberOfVariables
      self.movemaker=opengm.movemaker(gm)
      self.adj=gm.variablesAdjacency()
      self.localOpt=numpy.zeros(self.numVar,dtype=numpy.bool)
   def infer(self,verbose=False):
      changes=True
      while(changes):
         changes=False
         for v in  self.gm.variables():
            if(self.localOpt[v]==False):
               l=self.movemaker.label(v)
               nl=self.movemaker.moveOptimallyMin(v)
               self.localOpt[v]=True
               if(nl!=l):
                  if(verbose):print self.movemaker.value()
                  self.localOpt[self.adj[v]]=False
                  changes=True    
   def arg(self):
      argLabels=numpy.zeros(self.numVar,dtype=opengm.label_type)
      for v in  self.gm.variables():
         argLabels[v]=self.movemaker.label(v) 
      return argLabels

icm=IcmPurePython(gm)
icm.infer(verbose=False)
arg=icm.arg()
print arg.reshape(100,100)                    