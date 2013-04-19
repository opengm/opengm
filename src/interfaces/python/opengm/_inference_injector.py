from opengmcore import LabelVector

def _injectGenericInferenceInterface(solverClass):
   class InjectorGenericInference(object):
      class __metaclass__(solverClass.__class__):
         def __init__(self, name, bases, dict):
            for b in bases:
               if type(b) not in (self, type):
                  for k,v in dict.items():
                     setattr(b,k,v)
            return type.__init__(self, name, bases, dict)

   class PyAddon_GenericInference(InjectorGenericInference,solverClass):
      def arg(self,returnAsVector=False,out=None):
         if(out is None):
            outputVector=LabelVector()
            outputVector.resize(self.gm().numberOfVariables)
            self._arg(outputVector)
            if(returnAsVector == True): 
               return outputVector
            else:
               #print "get numpy"
               return outputVector.asNumpy()
            #return outputVector.view()
         elif(isinstance(out,LabelVector)):
            #print "is vector instance of length ",len(output)
            self._arg(out)
            if(returnAsVector == True): 
               return out
            else:
               return out.asNumpy()
         else:
            raise TypeError("if \"returnAsVector\"=\"True\" out has to be of the type \"opengm.LabelVector\"")

      def gm(self):
         return self.graphicalModel()
      def setStartingPoint(self,labels):
         if (isinstance(labels, LabelVector)):
            l=labels
         else:
            l=LabelVector(labels)
         self._setStartingPoint(l)

      def infer(self,visitor=None,releaseGil=True):
         if visitor is None:
            return self._infer_no_visitor(releaseGil=releaseGil)
         else:
            return self._infer(visitor=visitor,releaseGil=releaseGil)


