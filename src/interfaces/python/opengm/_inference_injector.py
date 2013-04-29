import numpy     
from opengmcore import LabelVector,IndependentFactor,index_type,value_type,label_type
from inference  import InferenceTermination

def _injectGenericInferenceInterface(solverClass):
    class InjectorGenericInference(object):
        class __metaclass__(solverClass.__class__):
            def __init__(self, name, bases, dict):
                for b in bases:
                    if type(b) not in (self, type):
                        for k, v in dict.items():
                            setattr(b, k, v)
                return type.__init__(self, name, bases, dict)

    # if solver has marginalization interface
    if hasattr(solverClass, "_marginals") and hasattr(solverClass, "_factorMarginals") :

        def marginals(self,vis):
            """get the marginals for a subset of variable indices

            Args:
                vis : variable indices  (for highest performance use a numpy.ndarray with ``opengm.index_type`` as dtype)

            Returns :
                a 2d numpy.ndarray where the first axis iterates over the variables passed by ``vis``

            Notes :
                All variables in ``vis`` must have the same number of labels
            """
            if isinstance(vis, numpy.ndarray):
                return self._marginals(numpy.require(vis,dtype=index_type))
            elif isinstance(vis, (int,long) ):
                return self._marginals(numpy.array([vis],dtype=index_type))
            else:
                return self._marginals(numpy.array(vis,dtype=index_type))




        def factorMarginals(self,fis):
            """get the marginals for a subset of variable indices

            Args:
                fis : factor indices  (for highest performance use a numpy.ndarray with ``opengm.index_type`` as dtype)

            Returns :
                a N-d numpy.ndarray where the first axis iterates over the factors passed by ``fis``

            Notes :
                All factors in ``fis`` must have the same number of variables and shape
            """
            if isinstance(fis, numpy.ndarray):
                return self._factorMarginals(numpy.require(fis,dtype=index_type))
            elif isinstance(fis, (int,long) ):
                return self._factorMarginals(numpy.array([fis],dtype=index_type))
            else:
                return self._factorMarginals(numpy.array(fis,dtype=index_type))

        setattr(solverClass, 'marginals', marginals)
        setattr(solverClass, 'factorMarginals', factorMarginals)

    # if solver has partialOptimality interface
    if hasattr(solverClass, "_partialOptimality")  :
        def partialOptimality(self):
            """get a numpy array of booleans which are true where the variables are optimal
            """
            return self._partialOptimality()
        setattr(solverClass, 'partialOptimality', partialOptimality)



    class PyAddon_GenericInference(InjectorGenericInference, solverClass):
        def arg(self, returnAsVector=False, out=None):
            if out is None:
                outputVector = LabelVector()
                outputVector.resize(self.gm().numberOfVariables)
                self._arg(outputVector)
                if returnAsVector:
                    return outputVector
                else:
                    # print "get numpy"
                    return outputVector.asNumpy()
                # return outputVector.view()
            elif isinstance(out, LabelVector):
                # print "is vector instance of length ",len(output)
                self._arg(out)
                if returnAsVector:
                    return out
                else:
                    return out.asNumpy()
            else:
                raise TypeError(
                    'if "returnAsVector"="True" out has to be of the type '
                    '"opengm.LabelVector"')

        def gm(self):
            return self.graphicalModel()

        def setStartingPoint(self, labels):
            if (isinstance(labels, LabelVector)):
                l = labels
            else:
                l = LabelVector(labels)
            self._setStartingPoint(l)

        def infer(self, visitor=None, releaseGil=True):
            if visitor is None:
                return self._infer_no_visitor(releaseGil=releaseGil)
            else:
                return self._infer(visitor=visitor, releaseGil=releaseGil)
