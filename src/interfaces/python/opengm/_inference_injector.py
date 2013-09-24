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
        
    # is solve has getEdgeLabeling interface
    if hasattr(solverClass, "_getEdgeLabeling")  :
        def getEdgeLabeling(self):
            return self._getEdgeLabeling()
        setattr(solverClass, 'getEdgeLabeling', getEdgeLabeling)

    # if solver has partialOptimality interface
    if hasattr(solverClass, "_partialOptimality")  :
        def partialOptimality(self):
            """get a numpy array of booleans which are true where the variables are optimal
            """
            return self._partialOptimality()
        setattr(solverClass, 'partialOptimality', partialOptimality)

    # if solver has lp interface
    if hasattr(solverClass,'_addConstraint'):
        def addConstraint(self,lpVariableIndices, coefficients, lowerBound, upperBound):
            lpVars  = numpy.require(lpVariableIndices,dtype=index_type)
            coeff   = numpy.require(coefficients     ,dtype=value_type)
            lb      = float(lowerBound)
            ub      = float(upperBound)
            if (coeff.ndim!=1):
                raise RuntimeError("coefficients.ndim must be 1")
            if (lpVars.ndim!=1):
                raise RuntimeError("lpVariableIndices.ndim must be 1")
            if (coeff.shape!=lpVars.shape):
                raise RuntimeError("lpVariableIndices.shape must match coefficients.shape")
            self._addConstraint(lpVars,coeff,lb,ub)

        def addConstraints(self,lpVariableIndices, coefficients, lowerBounds, upperBounds):
            lpVars  = numpy.require(lpVariableIndices,dtype=index_type)
            coeff   = numpy.require(lpVariableIndices,dtype=value_type)
            lbs     = numpy.require(lowerBounds,dtype=value_type)
            ubs     = numpy.require(upperBounds,dtype=value_type)

            if (coeff.ndim!=2):
                raise RuntimeError("coefficients.ndim must be 2")
            if (lpVars.ndim!=2):
                raise RuntimeError("lpVariableIndices.ndim must be 2")
            if (coeff.shape!=lpVars.shape):
                raise RuntimeError("lpVariableIndices.shape must match coefficients.shape")
            if (lbs.ndim!=1):
                raise RuntimeError("lowerBounds.ndim must be 1")
            if (ubs.ndim!=1):
                raise RuntimeError("upperBounds.ndim must be 1")
            if (lbs.shape!=ubs.shape):
                raise RuntimeError("lowerBounds.shape must match upperBounds.shape")
            if (lbs.shape[0]!=lpVars.shape[0]):
                raise RuntimeError("lowerBounds.shape[0] must match lpVars.shape[0]")

            self._addConstraints(lpVars,coeff,lbs,ubs)

        def lpNodeVariableIndex(self,variableIndex,label):
            return self._lpNodeVariableIndex(variableIndex,label)

        def lpFactorVariableIndex(self,factorIndex,labels):
            if isinstance(labels,(float,int,long)):
                l = long(labels)
                return self._lpFactorVariableIndex_Scalar(factorIndex,l)
            else:
                l = numpy.require(labels,dtype=label_type)
                return self._lpFactorVariableIndex_Numpy(factorIndex,l)


        setattr(solverClass, 'addConstraint', addConstraint)
        setattr(solverClass, 'addConstraints', addConstraints)
        setattr(solverClass, 'lpNodeVariableIndex', lpNodeVariableIndex)
        setattr(solverClass, 'lpFactorVariableIndex', lpFactorVariableIndex)
        

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
                    return numpy.array(outputVector)
                # return outputVector.view()
            elif isinstance(out, LabelVector):
                # print "is vector instance of length ",len(output)
                self._arg(out)
                if returnAsVector:
                    return out
                else:
                    return numpy.array(out)
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
