import numpy as np

import inspect
from cStringIO import StringIO
from _to_native_converter import to_native_class_converter
from _inference_parameter_injector import \
    _injectGenericInferenceParameterInterface
from _inference_injector import _injectGenericInferenceInterface
from _misc import defaultAccumulator
import sys
from opengmcore import index_type,value_type,label_type
from abc import ABCMeta, abstractmethod, abstractproperty
from optparse import OptionParser
import inspect
class InferenceBase:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, gm, accumulator, parameter):
        pass

    @abstractmethod
    def infer(self, visitor):
        pass

    #@abstractproperty
    #def gm(self):
    #    pass

    @abstractmethod
    def arg(self, out=None):
        pass

    #def bound(self, out=None):
    #    return self.gm.evaluate(self.arg(out))


class ImplementationPack(object):
    def __init__(self):
        self.implDict = {}

    def __hash__(self):
        return self.implDict.__hash__()

    def _check_consistency(self):
        hyperParamsKeywords = None   # as ['minStCut']
        hyperParamsHelp = None   # as ['minStCut implementation for graphcut']
        allowedHyperParams = set()  # as {['push-relabel'],['komolgorov'] }
        hasInterchangeableParameter = None
        # loop over all allowedHyperParams
        implDict = self.implDict
        for semiRingDict in implDict:
            hyperParameters = None
            # loop over all semi rings
            for algClass, paramClass in semiRingDict:
                hp = algClass.__hyperParameters()
                # check if the hyper parameter (as push-relabel)
                # is the same for all semi-rings
                if hyperParameters is not None:
                    raise RuntimeError("inconsistency in hyperParameters of %s"
                                       % algClass._algNames())
                    hyperParameters = hp
                    allowedHyperParams.add(hyperParameters)
                hpK = algClass._hyperParameterKeywords()
                hpH = algClass._hyperParametersHelp()

                icp = algClass._hasInterchangeableParameter()
                if hasInterchangeableParameter is not None:
                    assert (icp == hasInterchangeableParameter)
                else:
                    hasInterchangeableParameter = icp

                # check if the hyper parameter keywords are the same for all
                # algorithms within the implementation pack
                if (hyperParamsKeywords is not None
                        and hyperParamsHelp is not None):
                    if hpK != hyperParamsKeywords:
                        raise RuntimeError("inconsistency in hyperParamsKeywords of %s"
                                           % algClass._algNames())
                    if hpH != hyperParamsHelp:
                        raise RuntimeError("inconsistency in hyperParamsHelp of %s"
                                           % algClass._algNames())
                else:
                    hyperParamsKeywords = hpK
                    hyperParamsHelp = hpH

        if len(hyperParamsKeywords) != len(hyperParamsHelp):
            raise RuntimeError("inconsistency in hyperParamsHelp and "
                               "hyperParamsKeywords of %s"
                               % algClass._algNames())

    @ property
    def allowedHyperParameters(self):
        allowedHyperParams = set()  # as {['push-relabel'],['komolgorov'] }
        implDict = self.implDict
        for hyperParameters in implDict.keys():
            allowedHyperParams.add(hyperParameters)
        return allowedHyperParams

    @ property
    def hasHyperParameters(self):
        return len(self.hyperParameterKeywords) != 0

    @ property
    def hyperParameterKeywords(self):
        try:
            return dictDictElement(self.implDict)[0]._hyperParameterKeywords()
        except:
            raise RuntimeError(dictDictElement(self.implDict))

    @ property
    def hyperParametersDoc(self):
        return dictDictElement(self.implDict)[0]._hyperParametersDoc()

    @ property
    def hyperParameters(self):
        return dictDictElement(self.implDict)[0]._hyperParameters()

    @ property
    def hasInterchangeableParameter(self):
        return dictDictElement(self.implDict)[0]._hasInterchangeableParameter()

    @ property
    def anyParameterClass(self):
        return dictDictElement(self.implDict)[1]


def classGenerator(
    classname,
    inferenceClasses,
    defaultHyperParams,
    exampleClass,
):
    """ generates a high level class for each BASIC inference algorithm:
        There will be One class For Bp regardless what the operator
        and accumulator is .
        Also all classes with addidional templates lie
        GraphCut<PushRelabel> and GraphCut<komolgorov> will glued
        together to one class GraphCut
    """


    #print "className ",classname
    members =  inspect.getmembers(exampleClass, predicate=inspect.ismethod)







    def inference_init(self, gm, accumulator=None, parameter=None):
        # self._old_init()
        # set up basic properties
        self.gm = gm
        self.operator = gm.operator
        if accumulator is None:
            self.accumulator = defaultAccumulator(gm)
        else:
            self.accumulator = accumulator
        self._meta_parameter = parameter
        # get hyper parameter (as minStCut for graphcut, or the subsolver for
        # dualdec.)
        hyperParamKeywords = self._infClasses.hyperParameterKeywords
        numHyperParams = len(hyperParamKeywords)
        userHyperParams = [None]*numHyperParams
        collectedHyperParameters = 0
        # get the users hyper parameter ( if given)

        if(self._meta_parameter is not None):
            for hpIndex, hyperParamKeyword in enumerate(hyperParamKeywords):
                if hyperParamKeyword in self._meta_parameter.kwargs:
                    userHyperParams[hpIndex] = self._meta_parameter.kwargs.pop(
                        hyperParamKeyword)
                    collectedHyperParameters += 1

            # check if ZERO or ALL hyperParamerts have been collected
            if collectedHyperParameters != 0 and collectedHyperParameters != numHyperParams:
                raise RuntimeError("All or none hyper-parameter must be given")

        # check if the WHOLE tuple of hyperParameters is allowed
        if collectedHyperParameters != 0:
            if tuple(str(x) for x in userHyperParams) not in inferenceClasses.implDict:
                raise RuntimeError("%s is not an allowed hyperParameter\nAllowed hyperParameters are %s" % (
                    repr(userHyperParams), repr(inferenceClasses.implDict.keys())))
        else:
            userHyperParams = defaultHyperParams

        try:
            # get the selected inference class and the parameter
            if(numHyperParams == 0):
                
                self._selectedInfClass, self._selectedInfParamClass = inferenceClasses.implDict[
                        "__NONE__"][(self.operator, self.accumulator)]
            else:
                hp = tuple(str(x) for x in userHyperParams)
                self._selectedInfClass, self._selectedInfParamClass = inferenceClasses.implDict[
                    hp][(self.operator, self.accumulator)]
        except:
            dictStr=str(inferenceClasses.implDict)
            raise RuntimeError("given seminring (operator = %s ,accumulator = %s) is not implemented for this solver\n %s" % \
                (self.operator, self.accumulator,dictStr))

        if self._meta_parameter is None:
            self.parameter = self._selectedInfClass._parameter()
            self.parameter.set()
        else:
            self.parameter = to_native_class_converter(
                givenValue=self._meta_parameter, nativeClass=self._selectedInfParamClass)
            assert self.parameter is not None

        self.inference = self._selectedInfClass(self.gm, self.parameter)

    def verboseVisitor(self, printNth=1, multiline=True):
        """ factory function to get a verboseVisitor:

            A verboseVisitor will print some information while inference is running

        **Args**:
            printNth : call the visitor in each nth visit (default : ``1``)

            multiline : print the information in multiple lines or in one line (default: ``True``)

        **Notes**:
            The usage of a verboseVisitor can slow down inference a bit
        """
        return self.inference.verboseVisitor(printNth, multiline)

    def timingVisitor(self, visitNth=1,reserve=0,verbose=True, multiline=True,timeLimit=float('inf')):
        """ factory function to get a verboseVisitor:

            A verboseVisitor will print some information while inference is running

        **Args**:
            visitNth : call the python visitor  in each nth visit (default : ``1``)
            reserve  : reserve space for bounds,values,times, and iteratios (default: ``0``)
            verbose  : print information (default ``True``)
            multiline : print the information in multiple lines or in one line (default: ``True``)

        **Notes**:
            The usage of a timingVisitor can slow down inference a bit
        """
        return self.inference.timingVisitor(visitNth=visitNth,reserve=reserve,verbose=verbose, multiline=multiline,timeLimit=timeLimit)

    def pythonVisitor(self, callbackObject, visitNth):
        """ factory function to get a pythonVisitor:

            A python visitor can callback to pure python within the c++ inference

        **Args**:
            callbackObject : python function ( or class with implemented ``__call__`` function)

            visitNth : call the python function in each nth visit (default : 1)


        **Notes**:
            The usage of a pythonVisitor can slow down inference
        """
        return self.inference.pythonVisitor(callbackObject, visitNth)

    def infer(self, visitor=None, releaseGil=True):
        """ start the inference

        **Args**:
            visitor : run inference with an optional visitor (default : None)

        **Notes**:
            a call of infer will unlock the GIL
        """
        assert self.inference is not None
        return self.inference.infer(visitor=visitor, releaseGil=releaseGil)

    def arg(self, returnAsVector=False, out=None):
        """ get the result of the inference

        **Args**:
            returnAsVector : return the result as ``opengm.LabelVector`` (default : ``False``)

                To get a numpy ndarray ignore this argument or set it to ``False``

            out  : ``if returnAsVector==True`` a preallocated ``opengm.LabelVector`` can be passed to this function
        """
        return self.inference.arg(out=out, returnAsVector=returnAsVector)

    def partialOptimality(self):
        """get a numpy array of booleans which are true where the variables are optimal
        """
        return self.inference.partialOptimality()

    def setStartingPoint(self, labels):
        """ set a starting point / start labeling

        **Args**:
            labels : starting point labeling
        """
        numpyLabels=np.require(labels,dtype=label_type)
        self.inference.setStartingPoint(numpyLabels)

    def bound(self):
        """ get the bound"""
        return self.inference.bound()

    def value(self):
        """ get the value of inference.
        The same as ``gm.evaluate(inf.arg())``
        """
        return self.inference.value()

    def reset(self):
        """
        reset a inference solver (structure of gm must not change)
        """
        return self.inference.reset()
    def marginals(self,vis):
        """get the marginals for a subset of variable indices

        Args:
            vis : variable indices  (for highest performance use a numpy.ndarray with ``opengm.index_type`` as dtype)

        Returns :
            a 2d numpy.ndarray where the first axis iterates over the variables passed by ``vis``

        Notes :
            All variables in ``vis`` must have the same number of labels
        """
        return self.inference.marginals(vis)

    def factorMarginals(self,fis):
        """get the marginals for a subset of variable indices

        Args:
            fis : factor indices  (for highest performance use a numpy.ndarray with ``opengm.index_type`` as dtype)

        Returns :
            a N-d numpy.ndarray where the first axis iterates over the factors passed by ``fis``

        Notes :
            All factors in ``fis`` must have the same number of variables and shape
        """
        return self.inference.factorMarginals(fis)

    def addConstraint(self, lpVariableIndices, coefficients, lowerBound, upperBound):
        """
        Add a constraint to the lp

        **Args** :

            lpVariableIndices : variable indices w.r.t. the lp

            coefficients : coefficients of the constraint

            lowerBound : lowerBound of the constraint

            upperBound : upperBound of the constraint
        """
        self.inference.addConstraint(
            lpVariableIndices, coefficients, lowerBound, upperBound)

    def addConstraints(self, lpVariableIndices, coefficients, lowerBounds, upperBounds):
        """
        Add constraints to the lp

        **Args** :

            lpVariableIndices : variable indices w.r.t. the lp

            coefficients : coefficients of the constraints

            lowerBounds : lowerBounds of the constraints

            upperBounds : upperBounds of the constraints
        """
        self.inference.addConstraints(
            lpVariableIndices, coefficients, lowerBounds, upperBounds)
            
    def getEdgeLabeling(self):
        return self.inference.getEdgeLabeling()

    def lpNodeVariableIndex(self, variableIndex, label):
        """
        get the lp variable index from a gm variable index and the label

        **Args**:

            variableIndex : variable index w.r.t. the graphical model

            label : label of the variable

        **Returns**:

            variableIndex w.r.t. the lp

        """
        return self.inference.lpNodeVariableIndex(variableIndex, label)

    def lpFactorVariableIndex(self, factorIndex, labels):
        """
        get the lp factor index from a gm variable index and the labeling (or the scalar index of the labeling)

        **Args**:

            factorIndex : factor index w.r.t. the graphical model

            labels : labeling of the factor  (or a scalar index of the labeling)

        **Returns**:

            variableIndex w.r.t. the lp of the factor (and it's labeling )

        """
        return self.inference.lpFactorVariableIndex(factorIndex, labels)


    def generateParamHelp():

        # simple parameter
        if not inferenceClasses.hasHyperParameters:
            # get any parameter of this impl pack
            exampleParam = inferenceClasses.anyParameterClass()
            exampleParam.set()
            paramHelp = exampleParam._str_spaced_()
            return paramHelp
        # with hyper parameter(s)
        else:
            # the C++ parameter does NOT CHANGE if hyper parameters change
            if inferenceClasses.hasInterchangeableParameter:
                # get any parameter of this impl pack
                exampleParam = inferenceClasses.anyParameterClass()
                exampleParam.set()
                paramHelp = exampleParam._str_spaced_()
                # append  hyper parameter(s)
                # print to string!!!
                old_stdout = sys.stdout
                sys.stdout = mystdout = StringIO()
                # loop over all hp Keywords (usually there is max. 1 hyper parameter)
                # (should it be allowed to use more than 1 hp??? right now it is!)
                assert len(inferenceClasses.hyperParameterKeywords) == 1
                hyperParameterKeyword = inferenceClasses.hyperParameterKeywords[0]
                hyperParameterDoc = inferenceClasses.hyperParametersDoc[0]
                print "      * %s : %s" % (hyperParameterKeyword, hyperParameterDoc)
                #  loop over all hyperparamtersbound
                for hyperParameters in inferenceClasses.implDict.keys():
                    hyperParameter = hyperParameters[0]
                    # get an example for this hyperparameter class
                    classes = inferenceClasses.implDict[hyperParameters]
                    # get any semi ring solver
                    [solverC, paramC] = dictElement(classes)
                    assert len(hyperParameters) == 1
                    if(solverC._isDefault()):
                        print "          - ``'%s'`` (default)\n" % (hyperParameter,)
                    else:
                        print "          - ``'%s'``\n" % (hyperParameter,)

                sys.stdout = old_stdout
                hyperParamHelp = mystdout.getvalue()
                return paramHelp + "\n\n" + hyperParamHelp

            # the C++ parameter DOES CHANGE if hyper parameters change
            else:
                # print to string!!!
                old_stdout = sys.stdout
                sys.stdout = mystdout = StringIO()
                print "The parameter object of has internal dependencies:\n\n"

                assert len(inferenceClasses.hyperParameterKeywords) == 1
                hyperParameterKeyword = \
                    inferenceClasses.hyperParameterKeywords[0]
                hyperParameterDoc = inferenceClasses.hyperParametersDoc[0]
                print("      * %s : %s"
                      % (hyperParameterKeyword, hyperParameterDoc))
                #  loop over all hyperparamters
                for hyperParameters in inferenceClasses.implDict.keys():
                    hyperParameter = hyperParameters[0]
                    # get an example for this hyperparameter class
                    classes = inferenceClasses.implDict[hyperParameters]
                    # get any semi ring solver
                    [solverC, paramC] = dictElement(classes)
                    assert len(hyperParameters) == 1
                    if(solverC._isDefault()):
                        print("          - ``'%s'`` (default)\n"
                              % (hyperParameter,))
                    else:
                        print("          - ``'%s'``\n"
                              % (hyperParameter,))

                for hyperParameters in inferenceClasses.implDict.keys():
                    hyperParameter = hyperParameters[0]
                    # get an example for this hyperparameter class
                    classes = inferenceClasses.implDict[hyperParameters]
                    # get any semi ring solver
                    [solverC, paramC] = dictElement(classes)

                    hyperParameterKeywords = solverC._hyperParameterKeywords()
                    hyperParameters = solverC._hyperParameters()
                    assert len(hyperParameterKeywords) == 1
                    assert len(hyperParameters) == 1
                    hyperParameterKeyword = hyperParameterKeywords[0]
                    hyperParameter = hyperParameters[0]

                    print("        ``if %s == %s`` : \n\n"
                          % (hyperParameterKeyword, hyperParameter))
                    exampleParam = paramC()
                    exampleParam.set()
                    print exampleParam._str_spaced_('      ')

                sys.stdout = old_stdout
                return mystdout.getvalue()

    # exampleClass
    memberDict = {
        # public members
        '__init__': inference_init,
        'infer': infer,
        'arg': arg,
        'bound': bound,
        'value': value,
        'setStartingPoint': setStartingPoint,
        #
        'gm': None,
        'operator': None,
        'accumulator': None,
        'inference': None,
        'parameter': None,
        # 'protected' members
        '_meta_parameter': None,
        '_infClasses': inferenceClasses,
        '_selectedInfClass': None,
        '_selectedInfParamClass': None
    }



    def _generateFunction_(function,fname):
        def _f_(self,*args,**kwargs):
            attr  = getattr(self.inference, fname)
            return attr(*args,**kwargs)
        _f_.__doc__=function.__doc__
        return _f_

    for m in members:
        if m[0].startswith('_') or m[0].endswith('_') :
            pass
        else :
            memberDict[m[0]]=_generateFunction_(m[1],m[0])
            
    """
    if hasattr(exampleClass, "reset"):
        memberDict['reset'] = reset
    if hasattr(exampleClass, "verboseVisitor"):
        memberDict['verboseVisitor'] = verboseVisitor
    if hasattr(exampleClass, "timingVisitor"):
        memberDict['timingVisitor'] = timingVisitor
    if hasattr(exampleClass, "pythonVisitor"):
        memberDict['pythonVisitor'] = pythonVisitor
    if hasattr(exampleClass, "marginals") and hasattr(exampleClass, "factorMarginals"):
        memberDict['marginals'] = marginals
        memberDict['factorMarginals'] = factorMarginals

    if hasattr(exampleClass, "addConstraint") and hasattr(exampleClass, "addConstraints"):
        memberDict['addConstraints'] = addConstraints
        memberDict['addConstraint'] = addConstraint
        memberDict['lpNodeVariableIndex'] = lpNodeVariableIndex
        memberDict['lpFactorVariableIndex'] = lpFactorVariableIndex
       
    if hasattr(exampleClass, "partialOptimality") :
        memberDict['partialOptimality'] = partialOptimality
        
    if hasattr(exampleClass, "getEdgeLabeling") :
        memberDict['getEdgeLabeling'] = getEdgeLabeling

    """
    infClass = type(classname, (InferenceBase,), memberDict)

    infClass.__init__ = inference_init
    # print to string!!!
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    print """ %s is a  %s inference algorithm

    **Args** :
        gm : the graphical model to infere / optimize

        accumulator : accumulator used for inference can be:

            -``'minimizer'`` (default : ``if gm.operator is 'adder'==True:``)

            -``'maximizer'`` (default : ``if gm.operator is 'multiplier'==True:``)

            -``'integrator'``

            Not any accmulator can be used for any solver.
            Which accumulator can be used will be in the documentation soon.

        parameter : parameter object of the solver

    """ % (exampleClass._algName(), exampleClass._algType())

    print """
    **Parameter** :
      %s

    """ % (generateParamHelp(),)
    if(exampleClass._examples() != ''):
        print """    **Examples**: ::

        %s

        """ % (exampleClass._examples() .replace("\n", "\n        "),)
    if(exampleClass._guarantees() != ''):
        print """    **Guarantees** :

        %s

        """ % (exampleClass._guarantees(),)
    if(exampleClass._limitations() != ''):
        print """    **Limitations** :

        %s

        """ % (exampleClass._limitations(),)
    if(exampleClass._cite() != ''):
        print """    **Cite** :

        %s

        """ % (exampleClass._cite().replace("\n\n", "\n\n        "),)
    if(exampleClass._dependencies() != ''):
        print """    **Dependencies** :

        %s

        """ % (exampleClass._dependencies(),)
    if(exampleClass._notes() != ''):
        print """    **Notes** :

        %s

        """ % (exampleClass._notes().replace("\n\n", "\n\n        "),)
    sys.stdout = old_stdout
    infClass.__dict__['__init__'].__doc__ = mystdout.getvalue()

    return infClass, classname


def dictElement(aDict):
    return aDict.itervalues().next()


def dictDictElement(dictDict):
    return dictElement(dictElement(dictDict))


def _inject_interface(solverDicts):

    algs = dict()
    algDefaultHyperParams = dict()
    exampleClasses = dict()
    for solverDict, op, acc in solverDicts:
        semiRing = (op, acc)
        # inject raw interface to paramters and subparameters
        try:
            paramDict = solverDict['parameter'].__dict__
        except:
            raise RuntimeError(repr(solverDict))
        for key in paramDict:
            paramClass = paramDict[key]
            if inspect.isclass(paramClass):
                _injectGenericInferenceParameterInterface(
                    paramClass, infParam=not key.startswith('_SubParameter'),
                    subInfParam=key.startswith('_SubParameter'))

        for key in solverDict:

            elementInDict = solverDict[key]

            if (inspect.isclass(elementInDict) and not key.endswith('Visitor')
                    and hasattr(elementInDict, '_algName')
                    and hasattr(elementInDict, '_parameter')):
                solverClass = elementInDict

                param = solverClass._parameter()
                paramClass = param.__class__
                # inject raw interface to inference
                _injectGenericInferenceInterface(solverClass)

                # Get Properties to group algorithm
                algName = solverClass._algName()
                hyperParamKeywords = [str(
                    x) for x in solverClass._hyperParameterKeywords()]
                hyperParameters = tuple(str(
                    x) for x in solverClass._hyperParameters())

                assert hyperParamKeywords is not None

                exampleClasses[algName] = solverClass

                # algs['GraphCut']
                if algName in algs:
                    metaAlgs = algs[algName]
                else:
                    implPack = ImplementationPack()
                    algs[algName] = implPack
                    metaAlgs = algs[algName]

                metaAlgs = algs[algName]

                if(len(hyperParameters) == 0):
                    if '__NONE__' in metaAlgs.implDict:
                        semiRingAlgs = metaAlgs.implDict["__NONE__"]
                    else:
                        metaAlgs.implDict["__NONE__"] = dict()
                        semiRingAlgs = metaAlgs.implDict["__NONE__"]
                else:
                    if hyperParameters in metaAlgs.implDict:
                        semiRingAlgs = metaAlgs.asDict()[hyperParameters]
                    else:
                        metaAlgs.implDict[hyperParameters] = dict()
                        semiRingAlgs = metaAlgs.implDict[hyperParameters]

                semiRingAlgs[semiRing] = (solverClass, paramClass)

                if(len(hyperParameters) == 0):
                    metaAlgs.implDict["__NONE__"] = semiRingAlgs
                else:
                    metaAlgs.implDict[hyperParameters] = semiRingAlgs

                algs[algName] = metaAlgs
                # check if this implementation is the default
                if solverClass._isDefault():
                    algDefaultHyperParams[algName] = hyperParameters

    result = []
    # generate high level interface
    for algName in algs.keys():
        a = algs[algName]
        adhp = algDefaultHyperParams[algName]
        ec = exampleClasses[algName]
        result.append(classGenerator(algName, a, adhp, ec))
    return result
