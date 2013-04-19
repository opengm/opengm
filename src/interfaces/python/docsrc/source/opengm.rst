



:mod:`opengm` Package
============================

.. automodule:: opengm
  
  .. autoclass:: ExplicitFunction
    :members: __init__,__getitem__ , shape,dimension,ndim

  .. autoclass:: SparseFunction
    :members: __init__, __setitem__, __getitem__ , shape,dimension,ndim,defaultValue,keyToCoordinate

  .. autoclass:: PottsFunction
    :members: __init__, __getitem__ , shape,dimension,ndim

  .. autoclass:: PottsNFunction
    :members: __init__, __getitem__ , shape,dimension,ndim

  .. autoclass:: PottsGFunction
    :members: __init__, __getitem__ , shape,dimension,ndim

  .. autoclass:: AbsoluteDifferenceFunction
    :members: __init__, __getitem__ , shape,dimension,ndim

  .. autoclass:: SquaredDifferenceFunction
    :members: __init__, __getitem__ , shape,dimension,ndim

  .. autoclass:: TruncatedAbsoluteDifferenceFunction
    :members: __init__, __getitem__ , shape,dimension,ndim

  .. autoclass:: TruncatedSquaredDifferenceFunction
    :members: __init__, __getitem__ , shape,dimension,ndim

  .. autoclass:: PythonFunction
    :members: __init__, __getitem__ , shape,dimension,ndim

  .. autofunction:: modelViewFunction

  .. autofunction:: relabeledPottsFunction

  .. autofunction:: differenceFunction

  .. autofunction:: relabeledDifferenceFunction

  .. autofunction:: pottsFunction

  .. autoclass:: ExplicitFunctionVector
    :members: 

  .. autoclass:: SparseFunctionVector
    :members: 

  .. autoclass:: PottsFunctionVector
    :members: 

  .. autoclass:: PottsNFunctionVector
    :members: 

  .. autoclass:: PottsGFunctionVector
    :members: 

  .. autoclass:: AbsoluteDifferenceFunctionVector
    :members: 

  .. autoclass:: SquaredDifferenceFunctionVector
    :members: 

  .. autoclass:: TruncatedAbsoluteDifferenceFunctionVector
    :members: 

  .. autoclass:: TruncatedSquaredDifferenceFunctionVector
    :members: 

  .. autoclass:: PythonFunctionVector
    :members: 

  Factoy functions to generate some classes:

  .. autofunction:: gm
  
  .. autofunction:: graphicalModel

  .. autofunction:: movemaker

  .. autofunction:: shapeWalker

  .. autofunction:: visualizeGm


  .. autoclass:: IndependentFactor
    :members: 
    :undoc-members:



:mod:`opengm.inference` Package
================================

.. automodule:: opengm.inference

  .. autoclass:: Icm
     :members: infer, arg, bound,setStartingPoint,pythonVisitor,verboseVisitor

  .. autoclass:: LazyFlipper
     :members: infer, arg, bound,setStartingPoint,pythonVisitor,verboseVisitor

  .. autoclass:: Loc
     :members: infer, arg, bound,setStartingPoint,pythonVisitor,verboseVisitor

  .. autoclass:: Gibbs
     :members: infer, arg, bound,setStartingPoint,pythonVisitor,verboseVisitor

  .. autoclass:: BeliefPropagation
     :members: infer, arg, bound,pythonVisitor,verboseVisitor

  .. autoclass:: TreeReweightedBp
     :members: infer, arg, bound,setStartingPoint,pythonVisitor,verboseVisitor

  .. autoclass:: TrwsExternal
     :members: infer, arg, bound,pythonVisitor,verboseVisitor

  .. autoclass:: DynamicProgramming
     :members: infer, arg, bound,pythonVisitor,verboseVisitor

  .. autoclass:: AStar
     :members: infer, arg, bound,setStartingPoint,pythonVisitor,verboseVisitor

  .. autoclass:: GraphCut
     :members: infer, arg, bound,pythonVisitor,verboseVisitor

  .. autoclass:: Qpbo
     :members: infer, arg, bound,pythonVisitor,verboseVisitor

  .. autoclass:: QpboExternal
     :members: infer, arg, bound,pythonVisitor,verboseVisitor

  .. autoclass:: AlphaBetaSwap
     :members: infer, arg, bound,setStartingPoint,pythonVisitor,verboseVisitor

  .. autoclass:: AlphaExpansion
     :members: infer, arg, bound,setStartingPoint,pythonVisitor,verboseVisitor

  .. autoclass:: DualDecompositionSubgradient
     :members: infer, arg, bound,setStartingPoint,pythonVisitor,verboseVisitor

  .. autoclass:: LpCplex
     :members: infer, arg, bound,setStartingPoint,pythonVisitor,verboseVisitor , addConstraint, addConstraints, lpNodeVariableIndex, lpFactorVariableIndex

  .. autoclass:: Bruteforce
     :members: infer, arg, bound,pythonVisitor,verboseVisitor
   
  .. autoclass:: MrfLib
     :members: infer, arg, bound,pythonVisitor,verboseVisitor

  .. autoclass:: BeliefPropagationLibDai
     :members: infer, arg, bound,pythonVisitor,verboseVisitor  

  .. autoclass:: FractionalBpLibDai
     :members: infer, arg, bound,pythonVisitor,verboseVisitor  

  .. autoclass:: TreeReweightedBpLibDai
     :members: infer, arg, bound,pythonVisitor,verboseVisitor  

  .. autoclass:: JunctionTreeLibDai
     :members: infer, arg, bound,pythonVisitor,verboseVisitor 

  .. autoclass:: DecimationLibDai
     :members: infer, arg, bound,pythonVisitor,verboseVisitor 

  .. autoclass:: GibbsLibDai
     :members: infer, arg, bound,pythonVisitor,verboseVisitor  


:mod:`opengm.adder` Package
============================

.. automodule:: opengm.adder

  .. autoclass:: GraphicalModel
    :members: 
    :undoc-members:


  .. autoclass:: Factor
    :members: 
    :undoc-members:


  .. autoclass:: Movemaker
    :members: 
    :undoc-members:


:mod:`opengm.multiplier` Package
=================================
.. automodule:: opengm.multiplier

  .. autoclass:: GraphicalModel
    :members: 
    :undoc-members:


  .. autoclass:: Factor
    :members: 


  .. autoclass:: Movemaker
    :members: 
    :undoc-members:
