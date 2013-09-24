Important functions and classes
---------------------------------

Important functions:

* :func:`opengm.gm`
* :func:`opengm.shapeWalker`
* :func:`opengm.movemaker`
* :func:`opengm.visualizeGm`

Important classes:

* :class:`opengm.adder.GraphicalModel`
    * :class:`opengm.adder.Factor`
    * :class:`opengm.adder.Movemaker`
* :class:`opengm.multiplier.GraphicalModel`
    * :class:`opengm.multiplier.Factor`
    * :class:`opengm.multiplier.Movemaker`
* :class:`opengm.IndependentFactor`


Inference classes :

* :class:`opengm.inference.Icm`
* :class:`opengm.inference.LazyFlipper`
* :class:`opengm.inference.Gibbs`
* :class:`opengm.inference.BeliefPropagation`
* :class:`opengm.inference.TreeReweightedBp`
* :class:`opengm.inference.DynamicProgramming`
* :class:`opengm.inference.Bruteforce`
* :class:`opengm.inference.DualDecompositionSubgradient`
* :class:`opengm.inference.GraphCut`    
* :class:`opengm.inference.AlphaBetaSwap` 
* :class:`opengm.inference.AlphaExpansion` 
* :class:`opengm.inference.PartitionMove` 

* If compiled with CMake-Flag ``WITH_CPLEX`` set to ``ON``:
    * :class:`opengm.inference.LpCplex` 
    * :class:`opengm.inference.LpCplex2` 
    * :class:`opengm.inference.MultiCut`
* If compiled with CMake-Flag ``WITH_GUROBI`` set to ``ON``:
    * :class:`opengm.inference.LpGurobi` 
* If compiled with CMake-Flag ``WITH_QPBO`` set to ``ON``:
    * :class:`opengm.inference.QpboExternal` 
    * :class:`opengm.inference.Mqpbo`
    * :class:`opengm.inference.ReducedInference` 
    * :class:`opengm.inference.AlphaExpansionFusion` 
* If compiled with CMake-Flag ``WITH_TRWS`` set to ``ON``:
    * :class:`opengm.inference.TrwsExternal` 
* If compiled with CMake-Flag ``WITH_MRF`` set to ``ON``:
    * :class:`opengm.inference.MrfLib`
* If compiled with CMake-Flag ``WITH_FASTPD`` set to ``ON``:
    * :class:`opengm.inference.FastPd`
* If compiled with CMake-Flag ``WITH_AD3`` set to ``ON``:
    * :class:`opengm.inference.Ad3`
    * :class:`opengm.inference.Loc`
* If compiled with CMake-Flag ``WITH_LIBDAI`` set to ``ON``:
    * :class:`opengm.inference.BeliefPropagationLibDai` 
    * :class:`opengm.inference.FractionalBpLibDai` 
    * :class:`opengm.inference.TreeReweightedBpLibDai` 
    * :class:`opengm.inference.JunctionTreeLibDai` 
    * :class:`opengm.inference.DecimationLibDai` 
    * :class:`opengm.inference.GibbsLibDai` 


Function types  and factory functions:

* :class:`opengm.ExplicitFunction`
* :class:`opengm.TruncatedAbsoluteDifferenceFunction`
* :class:`opengm.TruncatedSquaredDifferenceFunction`
* :class:`opengm.PottsFunction`
* :class:`opengm.PottsNFunction`
* :class:`opengm.PottsGFunction`
* :class:`opengm.PythonFunction`
* :func:`opengm.modelViewFunction`
* :func:`opengm.differenceFunction`
* :func:`opengm.pottsFunction`
* :func:`opengm.relabeledDifferenceFunction`
* :func:`opengm.relabeledPottsFunction`

