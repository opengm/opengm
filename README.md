OpenGM 2
========

[![Build Status](https://travis-ci.org/opengm/opengm.png?branch=master)](https://travis-ci.org/opengm/opengm)


-----------------------------------------------------------------------------------------------

**Manual for OpenGM 2.0.2** -> http://hci.iwr.uni-heidelberg.de//opengm2/download/opengm-2.0.2-beta-manual.pdf

**Code-Documentation for OpenGM 2.0.2** -> http://hci.iwr.uni-heidelberg.de//opengm2/doxygen/opengm-2.0.2-beta/html/index.html

OpenGM is a C++ template library for discrete factor graph models and distributive operations on these models. It includes state-of-the-art optimization and inference algorithms beyond message passing. OpenGM handles large models efficiently, since (i) functions that occur repeatedly need to be stored only once and (ii) when functions require different parametric or non-parametric encodings, multiple encodings can be used alongside each other, in the same model, using included and custom C++ code. No restrictions are imposed on the factor graph or the operations of the model. OpenGM is modular and extendible. Elementary data types can be chosen to maximize efficiency. The graphical model data structure, inference algorithms and different encodings of functions interoperate through well-defined interfaces. The binary OpenGM file format is based on the HDF5 standard and incorporates user extensions automatically.

Features

    Factor Graph Models (Kschischang et al. 2001)
        Graphs of any order and structure, from second order grid graphs to irregular higher-order models
        Arbitrary (commutative and associative) operations, including sum, product, conjunction and disjunction
        Flexible number of labels (different variables can have differently many labels)
        Function sharing across factors
        Function type abstraction. Different (built-in and custom) encodings can be used alongside each other
    Functions
        Explicit function (multi-dimensional table)
        Sparse function (sparse multi-dimensional table)
        Potts functions (different types, including higher-order)
        Truncated absolute difference
        Truncated squared difference
        Views that treat one graphical model as a function within another graphical model
    Algorithms
        Loopy Belief Propagation (Pearl 1988, Yedidia et al. 2000)
            parallel and sequential min-sum and max-product message passing (also for higher-order models)
            message damping (Wainwright 2008)
        Tree-reweighted Belief Propagation (TRBP) (Wainwright et al. 2005)
            parallel and sequential min-sum and max-product message passing (also for higher-order models)
            message damping (Wainwright 2008)
        A-star branch-and-bound search (Bergtholdt et al. 2009)
        Dual Decomposition
            With sub-gradient methods (Kappes et al. 2010)
            With bundle methods (Kappes et al. 2012)
            Automated decomposition of arbitrary factor graphs
            Arbitrary sub-solvers via templates
        Graph Cut (Boykov et al. 2001).
            Push-Relabel (Goldberg and Tarjan 1986)
            Edmonds-Karp (Edmonds and Karp 1972)
            Kolmogorov (Boykov and Kolmogorov 2004)
        QPBO
        MQPBO
        Linear Programming Relaxations over the Local Polytope
        TRWS
        ADSAL
        CombiLP
        Integer Linear Programming
        Multicut (Kappes et al. 2011)
        Reduced-Inference (Kappes et al. 2013)
        Alpha-Expansion
        Alpha-Beta-Swap
        Alpha-Fusion
        Inf and Flip
        Iterated Conditional Modes (ICM) (Besag 1986)
        Lazy Flipper (Andres et al. 2010)
        Kerninghan Lin
        MCMC Metropolis-Hastings algorithms (Metropolis et al. 1953)
            Gibbs sampling (Geman and Geman 1984)
            Swendsen-Wang sampling (Swendsen and Wang 1987)
        Wrappers around other graphical model libraries
            MRF-LIB
            LIB-DAI
            TRW-S
            QPBO
            GCO
            FastPD
            AD3
            DAOOPT
            MPLP, MPLP-C
    Binary HDF5 file format
    Command Line Optimizer with built-in protocol mode for runtime and convergence analyses
    Python Module with OpenGM C++ API exported to Python with boost::python
        Allmost the complete C++ API is exported to Python
        Allmost all C++ inference algorithms wrapped to Python
        Vectorized API to add multiple functions and factors at once
        Add functions via numpy ndarrays
        Add functions via all default opengm function types
        Extendibility through interfaces for
            custom pure python function types
            custom pure python visitor for inference
                visualization of current inference state with matplotlib 
        Visualize Factor Graph (needs networkx and graphviz)
    High performance
        Graphical models with more than 10,000,000 factors
        Specialized functions for optimized cache usage
    Extendibility through interfaces for
        custom algorithms
        custom functions
        custom label spaces



opengm/opengm - master

[![Build Status](https://travis-ci.org/opengm/opengm.png?branch=master)](https://travis-ci.org/opengm/opengm)

DerThorsten/opengm - master  (opengm-python dev.)

[![Build Status](https://travis-ci.org/DerThorsten/opengm.png?branch=master)](https://travis-ci.org/DerThorsten/opengm)

Copyright (C) 2012 Bjoern Andres, Thorsten Beier and Joerg H.~Kappes.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
