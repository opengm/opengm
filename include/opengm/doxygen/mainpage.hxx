// DOXYGEN GROUP STRUCTURE
/*!
 * \defgroup graphical_models Graphical Models
 * \defgroup inference Inference Algorithms
 * \defgroup spaces Space Types
 * \defgroup operators Operators
 * \defgroup functions Function Types
 */

// DOXYGEN EXAMPLES
/*!
 * \example quick_start.cxx
 * \example grid_potts.cxx  
 * \example interpixel_boundary_segmentation.cxx
 * \example one_to_one_matching.cxx
 * \example io_graphical_model.cxx
 * \example gibbs.cxx
 * \example inference_types.cxx
 * \example markov-chain.cxx 
 * \example space_types.cxx 
 * \example swendsenwang.cxx
 * \example opengmBuild.cxx
 */


// DOXYGEN MAINPAGE
/*! \mainpage OpenGM
 *
 * \section Introduction
 * OpenGM is a C++ template library for defining discrete graphical models and
 * performing inference on these models, using a wide range of state-of-the-art
 * algorithms. 
 * \n
 * No restrictions are imposed on the factor graph to allow for higher-order
 * factors and arbitrary neighborhood structures.
 * \n
 * Large models with repetitive structure are handled efficiently because
 * (i) functions that occur repeatedly need to be stored only once, and (ii)
 * distinct functions can be implemented differently, using different encodings
 * alongside each other in the same model.
 * \n
 * Several parametric functions (e.g.~metrics), sparse and dense value tables are
 * provided and so is an interface for custom C++ code.
 * \n
 * Algorithms are separated by design from the representation of graphical models
 * and are easily exchangeable.
 * \n
 * OpenGM, its algorithms, HDF5 file format and command line tools are modular and 
 * extendible.
 * 
 * \section Handbook
 * A very detailed <B>OpenGM-handbook for users and developers </B>  is available at the 
 * <A HREF="http://hci.iwr.uni-heidelberg.de/opengm2"> OpenGM webside </A HREF>
 * or use the 
 * <A HREF="http://hci.iwr.uni-heidelberg.de/opengm2/download/opengm-2.0.2-beta-manual.pdf"> direct link  </A HREF> 
 * to download the handbook in pdf form. 
 * \n
 * The handbook covers everything from installation instructions to usage examples but also gives insides in the 
 *  mathematical foundations of graphical models.
 * 
 * 
 * \section Modules
 * 
 * - \ref graphical_models
 * - \ref inference 
 * - \ref spaces 
 * - \ref operators 
 * - \ref functions
 *    .
 * .
 * \section examples Examples
 * 
 * \subsection cppexamples C++ Examples
 * - \link quick_start.cxx Quick Start                                                       \endlink
 * - \link grid_potts.cxx  N-class segmentation on a 2d grid with a Potts model              \endlink
 * - \link interpixel_boundary_segmentation.cxx Segmentation in the dual / boundary domaine  \endlink
 * - \link one_to_one_matching.cxx One-to-one Matching                                       \endlink
 * - \link markov-chain.cxx Simple markov chain                                              \endlink
 * - \link inference_types.cxx Usage of different inference algorithms                       \endlink
 * - \link space_types.cxx Usage of different space types                                    \endlink
 * - \link io_graphical_model.cxx  save / load a graphical model from / to hdf5              \endlink
 * .
 * 
 * \subsection matlabexamples MatLab Examples
 * - \link testGridCreation.m   Create a grid-structured model                               \endlink 
 * - \link testAddUnaries.m    Adding several unaries at ones (this is faster)               \endlink 
 *
 * - \link opengmBuild.cxx Example for build models from Matlab with own mex-file            \endlink
 * - \link opengmBuildGrid.cxx Build grid models from Matlab with own mex-file               \endlink
 * .
 *
 * \subsection pythonexamples Python Examples
 * - \link add_functions.py   Adding functions to a model                                    \endlink
 * - \link add_factors_and_functions.py   Adding factors and functions to a model            \endlink
 * - \link add_multiple_unaries.py   Adding several unaries at ones (this is faster)         \endlink
 * - \link markov_chain.py  Simple markov chain                                              \endlink
 * - \link inference_bp.py  Inference with LBP                                               \endlink
 * - \link inference_graphcut.py  Inference with GraphCut                                    \endlink
 * .
 *  
 */
