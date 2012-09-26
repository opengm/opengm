#ifndef OPENGM_PYTHON_INTERFACE
#define OPENGM_PYTHON_INTERFACE 1
#endif

#include <stdexcept>
#include <stddef.h>
#include <string>
#include <boost/python.hpp>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/inference/inference.hxx>
#include <opengm/inference/graphcut.hxx>
#include <opengm/inference/alphabetaswap.hxx>
#ifdef WITH_BOOST
#  include <opengm/inference/auxiliary/minstcutboost.hxx>
#endif
#ifdef WITH_MAXFLOW
#  include <opengm/inference/auxiliary/minstcutkolmogorov.hxx>
#endif


#include "nifty_iterator.hxx"
#include "inferencehelpers.hxx"
#include "../export_typedes.hxx"


#define AB_SWAP_EXPORT_HELPER(AB_CLASS,GC_STRING,P_STRING,V_STRING,DOC_STRING)\
class_<typename AB_CLASS::Parameter > (P_STRING, init<>() ) \
      .def_readwrite("steps", & AB_CLASS::Parameter::maxNumberOfIterations_,\
      "steps: Maximum number of iterations"\
      )\
      .def ("set", &pyabswap::set<typename AB_CLASS::Parameter>, \
            ( \
            arg("steps")=1000\
            ), \
      "Set the parameters values.\n\n"\
      "All values of the parameter have a default value.\n\n"\
      "Args:\n\n"\
      "  steps: Maximum number of iterations (default=1000)\n\n"\
      "Returns:\n"\
      "  None\n\n"\
      ) \
      ; \
   OPENGM_PYTHON_VERBOSE_VISITOR_EXPORTER( typename AB_CLASS::VerboseVisitorType,V_STRING );\
   OPENGM_PYTHON_INFERENCE_NO_RESET_EXPORTER(AB_CLASS,GC_STRING,DOC_STRING)   



namespace  pyabswap{
   template<class PARAM>
   void set
   (
      PARAM & p,
      const size_t steps     
   ){
      p.maxNumberOfIterations_=steps;
   }
}


using namespace boost::python;

template<class GM,class ACC>
void export_abswap(){
   import_array(); 
   typedef GM PyGm;
   typedef typename PyGm::ValueType ValueType;


   typedef opengm::MinSTCutBoost<size_t, ValueType, opengm::KOLMOGOROV> MinStCutBoostKolmogorov;
   typedef opengm::GraphCut<PyGm, ACC, MinStCutBoostKolmogorov> PyGraphCutBoostKolmogorov;
   typedef opengm::AlphaBetaSwap<PyGm, PyGraphCutBoostKolmogorov> PyAlphaBetaSwapBoostKolmogorov;



   AB_SWAP_EXPORT_HELPER(PyAlphaBetaSwapBoostKolmogorov,
      "AlphaBetaSwapBoostKolmogorov",
      "AlphaBetaSwapBoostKolmogorovParameter",
      "AlphaBetaSwapBoostKolmogorovVerboseVisitor",
   "Alpha Beta Swap:\n\n"
   "cite: ???: \"`title <paper_url>`_\"," 
   "Journal.\n\n"
   "limitations: TODO\n\n"
   "guarantees:  TODO"
   );

}

template void export_abswap<GmAdder,opengm::Minimizer>();
