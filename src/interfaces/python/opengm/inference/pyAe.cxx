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
#include <opengm/inference/alphaexpansion.hxx>
#ifdef WITH_BOOST
#  include <opengm/inference/auxiliary/minstcutboost.hxx>
#endif
#ifdef WITH_MAXFLOW
#  include <opengm/inference/auxiliary/minstcutkolmogorov.hxx>
#endif

#include "nifty_iterator.hxx"
#include "inferencehelpers.hxx"
#include "../export_typedes.hxx"

#define AE_EXPORT_HELPER(AE_CLASS,GC_STRING,P_STRING,V_STRING)\
class_<typename AE_CLASS::Parameter > (P_STRING, init<>() ) \
      .def(init<const size_t ,const typename AE_CLASS::Parameter::InferenceParameter & >())\
      .def(init<const size_t>())\
      .def_readwrite("steps", & AE_CLASS::Parameter::maxNumberOfSteps_)\
      .def_readwrite("graphCutParameter", & AE_CLASS::Parameter::parameter_)\
      .def ("set", &pyae::set<AE_CLASS,typename AE_CLASS::Parameter>, \
            ( \
            arg("steps")=1000,\
            arg("graphCutParameter")=typename AE_CLASS::Parameter::InferenceParameter()\
            ) \
      ) \
      ; \
   OPENGM_PYTHON_VERBOSE_VISITOR_EXPORTER(typename AE_CLASS::VerboseVisitorType,V_STRING );\
   OPENGM_PYTHON_INFERENCE_NO_RESET_EXPORTER(AE_CLASS,GC_STRING)

using namespace boost::python;

namespace  pyae{
   template<class INF,class PARAM>
   void set
   (
      PARAM & p,
      const size_t steps,
      const typename INF::Parameter::InferenceParameter & parameter
   ){
      p.maxNumberOfSteps_=steps;
      p.parameter_=parameter;
   }
}


template<class GM,class ACC>
void export_ae(){
   import_array(); 
   typedef GM PyGm;
   typedef typename PyGm::ValueType ValueType;
   // Boost Graphcut  swap and expansion Types


   typedef opengm::MinSTCutBoost<size_t, ValueType, opengm::KOLMOGOROV> MinStCutBoostKolmogorov;
   typedef opengm::GraphCut<PyGm, ACC, MinStCutBoostKolmogorov> PyGraphCutBoostKolmogorov;
   typedef opengm::AlphaExpansion<PyGm, PyGraphCutBoostKolmogorov> PyAlphaExpansionBoostKolmogorov;

   AE_EXPORT_HELPER(PyAlphaExpansionBoostKolmogorov,
      "AlphaExpansionBoostKolmogorov",
      "AlphaExpansionBoostKolmogorovParameter",
      "AlphaExpansionBoostKolmogorovVerboseVisitor");


}

template void export_ae<GmAdder,opengm::Minimizer>();
