
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
#include <opengm/inference/alphaexpansion.hxx>
#ifdef WITH_BOOST
#  include <opengm/inference/auxiliary/minstcutboost.hxx>
#endif
#ifdef WITH_MAXFLOW
#  include <opengm/inference/auxiliary/minstcutkolmogorov.hxx>
#endif

#include "nifty_iterator.hxx"
#include "inferencehelpers.hxx"

#include "export_typedes.hxx"

#define GRAPH_CUT_EXPORT_HELPER(GC_CLASS,GC_STRING,P_STRING,V_STRING)\
class_<typename GC_CLASS::Parameter > (P_STRING, init<>() ) \
      .def(init<const typename  PyGm::ValueType>())\
      .def ("set", &pygc::set<typename GC_CLASS::Parameter>, \
            ( \
            arg("scale")=10 \
            ) \
      ) \
      .def_readwrite("scale", & GC_CLASS::Parameter::scale_)\
      ; \
   OPENGM_PYTHON_VERBOSE_VISITOR_EXPORTER(typename GC_CLASS::VerboseVisitorType,V_STRING );\
   OPENGM_PYTHON_INFERENCE_NO_RESET_EXPORTER(GC_CLASS,GC_STRING)   




using namespace boost::python;



namespace pygc{
   template<class PARAM>
   inline void set
   (
      PARAM & p,
      double scale     
   ){
      p.scale_=scale;
   }
}



template<class GM,class ACC>
void export_graphcut(){
   import_array(); 
   typedef GM PyGm;
   typedef typename PyGm::ValueType ValueType;
   
   typedef opengm::MinSTCutBoost<size_t, ValueType, opengm::KOLMOGOROV> MinStCutBoostKolmogorov;
   typedef opengm::GraphCut<PyGm, ACC, MinStCutBoostKolmogorov> PyGraphCutBoostKolmogorov;
  


   GRAPH_CUT_EXPORT_HELPER(PyGraphCutBoostKolmogorov,"GraphCutBoostKolmogorov", "GraphCutBoostKolmogorovParameter","GraphCutBoostKolmogorovVerboseVisitor");
   

}

template void export_graphcut<GmAdder,opengm::Minimizer>();
//template void export_graphcut<GmAdder,opengm::Maximizer>();

