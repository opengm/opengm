#define PY_ARRAY_UNIQUE_SYMBOL PyArrayHandleInference

#include <boost/python.hpp>
#include <stdexcept>

#include <stddef.h>
#include <string>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/inference/inference.hxx>

using namespace boost::python;


void export_inference(){
    //------------------------------------------------------------------------------------
   // InferenceTermination
   //------------------------------------------------------------------------------------
   enum_<opengm::InferenceTermination > ("InferenceTermination")
   .value("UNKNOWN", opengm::UNKNOWN)
   .value("NORMAL", opengm::NORMAL)
   .value("TIMEOUT", opengm::TIMEOUT)
   .value("CONVERGENCE", opengm::CONVERGENCE)
   .value("INFERENCE_ERROR", opengm::INFERENCE_ERROR)
   ;
}

