#ifdef WITH_QPBO

#include <boost/python.hpp>
#include <string>
#include "inf_def_visitor.hxx"
#include "partial_optimal_def_suite.hxx"


#define MQPBO_PYTHON_WRAPPER_HACK
#include <opengm/inference/mqpbo.hxx>
# include <param/mqpbo_param.hxx>

using namespace boost::python;



template<class GM,class ACC>
void export_mqpbo(){
   using namespace boost::python;
   import_array();
   append_subnamespace("solver");

   // setup 
   std::string srName = semiRingName  <typename GM::OperatorType,ACC >() ;
   InfSetup setup;
   setup.algType    = "qpbo";
   setup.guarantees = "partial optimal";
   setup.limitations= "max 2.order";
   setup.examples   = ">>> parameter = opengm.InfParam(TODO)\n"
                      ">>> inference = opengm.inference.Mqpbo(gm=gm,accumulator='minimizer',parameter=parameter)\n"
                      "\n\n";
   setup.dependencies = "This algorithm needs the Qpbo library from ??? , " 
                        "compile OpenGM with CMake-Flag ``WITH_QPBO`` set to ``ON`` ";

   typedef opengm::MQPBO<GM,ACC>  PyMqpbo;               
   const std::string enumName=std::string("_BpUpdateRuleLibDai")+srName;
   enum_<typename PyMqpbo::PermutationType> (enumName.c_str())
   .value("none", PyMqpbo::NONE)
   .value("random", PyMqpbo::RANDOM)
   .value("minarg", PyMqpbo::MINMARG)
   ;
   // export parameter
   exportInfParam<PyMqpbo>("_Mqpbo");
   // export inference
   class_< PyMqpbo>("_Mqpbo",init<const GM & >())  
   .def(InfSuite<PyMqpbo,false,true,false>(std::string("Mqpbo"),setup))
   .def(PartialOptimalitySuite2<PyMqpbo>())
   ;
}

template void export_mqpbo<opengm::python::GmAdder,opengm::Minimizer>();


#endif