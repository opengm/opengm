
#include <boost/python.hpp>
#include <string>
#include "inf_def_visitor.hxx"


#include <opengm/inference/trws/trws_adsal.hxx>
#include <param/adsal_param.hxx>
#include <opengm/python/opengmpython.hxx>
#include <opengm/python/converter.hxx>
#include <opengm/python/numpyview.hxx>
#include <opengm/python/pythonfunction.hxx>

using namespace boost::python;



template<class GM,class ACC>
void export_adsal(){
   using namespace boost::python;
   import_array();
   append_subnamespace("solver");

   // setup 
   std::string srName = semiRingName  <typename GM::OperatorType,ACC >() ;
   InfSetup setup;
   setup.algType    = "trws";
   setup.limitations= "max 2.order ( ? double check)";

   typedef opengm::ADSal<GM,ACC>  PyAdsal;               

   /*
   const std::string enumName=std::string("_BpUpdateRuleLibDai")+srName;
   enum_<typename PyMqpbo::PermutationType> (enumName.c_str())
   .value("none", PyMqpbo::NONE)
   .value("random", PyMqpbo::RANDOM)
   .value("minarg", PyMqpbo::MINMARG)
   ;
   */
   // export parameter

   exportInfParam<PyAdsal>("_Adsal");
   // export inference
   class_< PyAdsal>("_Adsal",init<const GM & >())  
   .def(InfSuite<PyAdsal,false>(std::string("Adsal"),setup))
   ;
}

template void export_adsal<opengm::python::GmAdder,opengm::Minimizer>();
template void export_adsal<opengm::python::GmMultiplier,opengm::Maximizer>();

