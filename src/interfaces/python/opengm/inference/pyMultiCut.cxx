#ifdef WITH_CPLEX
#include <boost/python.hpp>
#include <boost/python.hpp>
#include <string>
#include "inf_def_visitor.hxx"

#include <opengm/inference/multicut.hxx>
#include <param/multicut_param.hxx>

#include "multicut_def_suite.hxx"

using namespace boost::python;

template<class GM,class ACC>
void export_multicut(){
   using namespace boost::python;
   import_array();
   append_subnamespace("solver");

   // setup 
   InfSetup setup;
   setup.algType     = "multicut";
   setup.limitations = "model must be a generalized potts model";

   // export parameter
   typedef opengm::Multicut<GM, ACC>  PyMulticut;
   exportInfParam<PyMulticut>("_Multicut");
   // export inference
   class_< PyMulticut>("_Multicut",init<const GM & >())  
   .def(InfSuite<PyMulticut,false>(std::string("Multicut"),setup))
   .def(MulticutInferenceSuite<PyMulticut>())
   ;

}
template void export_multicut<opengm::python::GmAdder,opengm::Minimizer>();
#endif