#if defined(WITH_QPBO) || defined(WITH_BLOSSOM5) && defined(WITH_PLANARITY)
#include <boost/python.hpp>
#include <string>
#include "inf_def_visitor.hxx"



#include <opengm/inference/cgc.hxx>
#include <param/cgc_param.hxx>





template<class GM,class ACC>
void export_cgc(){
   using namespace boost::python;
   import_array();
   append_subnamespace("solver");

   // setup 
   InfSetup setup;
   setup.cite       = "Thorsten Beier";
   setup.algType    = "multicut";



   // export parameter
   typedef opengm::CGC<GM, ACC>  PyInf;
   exportInfParam<PyInf>("_Cgc");
   // export inference
   class_< PyInf>("_Cgc",init<const GM & >())  
   .def(InfSuite<PyInf>(std::string("Cgc"),setup))
   ;
}

template void export_cgc<opengm::python::GmAdder,opengm::Minimizer>();

#endif
