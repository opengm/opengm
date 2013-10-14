#include <boost/python.hpp>
#include <string>
#include "inf_def_visitor.hxx"

#include <opengm/inference/pbp.hxx>
#include <param/pbp_param.hxx>


template<class GM,class ACC>
void export_pbp(){
   using namespace boost::python;
   import_array();
   append_subnamespace("solver");

   // setup 
   InfSetup setup;
   setup.cite       = "todo";
   setup.algType    = "message-passing";
                      "\n\n";

   // export parameter
   typedef opengm::PBP<GM, ACC>  pyInference;
   exportInfParam<pyInference>("_Pbp");
   // export inference
   class_< pyInference>("_Pbp",init<const GM & >())  
   .def(InfSuite<pyInference,false>(std::string("Pbp"),setup))
   ;

}

template void export_pbp<opengm::python::GmAdder,opengm::Minimizer>();
