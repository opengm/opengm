#include <boost/python.hpp>
#include <string>
#include "inf_def_visitor.hxx"

#include <opengm/inference/dynamicprogramming.hxx>
#include <param/dynamic_programming_param.hxx>


#include <opengm/python/opengmpython.hxx>
#include <opengm/python/converter.hxx>
#include <opengm/python/numpyview.hxx>
#include <opengm/python/pythonfunction.hxx>

template<class GM,class ACC>
void export_dynp(){
   using namespace boost::python;
   import_array();
   append_subnamespace("solver");
   // setup 
   InfSetup setup;
   setup.algType     = "dynamic-programming";
   setup.guarantees  = "global optimal";
   setup.examples    = ">>> inference = opengm.inference.DynamicProgramming(gm=gm,accumulator='minimizer')\n"
                       "\n\n"; 
   setup.limitations = "graphical model must be a tree / must not have loops";

   // export parameter
   typedef opengm::DynamicProgramming<GM, ACC>  PyDynamicProgramming;
   exportInfParam<PyDynamicProgramming>("_DynamicProgramming");
   // export inference
   class_< PyDynamicProgramming>("_DynamicProgramming",init<const GM & >())  
   .def(InfSuite<PyDynamicProgramming,false>(std::string("DynamicProgramming"),setup))
   ;
}

template void export_dynp<opengm::python::GmAdder,opengm::Minimizer>();
template void export_dynp<opengm::python::GmAdder,opengm::Maximizer>();
template void export_dynp<opengm::python::GmMultiplier,opengm::Minimizer>();
template void export_dynp<opengm::python::GmMultiplier,opengm::Maximizer>();
