//#define GraphicalModelDecomposition TrBpInference_GraphicalModelDecomposition
#include <boost/python.hpp>
#include <string>
#include "inf_def_visitor.hxx"
#include "marginal_def_visitor.hxx"
#include <opengm/inference/messagepassing/messagepassing.hxx>
#include <param/message_passing_param.hxx>


template<class GM,class ACC>
void export_trbp(){
   using namespace boost::python;
   import_array();
   append_subnamespace("solver");

   // setup 
   InfSetup setup;
   setup.cite         = "";
   setup.algType      = "message-passing";
   setup.examples     = ">>> parameter = opengm.InfParam(steps=100,damping=0.5)\n"
                        ">>> inference = opengm.inference.TreeReweightedBp(gm=gm,accumulator='minimizer',parameter=parameter)\n"
                        "\n\n";
   setup.dependencies = "This algorithm needs the Trws library from ??? , " 
                        "compile OpenGM with CMake-Flag ``WITH_TRWS`` set to ``ON`` ";
   typedef opengm::TrbpUpdateRules<GM,ACC> UpdateRulesType;
   typedef opengm::MessagePassing<GM, ACC,UpdateRulesType, opengm::MaxDistance> PyTrBp;
   // export parameter
   exportInfParam<PyTrBp>("_TreeReweightedBp");
   // export inference
   class_< PyTrBp>("_TreeReweightedBp",init<const GM & >())  
   .def(InfSuite<PyTrBp>(std::string("TreeReweightedBp"),setup))
   .def(MarginalSuite<PyTrBp>())
   ;
}

template void export_trbp<opengm::python::GmAdder, opengm::Minimizer>();
template void export_trbp<opengm::python::GmAdder, opengm::Maximizer>();
template void export_trbp<opengm::python::GmAdder, opengm::Integrator>();
template void export_trbp<opengm::python::GmMultiplier, opengm::Minimizer>();
template void export_trbp<opengm::python::GmMultiplier, opengm::Maximizer>();
template void export_trbp<opengm::python::GmMultiplier, opengm::Integrator>();

