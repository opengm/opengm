#define GraphicalModelDecomposition BpInference_GraphicalModelDecomposition
#include <boost/python.hpp>
#include <string>
#include "inf_def_visitor.hxx"

#include <opengm/inference/messagepassing/messagepassing.hxx>
#include <param/message_passing_param.hxx>


template<class GM,class ACC>
void export_bp(){
   using namespace boost::python;
   import_array();
   append_subnamespace("solver");

   // setup 
   InfSetup setup;
   setup.cite       =  "";
   setup.algType    =  "message-passing";
   setup.guarantees =  "";
   setup.examples   = ">>> parameter = opengm.InfParam(steps=100,damping=0.5)\n"
                      ">>> inference = opengm.inference.TreeReweightedBp(gm=gm,accumulator='minimizer',parameter=parameter)\n"
                      "\n\n";
   typedef opengm::BeliefPropagationUpdateRules<GM,ACC> UpdateRulesType;
   typedef opengm::MessagePassing<GM, ACC,UpdateRulesType, opengm::MaxDistance> PyBp;
   // export parameter
   exportInfParam<exportTag::NoSubInf,PyBp>("_BeliefPropagation");
   // export inference
   class_< PyBp>("_BeliefPropagation",init<const GM & >())  
   .def(InfSuite<PyBp>(std::string("BeliefPropagation"),setup))
   ;
}

template void export_bp<GmAdder, opengm::Minimizer>();
template void export_bp<GmAdder, opengm::Maximizer>();
template void export_bp<GmMultiplier, opengm::Minimizer>();
template void export_bp<GmMultiplier, opengm::Maximizer>();
template void export_bp<GmMultiplier, opengm::Integrator>();

