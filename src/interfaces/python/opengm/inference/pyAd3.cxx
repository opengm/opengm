#ifdef WITH_AD3
#include <boost/python.hpp>
#include <string>
#include "inf_def_visitor.hxx"

#include <opengm/inference/external/ad3.hxx>
#include <param/ad3_param.hxx>


using namespace boost::python;


template<class GM,class ACC>
void export_ad3(){
   using namespace boost::python;
   import_array();
   append_subnamespace("solver");

   std::string srName = semiRingName  <typename GM::OperatorType,ACC >() ;

   // setup 
   InfSetup setup;
   setup.algType     = "dual decomposition";
   setup.guarantees  = "global optimal if solverType='ac3_ilp'";
   setup.examples   = ">>> parameter = opengm.InfParam(solverType='ac3_ilp')\n"
                      ">>> inference = opengm.inference.AStar(gm=gm,accumulator='minimizer',parameter=parameter)\n"
                      "\n\n";

   typedef opengm::external::AD3Inf<GM, ACC>  PyInf;

   // export enums
   const std::string enumName1=std::string("_Ad3SolverType")+srName;
   enum_<typename PyInf::SolverType> (enumName1.c_str())
      .value("ad3_lp",   PyInf::AD3_LP)
      .value("ad3_ilp",  PyInf::AD3_ILP)
      .value("psdd_lp",  PyInf::PSDD_LP)
   ;

   // export parameter
   exportInfParam<PyInf>("_Ad3");
   // export inference
   class_< PyInf>("_Ad3",init<const GM & >())  
   .def(InfSuite<PyInf,false>(std::string("Ad3"),setup))
   ;
}

template void export_ad3<GmAdder, opengm::Minimizer>();
template void export_ad3<GmAdder, opengm::Maximizer>();
//template void export_ad3<GmMultiplier, opengm::Minimizer>();
//template void export_ad3<GmMultiplier, opengm::Maximizer>();
#endif