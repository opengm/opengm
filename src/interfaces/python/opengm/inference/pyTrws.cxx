#ifdef WITH_TRWS
// redefined symbol since pyTrws.cxx:(.text+0x0): multiple definition of `DefaultErrorFn(char*)'
#define DefaultErrorFn DefaultErrorFn_TrwsExternal 

#include <boost/python.hpp>
#include <string>
#include "inf_def_visitor.hxx"

#include <opengm/inference/external/trws.hxx>
#include <param/trws_external_param.hxx>

template<class GM,class ACC>
void export_trws(){
   using namespace boost::python;
   import_array();
   append_subnamespace("solver");

   // setup 
   std::string srName = semiRingName  <typename GM::OperatorType,ACC >() ;
   InfSetup setup;
   setup.algType    =   "message-passing";
   setup.examples   =   ">>> parameter = opengm.InfParam(steps=100)\n"
                        ">>> inference = opengm.inference.TrwsExternal(gm=gm,accumulator='minimizer',parameter=parameter)\n"
                        "\n\n";
   setup.dependencies = "This algorithm needs the Trws library from ??? , " 
                        "compile OpenGM with CMake-Flag ``WITH_TRWS` set to ``ON`` ";
   // export parameter
   typedef opengm::external::TRWS<GM>  PyTrws; 
   const std::string enumName=std::string("_TrwsExternalEnergyType")+srName;
   enum_<typename PyTrws::Parameter::EnergyType> (enumName.c_str())
      .value("view", PyTrws::Parameter::VIEW)
      .value("tables", PyTrws::Parameter::TABLES)
      .value("tl1", PyTrws::Parameter::TL1)
      .value("tl2", PyTrws::Parameter::TL2)
   ;

   exportInfParam<PyTrws>("_TrwsExternal");
   // export inference
   class_< PyTrws>("_TrwsExternal",init<const GM & >())  
      .def(InfSuite<PyTrws,false,true,false>(std::string("TrwsExternal"),setup))
   ;


}

template void export_trws<opengm::python::GmAdder,opengm::Minimizer>();

#endif //WITH_TRWS