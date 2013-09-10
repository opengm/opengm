#ifdef WITH_MRF

#include <boost/python.hpp>
#include <string>
#include "inf_def_visitor.hxx"

#include <opengm/inference/external/mrflib.hxx>
#include <param/mrf_param.hxx>

using namespace boost::python;

template<class GM,class ACC>
void export_mrf(){
   using namespace boost::python;
   import_array();
   append_subnamespace("solver");


   // setup 
   InfSetup setup;
   setup.algType     = "grid2d-algorithms";
   setup.limitations = "graph must be a second order 2d grid";
   setup.dependencies = "This algorithm needs the MrfLib library from ??? , " 
                        "compile OpenGM with CMake-Flag ``WITH_MRF` set to ``ON`` ";

   typedef typename opengm::external::MRFLIB<GM> PyMrfLib;
   std::string srName = semiRingName  <typename GM::OperatorType,ACC >() ;


   const std::string enumName1=std::string("_MrfLibEnergyType")+srName;
   enum_<typename PyMrfLib::Parameter::EnergyType> (enumName1.c_str())
      .value("view", PyMrfLib::Parameter::VIEW)
      .value("tables", PyMrfLib::Parameter::TABLES)
      .value("tl1", PyMrfLib::Parameter::TL1)
      .value("tl2", PyMrfLib::Parameter::TL2)
      .value("weightedTable", PyMrfLib::Parameter::WEIGHTEDTABLE)
   ;

   const std::string enumName2=std::string("_MrfLibInferenceType")+srName;
   enum_<typename PyMrfLib::Parameter::InferenceType> (enumName2.c_str())
      .value("icm", PyMrfLib::Parameter::ICM)
      .value("expansion", PyMrfLib::Parameter::EXPANSION)
      .value("swap", PyMrfLib::Parameter::SWAP)
      .value("maxProdBp", PyMrfLib::Parameter::MAXPRODBP)
      .value("trws", PyMrfLib::Parameter::TRWS)
      .value("bps", PyMrfLib::Parameter::BPS)
   ;
   // export parameter
   exportInfParam<PyMrfLib>("_MrfLib");
   // export inference
   class_< PyMrfLib>("_MrfLib",init<const GM & >())  
   .def(InfSuite<PyMrfLib,false>(std::string("MrfLib"),setup))
   ;

}
template void export_mrf<opengm::python::GmAdder,opengm::Minimizer>();
#endif