#ifdef WITH_QPBO

#include <boost/python.hpp>
#include <string>
#include "inf_def_visitor.hxx"

#include <opengm/inference/inference.hxx>
#include <opengm/inference/alphaexpansionfusion.hxx>
#include <param/alpha_expansion_fusion_param.hxx>



using namespace boost::python;

template<class GM,class ACC>
void export_ae_fusion(){

   import_array(); 
   typedef GM PyGm;
   typedef typename PyGm::ValueType ValueType;
   typedef typename PyGm::IndexType IndexType;
   
   append_subnamespace("solver");

   std::string srName = semiRingName  <typename GM::OperatorType,ACC >() ;
   // documentation 
   InfSetup setup;
   setup.cite       = "";
   setup.algType    = "fusion / movemaking";
   setup.dependencies = "This algorithm needs the Qpbo library from ??? , " 
                        "compile OpenGM with CMake-Flag ``WITH_QPBO`` set to ``ON`` ";
                        
   typedef opengm::AlphaExpansionFusion<PyGm, ACC> PyAlphaExpansionFusion;

   const std::string enumName1=std::string("_AlphaExpansionFusionLabelingIntitialType")+srName;
   enum_<typename PyAlphaExpansionFusion::Parameter::LabelingIntitialType> (enumName1.c_str())
      .value("default",  PyAlphaExpansionFusion::Parameter::DEFAULT_LABEL)
      .value("random",   PyAlphaExpansionFusion::Parameter::RANDOM_LABEL)
      .value("localOpt", PyAlphaExpansionFusion::Parameter::LOCALOPT_LABEL)
      .value("explicit", PyAlphaExpansionFusion::Parameter::EXPLICIT_LABEL)
   ;
   const std::string enumName2=std::string("_AlphaExpansionFusionOrderType")+srName;
   enum_<typename PyAlphaExpansionFusion::Parameter::OrderType> (enumName2.c_str())
      .value("default",  PyAlphaExpansionFusion::Parameter::DEFAULT_ORDER)
      .value("random",   PyAlphaExpansionFusion::Parameter::RANDOM_ORDER)
      .value("explicit", PyAlphaExpansionFusion::Parameter::EXPLICIT_ORDER)
   ;

   typedef opengm::AlphaExpansionFusion<PyGm, ACC> PyAlphaExpansionFusion;
   // export parameter
   exportInfParam<PyAlphaExpansionFusion>("_AlphaExpansionFusion");
   // export inference
   class_< PyAlphaExpansionFusion>("_AlphaExpansionFusion",init<const GM & >())  
   .def(InfSuite<PyAlphaExpansionFusion,false>(std::string("AlphaExpansionFusion"),setup))
   ;
   
}

template void export_ae_fusion<opengm::python::GmAdder,opengm::Minimizer>();

#endif