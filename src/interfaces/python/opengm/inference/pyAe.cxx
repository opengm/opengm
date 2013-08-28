#include <boost/python.hpp>
#include <string>
#include "inf_def_visitor.hxx"

#include <opengm/inference/inference.hxx>
#include <opengm/inference/graphcut.hxx>
#include <opengm/inference/alphaexpansion.hxx>
#ifdef WITH_BOOST
#  include <opengm/inference/auxiliary/minstcutboost.hxx>
#endif
#ifdef WITH_MAXFLOW
#  include <opengm/inference/auxiliary/minstcutkolmogorov.hxx>
#endif
#include <param/alpha_expansion_param.hxx>



using namespace boost::python;

template<class GM,class ACC>
void export_ae(){

   import_array(); 
   typedef GM PyGm;
   typedef typename PyGm::ValueType ValueType;
   typedef typename PyGm::IndexType IndexType;
   
   append_subnamespace("solver");


   #ifdef WITH_MAXFLOW
      const bool withMaxFlow=true;
   #else
      const bool withMaxFlow=false;
   #endif
   

   // documentation 
   InfSetup setup;
   setup.cite       = "";
   setup.algType    = "graphCut / movemaking";
   setup.limitations= "max 2.order, must be submodular";
   setup.hyperParameterKeyWords = StringVector(1,std::string("minStCut"));
   setup.hyperParametersDoc     = StringVector(1,std::string("minStCut implementation of graphcut"));
   setup.dependencies = "to use ``'kolmogorov'`` as minStCut the kolmogorov max flow library, " 
                        "compile OpenGM with CMake-Flag ``WITH_MAXFLOW`` set to ``ON`` ";

   #ifdef WITH_MAXFLOW
   {
      // set up hyper parameter name for this template
      setup.isDefault=withMaxFlow;
      setup.hyperParameters= StringVector(1,std::string("kolmogorov"));
      typedef opengm::external::MinSTCutKolmogorov<size_t,ValueType> MinStCutKolmogorov;
      typedef opengm::GraphCut<PyGm, ACC, MinStCutKolmogorov>        PyGraphCutKolmogorov;
      typedef opengm::AlphaExpansion<PyGm, PyGraphCutKolmogorov> PyAlphaExpansionKolmogorov;
      // export parameter
      exportInfParam<PyAlphaExpansionKolmogorov>("_AlphaExpansion_Kolmogorov");
      // export inference
      class_< PyAlphaExpansionKolmogorov>("_AlphaExpansion_Kolmogorov",init<const GM & >())  
      .def(InfSuite<PyAlphaExpansionKolmogorov,false>(std::string("AlphaExpansion"),setup))
      ;
   }
   #endif

   {
      // set up hyper parameter name for this template
      setup.isDefault=!withMaxFlow;
      setup.hyperParameters= StringVector(1,std::string("boost-kolmogorov"));
      typedef opengm::MinSTCutBoost<size_t, ValueType, opengm::KOLMOGOROV> MinStCutBoostKolmogorov;
      typedef opengm::GraphCut<PyGm, ACC, MinStCutBoostKolmogorov> PyGraphCutBoostKolmogorov;
      typedef opengm::AlphaExpansion<PyGm, PyGraphCutBoostKolmogorov> PyAlphaExpansionBoostKolmogorov;
      // export parameter
      exportInfParam<PyAlphaExpansionBoostKolmogorov>("_AlphaExpansion_Boost_Kolmogorov");
      // export inference
      class_< PyAlphaExpansionBoostKolmogorov>("_AlphaExpansion_Boost_Kolmogorov",init<const GM & >())  
      .def(InfSuite<PyAlphaExpansionBoostKolmogorov,false>(std::string("AlphaExpansion"),setup))
      ;
   }

   {
   // set up hyper parameter name for this template
   setup.isDefault=false;
   setup.hyperParameters= StringVector(1,std::string("push-relabel"));
   typedef opengm::MinSTCutBoost<size_t, ValueType, opengm::PUSH_RELABEL> MinStCutBoostPushRelabel;
   typedef opengm::GraphCut<PyGm, ACC, MinStCutBoostPushRelabel> PyGraphCutBoostPushRelabel;
   typedef opengm::AlphaExpansion<PyGm, PyGraphCutBoostPushRelabel> PyAlphaExpansionPushRelabel;
   // export parameter
   exportInfParam<PyAlphaExpansionPushRelabel>("_AlphaExpansion_Boost_Push_Relabel");
   // export inference
   class_< PyAlphaExpansionPushRelabel>("_AlphaExpansion_Boost_Push_Relabel",init<const GM & >())  
   .def(InfSuite<PyAlphaExpansionPushRelabel,false>(std::string("AlphaExpansion"),setup))
   ;
   }







}

template void export_ae<opengm::python::GmAdder,opengm::Minimizer>();
