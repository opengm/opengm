#include <boost/python.hpp>
#include <string>
#include "inf_def_visitor.hxx"

#include <opengm/inference/graphcut.hxx>
#include <opengm/inference/alphabetaswap.hxx>
#ifdef WITH_BOOST
#  include <opengm/inference/auxiliary/minstcutboost.hxx>
#endif
#ifdef WITH_MAXFLOW
#  include <opengm/inference/auxiliary/minstcutkolmogorov.hxx>
#endif

#include <param/alpha_beta_swap_param.hxx>


template<class GM,class ACC>
void export_abswap(){

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
      typedef opengm::AlphaBetaSwap<PyGm, PyGraphCutKolmogorov> PyAlphaBetaSwapKolmogorov;
      // export parameter
      exportInfParam<PyAlphaBetaSwapKolmogorov>("_AlphaBetaSwap_Kolmogorov");
      // export inference
      boost::python::class_< PyAlphaBetaSwapKolmogorov>("_AlphaBetaSwap_Kolmogorov",init<const GM & >())  
      .def(InfSuite<PyAlphaBetaSwapKolmogorov,false>(std::string("AlphaBetaSwap"),setup))
      ;
   }
   #endif
   {
      // set up hyper parameter name for this template
      setup.isDefault=!withMaxFlow;
      setup.hyperParameters= StringVector(1,std::string("boost-kolmogorov"));
      typedef opengm::MinSTCutBoost<size_t, ValueType, opengm::KOLMOGOROV> MinStCutBoostKolmogorov;
      typedef opengm::GraphCut<PyGm, ACC, MinStCutBoostKolmogorov> PyGraphCutBoostKolmogorov;
      typedef opengm::AlphaBetaSwap<PyGm, PyGraphCutBoostKolmogorov> PyAlphaBetaSwapBoostKolmogorov;
      // export parameter
      exportInfParam<PyAlphaBetaSwapBoostKolmogorov>("_AlphaBetaSwap_Boost_Kolmogorov");
      // export inference
      boost::python::class_< PyAlphaBetaSwapBoostKolmogorov>("_AlphaBetaSwap_Boost_Kolmogorov",init<const GM & >())  
      .def(InfSuite<PyAlphaBetaSwapBoostKolmogorov,false>(std::string("AlphaBetaSwap"),setup))
      ;
   }

   {
      // set up hyper parameter name for this template
      setup.isDefault=false;
      setup.hyperParameters= StringVector(1,std::string("push-relabel"));
      typedef opengm::MinSTCutBoost<size_t, ValueType, opengm::PUSH_RELABEL> MinStCutBoostPushRelabel;
      typedef opengm::GraphCut<PyGm, ACC, MinStCutBoostPushRelabel> PyGraphCutBoostPushRelabel;
      typedef opengm::AlphaBetaSwap<PyGm, PyGraphCutBoostPushRelabel> PyAlphaBetaSwapPushRelabel;
      // export parameter
      exportInfParam<PyAlphaBetaSwapPushRelabel>("_AlphaBetaSwap_Boost_Push_Relabel");
      // export inference
      boost::python::class_< PyAlphaBetaSwapPushRelabel>("_AlphaBetaSwap_Boost_Push_Relabel",init<const GM & >())  
      .def(InfSuite<PyAlphaBetaSwapPushRelabel,false>(std::string("AlphaBetaSwap"),setup))
   ;
   }
   


}

template void export_abswap<opengm::python::GmAdder,opengm::Minimizer>();
