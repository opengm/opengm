#include <boost/python.hpp>
#include <string>
#include "inf_def_visitor.hxx"
#include "partial_optimal_def_suite.hxx"

#include <opengm/inference/qpbo.hxx>
#ifdef WITH_QPBO
#include <opengm/inference/external/qpbo.hxx>
#endif //WITH_QPBO
#include <opengm/inference/auxiliary/minstcutboost.hxx>
#ifdef WITH_MAXFLOW
#  include <opengm/inference/auxiliary/minstcutkolmogorov.hxx>
#endif
# include <param/qpbo_param.hxx>
#ifdef WITH_QPBO
# include <param/qpbo_external_param.hxx>
#endif
using namespace boost::python;


#ifdef WITH_QPBO
template<class GM,class ACC>
void export_qpbo_external(){
   using namespace boost::python;
   import_array();
   append_subnamespace("solver");

   // setup 
   InfSetup setup;
   setup.algType    = "graph-cut";
   setup.guarantees = "partial optimal";
   setup.limitations= "max 2.order, binary labels";
   setup.examples   = ">>> parameter = opengm.InfParam(strongPersistency=True,useImproveing=False,useProbeing=False)\n"
                      ">>> inference = opengm.inference.QpboExternal(gm=gm,accumulator='minimizer',parameter=parameter)\n"
                      "\n\n";
   setup.notes      = ".. seealso::\n\n"
                      "   :class:`opengm.inference.Qpbo` ";
   setup.dependencies = "This algorithm needs the Qpbo library from ??? , " 
                        "compile OpenGM with CMake-Flag ``WITH_QPBO`` set to ``ON`` ";
   // export parameter
   typedef opengm::external::QPBO<GM>  PyQpboExternal;
   exportInfParam<PyQpboExternal>("_QpboExternal");
   // export inference
   class_< PyQpboExternal>("_QpboExternal",init<const GM & >())  
   .def(InfSuite<PyQpboExternal,false,true,false>(std::string("QpboExternal"),setup))
   .def(PartialOptimalitySuite<PyQpboExternal>())
   ;
}

template void export_qpbo_external<opengm::python::GmAdder,opengm::Minimizer>();
#endif //WITH_QPBO

template<class GM,class ACC>
void export_qpbo(){
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
   

   // setup 
   InfSetup setup;
   setup.algType    = "graph-cut";
   setup.guarantees = "partial optimal";
   setup.limitations= "max 2.order, binary labels";
   setup.examples   = ">>> inference = opengm.inference.Qpbo(gm=gm,accumulator='minimizer'\n\n"; 
   setup.notes      = ".. seealso::\n\n"
                      "   :class:`opengm.inference.QpboExternal`";

   setup.hyperParameterKeyWords = StringVector(1,std::string("minStCut"));
   setup.hyperParametersDoc     = StringVector(1,std::string("minStCut implementation of graphcut"));


   #ifdef WITH_MAXFLOW
      // set up hyper parameter name for this template
      setup.isDefault=withMaxFlow;
      setup.hyperParameters= StringVector(1,std::string("kolmogorov"));
      typedef opengm::external::MinSTCutKolmogorov<size_t,ValueType> MinStCutKolmogorov;
      typedef opengm::QPBO<PyGm, MinStCutKolmogorov>        PyGraphCutKolmogorov;
      // export parameter
      exportInfParam<PyGraphCutKolmogorov>("_Qpbo_Kolmogorov");
      // export inference
      class_< PyGraphCutKolmogorov>("_Qpbo_Kolmogorov",init<const GM & >())  
      .def(InfSuite<PyGraphCutKolmogorov,false,true,false>(std::string("Qpbo"),setup))
      .def(PartialOptimalitySuite<PyGraphCutKolmogorov>())
      ;
   #endif


   // set up hyper parameter name for this template
   setup.isDefault=!withMaxFlow;
   setup.hyperParameters= StringVector(1,std::string("boost-kolmogorov"));
   typedef opengm::MinSTCutBoost<size_t, ValueType, opengm::KOLMOGOROV> MinStCutBoostKolmogorov;
   typedef opengm::QPBO<PyGm, MinStCutBoostKolmogorov> PyGraphCutBoostKolmogorov;
   // export parameter
   exportInfParam<PyGraphCutBoostKolmogorov>("_Qpbo_Boost_Kolmogorov");
   // export inference
   class_< PyGraphCutBoostKolmogorov>("_Qpbo_Boost_Kolmogorov",init<const GM & >())  
   .def(InfSuite<PyGraphCutBoostKolmogorov,false,true,false>(std::string("Qpbo"),setup))
   .def(PartialOptimalitySuite<PyGraphCutBoostKolmogorov>())
   ;

   // set up hyper parameter name for this template
   setup.isDefault=false;
   setup.hyperParameters= StringVector(1,std::string("push-relabel"));
   typedef opengm::MinSTCutBoost<size_t, ValueType, opengm::PUSH_RELABEL> MinStCutBoostPushRelabel;
   typedef opengm::QPBO<PyGm, MinStCutBoostPushRelabel> PyGraphCutBoostPushRelabel;
   // export parameter
   exportInfParam<PyGraphCutBoostPushRelabel>("_Qpbo_Boost_Push_Relabel");
   // export inference
   class_< PyGraphCutBoostPushRelabel>("_Qpbo_Boost_Push_Relabel",init<const GM & >())  
   .def(InfSuite<PyGraphCutBoostPushRelabel,false,true,false>(std::string("Qpbo"),setup))
   .def(PartialOptimalitySuite<PyGraphCutBoostPushRelabel>())
   ;


}
      
