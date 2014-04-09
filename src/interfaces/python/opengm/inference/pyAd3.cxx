#ifdef WITH_AD3
#include <boost/python.hpp>
#include <string>
#include "inf_def_visitor.hxx"

#include <opengm/inference/external/ad3.hxx>
#include <param/ad3_param.hxx>


using namespace boost::python;


template<class INF>
boost::python::object pyAd3Posteriors(const INF & inf){
   // get maximum label
   typedef typename INF::GraphicalModelType GraphicalModelType;
   typedef typename INF::AccumulationType AccumulationType;
   OPENGM_GM_TYPE_TYPEDEFS;

   const GraphicalModelType gm = inf.graphicalModel();
   LabelType maxNLabels=0;
   for(IndexType vi=0;vi<gm.numberOfVariables();++vi){
     maxNLabels = std::max(maxNLabels,gm.numberOfLabels(vi));
   }
   boost::python::object obj = opengm::python::get2dArray<double>(gm.numberOfVariables(),maxNLabels);
   double * castedPtr =opengm::python::getCastedPtr<ValueType>(obj);
   std::fill(castedPtr,castedPtr+maxNLabels*gm.numberOfVariables(),0.0);

   const std::vector<double> posteriors = inf.posteriors();
   opengm::UInt64Type cPost=0;
   opengm::UInt64Type cTot=0;
   for(IndexType vi=0;vi<gm.numberOfVariables();++vi){
      for(LabelType l=0;l<gm.numberOfLabels(vi);++l){
         // fixme: this can be implemented way faster
         if(l<gm.numberOfLabels(vi)){
            castedPtr[cTot]=posteriors[cPost];
            ++cPost;
         }
         ++cTot;
      }
   }
   return obj;   
}



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
   .def("posteriors",pyAd3Posteriors<PyInf>,
      "get ad3 posteriors, shape of result is  (nvar x maxNumberOfLabels)"
   )
   ;
}

template void export_ad3<opengm::python::GmAdder, opengm::Minimizer>();
template void export_ad3<opengm::python::GmAdder, opengm::Maximizer>();
//template void export_ad3<GmMultiplier, opengm::Minimizer>();
//template void export_ad3<GmMultiplier, opengm::Maximizer>();
#endif