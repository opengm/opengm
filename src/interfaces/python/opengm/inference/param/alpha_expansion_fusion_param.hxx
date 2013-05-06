#ifndef ALPHA_EXPANSION_FUSION_PARAM
#define ALPHA_EXPANSION_FUSION_PARAM

#include "param_exporter_base.hxx"
//solver specific
#include <opengm/inference/alphaexpansionfusion.hxx>

using namespace boost::python;

template<class INFERENCE>
class InfParamExporterAlphaExpansionFusion{

public:
   typedef typename INFERENCE::ValueType ValueType;
   typedef typename INFERENCE::LabelType LabelType;
   typedef typename INFERENCE::Parameter Parameter;
   typedef InfParamExporterAlphaExpansionFusion<INFERENCE> SelfType;

   static void set
   (
      Parameter & p,
      const size_t steps,
      const typename Parameter::LabelingIntitialType labelInitialType,
      const typename Parameter::OrderType orderType,
      const unsigned int randSeedOrder,
      const unsigned int randSeedLabel,
      const std::vector<LabelType> & labelOrder
   ){
      p.maxNumberOfSteps_=steps;
      p.labelInitialType_=labelInitialType;
      p.orderType_=orderType;
      p.randSeedOrder_=randSeedOrder;
      p.randSeedLabel_=randSeedLabel;
      p.labelOrder_=labelOrder;
   }

   void static exportInfParam(const std::string & className){
      class_<Parameter > (className.c_str(), init<>() ) 
         .def_readwrite("steps", & Parameter::maxNumberOfSteps_)
         .def_readwrite("labelInitialType", & Parameter::labelInitialType_)
         .def_readwrite("orderType", & Parameter::orderType_)
         .def_readwrite("randSeedOrder", & Parameter::randSeedOrder_)
         .def_readwrite("randSeedLabel", & Parameter::randSeedLabel_)
         .def_readwrite("labelOrder", & Parameter::labelOrder_)

         .def ("set", &SelfType::set, 
            ( 
               arg("steps")=1000,
               arg("labelInitialType")= Parameter::DEFAULT_LABEL,
               arg("orderType")=Parameter::DEFAULT_ORDER ,
               arg("randSeedOrder")=0 ,
               arg("randSeedLabel")=0,
               arg("labelOrder")=std::vector<LabelType>()
            )
         ) 
      ; 
   }
};



template<class GM,class ACC>
class InfParamExporter<opengm::AlphaExpansionFusion<GM, ACC > >  
:  public  
   InfParamExporterAlphaExpansionFusion<opengm::AlphaExpansionFusion< GM,ACC > > {
};


#endif