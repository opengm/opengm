#ifndef MESSAGE_PASSING_PARAMETER
#define MESSAGE_PASSING_PARAMETER

#include "param_exporter_base.hxx"
//solver specific
#include <opengm/inference/messagepassing/messagepassing.hxx>

using namespace boost::python;

template<class INFERENCE>
class InfParamExporterMessagePassing{

public:
   typedef typename INFERENCE::ValueType ValueType;
   typedef typename INFERENCE::Parameter Parameter;
   typedef InfParamExporterMessagePassing<INFERENCE> SelfType;

   static std::string paramAsString(const Parameter & param) {
      std::stringstream ss;
      ss<<"[ steps="<<param.maximumNumberOfSteps_;
      ss<<", damping="<<param.damping_;
      ss<<", convergenceBound="<<param.bound_<<" ]";
      ss<<", isAcyclic="<<param.isAcyclic_<<" ]";
      return ss.str();
   }

   static void set
   (
      Parameter & p,
      const size_t maximumNumberOfSteps,
      const double damping,
      const double convergenceBound,
      const opengm::Tribool isAcyclic
   ){
      p.maximumNumberOfSteps_=maximumNumberOfSteps;
      p.damping_=damping;
      p.bound_=convergenceBound;
      p.isAcyclic_=isAcyclic;
   }


    void static exportInfParam(const std::string & className){
         class_<Parameter > (className.c_str(), init< >() )
         .def_readwrite("steps", &Parameter::maximumNumberOfSteps_,
         "Number of message passing updates"
         )
         .def_readwrite("damping", &Parameter::damping_,
         "Damping must be in [0,1]"
         )
         .def_readwrite("convergenceBound", &Parameter::bound_,
         "Convergence bound stops message passing updates when message change is smaller than ``convergenceBound``"
         )
         .def_readwrite("isAcyclic", &Parameter::isAcyclic_,
         //.add_property("isAcyclic", &pytrbp::getIsAcyclic<Parameter>, pytrbp::setIsAcyclic<Parameter>,
         "isAcyclic can be:\n\n"
         "  -``'maybe'`` : if its unknown that the gm is acyclic (default)\n\n"
         "  -``True`` : if its known that the gm is acyclic (gm has no loops)\n\n"
         "  -``False`` : if its known that the gm is not acyclic (gm has loops)\n\n"
         )
         .def ("set", &SelfType::set, 
               (
               boost::python::arg("steps")=100,
               boost::python::arg("damping")= 0,
               boost::python::arg("convergenceBound")=0,
               boost::python::arg("isAcyclic")=opengm::Tribool(opengm::Tribool::Maybe)
               )
         )
         ;
    }
};


 //typedef opengm::BeliefPropagationUpdateRules<GM,ACC> UpdateRulesType;
 //typedef opengm::MessagePassing<GM, ACC,UpdateRulesType, opengm::MaxDistance> PyBp;

 //typedef opengm::TrbpUpdateRules<GM,ACC> UpdateRulesType2;
 //typedef opengm::MessagePassing<GM, ACC,UpdateRulesType2, opengm::MaxDistance> PyTrBp;







template<class GM,class ACC>
class                     InfParamExporter<opengm::MessagePassing<GM,ACC,opengm::BeliefPropagationUpdateRules<GM,ACC> > > 
: public   InfParamExporterMessagePassing<opengm::MessagePassing< GM,ACC,opengm::BeliefPropagationUpdateRules<GM,ACC> > > {

};

template<class GM,class ACC>
class                     InfParamExporter<opengm::MessagePassing<GM,ACC,opengm::TrbpUpdateRules<GM,ACC> > > 
: public   InfParamExporterMessagePassing<opengm::MessagePassing< GM,ACC,opengm::TrbpUpdateRules<GM,ACC> > > {

};


#endif