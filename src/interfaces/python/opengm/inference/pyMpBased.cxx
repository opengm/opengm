#ifndef OPENGM_PYTHON_INTERFACE
#define OPENGM_PYTHON_INTERFACE 1
#endif

#include <stdexcept>
#include <stddef.h>
#include <string>
#include <boost/python.hpp>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/inference/inference.hxx>
#include <opengm/inference/astar.hxx>
#include <opengm/inference/messagepassing/messagepassing.hxx>
#include <opengm/inference/loc.hxx>

#include "nifty_iterator.hxx"
#include "inferencehelpers.hxx"
#include"../export_typedes.hxx"

using namespace boost::python;

namespace pybp{
   template< class PARAM>
   std::string paramAsString(const PARAM & param) {
      std::stringstream ss;
      ss<<"[ steps="<<param.maximumNumberOfSteps_;
      ss<<", damping="<<param.damping_;
      ss<<", convergenceBound="<<param.bound_<<" ]";
      ss<<", isAcyclic="<<param.isAcyclic_<<" ]";
      return ss.str();
   }
   template<class PARAM>
   void set
   (
      PARAM & p,
      const size_t maximumNumberOfSteps,
      const double damping,
      const double convergenceBound,
      const opengm::Tribool::State isAcyclic
   ){
      p.maximumNumberOfSteps_=maximumNumberOfSteps;
      p.damping_=damping;
      p.bound_=convergenceBound;
      p.isAcyclic_=isAcyclic;
   }
   
   template<class PARAM>
   void setIsAcyclic
   (
      PARAM & p,
      const opengm::Tribool::State isAcyclic
   ){
      p.isAcyclic_=isAcyclic;
   }
   template<class PARAM>
   opengm::Tribool::State getIsAcyclic
   (
      PARAM & p
   ){
      if (p.isAcyclic_.maybe())
         return opengm::Tribool::Maybe;
      else if (p.isAcyclic_==true)
         return opengm::Tribool::True;
      else
         return opengm::Tribool::False;
   }
}
namespace pytrbp{
   template< class PARAM>
   std::string paramAsString(const PARAM & param) {
      std::stringstream ss;
      ss<<"[ steps="<<param.maximumNumberOfSteps_;
      ss<<", damping="<<param.damping_;
      ss<<", convergenceBound="<<param.bound_<<" ]";
      ss<<", isAcyclic="<<param.isAcyclic_<<" ]";
      return ss.str();
   }
   template<class PARAM>
   void set
   (
      PARAM & p,
      const size_t maximumNumberOfSteps,
      const double damping,
      const double convergenceBound,
      const opengm::Tribool::State isAcyclic
   ){
      p.maximumNumberOfSteps_=maximumNumberOfSteps;
      p.damping_=damping;
      p.bound_=convergenceBound;
      p.isAcyclic_=isAcyclic;
   }
   
   template<class PARAM>
   void setIsAcyclic
   (
      PARAM & p,
      const opengm::Tribool::State isAcyclic
   ){
      p.isAcyclic_=isAcyclic;
   }
   template<class PARAM>
   opengm::Tribool::State getIsAcyclic
   (
      PARAM & p
   ){
      if (p.isAcyclic_.maybe())
         return opengm::Tribool::Maybe;
      else if (p.isAcyclic_==true)
         return opengm::Tribool::True;
      else
         return opengm::Tribool::False;
   }
}
namespace pyastar{   
   template<class PARAM,class INF>
   typename pyenums::AStarHeuristic getHeuristic(const PARAM & p){
      if(p.heuristic_==PARAM::DEFAULTHEURISTIC)
         return pyenums::DEFAULT_HEURISTIC;
      else if(p.heuristic_==PARAM::FASTHEURISTIC)
         return pyenums::FAST_HEURISTIC;
      else
         return pyenums::STANDARD_HEURISTIC;
   }
   template<class PARAM,class INF>
   void setHeuristic( PARAM & p,const pyenums::AStarHeuristic h){
      if(h==pyenums::DEFAULT_HEURISTIC)
         p.heuristic_=PARAM::DEFAULTHEURISTIC;
      else if(h==pyenums::FAST_HEURISTIC)
         p.heuristic_=PARAM::FASTHEURISTIC;
      else
         p.heuristic_=PARAM::STANDARDHEURISTIC;
   }
   
   template<class PARAM,class INF>
   void set
   ( 
      PARAM & p,
      const pyenums::AStarHeuristic h,
      const typename INF::ValueType bound ,
      const size_t maxHeapSize,
      const size_t numberOfOpt
   ){
      if(h==pyenums::DEFAULT_HEURISTIC)
         p.heuristic_=PARAM::DEFAULTHEURISTIC;
      else if(h==pyenums::FAST_HEURISTIC)
         p.heuristic_=PARAM::FASTHEURISTIC;
      else
         p.heuristic_=PARAM::STANDARDHEURISTIC;
      p.objectiveBound_=bound;
      p.maxHeapSize_=maxHeapSize;
      p.numberOfOpt_=numberOfOpt;
   }
}

namespace pyloc{

   template<class PARAM>
   inline void set
   (
      PARAM & p,
      const double phi,
      const size_t maxRadius,
      const size_t maxIterations,
      const size_t aStarThreshold
   ){
      p.phi_=phi;
      p.maxRadius_=maxRadius;
      p.maxIterations_=maxIterations;
      p.aStarThreshold_=aStarThreshold;
   }

}

template<class GM,class ACC>
void export_mp_based(){
import_array();    
   
typedef GM PyGm;
typedef typename PyGm::ValueType ValueType;
typedef typename PyGm::IndexType IndexType;
typedef typename PyGm::LabelType LabelType;

typedef opengm::BeliefPropagationUpdateRules<GM,ACC> UpdateRulesType;
typedef opengm::MessagePassing<GM, ACC,UpdateRulesType, opengm::MaxDistance> PyBp;
typedef typename PyBp::Parameter PyBpParameter;
typedef typename PyBp::VerboseVisitorType PyBpVerboseVisitor;

typedef opengm::TrbpUpdateRules<GM,ACC> UpdateRulesType2;
typedef opengm::MessagePassing<GM, ACC,UpdateRulesType2, opengm::MaxDistance> PyTrBp;
typedef typename PyTrBp::Parameter PyTrBpParameter;
typedef typename PyTrBp::VerboseVisitorType PyTrBpVerboseVisitor;

typedef opengm::AStar<PyGm, ACC>  PyAStar;
typedef typename PyAStar::Parameter PyAStarParameter;
typedef typename PyAStar::VerboseVisitorType PyAStarVerboseVisitor;

typedef opengm::LOC<GM, ACC>  PyLOC;
typedef typename PyLOC::Parameter PyLOCParameter;
typedef typename PyLOC::VerboseVisitorType PyLOCVerboseVisitor;


   class_<PyAStarParameter > ("AStarParameter", init< >() )
      .def_readwrite("obectiveBound", &PyAStarParameter::objectiveBound_)
      .def_readwrite("maxHeapSize", &PyAStarParameter::maxHeapSize_)
      .def_readwrite("numberOfOpt", &PyAStarParameter::numberOfOpt_)
      .add_property("heuristic", 
           &pyastar::getHeuristic<PyAStarParameter,PyAStar>, pyastar::setHeuristic<PyAStarParameter,PyAStar>)
      .def ("set", &pyastar::set<PyAStarParameter,PyAStar>, 
            (
            arg("heuristic")=pyenums::DEFAULT_HEURISTIC,
            arg("bound")= ACC::template neutral<ValueType>(),
            arg("maxHeapSize")=3000000,
            arg("numberOfOpt")=1
            ) 
      ) 
      ;


   OPENGM_PYTHON_VERBOSE_VISITOR_EXPORTER(PyAStarVerboseVisitor,"AStarVerboseVisitor" );
   OPENGM_PYTHON_INFERENCE_EXPORTER(PyAStar,"AStar");

   // export inference parameter
   class_<PyBpParameter > ("BpParameter", init< const size_t,const typename PyBp::ValueType,const typename PyBp::ValueType  >() )
      .def(init<const size_t,const typename PyBp::ValueType>())
      .def(init<const size_t>())
      .def(init<>())
      .def ("set", &pybp::set<PyBpParameter>, 
            (
            arg("steps")=100,
            arg("damping")= 0,
            arg("convergenceBound")=0,
            arg("isAcyclic")=opengm::Tribool::Maybe
            ) 
      )
      .add_property("isAcyclic", &pybp::getIsAcyclic<PyBpParameter>, pybp::setIsAcyclic<PyBpParameter>)
      .def("__str__",pybp::paramAsString<PyBpParameter>)
      .def_readwrite("steps", &PyBpParameter::maximumNumberOfSteps_)
      .def_readwrite("damping", &PyBpParameter::damping_)
      .def_readwrite("convergenceBound", &PyBpParameter::bound_)
      ;
   // export inference verbose visitor via macro
   OPENGM_PYTHON_VERBOSE_VISITOR_EXPORTER(PyBpVerboseVisitor,"BpVerboseVisitor" );
   // export inference via macro
   OPENGM_PYTHON_INFERENCE_EXPORTER(PyBp,"Bp");

      class_<PyTrBpParameter > ("TrBpParameter", init< const size_t,const typename PyTrBp::ValueType,const typename PyTrBp::ValueType  >() )
      .def(init<const size_t,const typename PyTrBp::ValueType>())
      .def(init<const size_t>())
      .def(init<>())
      .def_readwrite("steps", &PyTrBpParameter::maximumNumberOfSteps_)
      .def_readwrite("damping", &PyTrBpParameter::damping_)
      .def_readwrite("convergenceBound", &PyTrBpParameter::bound_)
      .def ("set", &pytrbp::set<PyTrBpParameter>, 
            (
            arg("steps")=100,
            arg("damping")= 0,
            arg("convergenceBound")=0,
            arg("isAcyclic")=opengm::Tribool::Maybe
            ) 
      )
      .add_property("isAcyclic", &pytrbp::getIsAcyclic<PyTrBpParameter>, pytrbp::setIsAcyclic<PyTrBpParameter>)
      ;


   OPENGM_PYTHON_VERBOSE_VISITOR_EXPORTER(PyTrBpVerboseVisitor,"TrBpVerboseVisitor" );
   OPENGM_PYTHON_INFERENCE_EXPORTER(PyTrBp,"TrBp");

     class_<PyLOCParameter > ( "LOCParameter" , init< double ,size_t,size_t,size_t > (args("phi,maxRadius,maxIteration,aStarThreshold")))
   .def(init<>())
   .def_readwrite("phi", &PyLOCParameter::phi_)
   .def_readwrite("maxRadius", &PyLOCParameter::maxRadius_)
   .def_readwrite("maxSubgraphSize", &PyLOCParameter::maxIterations_)
   .def_readwrite("aStarThreshold", &PyLOCParameter::aStarThreshold_)
   .def ("set", &pyloc::set<PyLOCParameter>, 
            (
            arg("phi")=0.5,
            arg("maxRadius")=5,
            arg("steps")=0,
            arg("aStarThreshold")=10
            ) 
      ) 
   ;

   OPENGM_PYTHON_VERBOSE_VISITOR_EXPORTER(PyLOCVerboseVisitor,"LOCVerboseVisitor" );
   OPENGM_PYTHON_INFERENCE_EXPORTER(PyLOC,"LOC");

}

template void export_mp_based<GmAdder, opengm::Minimizer>();
template void export_mp_based<GmAdder, opengm::Maximizer>();
template void export_mp_based<GmMultiplier, opengm::Minimizer>();
template void export_mp_based<GmMultiplier, opengm::Maximizer>();
