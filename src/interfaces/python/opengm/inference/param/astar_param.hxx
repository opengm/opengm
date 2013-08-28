#ifndef ASTAR_PARAMETER
#define ASTAR_PARAMETER

#include "param_exporter_base.hxx"
//solver specific
#include <opengm/inference/astar.hxx>

using namespace boost::python;

template<class INFERENCE>
class InfParamExporterAStar{

public:
   typedef typename INFERENCE::ValueType ValueType;
   typedef typename INFERENCE::Parameter Parameter;
   typedef InfParamExporterAStar<INFERENCE> SelfType;
   typedef typename INFERENCE::AccumulationType AccumulationType;
   static void set
   ( 
      Parameter & p,
      const opengm::python::pyenums::AStarHeuristic h,
      const ValueType bound ,
      const size_t maxHeapSize,
      const size_t numberOfOpt
   ){
      if(h==opengm::python::pyenums::DEFAULT_HEURISTIC)
         p.heuristic_=Parameter::DEFAULTHEURISTIC;
      else if(h==opengm::python::pyenums::FAST_HEURISTIC)
         p.heuristic_=Parameter::FASTHEURISTIC;
      else
         p.heuristic_=Parameter::STANDARDHEURISTIC;
      p.objectiveBound_=bound;
      p.maxHeapSize_=maxHeapSize;
      p.numberOfOpt_=numberOfOpt;
   }

   static typename opengm::python::pyenums::AStarHeuristic getHeuristic(const Parameter & p){
      if(p.heuristic_==Parameter::DEFAULTHEURISTIC)
         return opengm::python::pyenums::DEFAULT_HEURISTIC;
      else if(p.heuristic_==Parameter::FASTHEURISTIC)
         return opengm::python::pyenums::FAST_HEURISTIC;
      else
         return opengm::python::pyenums::STANDARD_HEURISTIC;
   }

   static void setHeuristic( Parameter & p,const opengm::python::pyenums::AStarHeuristic h){
      if(h==opengm::python::pyenums::DEFAULT_HEURISTIC)
         p.heuristic_=Parameter::DEFAULTHEURISTIC;
      else if(h==opengm::python::pyenums::FAST_HEURISTIC)
         p.heuristic_=Parameter::FASTHEURISTIC;
      else
         p.heuristic_=Parameter::STANDARDHEURISTIC;
   }

   void static exportInfParam(const std::string & className){
      class_<Parameter > (className.c_str(), init< >() )
         .def_readwrite("obectiveBound", &Parameter::objectiveBound_,
         "AStar objective bound.\n\n"
         "  A good bound will speedup inference"
         )
         .def_readwrite("maxHeapSize", &Parameter::maxHeapSize_,
         "Maximum size of the heap which is used while inference"
         )
         .def_readwrite("numberOfOpt", &Parameter::numberOfOpt_,
         "Select which n best states should be searched for while inference:"
         )
         .add_property("heuristic", 
              &SelfType::getHeuristic, SelfType::setHeuristic,
         "heuristic can be:\n\n"
         "  -``'default'`` :default AStar heuristc (default)\n\n"
         "  -``'standart'`` : standart AStar heuristic \n\n"
         "  -``'fast'`` : fast AStar heuristic for second order gm's"     
         )
         .def ("set", &SelfType::set, 
               (
               boost::python::arg("heuristic")=opengm::python::pyenums::DEFAULT_HEURISTIC,
               boost::python::arg("obectiveBound")= AccumulationType::template neutral<ValueType>(),
               boost::python::arg("maxHeapSize")=3000000,
               boost::python::arg("numberOfOpt")=1
               ),
         "Set the parameters values.\n\n"
         "All values of the parameter have a default value.\n\n"
         "Args:\n\n"
         "  heuristic: Number of message passing updates\n\n"
         "     -``opengm.AStarHeuristic.default`` :default AStar heuristc (default)\n\n"
         "     -``opengm.AStarHeuristic.standart`` : stanart AStar heuristic \n\n"
         "     -``opengm.AStarHeuristic.fast`` : fast AStar heuristic for second order gm's\n\n"
         "  bound: AStar objective bound.\n\n"
         "     A good bound will speedup inference (default = neutral value)\n\n"
         "  maxHeapSize: Maximum size of the heap which is used while inference (default=3000000)\n\n"
         "  numberOfOpt: Select which n best states should be searched for while inference (default=1):\n\n"
         "Returns:\n"
         "  None\n\n"
         ) 
      ;

   }
};

template<class GM,class ACC>
class InfParamExporter<opengm::AStar<GM,ACC> >  : public  InfParamExporterAStar<opengm::AStar< GM,ACC> > {

};

#endif