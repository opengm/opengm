#ifndef ASTAR_PARAMETER
#define ASTAR_PARAMETER

#include "param_exporter_base.hxx"
//solver specific
#include <opengm/inference/astar.hxx>

using namespace boost::python;

template<class DEPTH,class INFERENCE>
class InfParamExporterAStar{

public:
   typedef typename INFERENCE::ValueType ValueType;
   typedef typename INFERENCE::Parameter Parameter;
   typedef InfParamExporterAStar<DEPTH,INFERENCE> SelfType;
   typedef typename INFERENCE::AccumulationType AccumulationType;
   static void set
   ( 
      Parameter & p,
      const pyenums::AStarHeuristic h,
      const ValueType bound ,
      const size_t maxHeapSize,
      const size_t numberOfOpt
   ){
      if(h==pyenums::DEFAULT_HEURISTIC)
         p.heuristic_=Parameter::DEFAULTHEURISTIC;
      else if(h==pyenums::FAST_HEURISTIC)
         p.heuristic_=Parameter::FASTHEURISTIC;
      else
         p.heuristic_=Parameter::STANDARDHEURISTIC;
      p.objectiveBound_=bound;
      p.maxHeapSize_=maxHeapSize;
      p.numberOfOpt_=numberOfOpt;
   }

   static typename pyenums::AStarHeuristic getHeuristic(const Parameter & p){
      if(p.heuristic_==Parameter::DEFAULTHEURISTIC)
         return pyenums::DEFAULT_HEURISTIC;
      else if(p.heuristic_==Parameter::FASTHEURISTIC)
         return pyenums::FAST_HEURISTIC;
      else
         return pyenums::STANDARD_HEURISTIC;
   }

   static void setHeuristic( Parameter & p,const pyenums::AStarHeuristic h){
      if(h==pyenums::DEFAULT_HEURISTIC)
         p.heuristic_=Parameter::DEFAULTHEURISTIC;
      else if(h==pyenums::FAST_HEURISTIC)
         p.heuristic_=Parameter::FASTHEURISTIC;
      else
         p.heuristic_=Parameter::STANDARDHEURISTIC;
   }

   void static exportInfParam(const std::string & className,const std::vector<std::string> & subInfParamNames){
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
               arg("heuristic")=pyenums::DEFAULT_HEURISTIC,
               arg("obectiveBound")= AccumulationType::template neutral<ValueType>(),
               arg("maxHeapSize")=3000000,
               arg("numberOfOpt")=1
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

template<class DEPTH,class GM,class ACC>
class InfParamExporter<DEPTH,opengm::AStar<GM,ACC> >  : public  InfParamExporterAStar<DEPTH,opengm::AStar< GM,ACC> > {

};

#endif