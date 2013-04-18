#include <boost/python.hpp>
#include <stdexcept>
#include <stddef.h>
#include <string>

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/utilities/tribool.hxx>
#include <opengm/inference/inference.hxx>
#include <opengm/inference/icm.hxx>
#include "nifty_iterator.hxx"
#include"../export_typedes.hxx"

#ifdef WITH_TRWS
#include <opengm/inference/external/trws.hxx>
#endif


using namespace boost::python;


std::string printTribool(const opengm::Tribool & tb){
   if(tb==true && tb.maybe()==false){
      return std::string("True");
   }
   else if(tb==false && tb.maybe()==false){
      return std::string("False");
   }
   else{
      return std::string("Maybe");
   }
}


void export_enum(){
   enum_<pyenums::AStarHeuristic > ("AStarHeuristic")
      .value("fast", pyenums::FAST_HEURISTIC)
      .value("standard", pyenums::STANDARD_HEURISTIC)
      .value("default", pyenums::DEFAULT_HEURISTIC)
      ;
   enum_<pyenums::IcmMoveType > ("IcmMoveType")
      .value("variable", pyenums::SINGLE_VARIABLE)
      .value("factor", pyenums::FACTOR)
      ;
   enum_<pyenums::GibbsVariableProposal > ("GibbsVariableProposal")
      .value("random", pyenums::RANDOM)
      .value("cyclic", pyenums::CYCLIC)
      ;
   enum_<opengm::Tribool::State> ("TriboolStates")
      .value("true", opengm::Tribool::True)
      .value("false", opengm::Tribool::False)
      .value("maybe", opengm::Tribool::Maybe)
      ;

   class_<opengm::Tribool > ( "Tribool", init<opengm::Tribool::State>())
   .def(init<bool>())
   .def(init<int>())
   .def("__str__",&printTribool)
   ;




}
