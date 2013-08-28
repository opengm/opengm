#include <boost/python.hpp>
#include <stdexcept>
#include <stddef.h>
#include <string>

#include <opengm/utilities/tribool.hxx>
#include <opengm/inference/icm.hxx>
#include <opengm/python/opengmpython.hxx>
#include <opengm/python/converter.hxx>
#include <opengm/python/numpyview.hxx>

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
   enum_<opengm::python::pyenums::AStarHeuristic > ("AStarHeuristic")
      .value("fast", opengm::python::pyenums::FAST_HEURISTIC)
      .value("standard", opengm::python::pyenums::STANDARD_HEURISTIC)
      .value("default", opengm::python::pyenums::DEFAULT_HEURISTIC)
      ;
   enum_<opengm::python::pyenums::IcmMoveType > ("IcmMoveType")
      .value("variable", opengm::python::pyenums::SINGLE_VARIABLE)
      .value("factor", opengm::python::pyenums::FACTOR)
      ;
   enum_<opengm::python::pyenums::GibbsVariableProposal > ("GibbsVariableProposal")
      .value("random", opengm::python::pyenums::RANDOM)
      .value("cyclic", opengm::python::pyenums::CYCLIC)
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
