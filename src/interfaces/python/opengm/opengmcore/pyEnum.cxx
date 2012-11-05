#ifndef OPENGM_PYTHON_INTERFACE
#define OPENGM_PYTHON_INTERFACE 1
#endif

#include <stdexcept>
#include <stddef.h>
#include <string>
#include <boost/python.hpp>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/utilities/tribool.hxx>
#include <opengm/inference/inference.hxx>
#include <opengm/inference/icm.hxx>
#include "nifty_iterator.hxx"
#include"../export_typedes.hxx"
using namespace boost::python;




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
   enum_<opengm::Tribool::State> ("Tribool")
      .value("true", opengm::Tribool::True)
      .value("false", opengm::Tribool::False)
      .value("maybe", opengm::Tribool::Maybe)
      ;
}
