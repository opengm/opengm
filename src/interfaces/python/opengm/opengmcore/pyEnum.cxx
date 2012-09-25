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


enum_<pyenums::AStarHeuristic > ("Heuristic")
   .value("fast", pyenums::FAST_HEURISTIC)
   .value("standard", pyenums::STANDARD_HEURISTIC)
   .value("default", pyenums::DEFAULT_HEURISTIC)
   ;
enum_<pyenums::IcmMoveType > ("MoveType")
   .value("variable", pyenums::SINGLE_VARIABLE)
   .value("factor", pyenums::FACTOR)
   ;
enum_<pyenums::GibbsVariableProposal > ("VariableProposal")
   .value("random", pyenums::RANDOM)
   .value("cyclic", pyenums::CYCLIC)
   ;
enum_<opengm::Tribool::State> ("Tribool")
   .value("true", opengm::Tribool::True)
   .value("false", opengm::Tribool::False)
   .value("maybe", opengm::Tribool::Maybe)
   ;
   
#ifdef WITH_LIBDAI
enum_<pyenums::libdai::UpdateRule> ("UpdateRule")
   .value("parall", pyenums::libdai::PARALL)
   .value("seqfix", pyenums::libdai::SEQFIX)
   .value("seqrnd", pyenums::libdai::SEQRND)
   .value("seqmax", pyenums::libdai::SEQMAX)
   ;
#endif
}
