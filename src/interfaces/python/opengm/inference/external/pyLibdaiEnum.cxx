#ifdef WITH_LIBDAI
#include <stdexcept>
#include <stddef.h>
#include <string>
#include <boost/python.hpp>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/inference/external/libdai/bp.hxx>
#include <opengm/inference/external/libdai/tree_reweighted_bp.hxx>
#include <opengm/inference/external/libdai/double_loop_generalized_bp.hxx>
#include <opengm/inference/external/libdai/junction_tree.hxx>
#include "nifty_iterator.hxx"
#include"../export_typedes.hxx"
using namespace boost::python;

void export_libdai_enums(){
   enum_<opengm::external::libdai::BpUpdateRule> ("BpUpdateRule")
      .value("parall", opengm::external::libdai::PARALL)
      .value("seqfix", opengm::external::libdai::SEQFIX)
      .value("seqrnd", opengm::external::libdai::SEQRND)
      .value("seqmax", opengm::external::libdai::SEQMAX)
      ;
   enum_<opengm::external::libdai::Init> ("Init")
      .value("uniform", opengm::external::libdai::UNIFORM)
      .value("random", opengm::external::libdai::RANDOM)
      ;
   enum_<opengm::external::libdai::Clusters> ("Clusters")
      .value("min", opengm::external::libdai::MIN)
      .value("bethe", opengm::external::libdai::BETHE)
      .value("delta", opengm::external::libdai::DELTA)
      .value("loop", opengm::external::libdai::LOOP)
      ;       
 
   enum_<opengm::external::libdai::JunctionTreeUpdateRule> ("JunctionTreeUpdateRule")
      .value("hugin", opengm::external::libdai::HUGIN)
      .value("shsh", opengm::external::libdai::SHSH)
      ;
   enum_<opengm::external::libdai::JunctionTreeHeuristic> ("JunctionTreeHeuristic")
      .value("minfill", opengm::external::libdai::MINFILL)
      .value("weightedminfill", opengm::external::libdai::WEIGHTEDMINFILL)
      .value("minweight", opengm::external::libdai::MINWEIGHT)
      .value("minneighbors", opengm::external::libdai::MINNEIGHBORS)
      ;         
}

#endif
