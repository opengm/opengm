#include <boost/python.hpp>
#include <boost/python.hpp>
#include <string>
#include "inf_def_visitor.hxx"

#include <opengm/inference/partition-move.hxx>
#include <param/partition_move_param.hxx>

using namespace boost::python;

template<class GM,class ACC>
void export_partition_move(){
   using namespace boost::python;
   import_array();
   append_subnamespace("solver");

   // setup 
   InfSetup setup;
   setup.algType     = "multicut";
   setup.limitations = "2-order potts model without unaries where each variable must have as many labels as there are variables";

   // export parameter
   typedef opengm::PartitionMove<GM, ACC>  PyPartitionMove;
   exportInfParam<PyPartitionMove>("_PartitionMove");
   // export inference
   class_< PyPartitionMove>("_PartitionMove",init<const GM & >())  
   .def(InfSuite<PyPartitionMove,false>(std::string("PartitionMove"),setup))
   ;

}
template void export_partition_move<opengm::python::GmAdder,opengm::Minimizer>();
