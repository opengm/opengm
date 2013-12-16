//#define GraphicalModelDecomposition AStarInference_GraphicalModelDecomposition

#include <boost/python.hpp>
#include <string>
#include "inf_def_visitor.hxx"

#include <opengm/inference/astar.hxx>
#include <param/astar_param.hxx>


using namespace boost::python;


template<class GM,class ACC>
void export_astar(){
   using namespace boost::python;
   import_array();
   append_subnamespace("solver");

   // setup 
   InfSetup setup;
   setup.cite       = "Kappes, J. H. :\"Inference on Highly-Connected Discrete Graphical Models with Applications to Visual Object Recognition \"," 
   "Ph.D. Thesis 2011.\n\n"
   "Bergtholdt, M. & Kappes, J. H. & Schnoerr, C.:\"`Learning of Graphical Models and Efficient Inference for Object Class Recognition"
   " <http://hci.iwr.uni-heidelberg.de/Staff/jkappes/publications/dagm2006.pdf>`_\"," 
   "  DAGM 2006\n\n"
   "Bergtholdt, M. & Kappes, J. H. & Schmidt, S. & Schnoerr, C.: \"`A Study of Parts-Based Object Class Detection Using Complete Graphs"
   " <https://www.inf.tu-dresden.de/content/institutes/ki/is/HS_SS09_Papers/A_Study_of_Parts_Based_Object_Class_Detection_Using_Complete_Graphs.pdf>`_\"," 
   "  DAGM 2006";
   setup.notes     =
   "The AStar-Algo transform the problem into a shortest path problem in an exponentially large graph.\n\n"
   "Due to the problem structure, this graph can be represented implicitly!\n\n"
   "To find the shortest path we perform a best first search and use a admissable tree-based heuristic\n\n"
   "to underestimate the cost to a goal node.\n\n"
   "This lower bound allows us to reduce the search to an manageable \n\n"
   "subspace of the exponentially large search-space. ";
   setup.algType     = "searching";
   setup.guarantees  = "global optimal";
   setup.limitations = "graphical model must be small";
   setup.examples   = ">>> parameter = opengm.InfParam(heuristic='fast')\n"
                      ">>> inference = opengm.inference.AStar(gm=gm,accumulator='minimizer',parameter=parameter)\n"
                      "\n\n";
                      ; 
   // export parameter
   typedef opengm::AStar<GM, ACC>  PyAStar;
   exportInfParam<PyAStar>("_AStar");
   // export inference
   class_< PyAStar>("_AStar",init<const GM & >())  
   .def(InfSuite<PyAStar>(std::string("AStar"),setup))
   ;
}

template void export_astar<opengm::python::GmAdder, opengm::Minimizer>();
template void export_astar<opengm::python::GmAdder, opengm::Maximizer>();
template void export_astar<opengm::python::GmMultiplier, opengm::Minimizer>();
template void export_astar<opengm::python::GmMultiplier, opengm::Maximizer>();
