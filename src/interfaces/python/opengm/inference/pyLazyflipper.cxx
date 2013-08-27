#include <boost/python.hpp>
#include <string>
#include "inf_def_visitor.hxx"

#include <opengm/inference/lazyflipper.hxx>
#include <param/lazyflipper_param.hxx>


template<class GM,class ACC>
void export_lazyflipper(){
   using namespace boost::python;
   import_array();
   append_subnamespace("solver");

   // setup 
   InfSetup setup;
   setup.cite       = "Boern Andres, Joerg H. Kappes, Thorsten Beier, Ullrich Koethe, Fred A. Hamprecht: \n\n"
                      "   The Lazy Flipper: Efficient Depth-Limited Exhaustive Search in Discrete Graphical Models. ECCV (7) 2012: 154-166";
   setup.algType    = "movemaking";
   setup.guarantees = "optimal within a hamming distance of the given subgraph size";
   setup.examples   = ">>> parameter = opengm.InfParam(maxSubgraphSize=2)\n"
                      ">>> inference = opengm.inference.LazyFlippper(gm=gm,accumulator='minimizer',parameter=parameter)\n"
                      "\n\n";

   // export parameter
   typedef opengm::LazyFlipper<GM, ACC>  PyLazyFlipper;
   exportInfParam<PyLazyFlipper>("_LazyFlipper");
   // export inference
   class_< PyLazyFlipper>("_LazyFlipper",init<const GM & >())  
   .def(InfSuite<PyLazyFlipper>(std::string("LazyFlipper"),setup))
   ;
}

template void export_lazyflipper<opengm::python::GmAdder,opengm::Minimizer>();
template void export_lazyflipper<opengm::python::GmAdder,opengm::Maximizer>();
template void export_lazyflipper<opengm::python::GmMultiplier,opengm::Minimizer>();
template void export_lazyflipper<opengm::python::GmMultiplier,opengm::Maximizer>();
