#ifdef WITH_AD3

//#define GraphicalModelDecomposition LOC_Inference_GraphicalModelDecomposition
#include <boost/python.hpp>
#include <string>
#include "inf_def_visitor.hxx"

#include <opengm/inference/loc.hxx>
# include <param/loc_param.hxx>

using namespace boost::python;


template<class GM,class ACC>
void export_loc(){
   using namespace boost::python;
   import_array();
   append_subnamespace("solver");

   // setup 
   InfSetup setup;
   setup.cite       = "K. Jung, P. Kohli and D. Shah:\"`Local Rules for Global MAP: When Do They Work? "
                      "<http://research.microsoft.com/en-us/um/people/pkohli/papers/jks_nips09_TR.pdf>`_\"," 
                      "NIPS 2009.\n\n";
   setup.algType    = "movemaking";
   setup.guarantees = "epsilon approximation for planar graphical models";
   setup.examples   = ">>> parameter = opengm.InfParam(phi=0.3,maxRadius=20)\n"
                      ">>> inference = opengm.inference.Loc(gm=gm,accumulator='minimizer',parameter=parameter)\n\n"
                      "\n\n";
   setup.dependencies = "needs AD3 / WITH_AD3";               

   // export parameter
   typedef opengm::LOC<GM, ACC>  PyLOC;
   exportInfParam<PyLOC>("_Loc");
   // export inference
   class_< PyLOC>("_Loc",init<const GM & >())  
   .def(InfSuite<PyLOC>(std::string("Loc"),setup))
   ;
}

template void export_loc<opengm::python::GmAdder, opengm::Minimizer>();
template void export_loc<opengm::python::GmAdder, opengm::Maximizer>();
//template void export_loc<GmMultiplier, opengm::Minimizer>();
//template void export_loc<GmMultiplier, opengm::Maximizer>();

#endif