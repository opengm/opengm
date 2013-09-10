#include <boost/python.hpp>
#include <string>
#include "inf_def_visitor.hxx"

#include <opengm/inference/icm.hxx>
#include <param/icm_param.hxx>


// export function
template<class GM, class ACC>
void export_icm() {

   using namespace boost::python;
   //Py_Initialize();
   //PyEval_InitThreads();
   import_array();
   append_subnamespace("solver");

   // setup 
   InfSetup setup;
   setup.cite       = "J. E. Besag: On the Statistical Analysis of Dirty Pictures" 
                       "Journal of the Royal Statistical Society, Series B 48(3):259-302, 1986.";
   setup.algType    = "movemaking";
   setup.guarantees = "optimal within a hamming distance of 1";
   setup.examples   = ">>> parameter = opengm.InfParam(moveType='variable')\n"
                      ">>> inference = opengm.inference.Icm(gm=gm,accumulator='minimizer',parameter=parameter)\n"
                      "\n\n"; 
   setup.notes      = ".. seealso::\n\n"
                      "   :class:`opengm.inference.LazyFlipper` a generalization of Icm";
   // export parameter
   typedef opengm::ICM<GM, ACC> PyICM;
   exportInfParam<PyICM>("_Icm");
   // export inference
   class_< PyICM>("_Icm",init<const GM & >())  
   .def(InfSuite<PyICM>(std::string("Icm"),setup))
   ;
}
// explicit template instantiation for the supported semi-rings
template void export_icm<opengm::python::GmAdder, opengm::Minimizer>();
template void export_icm<opengm::python::GmAdder, opengm::Maximizer>();
template void export_icm<opengm::python::GmMultiplier, opengm::Minimizer>();
template void export_icm<opengm::python::GmMultiplier, opengm::Maximizer>();
