#include <boost/python.hpp>
#include <string>
#include "inf_def_visitor.hxx"

#include <opengm/inference/bruteforce.hxx>
#include <param/bruteforce_param.hxx>


#include <opengm/python/opengmpython.hxx>
#include <opengm/python/converter.hxx>
#include <opengm/python/numpyview.hxx>
#include <opengm/python/pythonfunction.hxx>


using namespace boost::python;

template<class GM,class ACC>
void export_bruteforce(){
   using namespace boost::python;
   import_array();
   append_subnamespace("solver");

   // setup 
   InfSetup setup;
   setup.algType     = "searching";
   setup.guarantees  = "global optimal";
   setup.examples    = ">>> inference = opengm.inference.Bruteforce(gm=gm,accumulator='minimizer')\n\n"; 
   setup.limitations = "graphical model must be very small";
   setup.notes      = ".. seealso::\n\n"
                      "   :class:`opengm.inference.AStar` an global optimal solver for small graphical models";

   // export parameter
   typedef opengm::Bruteforce<GM, ACC>  PyBruteforce;
   exportInfParam<PyBruteforce>("_Bruteforce");
   // export inference
   class_< PyBruteforce>("_Bruteforce",init<const GM & >())  
   .def(InfSuite<PyBruteforce>(std::string("Bruteforce"),setup))
   ;

}
template void export_bruteforce<opengm::python::GmAdder,opengm::Minimizer>();
template void export_bruteforce<opengm::python::GmAdder,opengm::Maximizer>();
template void export_bruteforce<opengm::python::GmMultiplier,opengm::Minimizer>();
template void export_bruteforce<opengm::python::GmMultiplier,opengm::Maximizer>();
