#ifdef WITH_FASTPD_
#include <boost/python.hpp>
#include <string>
#include "inf_def_visitor.hxx"

#include <opengm/inference/external/fastPD.hxx>
#include <param/fastpd_external_param.hxx>

using namespace boost::python;


template<class GM,class ACC>
void export_fast_pd(){
   using namespace boost::python;
   import_array();
   append_subnamespace("solver");

   // setup 
   InfSetup setup;
   setup.algType      = "primal-dual (???)";
   setup.dependencies = "This algorithm needs the FastPD from  library from  http://www.csd.uoc.gr/~komod/FastPD/ " 
                        "compile OpenGM with CMake-Flag ``WITH_FASTPD`` set to ``ON`` ";

   // export parameter
   typedef opengm::external::FastPD<GM>  PyFastPd;
   exportInfParam<PyFastPd>("_FastPd");
   // export inference
   class_< PyFastPd>("_FastPd",init<const GM & >())  
   .def(InfSuite<PyFastPd,false>(std::string("FastPd"),setup))
   ;
}

template void export_fast_pd<GmAdder, opengm::Minimizer>();

#endif