#ifdef WITH_FASTPD
#include <boost/python.hpp>
#include <string>
#include "inf_def_visitor.hxx"

#include <opengm/python/opengmpython.hxx>
#include <opengm/python/converter.hxx>
#include <opengm/python/numpyview.hxx>
#include <opengm/python/pythonfunction.hxx>


using namespace boost::python;


#ifdef MIN
   #define OLD_MIN_DEF MIN
   #undef MIN
   #define MIN FAST_PD_MIN
#else 
   
#endif

#ifdef MAX
   #define OLD_MAX_DEF MAX
   #undef MAX
   #define MAX FAST_PD_MAX
#else 
   
#endif

#include <param/fastpd_external_param.hxx>
#include <opengm/inference/external/fastPD.hxx>


#ifdef OLD_MIN_DEF
   #undef MIN
   #define MIN OLD_MIN_DEF
#endif


#ifdef OLD_MAX_DEF
   #undef MAX
   #define MAX OLD_MIN_DEF
#endif


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

template void export_fast_pd<opengm::python::GmAdder, opengm::Minimizer>();


#endif