#ifndef OPENGM_PYTHON_INTERFACE
#define OPENGM_PYTHON_INTERFACE 1
#endif

#include <stdexcept>
#include <stddef.h>
#include <string>
#include <boost/python.hpp>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/inference/inference.hxx>
#include <opengm/inference/bruteforce.hxx>
#include "nifty_iterator.hxx"
#include "inferencehelpers.hxx"
#include "../export_typedes.hxx"
using namespace boost::python;

namespace pybruteforce{
   template<class PARAM>
   inline void set(PARAM & p){}
}

template<class GM,class ACC>
void export_bruteforce(){
   import_array(); 
   // Py Inference Types 
   typedef opengm::Bruteforce<GM, ACC>  PyBruteforce;
   typedef typename PyBruteforce::Parameter PyBruteforceParameter;
   typedef typename PyBruteforce::VerboseVisitorType PyBruteforceVerboseVisitor;
   
   class_<PyBruteforceParameter > ( "BruteforceParameter" , init< >())
   .def("set",&pybruteforce::set<PyBruteforceParameter>)
   ;

   OPENGM_PYTHON_VERBOSE_VISITOR_EXPORTER(PyBruteforceVerboseVisitor,"BruteforceVerboseVisitor" );
   OPENGM_PYTHON_INFERENCE_EXPORTER(PyBruteforce,"Bruteforce");
}
template void export_bruteforce<GmAdder,opengm::Minimizer>();
template void export_bruteforce<GmAdder,opengm::Maximizer>();
template void export_bruteforce<GmMultiplier,opengm::Minimizer>();
template void export_bruteforce<GmMultiplier,opengm::Maximizer>();
