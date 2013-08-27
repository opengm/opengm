#ifdef WITH_CPLEX
#include <boost/python.hpp>
#include <stdexcept>
#include <stddef.h>
#include <string>
#include "nifty_iterator.hxx"
#include "inf_def_visitor.hxx"
#include "lp_def_suite.hxx"

#include <opengm/python/opengmpython.hxx>
#include <opengm/python/converter.hxx>
#include <opengm/python/numpyview.hxx>
#include <opengm/python/pythonfunction.hxx>


#include <opengm/inference/lpcplex.hxx>
#include <param/lpcplex_param.hxx>



// export function
template<class GM, class ACC>
void export_cplex() {
   using namespace boost::python;
   import_array();
   append_subnamespace("solver");

   // setup 
   InfSetup setup;
   setup.algType    = "linear-programming";
   setup.guarantees = "global optimal";
   setup.examples   = ">>> parameter = opengm.InfParam(integerConstraint=True)\n\n"
                      ">>> inference = opengm.inference.LpCplex(gm=gm,accumulator='minimizer',parameter=parameter)\n\n"
                      "\n\n";  
   setup.dependencies = "This algorithm needs the IBM CPLEX Optimizer, compile OpenGM with CMake-Flag ``WITH_CPLEX`` set to ``ON`` ";
   // export parameter
   typedef typename GM::ValueType ValueType;
   typedef typename GM::IndexType IndexType;
   typedef opengm::LPCplex<GM, ACC> PyLPCplex;
   exportInfParam<PyLPCplex>("_LpCplex");
   // export inference
   class_< PyLPCplex>("_LpCplex",init<const GM & >())  
   .def(InfSuite<PyLPCplex,false,true,false>(std::string("LpCplex"),setup))
   .def(LpInferenceSuite<PyLPCplex>())
   ;
   /*
   // more members
   .def("addConstraint", &pycplex::addConstraintPythonNumpy<PyLPCplex,ValueType,IndexType>  )
   .def("addConstraint", &pycplex::addConstraintPythonList<PyLPCplex,ValueType,IndexType>  )
   .def("addConstraints", &pycplex::addConstraintsPythonListListOrListNumpy<PyLPCplex,ValueType,IndexType>  )
   .def("addConstraints", &pycplex::addConstraintsPythonNumpy<PyLPCplex,ValueType,IndexType>  )
   .def("lpNodeVariableIndex",&PyLPCplex::lpNodeVi)
   .def("lpFactorVariableIndex",&pycplex::lpFactorIter<PyLPCplex>)
   .def("lpFactorVariableIndex",&pycplex::lpFactorViScalar<PyLPCplex>)
   ;
   */
}
// explicit template instantiation for the supported semi-rings
template void export_cplex<opengm::python::GmAdder, opengm::Minimizer>();
#endif