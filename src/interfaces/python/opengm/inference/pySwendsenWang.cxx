#include <boost/python.hpp>
#include <stdexcept>
#include <stddef.h>
#include <string>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/inference/inference.hxx>
#include <opengm/inference/swendsenwang.hxx>
#include "nifty_iterator.hxx"
#include "inferencehelpers.hxx"
#include "../export_typedes.hxx"
using namespace boost::python;

namespace pyswendsenwang{
   template<class PARAM>
   inline void set 
   (
      PARAM & p,
      const size_t steps,
      const size_t burnInSteps,
      const double lowestAllowedProbability
   ) {
      p.maxNumberOfSamplingSteps_=steps;
      p.numberOfBurnInSteps_=burnInSteps;
      p.lowestAllowedProbability_=lowestAllowedProbability;
   }

   template<class PARAM>
   inline PARAM * construct
   (
      const size_t steps,
      const size_t burnInSteps,
      const double lowestAllowedProbability
   ){
      PARAM * p = new PARAM();
      pyswendsenwang::set(*p,steps,burnInSteps,lowestAllowedProbability);
      return p;
   }
}

template<class GM,class ACC>
void export_swendsen_wang(){
   import_array(); 
   // Py Inference Types 
   typedef opengm::SwendsenWang<GM, ACC>  PySwendsenWang;
   typedef typename PySwendsenWang::Parameter PySwendsenWangParameter;
   

   //const size_t maxNumberOfSamplingSteps = 1e5,
   //const size_t numberOfBurnInSteps = 1e5,
   //ProbabilityType lowestAllowedProbability = 1e-6,
   //const std::vector<LabelType>& initialState = std::vector<LabelType>()

   //size_t maxNumberOfSamplingSteps_;
   //size_t numberOfBurnInSteps_;
   //ProbabilityType lowestAllowedProbability_;
   //std::vector<LabelType> initialState_;

   class_<PySwendsenWangParameter > ( "Parameter_Swendsenwang_Opengm" , init< >())
   /*
   .def("__init__", make_constructor(&pyswendsenwang::construct<PySwendsenWangParameter> ,default_call_policies(),
      (
         arg("steps")=1e5,
         arg("numberOfBurnInSteps")=0,
         arg("lowestAllowedProbability")=1e-6
      )
   ),
   "Construtor of  the parameter.\n\n"
   "All values of the parameter have a default value.\n\n"
   "Args:\n\n"
   "  steps: maximum number of iterations (default: 1e5)\n\n"
   "  numberOfBurnInSteps: number of burn in steps (default: 0) \n\n"
   "  lowestAllowedProbability: lowest allowed probability (default: 1e-5)\n\n"
   )
   .def ("set", &pyswendsenwang::set<PySwendsenWangParameter>, 
      (
         arg("steps")=1e5,
         arg("numberOfBurnInSteps")=0,
         arg("lowestAllowedProbability")=1e-6
      )
   ,
   "Set the parameters values.\n\n"
   "All values of the parameter have a default value.\n\n"
   "Args:\n\n"
   "  steps: maximum number of iterations (default: 1e5)\n\n"
   "  numberOfBurnInSteps: number of burn in steps (default: 0) \n\n"
   "  lowestAllowedProbability: lowest allowed probability (default: 1e-5)\n\n"
   "Returns:\n"
   "  None\n\n"
   )
   */
   ;

   OPENGM_PYTHON_INFERENCE_NO_RESET_EXPORTER(PySwendsenWang ,Swendsenwang,Opengm,"swendsen-wang","sw","opengm","ogm","sw docstring");  

}
template void export_swendsen_wang<GmAdder,opengm::Minimizer>();
template void export_swendsen_wang<GmMultiplier,opengm::Maximizer>();
