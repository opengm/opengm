#ifndef OPENGM_PYTHON_INTERFACE
#define OPENGM_PYTHON_INTERFACE 1
#endif

#include <stdexcept>
#include <stddef.h>
#include <string>
#include <boost/python.hpp>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/inference/inference.hxx>
#include <opengm/inference/gibbs.hxx>
#include "nifty_iterator.hxx"
#include "inferencehelpers.hxx"
#include"../export_typedes.hxx"
using namespace boost::python;

namespace pygibbs{
   template<class PARAM>
   inline void set
   (
      PARAM & p,
      const size_t steps,
      const size_t burnInSteps,
      const pyenums::GibbsVariableProposal  proposal     
   ){
      p.maxNumberOfSamplingSteps_=steps;
      p.numberOfBurnInSteps_=burnInSteps;
      if(proposal==pyenums::RANDOM)
         p.variableProposal_=PARAM::RANDOM;
      else
         p.variableProposal_=PARAM::CYCLIC;
   }
   template<class PARAM>
   inline void setProposal
   (
      PARAM & p,
      const pyenums::GibbsVariableProposal  proposal     
   ){
      if(proposal==pyenums::RANDOM)
         p.variableProposal_=PARAM::RANDOM;
      else
         p.variableProposal_=PARAM::CYCLIC;
   }
   template<class PARAM>
   inline pyenums::GibbsVariableProposal getProposal
   (
      PARAM & p,
      const pyenums::GibbsVariableProposal  proposal     
   ){
      if(p.variableProposal_==PARAM::RANDOM)
         return pyenums::RANDOM;
      else
         return pyenums::CYCLIC;
   }
}

template<class GM,class ACC>
void export_gibbs(){
   import_array(); 
// Py Inference Types 
   typedef opengm::Gibbs<GM, ACC>  PyGibbs;
   typedef typename PyGibbs::Parameter PyGibbsParameter;
   typedef typename PyGibbs::VerboseVisitorType PyGibbsVerboseVisitor;
   
//   enum_<typename PyGibbsParameter::VariableProposal > ("GibbsVariableProposal")
//           .value("RANDOM", PyGibbsParameter::RANDOM)
//           .value("CYCLIC", PyGibbsParameter::CYCLIC)
           ;
   class_<PyGibbsParameter > ("GibbsParameter", init<const size_t ,const size_t,const typename PyGibbsParameter::VariableProposal> (args("numberOfSamplingSteps,numberOfburnInSteps,variableProposal")))
   .def(init<>())
   .def_readwrite("numberOfSamplingSteps", &PyGibbsParameter::maxNumberOfSamplingSteps_)
   .def_readwrite("numberOfburnInSteps", &PyGibbsParameter::numberOfBurnInSteps_)
   .add_property("variableProposal",&pygibbs::getProposal<PyGibbsParameter>, pygibbs::setProposal<PyGibbsParameter>)
   .def ("set", &pygibbs::set<PyGibbsParameter>, 
         (
         arg("steps")=1e5,
         arg("burnInSteps")= 0,
         arg("variableProposal")=pyenums::RANDOM
         )
   )
   ;

   OPENGM_PYTHON_VERBOSE_VISITOR_EXPORTER(PyGibbsVerboseVisitor,"GibbsVerboseVisitor" );
   OPENGM_PYTHON_INFERENCE_EXPORTER(PyGibbs,"Gibbs");

}

template void export_gibbs<GmAdder,opengm::Minimizer>();
//template void export_gibbs<GmAdder,opengm::Maximizer>();
//template void export_gibbs<GmMultiplier,opengm::Minimizer>();
template void export_gibbs<GmMultiplier,opengm::Maximizer>();