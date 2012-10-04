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
   template<class PARAM,class INF>
   inline void set
   (
      PARAM & p,
      const size_t steps,
      bool useTemp,
      const typename INF::ValueType tempMin,
      const typename INF::ValueType tempMax,
      const size_t periodes,
      const pyenums::GibbsVariableProposal  proposal     
   ){
      p.useTemp_=useTemp,
      p.maxNumberOfSamplingSteps_=steps;
      p.numberOfBurnInSteps_=0;
      p.tempMin_=tempMin;
      p.tempMax_=tempMax;
      p.periods_=periodes;
      p.p_=static_cast<typename INF::ValueType>(steps/periodes);
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
   

   
   
   class_<PyGibbsParameter > ("GibbsParameter", init<>())
   .def_readwrite("steps", &PyGibbsParameter::maxNumberOfSamplingSteps_,
   "Number of sampling steps"
   )
   .def_readwrite("useTemp", &PyGibbsParameter::useTemp_,
   "use temperature"
   )
   .def_readwrite("tempMin", &PyGibbsParameter::tempMin_,
   "Min Temperature in (0,1]"
   )
   .def_readwrite("tempMax", &PyGibbsParameter::tempMax_,
   "Min Temperature in (0,1]"
   )
   .def_readwrite("periodeLength", &PyGibbsParameter::p_,
   "periode length"
   )
   .add_property("variableProposal",&pygibbs::getProposal<PyGibbsParameter>, pygibbs::setProposal<PyGibbsParameter>,
   "variableProposal can be:\n\n"
   "  -``opengm.GibbsVariableProposal.random`` : The variable which is sampled is drawn randomly (default)\n\n"
   "  -``opengm.GibbsVariableProposal.cyclic`` : All variables will be sampled in a permuted order.\n\n"
   "     After all variables have been sampled the permutation is changed."
   )
   
   
   .def ("set", &pygibbs::set<PyGibbsParameter,PyGibbs>, 
      (
         arg("steps")=1e5,
         arg("useTemp")=true,
         arg("tempMin")= 0.001,
         arg("tempMax")= 1.0,
         arg("periodes")= 10,
         arg("variableProposal")=pyenums::RANDOM
      )
   ,
   "Set the parameters values.\n\n"
   "All values of the parameter have a default value.\n\n"
   "Args:\n\n"
   "  steps: Number of sampling steps (default=100)\n\n"
   "  useTemp: use temperature (default=True)\n\n"
   "  tempMin: Temperature in (0,1] (default=0.001)\n\n"
   "  tempMax: Temperature in (0,1] (default=1.0)\n\n"
   "  periodes: Number of periodes (default=10)\n\n"
   "  variableProposal: variableProposal can be:\n\n"
   "     -``opengm.GibbsVariableProposal.random`` : The variable which is sampled is drawn randomly (default)\n\n"
   "     -``opengm.GibbsVariableProposal.cyclic`` : All variables will be sampled in a permuted order.\n\n"
   "        After all variables have been sampled the permutation is changed.\n\n"
   "Returns:\n"
   "  None\n\n"
   )
   ;

   OPENGM_PYTHON_VERBOSE_VISITOR_EXPORTER(PyGibbsVerboseVisitor,"GibbsVerboseVisitor" );
   OPENGM_PYTHON_INFERENCE_EXPORTER(PyGibbs,"Gibbs",
   "Gibbs Sampler :\n\n"
   "cite: ???: \"`title <paper_url>`_\"," 
   "Journal.\n\n"
   "limitations: -\n\n"
   "guarantees: -\n"
   );

}

template void export_gibbs<GmAdder,opengm::Minimizer>();
//template void export_gibbs<GmAdder,opengm::Maximizer>();
//template void export_gibbs<GmMultiplier,opengm::Minimizer>();
template void export_gibbs<GmMultiplier,opengm::Maximizer>();
