#ifndef HELPER_HXX
#define HELPER_HXX

#include <boost/python.hpp>
#include <boost/python/module.hpp>
#include <opengm/python/opengmpython.hxx>
#include <opengm/python/converter.hxx>
#include <opengm/python/numpyview.hxx>

#include <opengm/inference/icm.hxx>
#include <opengm/learning/gridsearch-learning.hxx>
#include <opengm/inference/messagepassing/messagepassing.hxx>

namespace opengm{

template<class LEARNER>
class LearnerInferenceSuite: public boost::python::def_visitor<LearnerInferenceSuite<LEARNER> >{
public:
   friend class boost::python::def_visitor_access;

   LearnerInferenceSuite(){

   }

   template<class INF>
   static void pyLearnWithInf(LEARNER & learner, const typename INF::Parameter & param)
   {
       learner. template learn<INF>(param);
   }

   template <class classT>
   void visit(classT& c) const{
       // SOME INFERENCE METHODS
       typedef typename LEARNER::GMType GMType;
       typedef typename LEARNER::Parameter PyLearnerParam;
       typedef typename LEARNER::DatasetType DatasetType;
       typedef opengm::Minimizer ACC;

       typedef opengm::ICM<GMType, ACC> IcmInf;
       typedef opengm::BeliefPropagationUpdateRules<GMType, ACC> UpdateRulesType;
       typedef opengm::MessagePassing<GMType, ACC, UpdateRulesType, opengm::MaxDistance> BpInf;

      c
          .def("_learn",&pyLearnWithInf<IcmInf>)
          .def("_learn",&pyLearnWithInf<BpInf>)
      ;
   }
};

} // namespace opengm

#endif // HELPER_HXX

