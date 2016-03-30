#pragma once
#ifndef OPENGM_INFERENCE_SAT_HXX
#define OPENGM_INFERENCE_SAT_HXX

#include <iostream>
#include <vector>

#include <boost/config.hpp>
#include <boost/graph/strong_components.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/foreach.hpp>

#include "opengm/inference/inference.hxx"
#include "opengm/operations/and.hxx"
#include "opengm/operations/or.hxx"
#include "opengm/inference/visitors/visitors.hxx"

namespace opengm {

   /// 2-SAT solver
   ///
   /// \ingroup inference
   template<class GM>
   class SAT : Inference<GM, opengm::Or> {
   public:
      typedef opengm::Or AccumulationType;
      typedef GM GraphicalModelType;
      OPENGM_GM_TYPE_TYPEDEFS;

      struct Parameter {};

      SAT(const GraphicalModelType&, const Parameter& = Parameter());
      std::string name() const;
      const GraphicalModelType& graphicalModel() const;
      InferenceTermination infer();
      template<class VISITOR>
      InferenceTermination infer(VISITOR &);
      virtual void reset();
      ValueType value() const;
      typedef visitors::VerboseVisitor<SAT<GM> >        VerboseVisitorType;
      typedef visitors::TimingVisitor<SAT<GM> >         TimingVisitorType;
		typedef visitors::EmptyVisitor<SAT<GM> >          EmptyVisitorType;
   private:
      const GraphicalModelType& gm_;
      std::vector<int> component_;
   };

   template<class GM>
   inline SAT<GM>::SAT
   (
      const GraphicalModelType& gm,
      const Parameter& para
   )
   :  gm_(gm)
   {
      if(!NO_DEBUG) {
         OPENGM_ASSERT(gm_.factorOrder() <= 2);
         OPENGM_ASSERT(typeid(OperatorType) == typeid(opengm::And));
         for(size_t i=0; i<gm_.numberOfVariables();++i) {
            OPENGM_ASSERT(gm_.numberOfLabels(i) == 2);
         }
      }
   }
   template<class GM>
   void
   inline SAT<GM>::reset()
   {
   }

   template<class GM>
   inline std::string
   SAT<GM>::name() const
   {
      return "2Sat";
   }

   template<class GM>
   inline const GM&
   SAT<GM>::graphicalModel() const
   {
      return gm_;
   }
   
   template<class GM>
   InferenceTermination
   SAT<GM>::infer() {
      EmptyVisitorType v;
      return infer(v);
   }
   
   template<class GM>
   template<class VISITOR>
   InferenceTermination
   SAT<GM>::infer
   (
      VISITOR & visitor
   ) {
      visitor.begin(*this);
      typedef std::pair<int, int> clause;
      typedef boost::adjacency_list<> Graph; // properties of our graph. by default: oriented graph
      // build graph
      Graph g(gm_.numberOfVariables() * 2);
      for(size_t f=0; f<gm_.numberOfFactors(); ++f) {
         if(gm_[f].numberOfVariables() != 2) {
            throw RuntimeError("This implementation of the 2-SAT solver supports only factors of order 2.");
         }
         std::vector<size_t> vec(2);
         for(vec[0]=0; vec[0]<2; ++vec[0]) {
            for(vec[1]=0; vec[1]<2; ++vec[1]) {
               if(!gm_[f](vec.begin())) {
                  const int  v1=gm_[f].variableIndex(0)+(1-vec[0])*gm_.numberOfVariables();
                  const int nv1=gm_[f].variableIndex(0)+(0+vec[0])*gm_.numberOfVariables();
                  const int  v2=gm_[f].variableIndex(1)+(1-vec[1])*gm_.numberOfVariables();
                  const int nv2=gm_[f].variableIndex(1)+(0+vec[1])*gm_.numberOfVariables();
                  boost::add_edge(nv1,v2,g);
                  boost::add_edge(nv2,v1,g);
               }
            }
         }
      }
      component_.resize(num_vertices(g));
      strong_components(g, make_iterator_property_map(component_.begin(), get(boost::vertex_index, g)));
      visitor.end(*this);
      return NORMAL;
   }

   template<class GM>
   inline typename GM::ValueType
   SAT<GM>::value() const
   {
      bool satisfied = true;
      for(IndexType i=0; i<gm_.numberOfVariables(); i++) {
         if(component_[i] == component_[i+gm_.numberOfVariables()]) {
            satisfied = false;
         }
      }
      return satisfied;
   }

} // namespace opengm

#endif // #ifndef OPENGM_INFERENCE_SAT_HXX

