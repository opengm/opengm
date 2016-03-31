#pragma once
#ifndef OPENGM_GREEDYGREMLIN_HXX
#define OPENGM_GREEDYGREMLIN_HXX

#include <cmath>
#include <vector>
#include <list>
#include <set>
#include <iostream>
#include <algorithm>
#include <iostream>
#include <functional>

#include "opengm/opengm.hxx"
#include "opengm/graphicalmodel/graphicalmodel.hxx"
#include "opengm/inference/inference.hxx"
#include "opengm/inference/visitors/visitors.hxx"
namespace opengm {



/// \endcond 

   /// \brief GREEDY GREMLIN
   ///
   /// The greedy gremlin is a simple greedy algorithm for inference on graphical models.
   /// It itteratively fix a variable that is the best given all so far fixed variables,
   /// by ingoring all factors that include variables that are not fixed so far (exclude the current variable)
   ///
   /// The greedy gremlin defines a baseline for other algorithms.
   ///
   /// \ingroup inference
   template<class GM,class ACC>
   class GreedyGremlin : public Inference<GM,ACC>
   {
   public:
      ///graphical model type
      typedef GM                                          GraphicalModelType;
      ///accumulation type
      typedef ACC                                         AccumulationType;
      OPENGM_GM_TYPE_TYPEDEFS;
      /// visitor 
      typedef visitors::VerboseVisitor<GreedyGremlin<GM, ACC> > VerboseVisitorType;
      typedef visitors::EmptyVisitor<GreedyGremlin<GM, ACC> >   EmptyVisitorType;
      typedef visitors::TimingVisitor<GreedyGremlin<GM, ACC> >  TimingVisitorType;
      
      struct Parameter {
       
      };
      GreedyGremlin(const GM& gm, Parameter para = Parameter());
      virtual std::string name() const {return "GreedyGremlin";}
      const GraphicalModelType& graphicalModel() const;
      virtual InferenceTermination infer();
      virtual void reset();
      template<class VisitorType> InferenceTermination infer(VisitorType& vistitor);
      virtual InferenceTermination marginal(const size_t,IndependentFactorType& out)const        {return UNKNOWN;}
      virtual InferenceTermination factorMarginal(const size_t, IndependentFactorType& out)const {return UNKNOWN;}
      virtual InferenceTermination arg(std::vector<LabelType>& v, const size_t = 1)const;
      virtual InferenceTermination args(std::vector< std::vector<LabelType> >& v)const;

   private:
      const GM&                                   gm_;
      Parameter                                   parameter_;
      std::vector<LabelType>                      conf_;
   };


//*******************
//** Impelentation **
//*******************

/// \brief constructor
/// \param gm graphical model
/// \param para GreedyGremlin parameter
   template<class GM, class ACC >
   GreedyGremlin<GM,ACC>::GreedyGremlin
   (
      const GM& gm,
      Parameter para
      ):gm_(gm), parameter_(para)
   {
      conf_.resize(gm.numberOfVariables(),0);
   }
  
   /// \brief reset
   ///
   /// \warning  reset assumes that the structure of
   /// the graphical model has not changed
   ///
   /// TODO
   template<class GM, class ACC >
   void
   GreedyGremlin<GM,ACC>::reset()
   {
      ///todo
   }

   template <class GM, class ACC>
   InferenceTermination
   GreedyGremlin<GM,ACC>::infer()
   { 
      EmptyVisitorType v;
      return infer(v);
   }

/// \brief inference with visitor
/// \param visitor visitor
   template<class GM, class ACC>
   template<class VisitorType>
   InferenceTermination GreedyGremlin<GM,ACC>::infer(VisitorType& visitor)
   {
      std::vector<bool>       nodeColor(gm_.numberOfVariables(),false);
      std::vector<IndexType>  waitingList(gm_.numberOfVariables());
      waitingList[0] = 0;
      nodeColor[0]   = true;
      IndexType waitingListFirst = 0;
      IndexType waitingListLast  = 0;
      visitor.begin(*this);    
      const ValueType neutral = GM::OperatorType::template neutral<ValueType>();
      while(waitingListFirst<waitingList.size()){
         OPENGM_ASSERT(waitingListFirst<=waitingListLast);
         IndexType var = waitingList[waitingListFirst++];
         std::vector<ValueType> vals(gm_.numberOfLabels(var),neutral);
         //for all neigboured factors
         for(typename GM::ConstFactorIterator fit=gm_.factorsOfVariableBegin(var); fit!=gm_.factorsOfVariableEnd(var); ++fit){
            bool useIt = true;
            for(typename GM::ConstVariableIterator vit=gm_.variablesOfFactorBegin(*fit); vit!=gm_.variablesOfFactorEnd(*fit); ++vit){
               if(nodeColor[*vit]==false){
                  useIt = false;
                  break;
               }
            }
            if(useIt){
               std::vector<LabelType> l(gm_[*fit].numberOfVariables());
               size_t p;
               for(size_t i=0; i<l.size();++i){
                  if(gm_[*fit].variableIndex(i)==var)
                     p=i;
                  else
                     l[i] = conf_[gm_[*fit].variableIndex(i)];
               }
               for(l[p]=0; l[p]<gm_.numberOfLabels(var);++l[p]){
                  const ValueType v = gm_[*fit](l.begin());
                  GM::OperatorType::op(v,vals[l[p]]);
               }
            } 
         }

         //find best and fix
         for(size_t i=0; i<vals.size();++i){
            if(ACC::bop(vals[i],vals[conf_[var]]))
               conf_[var]=i;                    
         }

         //add white neighbours to waitingslist
         for(typename GM::ConstFactorIterator fit=gm_.factorsOfVariableBegin(var); fit!=gm_.factorsOfVariableEnd(var); ++fit){
            for(typename GM::ConstVariableIterator vit=gm_.variablesOfFactorBegin(*fit); vit!=gm_.variablesOfFactorEnd(*fit); ++vit){
               if(nodeColor[*vit]==false){
                  nodeColor[*vit]=true;
                  waitingList[++waitingListLast] = *vit;
               }
            }
         }
         if( visitor(*this) != visitors::VisitorReturnFlag::ContinueInf ){
            break;
         }
      }
           
      visitor.end(*this);     
      return NORMAL;
   }

   template<class GM, class ACC>
   InferenceTermination GreedyGremlin<GM, ACC>
   ::arg(std::vector<LabelType>& conf, const size_t n)const
   {
      if(n==1) {
         conf=conf_;
         return NORMAL;
      }else{
         conf.resize(0);
         return UNKNOWN;
      }
   }

/// \brief args
/// \param[out]conf state vectors
///
///get the inference solutions
   template<class GM, class ACC>
   InferenceTermination GreedyGremlin<GM,ACC>
   ::args(std::vector<std::vector<typename GreedyGremlin<GM,ACC>::LabelType> >& conf)const
   { 
      return UNKNOWN;
   }


   template<class GM, class ACC>
   inline const typename GreedyGremlin<GM, ACC>::GraphicalModelType&
   GreedyGremlin<GM, ACC>::graphicalModel() const
   {
      return gm_;
   }


} // namespace opengm

#endif // #ifndef OPENGM_GREEDYGREMLIN_HXX

