#pragma once
#ifndef OPENGM_EXTERNAL_AD3_HXX
#define OPENGM_EXTERNAL_AD3_HXX

#include "opengm/graphicalmodel/graphicalmodel.hxx"
#include "opengm/inference/inference.hxx"
#include "opengm/inference/visitors/visitor.hxx"


#include "ad3/FactorGraph.h"
//#include "FactorSequence.h"


namespace opengm {
   namespace external {


      template<class GM,class ACC>
      class AD3Inf : public Inference<GM, ACC> {

      public:
         typedef GM GraphicalModelType;
         typedef ACC AccumulationType;
         OPENGM_GM_TYPE_TYPEDEFS;
         typedef VerboseVisitor<AD3Inf<GM,ACC> > VerboseVisitorType;
         typedef TimingVisitor<AD3Inf<GM,ACC> > TimingVisitorType;
         typedef EmptyVisitor<AD3Inf<GM,ACC> > EmptyVisitorType;
         
         enum SolverType{
            AD3_LP,
            AD3_ILP,
            PSDD_LP
         };

         struct Parameter {
            Parameter(
               const SolverType  solverType        = AD3_ILP,
               const double      eta               = 0.1,
               const bool        adaptEta          = true,
               UInt64Type        steps             = 1000,
               const double      residualThreshold = 1e-6,
               const int         verbosity         = 0 
            ) :
               solverType_(solverType),
               eta_(eta),
               adaptEta_(adaptEta),
               steps_(steps),
               residualThreshold_(residualThreshold),
               verbosity_(verbosity)
            {  
            }

            SolverType  solverType_;

            double      eta_;
            bool        adaptEta_;
            UInt64Type  steps_;
            double      residualThreshold_;
            int         verbosity_;
         };

         // construction
         AD3Inf(const GraphicalModelType& gm, const Parameter para = Parameter());
         ~AD3Inf();

         // query
         std::string name() const;
         const GraphicalModelType& graphicalModel() const;
         // inference
         InferenceTermination infer();
         template<class VisitorType>
         InferenceTermination infer(VisitorType&);
         InferenceTermination arg(std::vector<LabelType>&, const size_t& = 1) const;

         ValueType value()const{
            return gm_.evaluate(arg_);
         }

         ValueType bound()const{
            if(inferenceDone_ && parameter_.solverType_==AD3_ILP ){
               return this->value();
            }
            else{
               return bound_;
            }
         }


         ValueType valueToMaxSum(const ValueType val)const{
            if( meta::Compare<OperatorType,Adder>::value && meta::Compare<AccumulationType,Minimizer>::value){
               return val*(-1.0);
            }
            else if( meta::Compare<OperatorType,Adder>::value && meta::Compare<AccumulationType,Maximizer>::value){
               return val;
            }
         }

         ValueType valueFromMaxSum(const ValueType val)const{
            if( meta::Compare<OperatorType,Adder>::value && meta::Compare<AccumulationType,Minimizer>::value){
               return val*(-1.0);
            }
            else if( meta::Compare<OperatorType,Adder>::value && meta::Compare<AccumulationType,Maximizer>::value){
               return val;
            }
         }



      private:
         const GraphicalModelType& gm_;
         Parameter parameter_;

         // AD3Inf MEMBERS
         AD3::FactorGraph factor_graph_;
         std::vector<AD3::MultiVariable*>  multi_variables_;

         std::vector<double> posteriors_;
         std::vector<double> additional_posteriors_;
         double bound_;

         std::vector<LabelType> arg_;

         bool inferenceDone_;

      };
      // public interface
      /// \brief Construcor
      /// \param gm graphical model
      /// \param para belief propargation paramaeter

      template<class GM,class ACC>
      AD3Inf<GM,ACC>
      ::AD3Inf(
         const typename AD3Inf::GraphicalModelType& gm,
         const Parameter para
      )  : 
         gm_(gm),
         parameter_(para),
         factor_graph_(),
         multi_variables_(gm.numberOfVariables()),
         posteriors_(),
         additional_posteriors_(),
         bound_(),
         arg_(gm.numberOfVariables(),static_cast<LabelType>(0)),
         inferenceDone_(false)
      {

         if(meta::Compare<OperatorType,Adder>::value==false){
            throw RuntimeError("AD3 does not only support opengm::Adder as Operator");
         }

         if(meta::Compare<AccumulationType,Minimizer>::value==false and meta::Compare<AccumulationType,Maximizer>::value==false ){
            throw RuntimeError("AD3 does not only support opengm::Minimizer and opengm::Maximizer as Accumulatpr");
         }


         bound_ =  ACC::template ineutral<ValueType>();



         factor_graph_.SetVerbosity(parameter_.verbosity_);
         UInt64Type maxFactorSize = 0 ; 
         for(IndexType fi=0;fi<gm_.numberOfFactors();++fi){
            maxFactorSize=std::max(static_cast<UInt64Type>(gm_[fi].size()),maxFactorSize);
         }

         ValueType * facVal = new ValueType[maxFactorSize];


         // fill space :
         //  - Create a multi-valued variable for variable of gm 
         //    and initialize unaries with 0
         for(IndexType vi=0;vi<gm_.numberOfVariables();++vi){
            multi_variables_[vi] = factor_graph_.CreateMultiVariable(gm_.numberOfLabels(vi));
            for(LabelType l=0;l<gm_.numberOfLabels(vi);++l){
               multi_variables_[vi]->SetLogPotential(l,0.0);
            }
         }


         // - add higher order factors
         // - setup values for 1. order and higher order factors
         for(IndexType fi=0;fi<gm_.numberOfFactors();++fi){
            //gm_[fi].copyValues(facVal);
            gm_[fi].copyValuesSwitchedOrder(facVal);
            const IndexType nVar=gm_[fi].numberOfVariables();

            if(nVar==1){
               const IndexType vi0 = gm_[fi].variableIndex(0);
               const IndexType nl0 = gm_.numberOfLabels(vi0);

               for(LabelType l=0;l<nl0;++l){
                  const ValueType logP = multi_variables_[vi0]->GetLogPotential(l);
                  const ValueType val  = this->valueToMaxSum(facVal[l]); 
                  multi_variables_[vi0]->SetLogPotential(l,logP+val);
               }
            }
            else if (nVar>1){
               // std::cout<<"factor size "<<gm_[fi].size()<<"\n";
               // create higher order factor function
               std::vector<double> additional_log_potentials(gm_[fi].size());
               for(IndexType i=0;i<gm_[fi].size();++i){
                  additional_log_potentials[i]=this->valueToMaxSum(facVal[i]);
               }

               // create high order factor vi
               std::vector<AD3::MultiVariable*> multi_variables_local(nVar);
               for(IndexType v=0;v<nVar;++v){
                  multi_variables_local[v]=multi_variables_[gm_[fi].variableIndex(v)];
               }

               // create higher order factor
               factor_graph_.CreateFactorDense(multi_variables_local,additional_log_potentials);
            }
            else{
               OPENGM_CHECK(false,"const factors are not yet implemented");
            }

         }

         // delete buffer
         delete[] facVal;
      }

      template<class GM,class ACC>
      AD3Inf<GM,ACC>
      ::~AD3Inf() {

      }

      template<class GM,class ACC>
      inline std::string
      AD3Inf<GM,ACC>
      ::name() const {
         return "AD3Inf";
      }

      template<class GM,class ACC>
      inline const typename AD3Inf<GM,ACC>::GraphicalModelType&
      AD3Inf<GM,ACC>
      ::graphicalModel() const {
         return gm_;
      }

      template<class GM,class ACC>
      inline InferenceTermination
      AD3Inf<GM,ACC>
      ::infer() {
         EmptyVisitorType v;
         return infer(v);
      }

      template<class GM,class ACC>
      template<class VisitorType>
      InferenceTermination 
      AD3Inf<GM,ACC>::infer(VisitorType& visitor)
      { 
         visitor.begin(*this);

         // set parameters
         if(parameter_.solverType_ == AD3_LP || parameter_.solverType_ == AD3_ILP){
            factor_graph_.SetEtaAD3(parameter_.eta_);
            factor_graph_.AdaptEtaAD3(parameter_.adaptEta_);
            factor_graph_.SetMaxIterationsAD3(parameter_.steps_);
            factor_graph_.SetResidualThresholdAD3(parameter_.residualThreshold_);
         }
         if(parameter_.solverType_ == PSDD_LP){
            factor_graph_.SetEtaPSDD(parameter_.eta_);
            factor_graph_.SetMaxIterationsPSDD(parameter_.steps_);
         }


         // solve
         if ( parameter_.solverType_ == AD3_LP){
            factor_graph_.SolveLPMAPWithAD3(&posteriors_, &additional_posteriors_, &bound_);
         }
         if ( parameter_.solverType_ == AD3_ILP){
            factor_graph_.SolveExactMAPWithAD3(&posteriors_, &additional_posteriors_, &bound_);
         }
         if (parameter_.solverType_ == PSDD_LP){
            factor_graph_.SolveExactMAPWithAD3(&posteriors_, &additional_posteriors_, &bound_);
         }

         // transform bound
         bound_ =this->valueFromMaxSum(bound_);

         // make gm arg
         UInt64Type c=0;
         for(IndexType vi = 0; vi < gm_.numberOfVariables(); ++vi) {
            LabelType bestLabel = 0 ;
            double    bestVal   = -100000;
            for(LabelType l=0;l<gm_.numberOfLabels(vi);++l){
               const double val = posteriors_[c];
               //std::cout<<"vi= "<<vi<<" l= "<<l<<" val= "<<val<<"\n";
            
               if(bestVal<0 || val>bestVal){
                  bestVal=val;
                  bestLabel=l;
               }
               ++c;
            }
            arg_[vi]=bestLabel;
         }
         inferenceDone_=true;


         visitor.end(*this);
         return NORMAL;
      }

      template<class GM,class ACC>
      inline InferenceTermination
      AD3Inf<GM,ACC>
      ::arg(std::vector<LabelType>& arg, const size_t& n) const {
         if(n > 1) {
            return UNKNOWN;
         }
         else {
            arg.resize(gm_.numberOfVariables());
            std::copy(arg_.begin(),arg_.end(),arg.begin());
            return NORMAL;
         }
      }


   } // namespace external
} // namespace opengm

#endif // #ifndef OPENGM_EXTERNAL_AD3Inf_HXX

