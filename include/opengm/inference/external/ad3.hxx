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


      template<class GM>
      class AD3Inf : public Inference<GM, opengm::Maximizer> {

      public:
         typedef GM GraphicalModelType;
         typedef opengm::Maximizer AccumulationType;
         OPENGM_GM_TYPE_TYPEDEFS;
         typedef VerboseVisitor<AD3Inf<GM> > VerboseVisitorType;
         typedef TimingVisitor<AD3Inf<GM> > TimingVisitorType;
         typedef EmptyVisitor<AD3Inf<GM> > EmptyVisitorType;
    

         struct Parameter {
            Parameter(
               const double eta=0.1,
               const bool adaptEta=true,
               UInt64Type steps=1000
            ) :
               eta_(eta),
               adaptEta_(adaptEta),
               steps_(steps){  
            }

            double eta_;
            bool adaptEta_;
            UInt64Type steps_;
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
            std::vector<LabelType> arg;
            this->arg(arg);
            return gm_.evaluate(arg);
         }

         ValueType bound()const{
            std::vector<LabelType> arg;
            this->arg(arg);
            return gm_.evaluate(arg);
         }

      private:
         const GraphicalModelType& gm_;
         Parameter parameter_;

         // AD3Inf MEMBERS
         AD3::FactorGraph factor_graph_;
         std::vector<AD3::MultiVariable*>  multi_variables_;

         std::vector<double> posteriors_;
         std::vector<double> additional_posteriors_;
         double value_;


      };
      // public interface
      /// \brief Construcor
      /// \param gm graphical model
      /// \param para belief propargation paramaeter

      template<class GM>
      AD3Inf<GM>
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
         value_()
      {
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
            gm_[fi].copyValues(facVal);
            //gm_[fi].copyValuesSwitchedOrder(facVal);
            const IndexType nVar=gm_[fi].numberOfVariables();

            if(nVar==1){
               const IndexType vi0 = gm_[fi].variableIndex(0);
               const IndexType nl0 = gm_.numberOfLabels(vi0);

               for(LabelType l=0;l<nl0;++l){
                  const ValueType logP = multi_variables_[vi0]->GetLogPotential(l);
                  const ValueType val  = facVal[l]; 
                  multi_variables_[vi0]->SetLogPotential(l,logP+val);
               }
            }
            else if (nVar>1){
               std::cout<<"factor size "<<gm_[fi].size()<<"\n";
               // create higher order factor function
               std::vector<double> additional_log_potentials(gm_[fi].size());
               std::copy(facVal,facVal+gm_[fi].size(),additional_log_potentials.begin());
               OPENGM_CHECK_OP(facVal[0],==,additional_log_potentials[0],"");
               OPENGM_CHECK_OP(facVal[1],==,additional_log_potentials[1],"");
               OPENGM_CHECK_OP(facVal[2],==,additional_log_potentials[2],"");
               OPENGM_CHECK_OP(facVal[3],==,additional_log_potentials[3],"");

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

      template<class GM>
      AD3Inf<GM>
      ::~AD3Inf() {

      }

      template<class GM>
      inline std::string
      AD3Inf<GM>
      ::name() const {
         return "AD3Inf";
      }

      template<class GM>
      inline const typename AD3Inf<GM>::GraphicalModelType&
      AD3Inf<GM>
      ::graphicalModel() const {
         return gm_;
      }

      template<class GM>
      inline InferenceTermination
      AD3Inf<GM>
      ::infer() {
         EmptyVisitorType v;
         return infer(v);
      }

      template<class GM>
      template<class VisitorType>
      InferenceTermination 
      AD3Inf<GM>::infer(VisitorType& visitor)
      { 
         visitor.begin(*this);
         
         factor_graph_.SetEtaAD3(parameter_.eta_);
         factor_graph_.AdaptEtaAD3(parameter_.adaptEta_);
         factor_graph_.SetMaxIterationsAD3(parameter_.steps_);
         factor_graph_.SolveLPMAPWithAD3(&posteriors_, &additional_posteriors_, &value_);

         visitor.end(*this);
         return NORMAL;
      }

      template<class GM>
      inline InferenceTermination
      AD3Inf<GM>
      ::arg(std::vector<LabelType>& arg, const size_t& n) const {
         if(n > 1) {
            return UNKNOWN;
         }
         else {
            arg.resize(gm_.numberOfVariables());
            if (posteriors_.size()<gm_.numberOfVariables() ){
               return NORMAL;
               OPENGM_CHECK(false,"");
            }
            else{
               UInt64Type c=0;
               for(IndexType vi = 0; vi < gm_.numberOfVariables(); ++vi) {
                  LabelType bestLabel = 0 ;
                  double    bestVal   = -100000;
                  for(LabelType l=0;l<gm_.numberOfLabels(vi);++l){
                     const double val = posteriors_[c];
                     std::cout<<"vi= "<<vi<<" l= "<<l<<" val= "<<val<<"\n";
                  
                     if(bestVal<0 || val>bestVal){
                        bestVal=val;
                        bestLabel=l;
                     }
                     ++c;
                  }
                  arg[vi]=bestLabel;
               }
            }
            return NORMAL;
         }
      }


   } // namespace external
} // namespace opengm

#endif // #ifndef OPENGM_EXTERNAL_AD3Inf_HXX

