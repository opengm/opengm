#pragma once
#ifndef OPENGM_EXTERNAL_AD3_HXX
#define OPENGM_EXTERNAL_AD3_HXX

#include "opengm/graphicalmodel/graphicalmodel.hxx"
#include "opengm/inference/inference.hxx"
#include "opengm/inference/visitors/visitors.hxx"



#include "ad3/FactorGraph.h"
//#include "FactorSequence.h"


namespace opengm {
   namespace external {

      /// \brief AD3\n
      /// \ingroup inference 
      /// \ingroup external_inference

      template<class GM,class ACC>
      class AD3Inf : public Inference<GM, ACC> {

      public:
         typedef GM GraphicalModelType;
         typedef ACC AccumulationType;
         OPENGM_GM_TYPE_TYPEDEFS;
         typedef visitors::VerboseVisitor<AD3Inf<GM,ACC> > VerboseVisitorType;
         typedef visitors::EmptyVisitor<AD3Inf<GM,ACC> >   EmptyVisitorType;
         typedef visitors::TimingVisitor<AD3Inf<GM,ACC> >  TimingVisitorType;
         
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
               return bound_;
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


         // iterface to create a ad3 gm without a gm

         template<class N_LABELS_ITER>
         AD3Inf(N_LABELS_ITER nLabelsBegin,N_LABELS_ITER nLabelsEnd, const Parameter para = Parameter());


         AD3Inf(const UInt64Type nVar,const UInt64Type nLabels, const Parameter para,const bool foo);


         template<class VI_ITERATOR,class FUNCTION>
         void addFactor(VI_ITERATOR viBegin,VI_ITERATOR viEnd,const FUNCTION & function);


         const std::vector<double> & posteriors()const{
            return posteriors_;
         }

         const std::vector<double> & higherOrderPosteriors()const{
            return additional_posteriors_;
         }  


      private:
         const GraphicalModelType& gm_;
         Parameter parameter_;
         IndexType numVar_;

         // AD3Inf MEMBERS
         AD3::FactorGraph factor_graph_;
         std::vector<AD3::MultiVariable*>  multi_variables_;

         std::vector<double> posteriors_;
         std::vector<double> additional_posteriors_;
         double bound_;

         std::vector<LabelType> arg_;

         bool inferenceDone_;

         std::vector<LabelType> space_;  // only used if setup without gm

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
         numVar_(gm.numberOfVariables()),
         factor_graph_(),
         multi_variables_(gm.numberOfVariables()),
         posteriors_(),
         additional_posteriors_(),
         bound_(),
         arg_(gm.numberOfVariables(),static_cast<LabelType>(0)),
         inferenceDone_(false),
         space_(0)
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
      template<class N_LABELS_ITER>
      AD3Inf<GM,ACC>::AD3Inf(
         N_LABELS_ITER nLabelsBegin,
         N_LABELS_ITER nLabelsEnd,
         const Parameter para
      ) :
      gm_(GM()), // DIRTY
      parameter_(para),
      numVar_(std::distance(nLabelsBegin,nLabelsEnd)),
      factor_graph_(),
      multi_variables_(std::distance(nLabelsBegin,nLabelsEnd)),
      posteriors_(),
      additional_posteriors_(),
      bound_(),
      arg_(std::distance(nLabelsBegin,nLabelsEnd),static_cast<LabelType>(0)),
      space_(nLabelsBegin,nLabelsEnd)
      {

         if(meta::Compare<OperatorType,Adder>::value==false){
            throw RuntimeError("AD3 does not only support opengm::Adder as Operator");
         }
         if(meta::Compare<AccumulationType,Minimizer>::value==false and meta::Compare<AccumulationType,Maximizer>::value==false ){
            throw RuntimeError("AD3 does not only support opengm::Minimizer and opengm::Maximizer as Accumulatpr");
         }
         bound_ =  ACC::template ineutral<ValueType>();
         factor_graph_.SetVerbosity(parameter_.verbosity_);

         //  and initialize unaries with 0
         for(IndexType vi=0;vi<numVar_;++vi){
            multi_variables_[vi] = factor_graph_.CreateMultiVariable(space_[vi]);
            for(LabelType l=0;l<space_[vi];++l){
               multi_variables_[vi]->SetLogPotential(l,0.0);
            }
         }
      }

      template<class GM,class ACC>
      AD3Inf<GM,ACC>::AD3Inf(
         const UInt64Type nVar,
         const UInt64Type nLabels,
         const Parameter para,
         const bool foo
      ) :
      gm_(GM()), // DIRTY
      parameter_(para),
      numVar_(nVar),
      factor_graph_(),
      multi_variables_(nVar),
      posteriors_(),
      additional_posteriors_(),
      bound_(),
      arg_(nVar,static_cast<LabelType>(0)),
      space_(nVar,nLabels)
      {

         if(meta::Compare<OperatorType,Adder>::value==false){
            throw RuntimeError("AD3 does not only support opengm::Adder as Operator");
         }
         if(meta::Compare<AccumulationType,Minimizer>::value==false and meta::Compare<AccumulationType,Maximizer>::value==false ){
            throw RuntimeError("AD3 does not only support opengm::Minimizer and opengm::Maximizer as Accumulatpr");
         }
         bound_ =  ACC::template ineutral<ValueType>();
         factor_graph_.SetVerbosity(parameter_.verbosity_);
         for(IndexType vi=0;vi<numVar_;++vi){
            multi_variables_[vi] = factor_graph_.CreateMultiVariable(space_[vi]);
            for(LabelType l=0;l<space_[vi];++l){
               multi_variables_[vi]->SetLogPotential(l,0.0);
            }
         }
      }


      template<class GM,class ACC>
      template<class VI_ITERATOR,class FUNCTION>
      void 
      AD3Inf<GM,ACC>::addFactor(
         VI_ITERATOR visBegin,
         VI_ITERATOR visEnd,
         const FUNCTION & function
      ){
         const IndexType nVis = std::distance(visBegin,visEnd);
         OPENGM_CHECK_OP(nVis,==,function.dimension(),"functions dimension does not match number of variabole indices");

         for(IndexType v=0;v<nVis;++v){
            OPENGM_CHECK_OP(space_[visBegin[v]],==,function.shape(v),"functions shape does not match space");
         }


         if(nVis==1){
            LabelType l[1];
            for(l[0]=0; l[0]<space_[visBegin[0]]; ++l[0]){   
               const ValueType logP = multi_variables_[visBegin[0]]->GetLogPotential(l[0]);
               const ValueType val  = this->valueToMaxSum(function(l)); 
               multi_variables_[visBegin[0]]->SetLogPotential(l[0],logP+val);
            }
         }
         else if(nVis>=2){
            

            // create high order factor vi
            std::vector<AD3::MultiVariable*> multi_variables_local(nVis);
            for(IndexType v=0;v<nVis;++v){
               multi_variables_local[v]=multi_variables_[visBegin[v]];
            }

            // create higher order function (for dense factor)
            std::vector<double> additional_log_potentials(function.size());

            // FILL THE FUNCTION

            if(nVis==2){
               LabelType l[2];
               UInt64Type c=0;
               for(l[0]=0; l[0]<space_[visBegin[0]]; ++l[0])
               for(l[1]=0; l[1]<space_[visBegin[1]]; ++l[1]){
                  additional_log_potentials[c]=this->valueToMaxSum(function(l)); 
                  ++c;
               }  
            }
            else if(nVis==3){
               LabelType l[3];
               UInt64Type c=0;
               for(l[0]=0; l[0]<space_[visBegin[0]]; ++l[0])
               for(l[1]=0; l[1]<space_[visBegin[1]]; ++l[1])
               for(l[2]=0; l[2]<space_[visBegin[2]]; ++l[2]){
                  additional_log_potentials[c]=this->valueToMaxSum(function(l)); 
                  ++c;
               }  
            }
            else if(nVis==4){
               LabelType l[4];
               UInt64Type c=0;
               for(l[0]=0; l[0]<space_[visBegin[0]]; ++l[0])
               for(l[1]=0; l[1]<space_[visBegin[1]]; ++l[1])
               for(l[2]=0; l[2]<space_[visBegin[2]]; ++l[2])
               for(l[3]=0; l[3]<space_[visBegin[3]]; ++l[3]){
                  additional_log_potentials[c]=this->valueToMaxSum(function(l)); 
                  ++c;
               }  
            }
            else if(nVis==5){
               LabelType l[5];
               UInt64Type c=0;
               for(l[0]=0; l[0]<space_[visBegin[0]]; ++l[0])
               for(l[1]=0; l[1]<space_[visBegin[1]]; ++l[1])
               for(l[2]=0; l[2]<space_[visBegin[2]]; ++l[2])
               for(l[3]=0; l[3]<space_[visBegin[3]]; ++l[3])
               for(l[4]=0; l[4]<space_[visBegin[4]]; ++l[4]){
                  additional_log_potentials[c]=this->valueToMaxSum(function(l)); 
                  ++c;
               }  
            }
            else if(nVis==6){
               LabelType l[6];
               UInt64Type c=0;
               for(l[0]=0; l[0]<space_[visBegin[0]]; ++l[0])
               for(l[1]=0; l[1]<space_[visBegin[1]]; ++l[1])
               for(l[2]=0; l[2]<space_[visBegin[2]]; ++l[2])
               for(l[3]=0; l[3]<space_[visBegin[3]]; ++l[3])
               for(l[4]=0; l[4]<space_[visBegin[4]]; ++l[4])
               for(l[5]=0; l[5]<space_[visBegin[5]]; ++l[5]){
                  additional_log_potentials[c]=this->valueToMaxSum(function(l)); 
                  ++c;
               }  
            }
            else if(nVis==7){
               LabelType l[7];
               UInt64Type c=0;
               for(l[0]=0; l[0]<space_[visBegin[0]]; ++l[0])
               for(l[1]=0; l[1]<space_[visBegin[1]]; ++l[1])
               for(l[2]=0; l[2]<space_[visBegin[2]]; ++l[2])
               for(l[3]=0; l[3]<space_[visBegin[3]]; ++l[3])
               for(l[4]=0; l[4]<space_[visBegin[4]]; ++l[4])
               for(l[5]=0; l[5]<space_[visBegin[5]]; ++l[5])
               for(l[6]=0; l[6]<space_[visBegin[6]]; ++l[6]){
                  additional_log_potentials[c]=this->valueToMaxSum(function(l)); 
                  ++c;
               }  
            }
            else if(nVis==8){
               LabelType l[8];
               UInt64Type c=0;
               for(l[0]=0; l[0]<space_[visBegin[0]]; ++l[0])
               for(l[1]=0; l[1]<space_[visBegin[1]]; ++l[1])
               for(l[2]=0; l[2]<space_[visBegin[2]]; ++l[2])
               for(l[3]=0; l[3]<space_[visBegin[3]]; ++l[3])
               for(l[4]=0; l[4]<space_[visBegin[4]]; ++l[4])
               for(l[5]=0; l[5]<space_[visBegin[5]]; ++l[5])
               for(l[6]=0; l[6]<space_[visBegin[6]]; ++l[6])
               for(l[7]=0; l[7]<space_[visBegin[7]]; ++l[7])
               {
                  additional_log_potentials[c]=this->valueToMaxSum(function(l)); 
                  ++c;
               }  
            }
            else if(nVis==9){
               LabelType l[9];
               UInt64Type c=0;
               for(l[0]=0; l[0]<space_[visBegin[0]]; ++l[0])
               for(l[1]=0; l[1]<space_[visBegin[1]]; ++l[1])
               for(l[2]=0; l[2]<space_[visBegin[2]]; ++l[2])
               for(l[3]=0; l[3]<space_[visBegin[3]]; ++l[3])
               for(l[4]=0; l[4]<space_[visBegin[4]]; ++l[4])
               for(l[5]=0; l[5]<space_[visBegin[5]]; ++l[5])
               for(l[6]=0; l[6]<space_[visBegin[6]]; ++l[6])
               for(l[7]=0; l[7]<space_[visBegin[7]]; ++l[7])
               for(l[8]=0; l[8]<space_[visBegin[8]]; ++l[8])
               {
                  additional_log_potentials[c]=this->valueToMaxSum(function(l)); 
                  ++c;
               }  
            }
            else if(nVis==10){
               LabelType l[10];
               UInt64Type c=0;
               for(l[0]=0; l[0]<space_[visBegin[0]]; ++l[0])
               for(l[1]=0; l[1]<space_[visBegin[1]]; ++l[1])
               for(l[2]=0; l[2]<space_[visBegin[2]]; ++l[2])
               for(l[3]=0; l[3]<space_[visBegin[3]]; ++l[3])
               for(l[4]=0; l[4]<space_[visBegin[4]]; ++l[4])
               for(l[5]=0; l[5]<space_[visBegin[5]]; ++l[5])
               for(l[6]=0; l[6]<space_[visBegin[6]]; ++l[6])
               for(l[7]=0; l[7]<space_[visBegin[7]]; ++l[7])
               for(l[8]=0; l[8]<space_[visBegin[8]]; ++l[8])
               for(l[9]=0; l[9]<space_[visBegin[9]]; ++l[9])
               {
                  additional_log_potentials[c]=this->valueToMaxSum(function(l)); 
                  ++c;
               }  
            }
            else{
               throw RuntimeError("order must be <=10 for inplace building of Ad3Inf (call us if you need higher order)");
            }





            // create higher order factor
            factor_graph_.CreateFactorDense(multi_variables_local,additional_log_potentials);

         }

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
            //std::cout<<"ad3  lp\n";
            factor_graph_.SolveLPMAPWithAD3(&posteriors_, &additional_posteriors_, &bound_);
         }
         if ( parameter_.solverType_ == AD3_ILP){
            //std::cout<<"ad3 ilp\n";
            factor_graph_.SolveExactMAPWithAD3(&posteriors_, &additional_posteriors_, &bound_);
         }
         if (parameter_.solverType_ == PSDD_LP){
            //std::cout<<"ad3 psdd lp\n";
            factor_graph_.SolveExactMAPWithAD3(&posteriors_, &additional_posteriors_, &bound_);
         }

         // transform bound
         bound_ =this->valueFromMaxSum(bound_);

         // make gm arg
         UInt64Type c=0;
         for(IndexType vi = 0; vi < numVar_; ++vi) {
            LabelType bestLabel = 0 ;
            double    bestVal   = -100000;
            const LabelType nLabels = (space_.size()==0 ?  gm_.numberOfLabels(vi) : space_[vi] );
            for(LabelType l=0;l< nLabels;++l){
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
            arg.resize(numVar_);
            std::copy(arg_.begin(),arg_.end(),arg.begin());
            return NORMAL;
         }
      }


   } // namespace external
} // namespace opengm

#endif // #ifndef OPENGM_EXTERNAL_AD3Inf_HXX

