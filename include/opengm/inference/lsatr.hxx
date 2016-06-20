#pragma once
#ifndef OPENGM_LSATR_HXX
#define OPENGM_LSATR_HXX

#include <algorithm>
#include <vector>
#include <string>
#include <iostream>

#include "opengm/opengm.hxx"
#include "opengm/inference/inference.hxx"
#include "opengm/inference/visitors/visitors.hxx"

#include <maxflowlib.h>

#ifndef NOVIGRA
#include "vigra/multi_distance.hxx"
#include "vigra/multi_array.hxx"
#endif

namespace opengm {
  
/// \brief Local Submodular Approximation with Trust Region regularization\n\n
///
/// Coresponding author: Joerg Hendrik Kappes
///
/// * Corresponding Papers:
/// 1) Lena Gorelick, Yuri Boykov, Olga Veksler, Ismail Ben Ayed and Andrew Delong  
///    Submodularization for Binary Pairwise Energies (CVPR 2014)
/// 2) Lena Gorelick, Frank R. Scmidt, Yuri Boykov
///    Fast Trust Region for Segmentation
/// * Corresponding/Reimplemented Matlab Code:
///    http://www.csd.uwo.ca/~ygorelic/downloads.html
/// * Thanks to Lena Gorelick for very helpful comments
/// \ingroup inference 



   struct LSA_TR_WeightedEdge{
      LSA_TR_WeightedEdge(double aw, size_t au, size_t av): w(aw), u(au), v(av){}
      double w;
      size_t u;
      size_t v;
   }; 


   template<class LabelType>
   class LSA_TR_HELPER{
   public:
      enum DISTANCE {HAMMING, EUCLIDEAN};
      
      LSA_TR_HELPER() { distanceType_= EUCLIDEAN;};
      ~LSA_TR_HELPER(){ if(graph_!=NULL){delete graph_; delete changedList_;}  };
      template<class GM>
      void init(const GM&, const std::vector<LabelType>& );
      void set(const double);
      void set(const std::vector<LabelType>&, const double);
      double optimize(std::vector<LabelType>&);
      void setDistanceType(const DISTANCE d){ distanceType_=d; };

      double eval(const std::vector<LabelType>&) const; 
      double evalAprox(const std::vector<LabelType>&,const std::vector<LabelType>&, const double) const;
      void evalBoth(const std::vector<LabelType>& label, const std::vector<LabelType>& workingPoint, const double lambda, double& value, double& valueAprox) const;

   private: 
      typedef maxflowLib::Graph<double,double,double>          graph_type;
      typedef maxflowLib::Block<typename graph_type::node_id>  block_type;

      void updateDistance();

      size_t                            numVar_;
      double                            lambda_;
      double                            constTerm_;
      double                            constTermApproximation_;
      double                            constTermTrustRegion_;
      std::vector<LabelType>            workingPoint_;
      std::vector<double>               distance_;
      std::vector<double>               unaries_; 
      std::vector<double>               approxUnaries_;
      std::vector< LSA_TR_WeightedEdge> supEdges_; 
      std::vector< LSA_TR_WeightedEdge> subEdges_;
      graph_type*                       graph_; 
      block_type*                       changedList_;
      bool                              solved_;
      DISTANCE                          distanceType_;
      std::vector<size_t>               shape_;
   };


   template<class GM, class ACC>
   class LSA_TR : public Inference<GM, ACC>
   {
   public:
      typedef ACC AccumulationType;
      typedef GM GraphicalModelType;
      OPENGM_GM_TYPE_TYPEDEFS;
      typedef opengm::visitors::VerboseVisitor<LSA_TR<GM,ACC> > VerboseVisitorType;
      typedef opengm::visitors::EmptyVisitor<LSA_TR<GM,ACC> >  EmptyVisitorType;
      typedef opengm::visitors::TimingVisitor<LSA_TR<GM,ACC> > TimingVisitorType; 
   
      class Parameter {
      public:
         enum DISTANCE {HAMMING, EUCLIDEAN};
         size_t randSeed_;
         double maxLambda_;
         double initialLambda_;
         double precisionLambda_; 
         double lambdaMultiplier_;
         double reductionRatio_;
         DISTANCE distance_;

         Parameter(){
            randSeed_         = 42;
            maxLambda_        = 1e5;
            initialLambda_    = 0.1; 
            precisionLambda_  = 1e-9; // used to compare GEO lambda in parametric maxflow
            lambdaMultiplier_ = 2;    // used for jumps in backtracking;
            reductionRatio_   = 0.25; // used to decide whether to increase or decrease lambda using the multiplier
            distance_         = EUCLIDEAN; 
         }
      };

      LSA_TR(const GraphicalModelType&);
      LSA_TR(const GraphicalModelType&, const Parameter&);
      ~LSA_TR();
      std::string name() const;
      const GraphicalModelType& graphicalModel() const;
      InferenceTermination infer();
      void reset();
      template<class VisitorType>
      InferenceTermination infer(VisitorType&);
      void setStartingPoint(typename std::vector<LabelType>::const_iterator);
      virtual InferenceTermination arg(std::vector<LabelType>&, const size_t = 1) const ;
      virtual ValueType value()const{ return gm_.evaluate(curState_);}

   private:
      void init();
      double findMinimalChangeBrakPoint(const double lambda, const std::vector<LabelType>& workingPoint);

      LSA_TR_HELPER<LabelType>           helper_;
      const GraphicalModelType&          gm_;
      Parameter                          param_;
      std::vector<LabelType>             curState_; 
      size_t                             numVar_;
  
      ValueType                          constTerm_;
      std::vector<ValueType>             unaries_;
      std::vector<LSA_TR_WeightedEdge>   subEdges_;
      std::vector<LSA_TR_WeightedEdge>   supEdges_;
      std::vector<ValueType>             approxUnaries_;
     
   };

//////////////
   template<class LabelType>
   template<class GM>
   void LSA_TR_HELPER<LabelType>::init(const GM& gm, const std::vector<LabelType>& workingPoint){
      typedef size_t IndexType;
      solved_       = false;
      numVar_       = gm.numberOfVariables();
      workingPoint_ = workingPoint;
      lambda_       = 0.2;
      constTerm_    = 0; 
      unaries_.resize(numVar_,0); 
      distance_.resize(numVar_,0);
 
      const LabelType label00[] = {0,0};
      const LabelType label01[] = {0,1};
      const LabelType label10[] = {1,0};
      const LabelType label11[] = {1,1};
      for(IndexType f=0; f<gm.numberOfFactors();++f){
         OPENGM_ASSERT(gm[f].numberOfVariables() <= 2);
         if(gm[f].numberOfVariables() == 0){
            constTerm_ += gm[f](label00);
         }
         else  if(gm[f].numberOfVariables() == 1){
            const double v0   = gm[f](label00);
            const double v1   = gm[f](label11);
            const IndexType var0 = gm[f].variableIndex(0);
            constTerm_ += v0;
            unaries_[var0] += v1-v0;
         }
         else  if(gm[f].numberOfVariables() == 2){ 
            const double v00   = gm[f](label00); 
            const double v01   = gm[f](label01);
            const double v10   = gm[f](label10);
            const double v11   = gm[f](label11);
            const IndexType var0 = gm[f].variableIndex(0); 
            const IndexType var1 = gm[f].variableIndex(1);
            constTerm_ += v00;
            const double D = 0.5*(v11-v00);
            const double M = 0.5*(v10-v01);
            unaries_[var0] += D+M;
            unaries_[var1] += D-M;
            const double V = v10-v00-D-M;
            if(V>0){//submodular
               subEdges_.push_back( LSA_TR_WeightedEdge(V,var0,var1));
            }
            else if(V<0){//supermodular
               unaries_[var0] += V;
               unaries_[var1] += V;
               supEdges_.push_back( LSA_TR_WeightedEdge(-2*V,var0,var1));
            }
         }
      }
      std::cout <<  std::endl;
      std::cout <<  subEdges_.size() <<" submodular edges."<<std::endl;
      std::cout <<  supEdges_.size() <<" supermodular edges."<<std::endl;
           
      graph_       = new graph_type(gm.numberOfVariables(),subEdges_.size()+1);
      changedList_ = new block_type(gm.numberOfVariables());
     
      graph_->add_node(numVar_); 
      for(size_t i=0; i<subEdges_.size(); ++i){
         graph_->add_edge( subEdges_[i].u, subEdges_[i].v, subEdges_[i].w, subEdges_[i].w);
      }
      approxUnaries_.assign(unaries_.begin(),unaries_.end());
      for(size_t i=0; i<supEdges_.size(); ++i){
         const size_t var0 = supEdges_[i].u;
         const size_t var1 = supEdges_[i].v;
         const double w    = supEdges_[i].w;
         if(workingPoint[var0]==1)
            approxUnaries_[var1] += w;
         if(workingPoint[var1]==1)
            approxUnaries_[var0] += w; 
         if(workingPoint[var0]==1 && workingPoint[var1]==1)
            constTermApproximation_ -= w;
      } 

 
      shape_.resize(1,numVar_);
      std::vector<size_t> neigbor_count(numVar_,0);
      for(size_t i=0; i<supEdges_.size(); ++i){ 
         ++neigbor_count[supEdges_[i].u];
         ++neigbor_count[supEdges_[i].v];
      } 
      for(size_t i=0; i<subEdges_.size(); ++i){ 
         ++neigbor_count[subEdges_[i].u];
         ++neigbor_count[subEdges_[i].v];
      }
      size_t min_deg = *std::min_element(neigbor_count.begin(),neigbor_count.end());
      std::vector<size_t> corners;
      for(size_t i=0; i<neigbor_count.size(); ++i)
         if (neigbor_count[i] == min_deg)
            corners.push_back(i);
      if(corners.size()==4){
         if( !(corners[1]-corners[0] != corners[3]-corners[2])&&
             !(corners[0] != 0 || corners[3] != numVar_-1)  ){
            shape_.resize(2);
            shape_[0] = corners[1]-corners[0]+1;
            shape_[1] = numVar_ / shape_[0]; 
         }
      } 
      if(shape_.size() ==1 && distanceType_ == EUCLIDEAN)
         std::cout << "Warning : Shape of labeling is 1 and Euclidean distance does not make sense! Maybe autodetection of shape fails ..." <<std::endl;
          
    

      updateDistance();
      constTermTrustRegion_ = 0;
      for(int i=0; i<approxUnaries_.size(); ++i){
         approxUnaries_[i] += lambda_*distance_[i]; 
         graph_->add_tweights( i, 0,  approxUnaries_[i]); 
         if(distance_[i]<0)
            constTermTrustRegion_-=lambda_*distance_[i];
      }
   };

   template<class LabelType>
   void LSA_TR_HELPER<LabelType>::updateDistance() {
      if (distanceType_==HAMMING){
         for(int i=0; i<numVar_; ++i){
            if(workingPoint_[i]==0){ 
               distance_[i] = 1;      
            }
            else{
               distance_[i] = -1; 
            } 
         }
      }
#ifdef NOVIGRA 
      else if(distanceType_==EUCLIDEAN){
         std::cout << "Warning : The useage of euclidean distance requires VIGRA!" <<std::endl;
         std::cout << " Vigra is disabled -> Switch to Hamming distance!" <<std::endl;
         distanceType_=HAMMING;
         for(int i=0; i<numVar_; ++i){
            if(workingPoint_[i]==0){ 
               distance_[i] = 1;      
            }
            else{
               distance_[i] = -1; 
            } 
         }
      }
#else
      else if(distanceType_==EUCLIDEAN){
         std::vector<size_t> s = shape_;
         std::vector<double> dist0(numVar_,0); 
         std::vector<double> dist1(numVar_,0);
         if(s.size()==1){
            typedef vigra::MultiArrayView<1, LabelType> ArrayType; 
            typedef vigra::MultiArrayView<1, double>    DArrayType;
            typedef typename ArrayType::difference_type ShapeType;
            ShapeType shape(s[0]);
            ShapeType stride(1);
            
            
            ArrayType source( shape, stride, &workingPoint_[0] );
            DArrayType dest0( shape, stride, &dist0[0] );
            DArrayType dest1( shape, stride, &dist1[0] );
            
            vigra::separableMultiDistance(source, dest0, false);
            vigra::separableMultiDistance(source, dest1, true);
            for(int i=0; i<numVar_; ++i){
               if(workingPoint_[i]==0){ 
                  distance_[i] = (dist1[i]-0.5); 
               }
               else{
                  distance_[i] = -(dist0[i]-0.5);  
               } 
            }
         }
         else if(s.size()==2){
            typedef vigra::MultiArrayView<2, LabelType> ArrayType; 
            typedef vigra::MultiArrayView<2, double>    DArrayType;
            typedef typename ArrayType::difference_type ShapeType;
            ShapeType shape(s[0],s[1]);
            ShapeType stride(1,s[0]);
            
            
            ArrayType source( shape, stride, &workingPoint_[0] );
            DArrayType dest0( shape, stride, &dist0[0] );
            DArrayType dest1( shape, stride, &dist1[0] );
            
            vigra::separableMultiDistance(source, dest0, false);
            vigra::separableMultiDistance(source, dest1, true);
            for(int i=0; i<numVar_; ++i){
               if(workingPoint_[i]==0){ 
                  distance_[i] = (dist1[i]-0.5); 
               }
               else{
                  distance_[i] = -(dist0[i]-0.5); 
               } 
            }
         } 
         else if(s.size()==3){
            typedef vigra::MultiArrayView<3, LabelType> ArrayType; 
            typedef vigra::MultiArrayView<3, double>    DArrayType;
            typedef typename ArrayType::difference_type ShapeType;
            ShapeType shape(s[0],s[1],s[2]);
            ShapeType stride(1,s[0],s[0]*s[1]);
            
            
            ArrayType source( shape, stride, &workingPoint_[0] );
            DArrayType dest0( shape, stride, &dist0[0] );
            DArrayType dest1( shape, stride, &dist1[0] );
            
            vigra::separableMultiDistance(source, dest0, false);
            vigra::separableMultiDistance(source, dest1, true);
            for(int i=0; i<numVar_; ++i){
               if(workingPoint_[i]==0){ 
                  distance_[i] = (dist1[i]-0.5); 
               }
               else{
                  distance_[i] = -(dist0[i]-0.5); 
               } 
            }
         }
      }//end EUCLIDEAN
#endif
      else{
         std::cout <<"Unknown distance"<<std::endl;
      }
      return;
   } 

   template<class LabelType>
   double LSA_TR_HELPER<LabelType>::optimize(std::vector<LabelType>& label){
      double value;
      //std::cout << lambda_ <<std::endl;
      if(solved_){ //use warmstart
         value = graph_->maxflow(true,changedList_); 
         for (typename graph_type::node_id* ptr = changedList_->ScanFirst(); ptr; ptr = changedList_->ScanNext()) {
            typename graph_type::node_id var = *ptr; 
            OPENGM_ASSERT(var>=0 && var<numVar_);
            graph_->remove_from_changed_list(var);
         }
         
         for(size_t var=0; var<numVar_; ++var) {
            if (graph_->what_segment(var) == graph_type::SOURCE) { label[var]=1;}
            else                                                 { label[var]=0;}
         } 
         changedList_->Reset();
      }
      else{ //first round without warmstart
         value = graph_->maxflow();
         for(size_t var=0; var<numVar_; ++var) {
            if (graph_->what_segment(var) == graph_type::SOURCE) { label[var]=1;}
            else                                                 { label[var]=0;}
         }   
         solved_=true;
      } 
      return value + constTerm_ + constTermApproximation_ + constTermTrustRegion_; 
   }   

  template<class LabelType>
  void  LSA_TR_HELPER<LabelType>::set(const double newLambda){
     if( newLambda == lambda_ ) return;
     double difLambda  = newLambda - lambda_;
     lambda_ = newLambda;
     constTermTrustRegion_ = 0;
     if(solved_){
        for(int i=0; i<approxUnaries_.size(); ++i){
           double oldcap = graph_->get_trcap(i);
           approxUnaries_[i] += difLambda*distance_[i]; 
           graph_->add_tweights( i, 0, difLambda*distance_[i] ); 
           if(distance_[i]<0)
              constTermTrustRegion_ -= difLambda*distance_[i];
           double newcap = graph_->get_trcap(i);
           if (!((newcap > 0 && oldcap > 0)||(newcap < 0 && oldcap < 0))){
              graph_->mark_node(i);
           }
        }
     }else{
        for(int i=0; i<approxUnaries_.size(); ++i){ 
           approxUnaries_[i] += difLambda*distance_[i]; 
           graph_->add_tweights( i, 0, difLambda*distance_[i] );
           if(distance_[i]<0)
              constTermTrustRegion_ -= difLambda*distance_[i];  
        }
     }
  } 

   template<class LabelType>
   void  LSA_TR_HELPER<LabelType>::set(const std::vector<LabelType>& newWorkingPoint, const double newLambda){
      workingPoint_           = newWorkingPoint;
      lambda_                 = newLambda; 
      constTermTrustRegion_   = 0;
      constTermApproximation_ = 0;
 
      updateDistance();
 
      std::vector<double> newApproxUnaries = unaries_;
      for(size_t i=0; i<supEdges_.size(); ++i){
         const size_t var0 = supEdges_[i].u;
         const size_t var1 = supEdges_[i].v;
         const double w    = supEdges_[i].w;
         if(workingPoint_[var0]==1)
            newApproxUnaries[var1] += w;
         if(workingPoint_[var1]==1)
            newApproxUnaries[var0] += w; 
         if(workingPoint_[var0]==1 && workingPoint_[var1]==1)
            constTermApproximation_ -= w;
      } 
      if(solved_){
         for(int i=0; i<numVar_; ++i){
            double oldcap = graph_->get_trcap(i);
            newApproxUnaries[i] += lambda_*distance_[i]; 
            graph_->add_tweights( i, 0, newApproxUnaries[i]-approxUnaries_[i] ); 
            if(distance_[i]<0)
               constTermTrustRegion_ -= lambda_*distance_[i];
           double newcap = graph_->get_trcap(i);
           if (!((newcap > 0 && oldcap > 0)||(newcap < 0 && oldcap < 0))){
              graph_->mark_node(i);
           }
        }
     }else{
         for(int i=0; i<numVar_; ++i){ 
            newApproxUnaries[i] += lambda_*distance_[i]; 
            graph_->add_tweights( i, 0, newApproxUnaries[i]-approxUnaries_[i]);
            if(distance_[i]<0)
               constTermTrustRegion_ -= lambda_*distance_[i]; 
         }
      }
      approxUnaries_.assign(newApproxUnaries.begin(),newApproxUnaries.end());

   }


   template<class LabelType>
   double LSA_TR_HELPER<LabelType>::eval(const std::vector<LabelType>& label) const
   {
      typedef double ValueType;
      ValueType v = constTerm_;
      for(size_t var=0; var<numVar_;++var)
         if(label[var]==1)
            v += unaries_[var];
      for(size_t i=0; i<subEdges_.size(); ++i)
         if(label[subEdges_[i].u] != label[subEdges_[i].v])
            v += subEdges_[i].w;
      for(size_t i=0; i<supEdges_.size(); ++i)
         if(label[supEdges_[i].u] == 1 && label[supEdges_[i].v] == 1)
            v += supEdges_[i].w;
      return v;
   }
 
   template<class LabelType>
   double LSA_TR_HELPER<LabelType>::evalAprox(const std::vector<LabelType>& label, const std::vector<LabelType>& workingPoint, const double lambda) const
   { 
      typedef double ValueType;
      ValueType v = constTerm_;
      for(size_t var=0; var<numVar_;++var)
         if(label[var]==1)
            v += unaries_[var];
      for(size_t i=0; i<subEdges_.size(); ++i)
         if(label[subEdges_[i].u] != label[subEdges_[i].v])
            v += subEdges_[i].w;
      for(size_t i=0; i<supEdges_.size(); ++i){
         if(label[supEdges_[i].u]        == 1 && workingPoint[supEdges_[i].v] == 1  )
            v += supEdges_[i].w;  
         if(workingPoint[supEdges_[i].u] == 1 && label[supEdges_[i].v]        == 1  )
            v += supEdges_[i].w; 
         if(workingPoint[supEdges_[i].u] == 1 && workingPoint[supEdges_[i].v] == 1  )
            v -= supEdges_[i].w;
      } 
      for(size_t i=0; i<numVar_; ++i){
         if(label[i] != workingPoint[i])
            v += lambda * std::fabs(distance_[i]);
      }
      return v;
   }
  
   template<class LabelType>
   void LSA_TR_HELPER<LabelType>::evalBoth(const std::vector<LabelType>& label, const std::vector<LabelType>& workingPoint, const double lambda, double& value, double& valueAprox) const
   {
      value = constTerm_;
      for(size_t var=0; var<numVar_;++var)
         if(label[var]==1)
            value += unaries_[var];
      for(size_t i=0; i<subEdges_.size(); ++i)
         if(label[subEdges_[i].u] != label[subEdges_[i].v])
            value += subEdges_[i].w;
      valueAprox = value;
      for(size_t i=0; i<supEdges_.size(); ++i){
         if(label[supEdges_[i].u]         == 1 && label[supEdges_[i].v]         == 1  )
            value += supEdges_[i].w;
         if(label[supEdges_[i].u]         == 1 && workingPoint[supEdges_[i].v]  == 1  )
            valueAprox += supEdges_[i].w;  
         if(workingPoint[supEdges_[i].u]  == 1 && label[supEdges_[i].v]         == 1  )
            valueAprox += supEdges_[i].w; 
         if(workingPoint[supEdges_[i].u]  == 1 && workingPoint[supEdges_[i].v]  == 1  )
            valueAprox -= supEdges_[i].w;
      }
      for(size_t i=0; i<numVar_; ++i){
         if(label[i] != workingPoint[i])
            valueAprox += lambda * std::fabs(distance_[i]);
      }
   }



/////////////

   template<class GM, class ACC>
   LSA_TR<GM, ACC>::~LSA_TR(){}
   
   template<class GM, class ACC>
   inline
   LSA_TR<GM, ACC>::LSA_TR
   (
      const GraphicalModelType& gm
      )
   :  gm_(gm),
      param_(Parameter())
   {
      init();
   }

   template<class GM, class ACC>
   LSA_TR<GM, ACC>::LSA_TR
   (
      const GraphicalModelType& gm,
      const Parameter& parameter
      )
   :  gm_(gm),
      param_(parameter)
   {
      init();
   }
   
   template<class GM, class ACC>
   void LSA_TR<GM, ACC>::init()
   {
      srand(param_.randSeed_);
      numVar_ = gm_.numberOfVariables();
      curState_.resize(numVar_,1);
      for (size_t i=0; i<numVar_; ++i) curState_[i]= rand()%2;
      helper_.init(gm_, curState_);
      if(param_.distance_ == Parameter::HAMMING)
         helper_.setDistanceType(LSA_TR_HELPER<LabelType>::HAMMING); 
      else if(param_.distance_ == Parameter::EUCLIDEAN)
         helper_.setDistanceType(LSA_TR_HELPER<LabelType>::EUCLIDEAN);
      else
         std::cout << "Warning:  Unknown distance type !"<<std::endl;
   }

      
   template<class GM, class ACC>
   inline void
   LSA_TR<GM, ACC>::reset()
   {
      curState_.resize(numVar_,1);  
   }
   
   template<class GM, class ACC>
   inline void 
   LSA_TR<GM,ACC>::setStartingPoint(typename std::vector<typename LSA_TR<GM,ACC>::LabelType>::const_iterator begin) { 
      curState_.assign(begin, begin+numVar_);
   }
   
   template<class GM, class ACC>
   inline std::string
   LSA_TR<GM, ACC>::name() const
   {
      return "LSA_TR";
   }
   
   template<class GM, class ACC>
   inline const typename LSA_TR<GM, ACC>::GraphicalModelType&
   LSA_TR<GM, ACC>::graphicalModel() const
   {
      return gm_;
   }
   
   template<class GM, class ACC>
   inline InferenceTermination
   LSA_TR<GM,ACC>::infer()
   {
      EmptyVisitorType v;
      return infer(v);
   }
   
  
   template<class GM, class ACC>
   template<class VisitorType>
   InferenceTermination LSA_TR<GM,ACC>::infer
   (
      VisitorType& visitor
      )
   {
      const ValueType tau1 = 0;
      const ValueType tau2 = param_.reductionRatio_;
      bool exitInf=false;
      std::vector<LabelType> label(numVar_);
      std::vector<ValueType> energies;
      std::vector<ValueType> energiesAprox;
      double lambda = param_.initialLambda_;
      helper_.set(curState_,lambda);
      visitor.begin(*this); 

      ValueType curr_value_aprox = helper_.evalAprox(curState_, curState_, lambda); 
      ValueType curr_value       = helper_.eval(curState_); 
      bool changedWorkingpoint   = false; 
      bool changedLambda         = false;
      ValueType value_after;
      ValueType value_after_aprox; 

      OPENGM_ASSERT(std::fabs(curr_value-gm_.evaluate(curState_))<0.0001);
      for (size_t i=0; i<10000 ; ++i){
         //std::cout << "round "<<i<<" (lambda = "<<lambda<<"): " <<std::endl;
         if(lambda>param_.maxLambda_) break;
       
         if(changedWorkingpoint)
            helper_.set(curState_,lambda);
         else if(changedLambda)
            helper_.set(lambda);
         changedWorkingpoint = false;
         changedLambda       = false;
         helper_.optimize(label);
         helper_.evalBoth(label, curState_, lambda, value_after,  value_after_aprox);
        
         //if(std::fabs(curr_value_aprox-curr_value)>0.0001)
         //   std::cout << "WARNING : "<<  helper_.evalAprox(curState_, curState_, lambda) << " != " << helper_.eval(curState_) << " == " <<gm_.evaluate(curState_)<<std::endl;         
         OPENGM_ASSERT(std::fabs(helper_.eval(curState_)-gm_.evaluate(curState_))<0.0001); 
         OPENGM_ASSERT(helper_.eval(curState_) == curr_value);
         OPENGM_ASSERT(std::fabs(helper_.eval(label)-gm_.evaluate(label))<0.0001);
   
         const ValueType P = curr_value_aprox - value_after_aprox;
         const ValueType R = curr_value       - value_after;

     

         //std::cout  <<P  << "  " <<curr_value_aprox <<  "  " << value_after_aprox <<std::endl;
         if(P==0){
            // ** Search for smallest possible step (largest penalty that give progress)
            //std::cout << "Approximation does not improve energy ... searching for better lambda ... "<< std::flush;     
            lambda = findMinimalChangeBrakPoint(lambda,  curState_);
            helper_.set(lambda);
            helper_.optimize(label);
            //std::cout<<"set lambda to "<<  lambda <<std::endl;

            helper_.evalBoth(label, curState_, lambda, value_after,  value_after_aprox);
            const ValueType P = curr_value_aprox - value_after_aprox;
            const ValueType R = curr_value       - value_after;   
            if(R<=0){
               visitor(*this);
               break;
            }else if(R>0){
               // ** Update Working Point
               //std::cout<<"Update Working Point"<<std::endl;
               curState_.assign(label.begin(),label.end()); 
               changedWorkingpoint = true;
               curr_value       =  value_after;
               curr_value_aprox =  value_after;
               //OPENGM_ASSERT(std::fabs( curr_value_aprox-helper_.evalAprox(curState_, curState_, lambda) )<0.0001); 
            }
         }
         else{
            if(P<0) std::cout << "WARNING : "<< curr_value_aprox << " < " << value_after_aprox << std::endl;         
            if(R>tau1){
               // ** Update Working Point
               //std::cout<<"Update Working Point"<<std::endl;
               curState_.assign(label.begin(),label.end());
               changedWorkingpoint = true;
               //helper_.set(curState_,lambda);
               curr_value       =  value_after;
               curr_value_aprox =  value_after;
               //OPENGM_ASSERT(std::fabs( curr_value_aprox-helper_.evalAprox(curState_, curState_, lambda) )<0.0001); 
            } 
         }
         
         // ** Update trust region term
         if(R/P>tau2){ ;
            lambda = lambda / param_.lambdaMultiplier_; 
            changedLambda       = true;
            //helper_.set(lambda);
            //std::cout<<"Decrease TR to "<<  lambda <<std::endl;
         }
         else{
            lambda = lambda * param_.lambdaMultiplier_; 
            changedLambda       = true;
            //helper_.set(lambda);
            //std::cout<<"Increase TR to "<<  lambda<<std::endl;
         }   
         
         // ** Store values
         energies.push_back     (curr_value);
         energiesAprox.push_back(value_after_aprox);
         
         // ** Call Visitor
         if( visitor(*this) != visitors::VisitorReturnFlag::ContinueInf ) break;
      }
      
      visitor.end(*this);
      return NORMAL;
   }
   
   template<class GM, class ACC>
   inline InferenceTermination
   LSA_TR<GM,ACC>::arg
   (
      std::vector<LabelType>& x,
      const size_t N
      ) const
   {
      if(N==1) {
         //x.resize(gm_.numberOfVariables());
         //for (size_t i=0; i<gm_.numberOfVariables(); ++i)
         //   x[i] = curState_[i];
         x.assign(curState_.begin(), curState_.end());
         return NORMAL;
      }
      else {
         return UNKNOWN;
      }
   }
   

   template<class GM, class ACC>
   double LSA_TR<GM,ACC>::findMinimalChangeBrakPoint(const double lambda, const std::vector<LabelType>& workingPoint){
      
      ValueType topLambda    = lambda;
      ValueType bottomLambda = param_.precisionLambda_;
      std::vector<LabelType> topLabel(numVar_);
      std::vector<LabelType> bottomLabel(numVar_); 
      std::vector<LabelType> label(numVar_);
      // upper bound for best lambda
      while(true){
         helper_.set(topLambda);
         helper_.optimize(topLabel); 
         if(!std::equal(topLabel.begin(),topLabel.end(),workingPoint.begin()))
            topLambda = topLambda * 2;
         else
            break;
      }
      
      // lower bound for lambda 
      helper_.set(bottomLambda);
      helper_.optimize(bottomLabel); 
   
      // binary search for minimal change point
      while(true){
         double middleLambda = (topLambda + bottomLambda)/2.0;
         //std::cout <<"test "<< bottomLambda<<" < "<<middleLambda<<" < "<<topLambda<<std::endl;
         helper_.set(middleLambda);
         helper_.optimize(label); 
         
         if(!std::equal(label.begin(),label.end(),topLabel.begin())){
            bottomLambda   = middleLambda;
            bottomLabel    = label;
         }    
         else if(!std::equal(label.begin(),label.end(),bottomLabel.begin())){
            topLambda = middleLambda;
            topLabel  = label;
         }
         else{       
            return  bottomLambda;     
         }
         if((topLambda-bottomLambda) < param_.precisionLambda_){
            return bottomLambda;
         }
      }
   }

} // namespace opengm

#endif // #ifndef OPENGM_LSATR_HXX
