#pragma once
#ifndef OPENGM_FUSION_BASED_INF_HXX
#define OPENGM_FUSION_BASED_INF_HXX

#include <vector>
#include <string>
#include <iostream>

#include "opengm/opengm.hxx"
#include "opengm/inference/inference.hxx"
#include "opengm/inference/visitors/visitors.hxx"
#include "opengm/utilities/random.hxx"

// Fusion Move Solver (they solve binary problems)
#include "opengm/inference/astar.hxx"
#include "opengm/inference/lazyflipper.hxx"
#include "opengm/inference/infandflip.hxx"
#include "opengm/inference/messagepassing/messagepassing.hxx"
#include "opengm/inference/hqpbo.hxx"
#ifdef WITH_CPLEX
#include "opengm/inference/lpcplex.hxx"
#endif
#ifdef WITH_QPBO
#include "QPBO.h"
#include "opengm/inference/reducedinference.hxx"
#endif
#ifdef WITH_AD3
#include "opengm/inference/external/ad3.hxx"
#endif




#include "opengm/inference/lazyflipper.hxx"

// fusion move model generator
#include "opengm/inference/auxiliary/fusion_move/fusion_mover.hxx"

namespace opengm
{


namespace proposal_gen
{

template<class GM, class ACC>
class AlphaExpansionGen
{
public:
    typedef ACC AccumulationType;
    typedef GM GraphicalModelType;
    OPENGM_GM_TYPE_TYPEDEFS;
    struct Parameter
    {
        Parameter(){}
    };
    AlphaExpansionGen(const GM &gm, const Parameter &param)
        :  gm_(gm),
           param_(param),
           currentAlpha_(0)
    {
       maxLabel_ =0;
       for(size_t i=0; i<gm.numberOfVariables();++i){
          if(gm.numberOfLabels(i)>maxLabel_){
             maxLabel_ = gm.numberOfLabels(i);
          }
       }
    }
    void reset()
    {
        currentAlpha_ = 0;
    }
    
    size_t defaultNumStopIt() {return maxLabel_;}
   
    void getProposal(const std::vector<LabelType> &current , std::vector<LabelType> &proposal)
    {
        for (IndexType vi = 0; vi < gm_.numberOfVariables(); ++vi)
        {
            if (gm_.numberOfLabels(vi) > currentAlpha_ )
            {
                proposal[vi] = currentAlpha_;
            }
            else
            {
                proposal[vi] = current[vi];
            }
        }
        ++currentAlpha_;
        if(currentAlpha_>=maxLabel_){
           currentAlpha_ = 0;
        }
    } 
   LabelType currentAlpha(){return currentAlpha_;}
private:
    const GM &gm_;
    Parameter param_;
    LabelType maxLabel_;
    LabelType currentAlpha_;
};


template<class GM, class ACC>
class AlphaBetaSwapGen
{
public:
    typedef ACC AccumulationType;
    typedef GM GraphicalModelType;
    OPENGM_GM_TYPE_TYPEDEFS;
    struct Parameter
    {
        Parameter(){}
    };
private:   
    static size_t getMaxLabel(const GM &gm){
      size_t maxLabel = 0;
      for(size_t i=0; i<gm.numberOfVariables();++i){
         if(gm.numberOfLabels(i)>maxLabel ){
            maxLabel = gm.numberOfLabels(i);
         }
      } 
      return maxLabel;
    }
public: 
    AlphaBetaSwapGen(const GM &gm, const Parameter &param)
        :  gm_(gm),
           param_(param),
           maxLabel_(getMaxLabel(gm)),
           abShape_(2, maxLabel_),
           abWalker_(abShape_.begin(), 2)
    {
       // ++abWalker_;
    }
    void reset()
    {
        abWalker_.reset();
    }  

   size_t defaultNumStopIt() {return (maxLabel_*maxLabel_-maxLabel_)/2;}

    void getProposal(const std::vector<LabelType> &current , std::vector<LabelType> &proposal)
    {
       if( maxLabel_<2){
          for (IndexType vi = 0; vi < gm_.numberOfVariables(); ++vi)
             proposal[vi] = current[vi];
       }else{
          ++abWalker_;
          if(currentAlpha()+1 ==  maxLabel_ && currentBeta()+1== maxLabel_){
             reset();
          }
          while (abWalker_.coordinateTuple()[0] == abWalker_.coordinateTuple()[1])
          {
             ++abWalker_;
          }
          
          const LabelType alpha = abWalker_.coordinateTuple()[0];
          const LabelType beta  = abWalker_.coordinateTuple()[1];
          
          for (IndexType vi = 0; vi < gm_.numberOfVariables(); ++vi)
          {
             if ( current[vi] == alpha && gm_.numberOfLabels(vi) > beta )
             {
                proposal[vi] = beta;
             }
             else if ( current[vi] == beta && gm_.numberOfLabels(vi) > alpha )
             {
                proposal[vi] = alpha;
             }
             else
             {
                proposal[vi] = current[vi];
             }
          }
       }
    }

    LabelType currentAlpha()
    {
        return abWalker_.coordinateTuple()[0];
    }
    LabelType currentBeta()
    {
        return abWalker_.coordinateTuple()[1];
    }
private:

    const GM &gm_;
    Parameter param_; 
    LabelType maxLabel_;
    std::vector<LabelType> abShape_;
    ShapeWalker<typename std::vector<LabelType>::const_iterator>  abWalker_;
    
};

template<class GM, class ACC>
class RandomGen
{
public:
    typedef ACC AccumulationType;
    typedef GM GraphicalModelType;
    OPENGM_GM_TYPE_TYPEDEFS;
    struct Parameter
    {
        Parameter(){}
    };
    RandomGen(const GM &gm, const Parameter &param)
        :  gm_(gm),
           param_(param),
           currentStep_(0)
    {
    }
    void reset()
    {
        currentStep_ = 0;
    } 
    size_t defaultNumStopIt() {return 10;}
    void getProposal(const std::vector<LabelType> &current , std::vector<LabelType> &proposal)
    {
        for (IndexType vi = 0; vi < gm_.numberOfVariables(); ++vi){
            // draw label
           opengm::RandomUniform<size_t> randomLabel(0, gm_.numberOfLabels(vi),currentStep_+vi);
            proposal[vi] = randomLabel();
        }
        ++currentStep_;
    }
private:
    const GM &gm_;
    Parameter param_;
    LabelType currentStep_;
};



template<class GM, class ACC>
class RandomLFGen
{
public:
    typedef ACC AccumulationType;
    typedef GM GraphicalModelType;
    OPENGM_GM_TYPE_TYPEDEFS;
    struct Parameter
    {
        Parameter(){}
    };
    RandomLFGen(const GM &gm, const Parameter &param)
        :  gm_(gm),
           param_(param),
           currentStep_(0)
    {
    }
    void reset()
    {
        currentStep_ = 0;
    }
    size_t defaultNumStopIt() {return 10;}
    void getProposal(const std::vector<LabelType> &current , std::vector<LabelType> &proposal)
    {
        for (IndexType vi = 0; vi < gm_.numberOfVariables(); ++vi){
            // draw label
            opengm::RandomUniform<size_t> randomLabel(0, gm_.numberOfLabels(vi),currentStep_+vi);
            proposal[vi] = randomLabel();
        }
        typename opengm::LazyFlipper<GM,ACC>::Parameter para(1,proposal.begin(),proposal.end());
        opengm::LazyFlipper<GM,ACC> lf(gm_,para);
        lf.infer();
        lf.arg(proposal);
        ++currentStep_;
    }
private:
    const GM &gm_;
    Parameter param_;
    LabelType currentStep_;
};


template<class GM, class ACC>
class NonUniformRandomGen
{
    public:
    typedef ACC AccumulationType;
    typedef GM GraphicalModelType;
    OPENGM_GM_TYPE_TYPEDEFS;
    struct Parameter
    {
        Parameter(const float temp=1.0)
        :   temp_(temp){
        }
        float temp_;
    };

    NonUniformRandomGen(const GM &gm, const Parameter &param)
    :  gm_(gm),
    param_(param),
    currentStep_(0),
    randomGens_(gm.numberOfVariables())
    {
        std::vector<bool> hasUnary(gm.numberOfVariables(),false);

        for(IndexType fi=0; fi<gm_.numberOfFactors(); ++fi){

            if(gm_[fi].numberOfVariables()==1){

                const IndexType vi = gm_[fi].variableIndex(0);
                const LabelType numLabels = gm_.numberOfLabels(vi);
                std::vector<ValueType> weights(numLabels);
                gm_[fi].copyValues(&weights[0]);
                const ValueType minValue = *std::min_element(weights.begin(),weights.end());
                for(LabelType l=0; l<numLabels; ++l){
                   weights[l]-= minValue;
                }
                for(LabelType l=0; l<numLabels; ++l){
                   //OPENGM_CHECK_OP(weights[l],>=,0.0, "NonUniformRandomGen allows only positive unaries");
                    weights[l]=std::exp(-1.0*param_.temp_*weights[l]);
                }
                randomGens_[vi]=GenType(weights.begin(),weights.end());
                hasUnary[vi]=true;
            }
        }
        for(IndexType vi=0 ;vi<gm_.numberOfVariables(); ++vi){
            if(!hasUnary[vi]){
                const LabelType numLabels = gm_.numberOfLabels(vi);
                std::vector<ValueType> weights(numLabels,1.0);
                randomGens_[vi]=GenType(weights.begin(),weights.end());
            }
        }

    }

    void reset()
    {
        currentStep_ = 0;
    } 

    size_t defaultNumStopIt() {
        return 10;
    }
    void getProposal(const std::vector<LabelType> &current , std::vector<LabelType> &proposal)
    {
        for (IndexType vi = 0; vi < gm_.numberOfVariables(); ++vi){
            proposal[vi]=randomGens_[vi]();
        }
        ++currentStep_;
    }
    private:
    const GM &gm_;
    Parameter param_;
    LabelType currentStep_;

    typedef RandomDiscreteWeighted<LabelType,ValueType> GenType;

    std::vector < RandomDiscreteWeighted<LabelType,ValueType> > randomGens_;
};



/*
template<class GM, class ACC>
class BlurGen
{
public:
    typedef ACC AccumulationType;
    typedef GM GraphicalModelType;
    OPENGM_GM_TYPE_TYPEDEFS;
    struct Parameter
    {
       Parameter(double sigma = 20.0) : sigma_(sigma)
          {
          }
       const double sigma_;
    };
    BlurGen(const GM &gm, const Parameter &param)
        :  gm_(gm),
           param_(param),
           currentStep_(0)
    {
       const double pi = 3.1416;
       const double oneOverSqrt2PiSigmaSquared = 1.0 / (std::sqrt(2.0 * pi) * param_.sigma_);
       const double oneOverTwoSigmaSquared = 1.0 / (2.0* param_.sigma_ * param_.sigma_);
       const size_t radius = std::ceil(3*param_.sigma_);
       kernel_.resize(2*radius + 1);
       double sum = 0;
       for(double i = 0; i <= radius ; ++i) {
          double value = oneOverSqrt2PiSigmaSquared * std::exp(-(i*i)*oneOverTwoSigmaSquared);
          kernel_[radius+i] = value;
          kernel_[radius-i] = value;
          sum += 2*value;
       } 
       for(double i = 0; i <= radius ; ++i) {
          kernel_[radius+i] /= sum;
          kernel_[radius-i] /= sum;
       }

       size_t N = gm_.numberOfFactors(0);
       for(size_t i=1; i<gm_.numberOfVariables(); ++i){
          if(N==gm_.numberOfFactors(i)){
             height_ = i+1;
             break;
          }
       }

       width_  = gm_.numberOfVariables()/height_;

       OPENGM_ASSERT(height_*width_ == gm_.numberOfVariables());
    }
    void reset(){}
    size_t defaultNumStopIt() {return 10;}
   
    void getProposal(const std::vector<LabelType> &current , std::vector<LabelType> &proposal)
    { 
       if ((currentStep_ % 2) == 0){ 
          for (int i = 0; i < height_; ++i) {
             for (int j = 0; j < width_; ++j) { 
                const size_t var = ind(i,j);
                opengm::RandomUniform<size_t> randomLabel(0, gm_.numberOfLabels(var),currentStep_+i+j);
                proposal[var] = (LabelType)(randomLabel());
             }
          } 
       }else{
          std::vector<double> temp(gm_.numberOfVariables(),0.0);
          opengm::RandomUniform<double> randomLabel(-param_.sigma_*1.5, param_.sigma_*1.5,currentStep_);
          const int radius = (kernel_.size()-1)/2; 
          const int h = height_-1;
          const int w = width_ -1;
          for (int i = 0; i < height_; ++i) {
             for (int j = 0; j < width_; ++j) {
                double val = 0.0;
                for (int k = 0; k < 2*radius+1; ++k) {
                   int i2 = std::min( h,std::max(0,i-radius+k));
                   val += kernel_[k] * current[ind(i2,j)];
                }
                temp[ind(i,j)] = val;
             }
          }
          for (int i = 0; i < height_; ++i) {
             for (int j = 0; j < width_; ++j) {
                double val = 0.0;
                for (int k = 0; k < 2*radius+1; ++k) { 
                   int j2 = std::min(w,std::max(0,i-radius+k));
                   val += kernel_[k] * temp[ind(i, j2)];
                }
                proposal[ind(i,j)] = std::min(gm_.numberOfLabels(ind(i,j)),(LabelType)(std::max(0.0,val+randomLabel())));
             }
          } 
       }
       ++currentStep_;
    }
private:
    size_t ind(int i, int j){ return i+j*height_;}
    const GM &gm_;
    Parameter param_;
    size_t height_;
    size_t width_;
    std::vector<double> kernel_;

    LabelType currentStep_;
};
*/
template<class GM, class ACC>
class BlurGen
{
public:
    typedef ACC AccumulationType;
    typedef GM GraphicalModelType;
    OPENGM_GM_TYPE_TYPEDEFS;
    struct Parameter
    {
       Parameter(double sigma = 20.0) : sigma_(sigma)
          {
          }
       const double sigma_;
    };
    BlurGen(const GM &gm, const Parameter &param)
        :  gm_(gm),
           param_(param),
           currentStep_(0)
    {
       const double pi = 3.1416;
       const double oneOverSqrt2PiSigmaSquared = 1.0 / (std::sqrt(2.0 * pi) * param_.sigma_);
       const double oneOverTwoSigmaSquared = 1.0 / (2.0* param_.sigma_ * param_.sigma_);
       const size_t kradius = std::ceil(3*param_.sigma_);
       kernel_.resize(2*kradius + 1);
       double sum = 0;
       for(double i = 0; i <= kradius ; ++i) {
          double value = oneOverSqrt2PiSigmaSquared * std::exp(-(i*i)*oneOverTwoSigmaSquared);
          kernel_[kradius+i] = value;
          kernel_[kradius-i] = value;
          sum += 2*value;
       } 
       for(double i = 0; i <= kradius ; ++i) {
          kernel_[kradius+i] /= sum;
          kernel_[kradius-i] /= sum;
       }

       size_t N = gm_.numberOfFactors(0);
       for(size_t i=1; i<gm_.numberOfVariables(); ++i){
          if(N==gm_.numberOfFactors(i)){
             height_ = i+1;
             break;
          }
       }

       width_  = gm_.numberOfVariables()/height_;

       OPENGM_ASSERT(height_*width_ == gm_.numberOfVariables());

       //Generate blured label
       bluredLabel_.resize(gm_.numberOfVariables(),0);
       std::vector<double> temp(gm_.numberOfVariables(),0.0);
       std::vector<LabelType> localLabel(gm_.numberOfVariables(),0);
       for (size_t i=0; i<gm_.numberOfVariables(); ++i){
          for(typename GM::ConstFactorIterator it=gm_.factorsOfVariableBegin(i); it!=gm_.factorsOfVariableEnd(i);++it){
             if(gm_[*it].numberOfVariables() == 1){
                ValueType v;
                ACC::neutral(v);
                for(LabelType l=0; l<gm_.numberOfLabels(i); ++l){
                   if(ACC::bop(gm_[*it](&l),v)){
                      v=gm_[*it](&l);
                      localLabel[i]=l;
                   }
                }
                continue;
             }
          } 
       }
       const int radius = (kernel_.size()-1)/2; 
       const int h = height_-1;
       const int w = width_ -1;
       for (int i = 0; i < height_; ++i) {
          for (int j = 0; j < width_; ++j) {
             double val = 0.0;
             for (int k = 0; k < 2*radius+1; ++k) {
                int i2 = std::min( h,std::max(0,i-radius+k));
                val += kernel_[k] * localLabel[ind(i2,j)];
             }
             temp[ind(i,j)] = val;
          }
       }
       for (int i = 0; i < height_; ++i) {
          for (int j = 0; j < width_; ++j) {
             double val = 0.0;
             for (int k = 0; k < 2*radius+1; ++k) { 
                int j2 = std::min(w,std::max(0,i-radius+k));
                val += kernel_[k] * temp[ind(i, j2)];
             }
             bluredLabel_[ind(i,j)] = std::min(double(gm_.numberOfLabels(ind(i,j))),(std::max(0.0,val)));
          }
       } 
    }

    void reset(){}
    size_t defaultNumStopIt() {return 10;}
   
    void getProposal(const std::vector<LabelType> &current , std::vector<LabelType> &proposal)
    { 
       if ((currentStep_ % 2) == 0){ 
          for (int i = 0; i < height_; ++i) {
             for (int j = 0; j < width_; ++j) { 
                const size_t var = ind(i,j);
                opengm::RandomUniform<size_t> randomLabel(0, gm_.numberOfLabels(var),currentStep_+i+j);
                proposal[var] = (LabelType)(randomLabel());
             }
          } 
       }else{
          proposal.resize(gm_.numberOfVariables(),0.0);
          opengm::RandomUniform<double> randomLabel(-param_.sigma_*1.5, param_.sigma_*1.5,currentStep_);
          for(size_t i=0; i<proposal.size();++i){
             proposal[i] = std::min(gm_.numberOfLabels(i), (LabelType)(std::max(0.0,bluredLabel_[i] + randomLabel())));
          }
       }
       ++currentStep_;
    }
private:
    size_t ind(int i, int j){ return i+j*height_;}
    const GM &gm_;
    Parameter param_; 
    size_t height_;
    size_t width_;
    std::vector<double> kernel_;
    std::vector<double> bluredLabel_;
    LabelType currentStep_;


};



template<class GM, class ACC>
class EnergyBlurGen
{
public:
   typedef ACC AccumulationType;
   typedef GM GraphicalModelType;
   OPENGM_GM_TYPE_TYPEDEFS;
   struct Parameter
   {
      Parameter(double sigma = 20.0, bool useLocalMargs = false, double temp=1) : sigma_(sigma),  useLocalMargs_(useLocalMargs),  temp_(temp)
         {
         }
      const double sigma_;
      const bool   useLocalMargs_; 
      const double temp_;
      
   };
   EnergyBlurGen(const GM &gm, const Parameter &param)
      :  gm_(gm),
         param_(param),
         currentStep_(0)
      {
         const double pi = 3.1416;
         const double oneOverSqrt2PiSigmaSquared = 1.0 / (std::sqrt(2.0 * pi) * param_.sigma_);
         const double oneOverTwoSigmaSquared = 1.0 / (2.0* param_.sigma_ * param_.sigma_);
         const size_t kradius = std::ceil(3*param_.sigma_);
         std::vector<double> kernel; 
         kernel.resize(2*kradius + 1);
         double sum = 0;
         for(double i = 0; i <= kradius ; ++i) {
            double value = oneOverSqrt2PiSigmaSquared * std::exp(-(i*i)*oneOverTwoSigmaSquared);
            kernel[kradius+i] = value;
            kernel[kradius-i] = value;
            sum += 2*value;
         } 
         for(double i = 0; i <= kradius ; ++i) {
            kernel[kradius+i] /= sum;
            kernel[kradius-i] /= sum;
         }

         size_t N = gm_.numberOfFactors(0);
         for(size_t i=1; i<gm_.numberOfVariables(); ++i){
            if(N==gm_.numberOfFactors(i)){
               height_ = i+1;
               break;
            }
         }

         width_  = gm_.numberOfVariables()/height_;

         OPENGM_ASSERT(height_*width_ == gm_.numberOfVariables());

         //Generate energy-blured label
         size_t numLabels =gm_.numberOfLabels(0);
         std::vector<double> temp(gm_.numberOfVariables(),0.0);
         std::vector<double> bluredEnergy(gm_.numberOfVariables(),1000000000000.0); 
         std::vector<double> bluredOpt(gm_.numberOfVariables(),0); 
         std::vector<double> energy(gm_.numberOfVariables(),0.0);
         std::vector<IndexType> unaries(gm_.numberOfVariables());
         std::vector<std::vector<double> > margs;;
         if(param_.useLocalMargs_)
            margs.resize(gm_.numberOfVariables(),std::vector<double>(numLabels));
        
         for (size_t i=0; i<gm_.numberOfVariables(); ++i){
            bool found = false;
            for(typename GM::ConstFactorIterator it=gm_.factorsOfVariableBegin(i); it!=gm_.factorsOfVariableEnd(i);++it){
               if(gm_[*it].numberOfVariables() == 1){
                  unaries[i] = *it;
                  found = true;
                  if(gm_[*it].numberOfLabels(0) != numLabels)
                     throw RuntimeError("number of labels are not equal for all variables");             
                  continue;
               }
            } 
            if(!found)
               throw RuntimeError("missing unary");
         } 
        

         for(size_t l=0; l<numLabels; ++l){
            for (int i = 0; i < height_; ++i) {
               for (int j = 0; j < width_; ++j) { 
                  const size_t var = ind(i, j);
                  energy[var]  =gm_[unaries[ind(i, j)]](&l);
               }
            }

            const int radius = (kernel.size()-1)/2; 
            const int h = height_-1;
            const int w = width_ -1;
            for (int i = 0; i < height_; ++i) {
               for (int j = 0; j < width_; ++j) {
                  double val = 0.0;
                  const size_t var = ind(i, j);
                  for (int k = 0; k < 2*radius+1; ++k) {
                     int i2 = std::min( h,std::max(0,i-radius+k));
                     val += kernel[k] * energy[ind(i2,j)];
                  }
                  temp[var] = val;
               }
            }
            for (int i = 0; i < height_; ++i) {
               for (int j = 0; j < width_; ++j) {
                  double val = 0.0;
                  const size_t var = ind(i, j);
                  for (int k = 0; k < 2*radius+1; ++k) { 
                     int j2 = std::min(w,std::max(0,i-radius+k));
                     val += kernel[k] * temp[ind(i, j2)];
                  }
                  if(param_.useLocalMargs_){
                     margs[var][l]=val;
                  }else{
                     if(val < bluredEnergy[var]){
                        bluredEnergy[var] = val;
                        bluredOpt[var] = l;
                     }
                  }
               }
            }
         } 
         if(param_.useLocalMargs_){
            localMargGens_.reserve(bluredOpt.size());
            for(size_t var=0 ; var<bluredOpt.size(); ++var){
               const ValueType minValue = *std::min_element(margs[var].begin(),margs[var].end());
               for(LabelType l=0; l<numLabels; ++l){
                  margs[var][l]-= minValue;
               }
               for(LabelType l=0; l<numLabels; ++l){
                  margs[var][l]=std::exp(-1.0*param_.temp_*margs[var][l]);
               }
               localMargGens_[var]=opengm::RandomDiscreteWeighted<LabelType,ValueType>(margs[var].begin(),margs[var].end(),var);  
            }
         }else{
            uniformGens_.reserve(bluredOpt.size());
            for(size_t var=0 ; var<bluredOpt.size(); ++var){
               LabelType minVal = (LabelType)(std::max((double)(0)         , bluredOpt[var]-param_.sigma_*1.5));
               LabelType maxVal = (LabelType)(std::min((double)(numLabels) , bluredOpt[var]+param_.sigma_*1.5));
               uniformGens_[var] = opengm::RandomUniform<LabelType>(minVal, maxVal+1, var);
            }
         }   
      }

   void reset(){}
   size_t defaultNumStopIt() {return 10;}
   
   void getProposal(const std::vector<LabelType> &current , std::vector<LabelType> &proposal)
      {
         proposal.resize(gm_.numberOfVariables());  
         if(param_.useLocalMargs_){ 
            for(size_t i=0; i<proposal.size();++i){
               proposal[i] = localMargGens_[i](); 
            } 
         }
         else{
            opengm::RandomUniform<LabelType> randomLabel(0, gm_.numberOfLabels(0),currentStep_);
            if ((currentStep_ % 2) == 0){ 
               for(size_t i=0; i<proposal.size();++i){
                  proposal[i] = randomLabel();
               } 
            }else{
               for(size_t i=0; i<proposal.size();++i){
                  proposal[i] = uniformGens_[i]();
               }
            }
         }
         ++currentStep_;
      }
private:
   size_t ind(int i, int j){ return i+j*height_;}
   const GM &gm_;
   Parameter param_; 
   size_t height_;
   size_t width_;
   LabelType currentStep_;

   // Random Generators
   std::vector<opengm::RandomDiscreteWeighted<LabelType,ValueType> > localMargGens_;
   std::vector<opengm::RandomUniform<LabelType> >                    uniformGens_;
};

}


template<class GM, class ACC>
class FusionBasedInf : public Inference<GM, ACC>
{
public:

    typedef ACC AccumulationType;
    typedef GM GraphicalModelType;
    OPENGM_GM_TYPE_TYPEDEFS;

    typedef opengm::visitors::VerboseVisitor<FusionBasedInf<GM, ACC> > VerboseVisitorType;
    typedef opengm::visitors::EmptyVisitor<FusionBasedInf<GM, ACC> >  EmptyVisitorType;
    typedef opengm::visitors::TimingVisitor<FusionBasedInf<GM, ACC> > TimingVisitorType;


    typedef FusionMover<GraphicalModelType, AccumulationType>    FusionMoverType ;





    // solvers for the binary problem
    typedef typename FusionMoverType::SubGmType                                 SubGmType;
    typedef opengm::LazyFlipper<SubGmType, AccumulationType>                    LazyFlipperSubInf;
    typedef opengm::BeliefPropagationUpdateRules<SubGmType, AccumulationType>   BpUr;
    typedef opengm::MaxDistance                                                 BpMd;
    typedef opengm::MessagePassing<SubGmType, AccumulationType, BpUr,BpMd>      BpSubInf;
    typedef opengm::InfAndFlip<SubGmType, AccumulationType, BpSubInf>           BpLfSubInf;
    


    #ifdef WITH_QPBO
        typedef typename ReducedInferenceHelper<SubGmType>::InfGmType ReducedGmType;
        typedef opengm::LazyFlipper<ReducedGmType, AccumulationType>    _LazyFlipperSubInf;
        typedef ReducedInference<SubGmType,ACC,_LazyFlipperSubInf>                    LazyFlipperReducedSubInf;

        #ifdef WITH_CPLEX
          typedef opengm::LPCplex<ReducedGmType, AccumulationType>      _CplexSubInf;
          typedef ReducedInference<SubGmType,ACC,_CplexSubInf>          CplexReducedSubInf;
        #endif


    #endif 



    #ifdef WITH_AD3
    typedef opengm::external::AD3Inf<SubGmType, AccumulationType>               Ad3SubInf;
    #endif
    #ifdef WITH_QPBO
    typedef kolmogorov::qpbo::QPBO<double>                                      QpboSubInf;
    typedef opengm::external::QPBO<SubGmType>                                   QPBOSubInf;
    typedef opengm::HQPBO<SubGmType,ACC>                                        HQPBOSubInf;
    #endif
    #ifdef WITH_CPLEX
    typedef opengm::LPCplex<SubGmType, AccumulationType>                        CplexSubInf;
    #endif

    enum FusionSolver
    {
        QpboFusion,
        CplexFusion,
        LazyFlipperFusion,
        BpFusion,
        BpLfFusion
    };

    enum ProposalGen
    {
        AlphaExpansion,
        AlphaBetaSwap,
        Random,
        RandomLF,
        NonUniformRandom,
        Blur,
        EnergyBlur
    };

    class Parameter
    {
    public:
        Parameter(
            const ProposalGen proposalGen    = AlphaExpansion,
            const FusionSolver fusionSolver  = QpboFusion,
            const size_t numIt               = 100, //number of Iterations
            const size_t numStopIt           = 0,   //number of Iterations without change befor stopping
            const UInt64Type maxSubgraphSize = 3,   //max SubgraphSize for _F-Solver
            const double damping             = 0.5, //damping used for LBP-Solver in each round
            const UInt64Type solverSteps     = 10, //steps for LBP-solver in each round,
            const bool reducedInf            = false,
            const bool connectedComponents   = false,
            const bool tentacles             = false,
            const float temperatur           = 1.0
        )
            :   proposalGen_(proposalGen),
                fusionSolver_(fusionSolver),
                numIt_(numIt),
                numStopIt_(numStopIt),
                fusionTimeLimit_(100),
                maxSubgraphSize_(maxSubgraphSize),
                damping_(damping),
                solverSteps_(solverSteps),
                useDirectInterface_(false),
                reducedInf_(reducedInf),
                connectedComponents_(connectedComponents),
                tentacles_(tentacles),
                temperature_(temperatur),
                sigma_(20.0),
                useEstimatedMarginals_(false)
        {

        }
        ProposalGen proposalGen_;
        FusionSolver fusionSolver_;
        size_t numIt_;
        size_t numStopIt_;
        double fusionTimeLimit_;
        UInt64Type maxSubgraphSize_;
        double damping_;
        UInt64Type solverSteps_;
        bool useDirectInterface_; 
        bool reducedInf_;
        bool connectedComponents_;
        bool tentacles_;
        float temperature_;
        double sigma_;
        bool useEstimatedMarginals_;
    };


    FusionBasedInf(const GraphicalModelType &, const Parameter & = Parameter() );
    std::string name() const;
    const GraphicalModelType &graphicalModel() const;
    InferenceTermination infer();
    void reset();
    template<class VisitorType>
    InferenceTermination infer(VisitorType &);
    void setStartingPoint(typename std::vector<LabelType>::const_iterator);
    virtual InferenceTermination arg(std::vector<LabelType> &, const size_t = 1) const ;
    virtual ValueType value()const {return bestValue_;}
private:


    template<class PROPOSAL_GEN,class VISITOR>
    void inferWithGen(PROPOSAL_GEN & gen,VISITOR & visitor);


    const GraphicalModelType &gm_;
    Parameter param_;
    FusionMoverType fusionMover_;
    ValueType bestValue_;
    std::vector<LabelType> bestArg_;
    size_t maxOrder_;
};




template<class GM, class ACC>
FusionBasedInf<GM, ACC>::FusionBasedInf
(
    const GraphicalModelType &gm,
    const Parameter &parameter
)
    :  gm_(gm),
       param_(parameter),
       fusionMover_(gm),
       bestValue_(),
       bestArg_(gm_.numberOfVariables(), 0),
       maxOrder_(gm.factorOrder())
{
    ACC::neutral(bestValue_);   

    //set default starting point
    std::vector<LabelType> conf(gm_.numberOfVariables(),0);
    for (size_t i=0; i<gm_.numberOfVariables(); ++i){
       for(typename GM::ConstFactorIterator it=gm_.factorsOfVariableBegin(i); it!=gm_.factorsOfVariableEnd(i);++it){
          if(gm_[*it].numberOfVariables() == 1){
             ValueType v;
             ACC::neutral(v);
             for(LabelType l=0; l<gm_.numberOfLabels(i); ++l){
                if(ACC::bop(gm_[*it](&l),v)){
                   v=gm_[*it](&l);
                   conf[i]=l;
                }
             }
             continue;
          }
       } 
    }
    setStartingPoint(conf.begin());
}

template<class GM, class ACC>
inline void
FusionBasedInf<GM, ACC>::reset()
{
    throw RuntimeError("not implemented yet");
}

template<class GM, class ACC>
inline void
FusionBasedInf<GM, ACC>::setStartingPoint
(
    typename std::vector<typename FusionBasedInf<GM, ACC>::LabelType>::const_iterator begin
)
{
    std::copy(begin, begin + gm_.numberOfVariables(), bestArg_.begin());
    bestValue_ = gm_.evaluate(bestArg_.begin());
}

template<class GM, class ACC>
inline std::string
FusionBasedInf<GM, ACC>::name() const
{
    return "FusionBasedInf";
}

template<class GM, class ACC>
inline const typename FusionBasedInf<GM, ACC>::GraphicalModelType &
FusionBasedInf<GM, ACC>::graphicalModel() const
{
    return gm_;
}

template<class GM, class ACC>
inline InferenceTermination
FusionBasedInf<GM, ACC>::infer()
{
    EmptyVisitorType v;
    return infer(v);
}


template<class GM, class ACC>
template<class VisitorType>
InferenceTermination FusionBasedInf<GM, ACC>::infer
(
    VisitorType &visitor
)
{
    // evaluate the current best state
    bestValue_ = gm_.evaluate(bestArg_.begin());

    visitor.begin(*this);

    if(param_.proposalGen_ == AlphaExpansion){
        typedef opengm::proposal_gen::AlphaExpansionGen<GraphicalModelType,AccumulationType> Gen;
        typename Gen::Parameter genParam;
        Gen gen(gm_, genParam);
        inferWithGen(gen,visitor);
    }
    else if(param_.proposalGen_ == AlphaBetaSwap){
        typedef opengm::proposal_gen::AlphaBetaSwapGen<GraphicalModelType,AccumulationType> Gen;
        typename Gen::Parameter genParam;
        Gen gen(gm_, genParam);
        inferWithGen(gen,visitor);
    }
    else if(param_.proposalGen_ == Random){
        typedef opengm::proposal_gen::RandomGen<GraphicalModelType,AccumulationType> Gen;
        typename Gen::Parameter genParam;
        Gen gen(gm_, genParam);
        inferWithGen(gen,visitor);
    }
    else if(param_.proposalGen_ == RandomLF){
        typedef opengm::proposal_gen::RandomLFGen<GraphicalModelType,AccumulationType> Gen;
        typename Gen::Parameter genParam;
        Gen gen(gm_, genParam);
        inferWithGen(gen,visitor);
    } 
    else if(param_.proposalGen_ == NonUniformRandom){
        typedef opengm::proposal_gen::NonUniformRandomGen<GraphicalModelType,AccumulationType> Gen;
        typename Gen::Parameter genParam(param_.temperature_);
        Gen gen(gm_, genParam);
        inferWithGen(gen,visitor);
    }
    else if(param_.proposalGen_ == Blur){
        typedef opengm::proposal_gen::BlurGen<GraphicalModelType,AccumulationType> Gen;
        typename Gen::Parameter genParam;
        Gen gen(gm_, genParam);
        inferWithGen(gen,visitor);
    }  
    else if(param_.proposalGen_ == EnergyBlur){
        typedef opengm::proposal_gen::EnergyBlurGen<GraphicalModelType,AccumulationType> Gen;
        typename Gen::Parameter genParam(param_.sigma_,param_.useEstimatedMarginals_,param_.temperature_);
        Gen gen(gm_, genParam);
        inferWithGen(gen,visitor);
    }



    visitor.end(*this);
    return NORMAL;
}


template<class GM, class ACC>
template<class PROPOSAL_GEN,class VISITOR>
void FusionBasedInf<GM, ACC>::inferWithGen(PROPOSAL_GEN & gen,VISITOR & visitor){
  
   if(param_.numStopIt_ == 0){
      param_.numStopIt_ = gen.defaultNumStopIt();
   }

   std::vector<LabelType> proposedState(gm_.numberOfVariables());
   std::vector<LabelType> fusedState(gm_.numberOfVariables());
   
   size_t countRoundsWithNoImprovement = 0;
   
   for(size_t iteration=0; iteration<param_.numIt_; ++iteration){
      // store initial value before one proposal  round
      const ValueType valueBeforeRound = bestValue_;
      
      gen.getProposal(bestArg_,proposedState);
      
      // this might be to expensive
      ValueType proposalValue = gm_.evaluate(proposedState);
      //ValueType proposalValue = 100000000000000000000000.0;
      
      fusionMover_.setup(bestArg_,proposedState,fusedState,bestValue_,proposalValue);
      const IndexType nFuseMoveVar=fusionMover_.numberOfFusionMoveVariable();
        
      if(nFuseMoveVar>0){
         if(param_.fusionSolver_==LazyFlipperFusion){
#ifdef WITH_QPBO
            // NON reduced inference
            if(param_.reducedInf_==false){
#endif
              typedef LazyFlipperSubInf SubInf;
              typename SubInf::Parameter subInfParam(param_.maxSubgraphSize_);
              bestValue_ = fusionMover_. template fuse<SubInf> (subInfParam,true);
#ifdef WITH_QPBO
            }
            // reduced inference
            else{
              typedef LazyFlipperReducedSubInf SubInf;
              typename _LazyFlipperSubInf::Parameter _subInfParam(param_.maxSubgraphSize_);
              typename SubInf::Parameter subInfParam(true,param_.tentacles_,param_.connectedComponents_,_subInfParam);
              bestValue_ = fusionMover_. template fuse<SubInf> (subInfParam,true);
            }
#endif
         }
         else if(param_.fusionSolver_==BpFusion){
            typedef BpSubInf SubInf;
            typename SubInf::Parameter subInfParam(param_.solverSteps_,0.001,param_.damping_);
            subInfParam.isAcyclic_=false;
            bestValue_ = fusionMover_. template fuse<SubInf> (subInfParam,false);
         }
         else if(param_.fusionSolver_==BpLfFusion){
            typedef BpLfSubInf SubInf;
            typename SubInf::Parameter subInfParam(param_.maxSubgraphSize_);
            typename BpSubInf::Parameter bpParam(param_.solverSteps_,0.001,param_.damping_);
            subInfParam.subPara_=bpParam;
            bestValue_ = fusionMover_. template fuse<SubInf> (subInfParam,true);
         }

#ifdef WITH_QPBO
         else if(param_.fusionSolver_==QpboFusion ){
            if(maxOrder_<=2){
               //if(param_.useDirectInterface_==false){
               //   typename QPBOSubInf::Parameter subInfParam;
               //   bestValue_ = fusionMover_. template fuse<QPBOSubInf> (subInfParam,false); 
               //}else{
                  bestValue_ = fusionMover_. template fuseQpbo<QpboSubInf> ();
                  //}
            }
            else{ 
               if(param_.useDirectInterface_==false){
                  typename HQPBOSubInf::Parameter subInfParam;
                  bestValue_ = fusionMover_. template fuse<HQPBOSubInf> (subInfParam,true);
               }else{
                  bestValue_ = fusionMover_. template fuseFixQpbo<QpboSubInf> ();
               }
            }
         }
#endif
#ifdef WITH_CPLEX
         else if(param_.fusionSolver_==CplexFusion ){
#ifdef WITH_QPBO
            // NON reduced inference
            if(param_.reducedInf_==false){
#endif
               typedef CplexSubInf SubInf;
               typename SubInf::Parameter subInfParam;
               subInfParam.integerConstraint_ = true; 
               subInfParam.numberOfThreads_ = 1; 
               subInfParam.timeLimit_       = param_.fusionTimeLimit_;  
               bestValue_ = fusionMover_. template fuse<SubInf> (subInfParam,true); 
#ifdef WITH_QPBO
            }
            // reduced inference
            else{
              typedef  CplexReducedSubInf SubInf;
              typename _CplexSubInf::Parameter _subInfParam;
              _subInfParam.integerConstraint_ = true; 
              _subInfParam.numberOfThreads_ = 1; 
              _subInfParam.timeLimit_       = param_.fusionTimeLimit_; 
              typename SubInf::Parameter subInfParam(true,param_.tentacles_,param_.connectedComponents_,_subInfParam);
              bestValue_ = fusionMover_. template fuse<SubInf> (subInfParam,true);
            }
#endif
         }
#endif

         else{
            throw std::runtime_error("Unknown Fusion Type! Maybe caused by wrong configured CMakeLists.txt");
         }
         std::copy(fusedState.begin(),fusedState.end(),bestArg_.begin());
         // get the number of fusion-move variables
      }
      if(!ACC::bop(bestValue_,valueBeforeRound)){
         // No improvement 
         ++countRoundsWithNoImprovement;
      }
      else{
         // Improvement
         countRoundsWithNoImprovement = 0;
      }
   

      if(visitor(*this)!=0){
         break;
      }

      // check if converged or done
      if(countRoundsWithNoImprovement==param_.numStopIt_ && param_.numStopIt_ !=0 ){
         break;
      }  
   }

}



template<class GM, class ACC>
inline InferenceTermination
FusionBasedInf<GM, ACC>::arg
(
    std::vector<LabelType> &x,
    const size_t N
) const
{
    if (N == 1)
    {
        x.resize(gm_.numberOfVariables());
        for (size_t j = 0; j < x.size(); ++j)
        {
            x[j] = bestArg_[j];
        }
        return NORMAL;
    }
    else
    {
        return UNKNOWN;
    }
}

} // namespace opengm

#endif // #ifndef OPENGM_FUSION_BASED_INF_HXX
