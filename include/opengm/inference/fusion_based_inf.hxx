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

#ifdef WITH_CPLEX
#include "opengm/inference/lpcplex.hxx"
#endif
#ifdef WITH_QPBO
#include "QPBO.h"
#include "opengm/inference/hqpbo.hxx"
#include "opengm/inference/reducedinference.hxx"
#endif
#ifdef WITH_AD3
#include "opengm/inference/external/ad3.hxx"
#endif

#include <stdlib.h>     /* srand, rand */


#include "opengm/inference/lazyflipper.hxx"

// fusion move model generator
#include "opengm/inference/auxiliary/fusion_move/fusion_mover.hxx"

namespace opengm
{











namespace proposal_gen
{



template<class GM, class ACC>
class AutoTunedSmoothing{
public:
    typedef ACC AccumulationType;
    typedef GM GraphicalModelType;
    OPENGM_GM_TYPE_TYPEDEFS;
    struct Parameter
    {
        Parameter(){}
    };

    AutoTunedSmoothing(const GM & gm, const Parameter & param)
    : 
        gm_(gm),
        param_(param),
        unaries_(),
        hasUnaries_(gm.numberOfVariables(), false)
    {
        size_t shape[2] = {gm_.numberOfVariables(), gm_.maxNumberOfLabels()};
        hasUnaries_.resize(shape, shape+2,ACC:: template neutral<ValueType>());

        for(IndexType vi=0; vi<gm_.numberOfVariables(); ++vi){
            const IndexType nFac = gm_.numberOfFactors(vi);

            for(IndexType f=0; f<nFac; ++f){
                const IndexType fi = gm_.factorOfVariable(vi, f);
                if(gm_[fi].numberOfVariables()==1){
                    hasUnaries_[vi]=true;
                }
            }
        }


    }

private:
    const GM & gm_;
    Parameter param_;
    marray::Marray<ValueType>  unaries_;
    std::vector<unsigned char> hasUnaries_;
};


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
class MJumpUpDownGen
{
public:
    typedef ACC AccumulationType;
    typedef GM GraphicalModelType;
    OPENGM_GM_TYPE_TYPEDEFS;
    struct Parameter
    {
        Parameter(
            const std::string startDirection = std::string("up")
        )
        : startDirection_(startDirection)
        {

        }
        std::string startDirection_;
    };
    MJumpUpDownGen(const GM &gm, const Parameter &param)
        :  gm_(gm),
           param_(param),
           argBuffer_(gm.numberOfVariables(),0),
           direction_(gm.numberOfVariables()),
           jumpSize_(gm.numberOfVariables(),1)
    {
        this->reset();
    }
    void reset()
    {
        if(param_.startDirection_==  std::string("random")){
            for(size_t i=0; i<gm_.numberOfVariables();++i){
                direction_[i]=rand()%2 == 0 ? -1:1;
            }
        }
        else if(param_.startDirection_==  std::string("up")){
            for(size_t i=0; i<gm_.numberOfVariables();++i){
                direction_[i]=1;
            }
        }
        else if(param_.startDirection_==  std::string("down")){
            for(size_t i=0; i<gm_.numberOfVariables();++i){
                direction_[i]=-1;
            }
        }
        else{
            throw opengm::RuntimeError("wrong starting direction for JumpUpDownGen");
        }
    }
    
    size_t defaultNumStopIt() {return gm_.maxNumberOfLabels();}
   
    void getProposal(const std::vector<LabelType> &current , std::vector<LabelType> &proposal)
    {
        for (IndexType vi = 0; vi < gm_.numberOfVariables(); ++vi)
        {
            const size_t numL = gm_.numberOfLabels(vi);

            const LabelType ol = argBuffer_[vi];   
            const LabelType cl = current[vi];   
            
            std::copy(current.begin(), current.end(), argBuffer_.begin());

            // flip direction?
            if(ol == cl){
                if(jumpSize_[vi] == 1)
                    direction_[vi]*=-1;
                else{
                    jumpSize_[vi]/=2;
                }
            }
            else{
                jumpSize_[vi]*=2;
            }
            const LabelType d  = direction_[vi];
            const LabelType js = jumpSize_[vi];

            if(d>=1){

                if(cl+js < gm_.numberOfLabels(vi)){
                    proposal[vi] = cl + js;
                }
                else{
                    direction_[vi] = -1;
                    proposal[vi] = gm_.numberOfLabels(vi)-1;
                    jumpSize_[vi] = 1;
                }
            }
            else{
                if(cl>=js){
                    proposal[vi] = cl - js;
                }
                else{
                    direction_[vi] = 1;
                    proposal[vi] = 0 ;
                    jumpSize_[vi] = 1;
                }
            }
        }
    } 
private:
    const GM &gm_;
    Parameter param_;
    std::vector<LabelType> argBuffer_;
    std::vector<LabelType> direction_;
    std::vector<LabelType> jumpSize_;
};

template<class GM, class ACC>
class JumpUpDownGen
{
public:
    typedef ACC AccumulationType;
    typedef GM GraphicalModelType;
    OPENGM_GM_TYPE_TYPEDEFS;
    struct Parameter
    {
        Parameter(
            const std::string startDirection = std::string("up")
        )
        : startDirection_(startDirection)
        {

        }
        std::string startDirection_;
    };
    JumpUpDownGen(const GM &gm, const Parameter &param)
        :  gm_(gm),
           param_(param),
           argBuffer_(gm.numberOfVariables(),0),
           direction_(gm.numberOfVariables()),
           jumpSize_(gm.numberOfVariables(),1)
    {
        this->reset();
    }
    void reset()
    {
        if(param_.startDirection_==  std::string("random")){
            for(size_t i=0; i<gm_.numberOfVariables();++i){
                direction_[i]=rand()%2 == 0 ? -1:1;
            }
        }
        else if(param_.startDirection_==  std::string("up")){
            for(size_t i=0; i<gm_.numberOfVariables();++i){
                direction_[i]=1;
            }
        }
        else if(param_.startDirection_==  std::string("down")){
            for(size_t i=0; i<gm_.numberOfVariables();++i){
                direction_[i]=-1;
            }
        }
        else{
            throw opengm::RuntimeError("wrong starting direction for JumpUpDownGen");
        }
    }
    
    size_t defaultNumStopIt() {return gm_.maxNumberOfLabels();}
   
    void getProposal(const std::vector<LabelType> &current , std::vector<LabelType> &proposal)
    {
        for (IndexType vi = 0; vi < gm_.numberOfVariables(); ++vi)
        {
            const size_t numL = gm_.numberOfLabels(vi);

            const LabelType ol = argBuffer_[vi];   
            const LabelType cl = current[vi];   
            
            std::copy(current.begin(), current.end(), argBuffer_.begin());

            // flip direction?
            if(ol == cl){
                if(jumpSize_[vi] == 1)
                    direction_[vi]*=-1;
                else{
                    jumpSize_[vi]-=1;
                }
            }
            else{
                jumpSize_[vi]+=1;
            }
            const LabelType d  = direction_[vi];
            const LabelType js = jumpSize_[vi];

            if(d>=1){

                if(cl+js < gm_.numberOfLabels(vi)){
                    proposal[vi] = cl + js;
                }
                else{
                    direction_[vi] = -1;
                    proposal[vi] =  gm_.numberOfLabels(vi)-1;
                    jumpSize_[vi] = 1;
                }
            }
            else{
                if(cl>=js){
                    proposal[vi] = cl - js;
                }
                else{
                    direction_[vi] = 1;
                    proposal[vi] = 0 ;
                    jumpSize_[vi] = 1;
                }
            }
        }
    } 
private:
    const GM &gm_;
    Parameter param_;
    std::vector<LabelType> argBuffer_;
    std::vector<LabelType> direction_;
    std::vector<LabelType> jumpSize_;
};

template<class GM, class ACC>
class UpDownGen
{
public:
    typedef ACC AccumulationType;
    typedef GM GraphicalModelType;
    OPENGM_GM_TYPE_TYPEDEFS;
    struct Parameter
    {
        Parameter(
            const std::string startDirection = std::string("up")
        )
        : startDirection_(startDirection)
        {

        }
        std::string startDirection_;
    };
    UpDownGen(const GM &gm, const Parameter &param)
        :  gm_(gm),
           param_(param),
           argBuffer_(gm.numberOfVariables(),0),
           direction_(gm.numberOfVariables())
    {
        this->reset();
    }
    void reset()
    {
        if(param_.startDirection_==  std::string("random")){
            for(size_t i=0; i<gm_.numberOfVariables();++i){
                direction_[i]=rand()%2 == 0 ? -1:1;
            }
        }
        else if(param_.startDirection_==  std::string("up")){
            for(size_t i=0; i<gm_.numberOfVariables();++i){
                direction_[i]=1;
            }
        }
        else if(param_.startDirection_==  std::string("down")){
            for(size_t i=0; i<gm_.numberOfVariables();++i){
                direction_[i]=-1;
            }
        }
        else{
            throw opengm::RuntimeError("wrong starting direction for UpDownGen");
        }
    }
    
    size_t defaultNumStopIt() {return 2;}
   
    void getProposal(const std::vector<LabelType> &current , std::vector<LabelType> &proposal)
    {
        for (IndexType vi = 0; vi < gm_.numberOfVariables(); ++vi)
        {
            const size_t numL = gm_.numberOfLabels(vi);

            const LabelType ol = argBuffer_[vi];   
            const LabelType cl = current[vi];   
            
            std::copy(current.begin(), current.end(), argBuffer_.begin());

            // flip direction?
            if(ol == cl){
                direction_[vi]*=-1;
            }
            const LabelType d  = direction_[vi];
            if(d==1){

                if(cl+1<numL){
                    proposal[vi] = cl +1;
                }
                else{
                    direction_[vi] = -1;
                    proposal[vi] = cl - 1 ;
                }
            }
            else{
                if(cl>=1){
                    proposal[vi] = cl - 1;
                }
                else{
                    direction_[vi] = 1;
                    proposal[vi] = cl + 1 ;
                }
            }
        }
    } 
private:
    const GM &gm_;
    Parameter param_;
    std::vector<LabelType> argBuffer_;
    std::vector<LabelType> direction_;
    std::vector<LabelType> jumpSize_;
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
       double sigma_;
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
      double sigma_;
      bool   useLocalMargs_; 
      double temp_;
      
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


template<class GM, class ACC>
class DynamincGen{
public:
   typedef ACC AccumulationType;
   typedef GM GraphicalModelType;
   OPENGM_GM_TYPE_TYPEDEFS;
    enum GeneratorType{
        AlphaExpansion,
        AlphaBetaSwap,
        UpDown,
        Random,
        RandomLF,
        NonUniformRandom,
        Blur,
        EnergyBlur
    };

    struct Parameter{
        GeneratorType gen_;
    };

    DynamincGen(const GM & gm, const Parameter & param)
    : 
        gm_(gm),
        param_(param){
    }

    void reset(){
        if(param_.gen_ == AlphaExpansion)
            alphaExpansionGen_->reset();
        else if(param_.gen_ == AlphaBetaSwap)
            alphaBetaSwapGen_->reset();
        else if(param_.gen_ == UpDown)
            upDownGen_->reset();
        else if(param_.gen_ == Random)
            randomGen_->reset();
        else if(param_.gen_ == RandomLF)
            randomLFGen_->reset();
        else if(param_.gen_ == NonUniformRandom)
            nonUniformRandomGen_->reset();
        else if(param_.gen_ == Blur)
            blurGen_->reset();
        else if(param_.gen_ == EnergyBlur)
            energyBlurGen_->reset();
        else{
            throw RuntimeError("unknown generator type");
        }
    }
    size_t defaultNumStopIt() {
        if(param_.gen_ == AlphaExpansion)
            return alphaExpansionGen_->defaultNumStopIt();
        else if(param_.gen_ == AlphaBetaSwap)
            return alphaBetaSwapGen_->defaultNumStopIt();
        else if(param_.gen_ == UpDown)
            return upDownGen_->defaultNumStopIt();
        else if(param_.gen_ == Random)
            return randomGen_->defaultNumStopIt();
        else if(param_.gen_ == RandomLF)
            return randomLFGen_->defaultNumStopIt();
        else if(param_.gen_ == NonUniformRandom)
            return nonUniformRandomGen_->defaultNumStopIt();
        else if(param_.gen_ == Blur)
            return blurGen_->defaultNumStopIt();
        else if(param_.gen_ == EnergyBlur)
            return energyBlurGen_->defaultNumStopIt();
        else{
            throw RuntimeError("unknown generator type");
        }
    }
    void getProposal(const std::vector<LabelType> &current , std::vector<LabelType> &proposal){
        if(param_.gen_ == AlphaExpansion)
            return alphaExpansionGen_->getProposal(current, proposal);
        else if(param_.gen_ == AlphaBetaSwap)
            return alphaBetaSwapGen_->getProposal(current, proposal);
        else if(param_.gen_ == UpDown)
            return upDownGen_->getProposal(current, proposal);
        else if(param_.gen_ == Random)
            return randomGen_->getProposal(current, proposal);
        else if(param_.gen_ == RandomLF)
            return randomLFGen_->getProposal(current, proposal);
        else if(param_.gen_ == NonUniformRandom)
            return nonUniformRandomGen_->getProposal(current, proposal);
        else if(param_.gen_ == Blur)
            return blurGen_->getProposal(current, proposal);
        else if(param_.gen_ == EnergyBlur)
            return energyBlurGen_->getProposal(current, proposal);
        else{
            throw RuntimeError("unknown generator type");
        }
    }
private:
    const GM & gm_;
    Parameter param_;

    // generators
    AlphaExpansionGen<GM, ACC> *   alphaExpansionGen_;
    AlphaBetaSwapGen <GM, ACC> *   alphaBetaSwapGen_;
    UpDownGen<GM, ACC> *           upDownGen_;
    RandomGen<GM, ACC> *           randomGen_;
    RandomLFGen<GM, ACC> *         randomLFGen_;
    NonUniformRandomGen<GM, ACC> * nonUniformRandomGen_;
    BlurGen<GM, ACC> *             blurGen_;
    EnergyBlurGen<GM, ACC> *       energyBlurGen_;
};



}


template<class GM, class PROPOSAL_GEN>
class FusionBasedInf : public Inference<GM, typename  PROPOSAL_GEN::AccumulationType>
{
public:
    typedef PROPOSAL_GEN ProposalGen;
    typedef typename ProposalGen::AccumulationType AccumulationType;
    typedef AccumulationType ACC;
    typedef GM GraphicalModelType;
    OPENGM_GM_TYPE_TYPEDEFS;

    typedef opengm::visitors::VerboseVisitor<FusionBasedInf<GM, PROPOSAL_GEN> > VerboseVisitorType;
    typedef opengm::visitors::EmptyVisitor<FusionBasedInf<GM, PROPOSAL_GEN> >  EmptyVisitorType;
    typedef opengm::visitors::TimingVisitor<FusionBasedInf<GM, PROPOSAL_GEN> > TimingVisitorType;


    typedef HlFusionMover<GraphicalModelType, AccumulationType>    FusionMoverType ;
    typedef HlFusionMover<GraphicalModelType, AccumulationType>    FusionMover ;

    typedef typename ProposalGen::Parameter ProposalParameter;
    typedef typename FusionMoverType::Parameter FusionParameter;



    class Parameter
    {
    public:
        Parameter(
            const ProposalParameter & proposalParam = ProposalParameter(),
            const FusionParameter   & fusionParam = FusionParameter(),
            const size_t numIt=1000,
            const size_t numStopIt = 0
        )
            :   proposalParam_(proposalParam),
                fusionParam_(fusionParam),
                numIt_(numIt),
                numStopIt_(numStopIt)
        {

        }
        ProposalParameter proposalParam_;
        FusionParameter fusionParam_;
        size_t numIt_;
        size_t numStopIt_;
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


    const GraphicalModelType &gm_;
    Parameter param_;
    FusionMoverType fusionMover_;
    PROPOSAL_GEN proposalGen_;
    ValueType bestValue_;
    std::vector<LabelType> bestArg_;
    size_t maxOrder_;
};




template<class GM, class PROPOSAL_GEN>
FusionBasedInf<GM, PROPOSAL_GEN>::FusionBasedInf
(
    const GraphicalModelType &gm,
    const Parameter &parameter
)
    :  gm_(gm),
       param_(parameter),
       fusionMover_(gm,parameter.fusionParam_),
       proposalGen_(gm,parameter.proposalParam_),
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

template<class GM, class PROPOSAL_GEN>
inline void
FusionBasedInf<GM, PROPOSAL_GEN>::reset()
{
    throw RuntimeError("not implemented yet");
}

template<class GM, class PROPOSAL_GEN>
inline void
FusionBasedInf<GM, PROPOSAL_GEN>::setStartingPoint
(
    typename std::vector<typename FusionBasedInf<GM, PROPOSAL_GEN>::LabelType>::const_iterator begin
)
{
    std::copy(begin, begin + gm_.numberOfVariables(), bestArg_.begin());
    bestValue_ = gm_.evaluate(bestArg_.begin());
}

template<class GM, class PROPOSAL_GEN>
inline std::string
FusionBasedInf<GM, PROPOSAL_GEN>::name() const
{
    return "FusionBasedInf";
}

template<class GM, class PROPOSAL_GEN>
inline const typename FusionBasedInf<GM, PROPOSAL_GEN>::GraphicalModelType &
FusionBasedInf<GM, PROPOSAL_GEN>::graphicalModel() const
{
    return gm_;
}

template<class GM, class PROPOSAL_GEN>
inline InferenceTermination
FusionBasedInf<GM, PROPOSAL_GEN>::infer()
{
    EmptyVisitorType v;
    return infer(v);
}


template<class GM, class PROPOSAL_GEN>
template<class VisitorType>
InferenceTermination FusionBasedInf<GM, PROPOSAL_GEN>::infer
(
    VisitorType &visitor
)
{
    // evaluate the current best state
    bestValue_ = gm_.evaluate(bestArg_.begin());

    visitor.begin(*this);


    if(param_.numStopIt_ == 0){
        param_.numStopIt_ = proposalGen_.defaultNumStopIt();
    }

    std::vector<LabelType> proposedState(gm_.numberOfVariables());
    std::vector<LabelType> fusedState(gm_.numberOfVariables());

    size_t countRoundsWithNoImprovement = 0;

    for(size_t iteration=0; iteration<param_.numIt_; ++iteration){
        // store initial value before one proposal  round
        const ValueType valueBeforeRound = bestValue_;

        proposalGen_.getProposal(bestArg_,proposedState);

        // this might be to expensive
        ValueType proposalValue = gm_.evaluate(proposedState);
        //ValueType proposalValue = 100000000000000000000000.0;


        const bool anyVar = fusionMover_.fuse(bestArg_,proposedState, fusedState, 
                                              bestValue_, proposalValue, bestValue_);
        if(anyVar){
            if( !ACC::bop(bestValue_, valueBeforeRound)){
                ++countRoundsWithNoImprovement;
            }
            else{
                // Improvement
                countRoundsWithNoImprovement = 0;
                bestArg_ = fusedState;
            }
            if(visitor(*this)!=0){
                break;
            }
        }
        else{
            ++countRoundsWithNoImprovement;
        }
        // check if converged or done
        if(countRoundsWithNoImprovement==param_.numStopIt_ && param_.numStopIt_ !=0 )
            break;
    }
    visitor.end(*this);
    return NORMAL;
}




template<class GM, class PROPOSAL_GEN>
inline InferenceTermination
FusionBasedInf<GM, PROPOSAL_GEN>::arg
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
