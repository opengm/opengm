#pragma once
#ifndef OPENGM_LP_INTERFACE_HXX
#define OPENGM_LP_INTERFACE_HXX

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <typeinfo>

#include "opengm/datastructures/marray/marray.hxx"
#include "opengm/opengm.hxx"
#include "opengm/operations/adder.hxx"
#include "opengm/operations/minimizer.hxx"
#include "opengm/inference/inference.hxx"
#include "opengm/inference/visitors/visitor.hxx"




namespace opengm {
    
    
    enum Rounding{
        MaxLpVarFromVar=0
    };
    
    template<class LpValueType>
    struct LpInferenceHelperParameter{
        bool integerConstraint_;
    };

    template<class LP_SOLVER,class GM,class ACC,class LP_INDEX_TYPE>
    class LpInferenceHelper{
    public:
        typedef GM GraphicalModelType;
        typedef ACC AccumulationType;
        OPENGM_GM_TYPE_TYPEDEFS;
        typedef LP_INDEX_TYPE LpIndexType;
    public:
        LpInferenceHelper(const GraphicalModelType & gm);
        LpIndexType variableLabelToLpVariable(const IndexType vi,const LabelType l)const;
        LpIndexType factorLabelingToLpVariable(const IndexType fi,const LabelType configIndex)const;
        size_t numberOfLpVariables()const;
        size_t numberOfLpVariablesFromVariables()const;
        size_t numberOfLpVariablesFromFactor()const;
    protected:
        LabelType maxNumberOfLabels()const;
        size_t maxNumberOfFactorStates()const;
        size_t numberOfUnaryFactors()const;
        size_t numberOfHighOrderFactors()const;
        size_t numberOfBasicConstraints()const;
        IndexType highOrderFactorToGmFactor(const IndexType highOrderFactorIndex)const;
        IndexType gmFactorToHighOrderFactor(const IndexType highOrderFactorIndex)const;
    protected:
        template<class GM_LABEL_ITER,class LP_LABEL_ITER>
        void gmLabelingToLpLabeling(GM_LABEL_ITER gmLabeling,LP_LABEL_ITER lpLabeling)const;
        
        template<class LP_LABEL_ITER,class GM_LABEL_ITER>
        void lpLabelingToGmLabeling(LP_LABEL_ITER lpLabeling,GM_LABEL_ITER gmLabeling,const Rounding rounding,const bool integerConstraint)const;
    private:
        const GraphicalModelType & gm_;
        //
        size_t numberOfHighOrderFactors_;
        size_t numberOfUnaryFactors_;
        // indexing
        std::vector<LpIndexType> varToLpVarBegin_;
        std::vector<LpIndexType> highOrderFactorToLpVarBegin_;
        std::vector<IndexType> gmFactorIndexToHighOrderIndex_;
        std::vector<IndexType> highOrderIndexToGmIndex_;
        // max (for buffers)
        size_t maxVarLabel_;
        size_t maxFactorStates_;
        // lp var
        LpIndexType numLpVar_;
        LpIndexType numLpVarFromVar_;
        LpIndexType numLpVarFromFactor_;
        // constraints
        LpIndexType numBaseConstraints_;
    };
 

template<class LP_SOLVER,class GM,class ACC,class LP_INDEX_TYPE>
LpInferenceHelper<LP_SOLVER,GM,ACC,LP_INDEX_TYPE>::LpInferenceHelper
(
    const GM & gm
) :
gm_(gm),
numberOfHighOrderFactors_(0),
numberOfUnaryFactors_(0),
varToLpVarBegin_(gm.numberOfVariables()),
highOrderFactorToLpVarBegin_(),
gmFactorIndexToHighOrderIndex_(gm.numberOfFactors()),
highOrderIndexToGmIndex_(),
maxVarLabel_(0),
maxFactorStates_(0),
numLpVar_(0),
numLpVarFromVar_(0),
numLpVarFromFactor_(0),
numBaseConstraints_(0)
{
    // count high order factors
    for(IndexType fi=0;fi<gm.numberOfFactors();++fi){
        const IndexType numVar=gm[fi].numberOfVariables();
        if(numVar==1)
            ++numberOfUnaryFactors_;
        else if(numVar>1)
            ++numberOfHighOrderFactors_;
        else
            throw RuntimeError("constant factors ar not allowed in LpCoinOrOsi");
    }

    // GM VAR / FACTORS STATES  TO  LP VAR  (INDEXING)
    // reserve
    highOrderIndexToGmIndex_.resize(numberOfHighOrderFactors_);
    highOrderFactorToLpVarBegin_.resize(numberOfHighOrderFactors_);
    for(IndexType vi=0;vi<gm_.numberOfVariables();++vi){
        // var to lp var
        varToLpVarBegin_[vi]=numLpVarFromVar_;
        const LabelType numLabels=gm.numberOfLabels(vi);
        numLpVarFromVar_+=static_cast<LpIndexType>(numLabels);
        // max label
        maxVarLabel_= numLabels>maxVarLabel_? numLabels:maxVarLabel_;
    }
    // factor lp var begin with numLpVarFromVar_
    numLpVar_=numLpVarFromVar_;
    // factor to lp var (count number of high order factors first
    size_t hoFi=0;
    for(IndexType fi=0;fi<gm.numberOfFactors();++fi){
        const IndexType numVar=gm[fi].numberOfVariables();
        if(numVar==1){
            gmFactorIndexToHighOrderIndex_[fi]=gm.numberOfFactors();
        }
        else if(numVar>1){
            OPENGM_ASSERT(hoFi<this->numberOfHighOrderFactors());
            gmFactorIndexToHighOrderIndex_[fi]=hoFi;
            highOrderIndexToGmIndex_[hoFi]=fi;
            const size_t numFactorStates=gm[fi].size();
            highOrderFactorToLpVarBegin_[hoFi]=numLpVarFromVar_+numLpVarFromFactor_;
            numLpVarFromFactor_+=numFactorStates;
            // max factor states
            maxFactorStates_= numFactorStates>maxFactorStates_? numFactorStates:maxFactorStates_;
            ++hoFi;
        }
        else
            throw RuntimeError("constant factors ar not allowed in LpCoinOrOsi");
    }
    OPENGM_ASSERT(hoFi==this->numberOfHighOrderFactors());
    numLpVar_=numLpVarFromVar_+numLpVarFromFactor_;
    
    // get number of constraints
    numBaseConstraints_=0;
    // 1 constraint for each lp variable X to enforce  0 <= X <=0
    // (for all lp var ,those from variables and those from factors)
    
    //numBaseConstraints_+=this->numberOfLpVariables();
    
    // 1 constraint for each gm variable sum_ lpvar_gmvar = 1
    // (for each gm variable the sum of all corresponding lp vars must be 1)
    numBaseConstraints_+=gm_.numberOfVariables();
    // 1 constraint for each gm factor   sum_ lpvar_gmvar = 1
    // (for each gm factor the sum of all corresponding lp vars must be 1)
    numBaseConstraints_+=this->numberOfHighOrderFactors();
    
    // for each factor there are constraints to enforce that
    // the labeling of the factor machtes the labeling of variables
    for(size_t hofi=0;hofi<this->numberOfHighOrderFactors();++hofi){
        const IndexType fi=this->highOrderFactorToGmFactor(hofi);
        numBaseConstraints_+=gm[fi].size();
    }
}


template<class LP_SOLVER,class GM,class ACC,class LP_INDEX_TYPE>
inline LP_INDEX_TYPE 
LpInferenceHelper<LP_SOLVER,GM,ACC,LP_INDEX_TYPE>::variableLabelToLpVariable
(
    const typename GM::IndexType vi,
    const typename GM::LabelType l
)const{
    OPENGM_ASSERT(vi<gm_.numberOfVariables());
    OPENGM_ASSERT(vi<this->numberOfLpVariablesFromVariables());
    return varToLpVarBegin_[vi]+l;
}

template<class LP_SOLVER,class GM,class ACC,class LP_INDEX_TYPE>
LP_INDEX_TYPE 
LpInferenceHelper<LP_SOLVER,GM,ACC,LP_INDEX_TYPE>::factorLabelingToLpVariable
(
    const IndexType fi,
    const LabelType configIndex
)const{
    OPENGM_ASSERT(fi<gm_.numberOfFactors());
    OPENGM_ASSERT(gm_[fi].numberOfVariables()>1);
    OPENGM_ASSERT(configIndex<gm_[fi].size());
    OPENGM_ASSERT(fi<gmFactorIndexToHighOrderIndex_.size());
    const IndexType hoFi=gmFactorIndexToHighOrderIndex_[fi];
    OPENGM_ASSERT(hoFi<this->numberOfHighOrderFactors());
    OPENGM_ASSERT(highOrderFactorToLpVarBegin_[hoFi]+configIndex>=this->numberOfLpVariablesFromVariables());
    OPENGM_ASSERT(highOrderFactorToLpVarBegin_[hoFi]+configIndex -this->numberOfLpVariablesFromVariables() < this->numberOfLpVariablesFromFactor());
    OPENGM_ASSERT(highOrderFactorToLpVarBegin_[hoFi]+configIndex<this->numberOfLpVariables());
    return highOrderFactorToLpVarBegin_[hoFi]+configIndex;
}

template<class LP_SOLVER,class GM,class ACC,class LP_INDEX_TYPE>
size_t  
LpInferenceHelper<LP_SOLVER,GM,ACC,LP_INDEX_TYPE>::numberOfLpVariables()const{
    return numLpVar_;
}

template<class LP_SOLVER,class GM,class ACC,class LP_INDEX_TYPE>
inline size_t 
LpInferenceHelper<LP_SOLVER,GM,ACC,LP_INDEX_TYPE>::numberOfLpVariablesFromVariables()const{
    return numLpVarFromVar_;
}

template<class LP_SOLVER,class GM,class ACC,class LP_INDEX_TYPE>
inline size_t 
LpInferenceHelper<LP_SOLVER,GM,ACC,LP_INDEX_TYPE>::numberOfLpVariablesFromFactor()const{
    return numLpVarFromFactor_;
}

template<class LP_SOLVER,class GM,class ACC,class LP_INDEX_TYPE>
inline typename GM::LabelType 
LpInferenceHelper<LP_SOLVER,GM,ACC,LP_INDEX_TYPE>::maxNumberOfLabels()const{
    return maxVarLabel_;
}

template<class LP_SOLVER,class GM,class ACC,class LP_INDEX_TYPE>
inline size_t 
LpInferenceHelper<LP_SOLVER,GM,ACC,LP_INDEX_TYPE>::maxNumberOfFactorStates()const{
    return maxFactorStates_;
}

template<class LP_SOLVER,class GM,class ACC,class LP_INDEX_TYPE>
inline size_t  
LpInferenceHelper<LP_SOLVER,GM,ACC,LP_INDEX_TYPE>::numberOfUnaryFactors()const{
    return numberOfUnaryFactors_;
}

template<class LP_SOLVER,class GM,class ACC,class LP_INDEX_TYPE>
inline size_t  
LpInferenceHelper<LP_SOLVER,GM,ACC,LP_INDEX_TYPE>::numberOfHighOrderFactors()const{
    return numberOfHighOrderFactors_;
}

template<class LP_SOLVER,class GM,class ACC,class LP_INDEX_TYPE>
inline size_t  
LpInferenceHelper<LP_SOLVER,GM,ACC,LP_INDEX_TYPE>::numberOfBasicConstraints()const{
    return numBaseConstraints_;
}

template<class LP_SOLVER,class GM,class ACC,class LP_INDEX_TYPE>  
inline typename GM::IndexType 
LpInferenceHelper<LP_SOLVER,GM,ACC,LP_INDEX_TYPE>::highOrderFactorToGmFactor
(
    const typename GM::IndexType highOrderFactorIndex
)const{
    return highOrderIndexToGmIndex_[highOrderFactorIndex];
}

template<class LP_SOLVER,class GM,class ACC,class LP_INDEX_TYPE>
inline typename GM::IndexType 
LpInferenceHelper<LP_SOLVER,GM,ACC,LP_INDEX_TYPE>::gmFactorToHighOrderFactor
(
    const typename GM::IndexType factorIndex
)const{
    OPENGM_ASSERT(gm_[factorIndex].numberOfVariables()>1);
    return gmFactorIndexToHighOrderIndex_[factorIndex];
}

template<class LP_SOLVER,class GM,class ACC,class LP_INDEX_TYPE>
template<class GM_LABEL_ITER,class LP_LABEL_ITER>
void 
LpInferenceHelper<LP_SOLVER,GM,ACC,LP_INDEX_TYPE>::gmLabelingToLpLabeling
(
    GM_LABEL_ITER gmLabeling,
    LP_LABEL_ITER lpLabeling
)const{
    typedef typename std::iterator_traits<LP_LABEL_ITER>::value_type LpLabelType;
    // initialize all lp var with 0
    std::fill(lpLabeling,lpLabeling+this->numberOfLpVariables(),static_cast<LpLabelType>(0.0));
    
    // lp var from gm variables
    for(IndexType vi=0;vi<gm_.numberOfVariables();++vi){
        LabelType label=static_cast<LabelType>(gmLabeling[vi]);
        LpIndexType lpVar=this->variableLabelToLpVariable(vi,label);
        lpLabeling[lpVar]=static_cast<LpLabelType>(1.0);
    }
    // lp var from factor states
    for(IndexType hofi=0;hofi<this->numberOfHighOrderFactors();++hofi){
        IndexType fi=this->highOrderFactorToGmFactor(hofi);
        size_t numVar=gm_[fi].numberOfVariables();
        
        // compute the index of the labeling
        const IndexType vi1=gm_[fi].variableIndex(0);
        const LabelType l1=gmLabeling[vi1];
        const LabelType numL1=gm_[fi].numberOfLabels(0);
        size_t configIndex=l1;
        size_t strides=numL1;
        for(size_t v=1;v<numVar;++v){
            const IndexType vi=gm_[fi].variableIndex(v);
            const LabelType l=gmLabeling[vi];
            const LabelType numL=gm_[fi].numberOfLabels(v);
            configIndex+=static_cast<size_t>(l)*static_cast<size_t>(strides);
            strides*=numL;
        }
        // set the variable of this configuration to 1
        lpLabeling[configIndex]=static_cast<LpLabelType>(1.0);
    }
}

template<class LP_SOLVER,class GM,class ACC,class LP_INDEX_TYPE>
template<class LP_LABEL_ITER,class GM_LABEL_ITER>
void 
LpInferenceHelper<LP_SOLVER,GM,ACC,LP_INDEX_TYPE>::lpLabelingToGmLabeling
(
    LP_LABEL_ITER lpLabeling,
    GM_LABEL_ITER gmLabeling,
    const Rounding rounding,
    const bool integerConstraint
)const{
    typedef typename std::iterator_traits<LP_LABEL_ITER>::value_type LpLabelType;
    typedef typename std::iterator_traits<GM_LABEL_ITER>::value_type GmLabelType;
    if(integerConstraint==false ){
        for(IndexType vi=0;vi<gm_.numberOfVariables();++vi){
            
            LpLabelType maxLpLabel=static_cast<LpLabelType>(0.0);
            LabelType maxState=static_cast<LabelType>(0);
            LabelType numberOfLabels=gm_.numberOfLabels(vi);
            
            for(size_t l=0;l<numberOfLabels;++l){
                LpIndexType lpVar=this->variableLabelToLpVariable(vi,l);
                LpLabelType x=lpLabeling[lpVar];
                OPENGM_ASSERT(x>=static_cast<LpLabelType>(0.0) && x<=static_cast<LpLabelType>(1.0));
                if(x>maxLpLabel){
                    maxLpLabel=x;
                    maxState=l;
                }
            }
            gmLabeling[vi]=static_cast<GmLabelType>(maxState);
        }
    }
    else{
        for(IndexType vi=0;vi<gm_.numberOfVariables();++vi){
            
            bool found1Integer=false;
            LabelType bestLabel=0;
            LabelType numberOfLabels=gm_.numberOfLabels(vi);
            for(size_t l=0;l<numberOfLabels;++l){
                LpIndexType lpVar=this->variableLabelToLpVariable(vi,l);
                LpLabelType x=lpLabeling[lpVar];
                OPENGM_ASSERT(x>=static_cast<LpLabelType>(-0.001) && x<=static_cast<LpLabelType>(1.0));
                if(x>=1.0){
                    if(found1Integer==false){
                        found1Integer=true;
                        bestLabel=l;
                    }
                    else{
                        throw RuntimeError("solution is not valid");
                    }
                }
            }
            gmLabeling[vi]=static_cast<GmLabelType>(bestLabel);
        }
    }

}

} // end namespace opengm
#endif
