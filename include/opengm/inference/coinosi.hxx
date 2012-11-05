#pragma once
#ifndef OPENGM_CONIN_OSI_HXX
#define OPENGM_CONIN_OSI_HXX

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

#include "opengm/inference/lpinterface.hxx"
//COIN-OR-OSI

#include <coin/CoinPackedMatrix.hpp>
#include <coin/OsiDylpSolverInterface.hpp>
#include <coin/OsiClpSolverInterface.hpp>
#include <coin/OsiSolverInterface.hpp>



#include <coin/CoinPackedVector.hpp>
#include <coin/CoinWarmStartVector.hpp>
#include <map>

namespace opengm {


enum Solver{
    ClpSolver=0,
    DylpSolver=1
};


template<class GM,class ACC>
class LpCoinOrOsi :   
public Inference<GM,ACC>,
LpInferenceHelper<LpCoinOrOsi<GM,ACC>,GM,ACC,typename GM::IndexType > 
{
public:
    typedef LpInferenceHelper<LpCoinOrOsi<GM,ACC>,GM,ACC,typename GM::IndexType >  BaseType;
    typedef GM GraphicalModelType;
    typedef ACC AccumulationType;
    OPENGM_GM_TYPE_TYPEDEFS;
private:    
    typedef typename GraphicalModelType::FactorType::ShapeIteratorType  FactorShapeIteratorType;
    typedef opengm::ShapeWalker<FactorShapeIteratorType> FactorShapeWalkerType;
public:
    
    
    // Base Solver
    typedef OsiSolverInterface SolverBaseType;
    // Solvers:
    typedef OsiClpSolverInterface ClpSolverType;
    typedef OsiDylpSolverInterface DylpSolverType;
    
    typedef CoinPackedMatrix PackedMatrixType;
    typedef CoinPackedVector PackedVectorType;
    typedef double LpValueType;
    typedef IndexType LpIndexType;
    typedef LpIndexType ConstraintIndexType;
    
    
    typedef OsiIntParam IntegerParamEnum;
    typedef OsiDblParam DoubleParamEnum;
    typedef OsiStrParam StringParamEnum;
    typedef OsiHintParam HintParamEnum;
    typedef OsiHintStrength HintStrengthEnum;
    
    struct HintStrengthAndValue{
        HintStrengthEnum hintStrength_;
        bool value_;
    };
    
    typedef std::map<IntegerParamEnum,int> IntegerParameterMapType;
    typedef std::map<DoubleParamEnum,double> DoubleParameterMapType;
    typedef std::map<StringParamEnum,std::string> StringParameterMapType;
    typedef std::map<OsiHintParam,HintStrengthAndValue> HintParameterMapType;
    
    struct Parameter{
        Parameter
        (
            Solver solver=DylpSolver,
            const bool integerConstraint=true,
            const IntegerParameterMapType & intParamMap=IntegerParameterMapType(),
            const DoubleParameterMapType & doubleParamMap=DoubleParameterMapType(),
            const StringParameterMapType & stringParamMap=StringParameterMapType(),
            const HintParameterMapType & hintParamMap=HintParameterMapType()
        )
        :
        solver_(solver),
        integerConstraint_(integerConstraint),
        intParamMap_(intParamMap),
        doubleParamMap_(doubleParamMap),
        stringParamMap_(stringParamMap),
        hintParamMap_(hintParamMap){
        }
        Solver solver_;
        bool integerConstraint_;
        IntegerParameterMapType intParamMap_;
        DoubleParameterMapType doubleParamMap_;
        StringParameterMapType stringParamMap_;
        HintParameterMapType hintParamMap_;
    };
    
    LpCoinOrOsi(const GM & gm,const typename LpCoinOrOsi<GM,ACC>::Parameter & param= typename LpCoinOrOsi<GM,ACC>::Parameter());
    std::string name() const;
    const GraphicalModelType& graphicalModel() const;
    void reset();
    InferenceTermination infer();
    template<class VISITOR>
    InferenceTermination infer(VISITOR&);
    void setStartingPoint(typename std::vector<LabelType>::const_iterator);
    InferenceTermination arg(std::vector<LabelType>&, const size_t =1) const;
    //ValueType bound()const;
    
    
protected:
    

    
    void setUpParameters();
    const GraphicalModelType & gm_;  
    //CoinOSL data
    SolverBaseType * solverPtr_;
    LpValueType * objectivePtr_;
    LpValueType * lowerBounds_;
    LpValueType * upperBounds_;
    LpValueType * varLowerBounds_;
    LpValueType * varUpperBounds_;
    PackedMatrixType * constraintMatrix_;
    //Parameter
    Parameter param_;
};




template<class GM,class ACC>
LpCoinOrOsi<GM,ACC>::LpCoinOrOsi
(
    const GM & gm,
    const typename LpCoinOrOsi<GM,ACC>::Parameter & param
)
:BaseType(gm),gm_(gm),param_(param){
    //Create a problem pointer
    //When we instantiate the object, we need a specific derived class
    if(param_.solver_==ClpSolver){
        solverPtr_=new ClpSolverType();
    }
    else if(param_.solver_==DylpSolver){
        solverPtr_=new DylpSolverType();
    }
    else{
        throw RuntimeError("Solver not supported");
    }
    //// OBJECTIVE
    // ALLOCATE SPACE FOR OBJECTIVE 
    objectivePtr_=new LpValueType[this->numberOfLpVariables()];
    std::fill(objectivePtr_,objectivePtr_+this->numberOfLpVariables(),static_cast<LpValueType>(0));
    // objective of unaries:
    // allocate buffer for unary values
    {
        std::vector<ValueType> unaryBuffer(this->maxNumberOfLabels());
        for(IndexType fi=0;fi<gm.numberOfFactors();++fi){
            const FactorType & factor=gm[fi];
            if(factor.numberOfVariables()==1){
                const IndexType gmVi=factor.variableIndex(0);
                const LabelType numLabels=factor.numberOfLabels(0);
                // copy values 
                factor.copyValues(unaryBuffer.begin());
                for (LabelType l=0;l<numLabels;++l){
                    const LpIndexType lpVi=variableLabelToLpVariable(gmVi,l);
                    objectivePtr_[lpVi]=unaryBuffer[l];
                }
            }
        }
    }
    // objective of high order factors:
    // allocate buffer for high order factor
    {
        std::vector<ValueType> factorBuffer(this->maxNumberOfFactorStates());
        for(IndexType fi=0;fi<gm.numberOfFactors();++fi){
            const FactorType & factor=gm[fi];
            if(factor.numberOfVariables()>1){
                const size_t hoFi=this->gmFactorToHighOrderFactor(fi);
                OPENGM_ASSERT(hoFi<this->numberOfHighOrderFactors());
                const size_t numStates=factor.size();
                // copy values 
                factor.copyValues(factorBuffer.begin());
                for (LabelType s=0;s<numStates;++s){
                    const LpIndexType lpVi=this->factorLabelingToLpVariable(fi,s);
                    OPENGM_ASSERT(lpVi<this->numberOfLpVariables());
                    OPENGM_ASSERT(s<factorBuffer.size());
                    objectivePtr_[lpVi]=factorBuffer[s];
                }
            }
        }
    }
    ////////////////////////////////////////////////////////////////
    ////
    ////     CONSTRAINTS
    ////
    ////////////////////////////////////////////////////////////////
    
    // fill constraints in a sparse matrix,where each row is a constraint
    PackedMatrixType * constraintMatrix_ = new CoinPackedMatrix(false,0,0);
    
    size_t numConstraints=this->numberOfBasicConstraints(); 
    constraintMatrix_->setDimensions(0, this->numberOfLpVariables());

    LpValueType * lowerBounds_ = new LpValueType[numConstraints];
    LpValueType * upperBounds_ = new LpValueType[numConstraints];
    LpValueType * varLowerBounds_ = new LpValueType[this->numberOfLpVariables()];
    LpValueType * varUpperBounds_ = new LpValueType[this->numberOfLpVariables()];
    std::fill(varLowerBounds_,varLowerBounds_+this->numberOfLpVariables(),static_cast<LpValueType>(0.0));
    std::fill(varUpperBounds_,varUpperBounds_+this->numberOfLpVariables(),static_cast<LpValueType>(1.0));
    
    ConstraintIndexType cIndex=0;
    // constraints on variables 
    for(IndexType vi=0;vi<gm.numberOfVariables();++vi){
        const LabelType numLabels=gm_.numberOfLabels(vi);
        PackedVectorType sumStatesMustBeOne;
        // 1 equality constraint that summ must be 1
        for (LabelType l=0;l<numLabels;++l){
            OPENGM_ASSERT(cIndex<numConstraints);
            const LpIndexType lpVar=this->variableLabelToLpVariable(vi,l);
            OPENGM_ASSERT(lpVar<this->numberOfLpVariablesFromVariables());
            sumStatesMustBeOne.insert(lpVar,1.0);
        }
        OPENGM_ASSERT(cIndex<numConstraints);
        constraintMatrix_->appendRow(sumStatesMustBeOne);   
        //equality constraint
        lowerBounds_[cIndex]=static_cast<LpValueType>(1.0);
        upperBounds_[cIndex]=static_cast<LpValueType>(1.0);
        ++cIndex;
    }
    OPENGM_ASSERT(cIndex==gm.numberOfVariables());
    // constraints on high order factors
     for(IndexType fi=0;fi<gm.numberOfFactors();++fi){
        const FactorType & factor=gm[fi];
        const size_t numVar=factor.numberOfVariables();
        if(numVar>1){
            FactorShapeWalkerType walker(factor.shapeBegin(),numVar);
            const size_t factorSize=factor.size();
            // 1 constraints that summ must be 1
            // |factor|  constraints  that factor var matches unary vars
            PackedVectorType sumStatesMustBeOne;
            for (size_t confIndex=0;confIndex<factorSize;++confIndex,++walker){
                OPENGM_ASSERT(cIndex<numConstraints);
                const LpIndexType lpVar=this->factorLabelingToLpVariable(fi,confIndex);
                sumStatesMustBeOne.insert(lpVar,LpValueType(numVar));
                // loop over all var
                PackedVectorType factorMustMatchVariable;
                factorMustMatchVariable.insert(lpVar,static_cast<LpValueType>(1.0));
                for( size_t v=0;v<numVar;++v){
                    OPENGM_ASSERT(cIndex<numConstraints);
                    const size_t gmVi=factor.variableIndex(v);
                    const size_t gmViLabel=walker.coordinateTuple()[v];
                    const LpIndexType lpVarFromVar=this->variableLabelToLpVariable(gmVi,gmViLabel);
                    OPENGM_ASSERT(lpVarFromVar<this->numberOfLpVariablesFromVariables());
                    // constraint that lpVar (from factor) must match lpVar (fromVariable)
                    factorMustMatchVariable.insert(lpVarFromVar,static_cast<LpValueType>(-1.0));
                    
                }
                constraintMatrix_->appendRow(factorMustMatchVariable);
                // example for a factor with 3 variables
                // -2 <= 3(fvar) -v1 -v3-v3 <=0
                // if all var lp var are active fvar must also be active
                lowerBounds_[cIndex]=static_cast<LpValueType>(-1.0*numVar+1.0);
                upperBounds_[cIndex]=static_cast<LpValueType>(0.0);
                ++cIndex;
            }
            OPENGM_ASSERT(cIndex<numConstraints);
            constraintMatrix_->appendRow(sumStatesMustBeOne);
            lowerBounds_[cIndex]=static_cast<LpValueType>(1.0);
            upperBounds_[cIndex]=static_cast<LpValueType>(1.0);
            ++cIndex;
        }
        
    }
    std::cout<<"cindex="<<cIndex<<" numC="<<this->numberOfBasicConstraints()<<"\n";
    OPENGM_ASSERT(cIndex==this->numberOfBasicConstraints());
    // load problem
    solverPtr_->loadProblem(*constraintMatrix_, varLowerBounds_, varUpperBounds_, objectivePtr_, lowerBounds_, upperBounds_);

}


template<class GM,class ACC>
inline std::string 
LpCoinOrOsi<GM,ACC>::name() const{
    return "Lp-CoinOr-Osi";
}

template<class GM,class ACC>
inline const typename LpCoinOrOsi<GM,ACC>::GraphicalModelType & 
LpCoinOrOsi<GM,ACC>::graphicalModel() const{
    return gm_;
}
template<class GM,class ACC>
inline void 
LpCoinOrOsi<GM,ACC>::reset(){
    throw RuntimeError("reset is not yet implemented");
}

template<class GM,class ACC>
inline InferenceTermination 
LpCoinOrOsi<GM,ACC>::infer(){
    //throw RuntimeError("infer is not yet implemented"); 
    float a;
    return this->infer(a)  ;
}


template<class GM,class ACC>
inline void
LpCoinOrOsi<GM,ACC>::setUpParameters(){
    
        
    //INT PARAMETERS
    {
        typedef IntegerParameterMapType MapType;
        const MapType & paramMap=param_.intParamMap_;
        typedef typename MapType::const_iterator MapIteratorType;
        for(MapIteratorType iter=paramMap.begin();iter!=paramMap.end();++iter)
             solverPtr_->setIntParam( iter->first, iter->second);
    }
    //DOUBLE PARAMETERS
    {
        typedef DoubleParameterMapType MapType;
        const MapType & paramMap=param_.doubleParamMap_;
        typedef typename MapType::const_iterator MapIteratorType;
        for(MapIteratorType iter=paramMap.begin();iter!=paramMap.end();++iter)
             solverPtr_->setDblParam( iter->first, iter->second);
    }
    //STR PARAMETERS
    {
        typedef StringParameterMapType MapType;
        const MapType & paramMap=param_.stringParamMap_;
        typedef typename MapType::const_iterator MapIteratorType;
        for(MapIteratorType iter=paramMap.begin();iter!=paramMap.end();++iter){
             solverPtr_->setStrParam( iter->first, iter->second);
        }
    }
    //HINT PARAMETERS
    {
        typedef HintParameterMapType MapType;
        const MapType & paramMap=param_.hintParamMap_;
        typedef typename MapType::const_iterator MapIteratorType;
        for(MapIteratorType iter=paramMap.begin();iter!=paramMap.end();++iter)
             solverPtr_->setHintParam( iter->first, iter->second.value_,iter->second.hintStrength_);
    }
    // INTEGER CONSTRAINT
    if(this->param_.integerConstraint_||true){
        for(LpIndexType lpvar=0;lpvar<this->numberOfLpVariables();++lpvar){
             solverPtr_->setInteger(int(lpvar));
        }
    }
}
template<class GM,class ACC>
template<class VISITOR>
inline InferenceTermination 
LpCoinOrOsi<GM,ACC>::infer(VISITOR&){
    this->setUpParameters();
    solverPtr_->initialSolve();
    return NORMAL;
}

template<class GM,class ACC>
inline void 
LpCoinOrOsi<GM,ACC>::setStartingPoint
(
    typename std::vector<typename LpCoinOrOsi<GM,ACC>::LabelType>::const_iterator startingPointBegin
){
    typedef unsigned char WarmStartValueType;
    typedef CoinWarmStartVector<WarmStartValueType> WarmStartVectorType;
    //allocate raw pointer array for coin osi warm start vector
    WarmStartValueType * lpWarmStartPtr = new WarmStartValueType[this->numberOfLpVariables()];
    this->gmLabelingToLpLabeling(startingPointBegin,lpWarmStartPtr);
    WarmStartVectorType * coinWarmStartVector= new WarmStartVectorType(this->numberOfLpVariables(),lpWarmStartPtr);
    solverPtr_->setWarmStart(coinWarmStartVector);
}

template<class GM,class ACC>
InferenceTermination 
LpCoinOrOsi<GM,ACC>::arg(
    std::vector<LabelType>& arg, 
    const size_t numarg 
) const{
    
    // Check the solution
    if ( solverPtr_->isProvenOptimal() || true) {
        if ( solverPtr_->isProvenOptimal()){
            std::cout << "Found optimal solution!" << std::endl;
            std::cout << "Objective value is " << solverPtr_->getObjValue() << std::endl;
        }
        else{
            std::cout << "Didn’t find optimal solution." << std::endl;
            // Check other status functions. What happened?
            if (solverPtr_->isProvenPrimalInfeasible())
                std::cout << "-Problem is proven to be infeasible." << std::endl;
            if (solverPtr_->isProvenDualInfeasible())
                std::cout << "-Problem is proven dual infeasible." << std::endl;
            if (solverPtr_->isIterationLimitReached())
                std::cout << "-Reached iteration limit." << std::endl;
        }
        // Examine solution
        int n = solverPtr_->getNumCols();
        const double *solution;
        solution = solverPtr_->getColSolution();
        arg.resize(gm_.numberOfVariables());
        this->lpLabelingToGmLabeling(solution,arg.begin(),MaxLpVarFromVar,param_.integerConstraint_);
    }
    else{
        std::cout << "Didn’t find optimal solution." << std::endl;
        // Check other status functions. What happened?
        if (solverPtr_->isProvenPrimalInfeasible())
            std::cout << "-Problem is proven to be infeasible." << std::endl;
        if (solverPtr_->isProvenDualInfeasible())
            std::cout << "-Problem is proven dual infeasible." << std::endl;
        if (solverPtr_->isIterationLimitReached())
            std::cout << "-Reached iteration limit." << std::endl;
    }
    return NORMAL;

}

} // end namespace opengm
#endif

