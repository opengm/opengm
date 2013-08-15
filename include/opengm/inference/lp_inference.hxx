#pragma once
#ifndef OPENGM_GENERIC_LP_INFERENCE_HXX
#define OPENGM_GENERIC_LP_INFERENCE_HXX

#include <vector>
#include <string>
#include <sstream> 
#include <iostream>

#include "opengm/opengm.hxx"
#include "opengm/inference/visitors/visitor.hxx"
#include "opengm/inference/inference.hxx"
#include "opengm/datastructures/buffer_vector.hxx"


#include "lp_inference_base.hxx"




namespace opengm {
  




template<class GM, class ACC,class LP_SOLVER>
class LPInference : public LpInferenceBase<GM,ACC>,  public Inference<GM, ACC>
{
public:

   enum Relaxation { FirstOrder,FirstOrder2 };

   typedef ACC AccumulationType;
   typedef GM GraphicalModelType;
   OPENGM_GM_TYPE_TYPEDEFS;
   typedef opengm::ShapeWalker<typename FactorType::ShapeIteratorType> FactorShapeWalkerType;
   typedef VerboseVisitor<LPInference<GM,ACC,LP_SOLVER> >   VerboseVisitorType;
   typedef EmptyVisitor<LPInference<GM,ACC,LP_SOLVER> >     EmptyVisitorType;
   typedef TimingVisitor<LPInference<GM,ACC,LP_SOLVER> >    TimingVisitorType;

   typedef LP_SOLVER                        LpSolverType;
   typedef typename LpSolverType::Parameter LpSolverParameter;

   typedef double LpValueType;
   typedef int LpIndexType;
   typedef double LpArgType;


   



   class Parameter {
   public:
      Parameter(
         const bool integerConstraint = false,
         const bool integerConstraintFactorVar = false,
         const LpSolverParameter & lpSolverParamter= LpSolverParameter(),
         const Relaxation  relaxation = FirstOrder
      )
      :  
         integerConstraint_(integerConstraint),
         integerConstraintFactorVar_(integerConstraintFactorVar),
         lpSolverParameter_(lpSolverParamter),
         relaxation_(relaxation){
      }
      bool integerConstraint_;
      bool integerConstraintFactorVar_;
      LpSolverParameter lpSolverParameter_;
      Relaxation        relaxation_;
   };


   LPInference(const GraphicalModelType&, const Parameter& = Parameter());
   std::string name() const;
   const GraphicalModelType& graphicalModel() const;
   InferenceTermination infer();
   void reset();
   ValueType bound()const;
   ValueType value()const;
   template<class VisitorType>
   InferenceTermination infer(VisitorType&);
   void setStartingPoint(typename std::vector<LabelType>::const_iterator);
   virtual InferenceTermination arg(std::vector<LabelType>&, const size_t = 1) const ;

   template<class LPVariableIndexIterator,class CoefficientIterator>
   void addConstraint(LPVariableIndexIterator , LPVariableIndexIterator , CoefficientIterator ,const ValueType , const ValueType   , const std::string & name = std::string() );





private:

      void setupLPObjective();
      void addFirstOrderRelaxationConstraints();
      void addFirstOrderRelaxationConstraints2();


      const GraphicalModelType& gm_;
      Parameter param_;
      
      LpSolverType lpSolver_;
      std::vector<LabelType> gmArg_;
};




template<class GM, class ACC, class LP_SOLVER>
LPInference<GM,ACC,LP_SOLVER>::LPInference
(
      const GraphicalModelType& gm,
      const Parameter& parameter
)
:  LpInferenceBase<GM,ACC>(gm),
   gm_(gm),
   param_(parameter),
   lpSolver_(parameter.lpSolverParameter_),
   gmArg_(gm.numberOfVariables(),static_cast<LabelType>(0) )
{
   // ADD VARIABLES TO LP SOLVER
   lpSolver_.addVariables(this->numberOfNodeLpVariables(),
      param_.integerConstraint_ ? LpSolverType::Binary : LpSolverType::Continous, 0.0,1.0
   );
   lpSolver_.addVariables(this->numberOfFactorLpVariables(),
      param_.integerConstraintFactorVar_ ? LpSolverType::Binary : LpSolverType::Continous, 0.0,1.0
   );
   lpSolver_.updateObjective();
   // SET UP OBJECTIVE AND UPDATE MODEL (SINCE OBJECTIVE CHANGED)
   this->setupLPObjective();  
   lpSolver_.updateObjective();

   // ADD CONSTRAINTS 
   if (param_.relaxation_==FirstOrder)
      this->addFirstOrderRelaxationConstraints();
   if (param_.relaxation_==FirstOrder2)
      this->addFirstOrderRelaxationConstraints2();
   lpSolver_.updateConstraints();

   lpSolver_.setupFinished();
}


template<class GM, class ACC, class LP_SOLVER>
void
LPInference<GM,ACC,LP_SOLVER>::setupLPObjective()
{

   // max "value-table" size of factors
   // and buffer can store the "value-table" of any factor 
   const IndexType maxFactorSize = findMaxFactorSize(gm_);
   ValueType * factorValBuffer = new ValueType[maxFactorSize];


   // objective for lpNodeVariables
   for(IndexType vi = 0 ; vi<gm_.numberOfVariables();++vi){
      if(this->hasUnary(vi)){
         gm_[this->unaryFactorIndex(vi)].copyValues(factorValBuffer);
         for(LabelType label=0;label<gm_.numberOfLabels(vi);++label)
            lpSolver_.setObjective(this->lpNodeVi(vi,label),this->valueToMinSumValue(factorValBuffer[label]));
      }
      else{
         for(LabelType label=0;label<gm_.numberOfLabels(vi);++label)
            lpSolver_.setObjective(this->lpNodeVi(vi,label),0.0);
      }
   }

   // objective for lpFactorVariables
   for(IndexType fi = 0; fi<gm_.numberOfFactors();++fi){
      if(gm_[fi].numberOfVariables() > 1){
         gm_[fi].copyValues(factorValBuffer);
         for(LabelType labelingIndex=0;labelingIndex<gm_[fi].size();++labelingIndex)
            lpSolver_.setObjective(this->lpFactorVi(fi,labelingIndex),this->valueToMinSumValue(factorValBuffer[labelingIndex]));
      }
   }

   // delete buffer which stored the "value-table" of any factor 
   delete[] factorValBuffer;
}

template<class GM, class ACC, class LP_SOLVER>
inline typename GM::ValueType 
LPInference<GM,ACC,LP_SOLVER>::bound() const {
   return static_cast<ValueType>(this->valueFromMinSumValue(lpSolver_.lpValue()));
}

template<class GM, class ACC, class LP_SOLVER>
template<class LPVariableIndexIterator,class CoefficientIterator>
void LPInference<GM,ACC,LP_SOLVER>::addConstraint(
      LPVariableIndexIterator lpVarBegin, 
      LPVariableIndexIterator lpVarEnd, 
      CoefficientIterator     coeffBegin,
      const LPInference<GM,ACC,LP_SOLVER>::ValueType   lowerBound, 
      const LPInference<GM,ACC,LP_SOLVER>::ValueType   upperBound, 
      const std::string & name 
){
   lpSolver_.addConstraint(lpVarBegin,lpVarEnd,coeffBegin,lowerBound,upperBound,name);
}

template<class GM, class ACC, class LP_SOLVER>
void
LPInference<GM,ACC,LP_SOLVER>::addFirstOrderRelaxationConstraints2(){

   // find the max number of label for the graphical model
   const LabelType maxNumerOfLabels =  findMaxNumberOfLabels(gm_);
   std::vector<LpIndexType>   lpVarBuffer_(maxNumerOfLabels);
   std::vector<LpValueType>   valBuffer_(maxNumerOfLabels,1.0);

   // 1 equality constraint for each variable in the graphical model
   // that all lp variables related to this gm variable summ to 1
   for(IndexType vi=0;vi<gm_.numberOfVariables();++vi){
        const LabelType numLabels=gm_.numberOfLabels(vi);
        for (LabelType l=0;l<numLabels;++l){
            const LpIndexType lpVi=this->lpNodeVi(vi,l); 
            lpVarBuffer_[l]=lpVi; 
        }
        lpSolver_.addConstraint(lpVarBuffer_.begin(),lpVarBuffer_.begin()+numLabels,valBuffer_.begin(),1.0,1.0);
   }

   // constraints on high order factorslpVi
   for(IndexType fi=0;fi<gm_.numberOfFactors();++fi){
      const FactorType & factor=gm_[fi];
      const IndexType numVar=factor.numberOfVariables();
      if(numVar>1){
         // collect for each variables state all the factors lp var where 
         // a variable has a certain label to get the marginalization
         FactorShapeWalkerType walker(factor.shapeBegin(),numVar);
         const size_t factorSize=factor.size();
         FastSequence<LpIndexType,5> lpVars(numVar+1);
         FastSequence<LpIndexType,5> values(numVar+1); 
         FastSequence<LpIndexType,12> lpVars2(factorSize);
         FastSequence<LpIndexType,12> values2(factorSize); 

         for (size_t confIndex=0;confIndex<factorSize;++confIndex,++walker){


            const LpIndexType lpFactorVi=this->lpFactorVi(fi,confIndex);

            lpVars2[confIndex]=lpFactorVi;
            values2[confIndex]=1.0;
            lpVars[0]=lpFactorVi;
            values[0]=static_cast<LpValueType>(numVar);

            for( size_t v=0;v<numVar;++v){
               const LabelType gmLabel    = walker.coordinateTuple()[v];
               const LpIndexType lpNodeVi = this->lpNodeVi(factor.variableIndex(v),gmLabel);
               lpVars[v+1]=lpNodeVi;
               values[v+1]=static_cast<LpValueType>(-1.0);
            }
            lpSolver_.addConstraint(lpVars.begin(),lpVars.end(),values.begin(),
               static_cast<LpValueType>(-1.0)*static_cast<LpValueType>(numVar-1),static_cast<LpValueType>(0.0));
         }
         lpSolver_.addConstraint(lpVars2.begin(),lpVars2.end(),values2.begin(),
               static_cast<LpValueType>(1.0),static_cast<LpValueType>(1.0));
      }
   }
}


template<class GM, class ACC, class LP_SOLVER>
void
LPInference<GM,ACC,LP_SOLVER>::addFirstOrderRelaxationConstraints(){

   // find the max number of label for the graphical model
   const LabelType maxNumerOfLabels =  findMaxNumberOfLabels(gm_);
   std::vector<LpIndexType>   lpVarBuffer_(maxNumerOfLabels);
   std::vector<LpValueType>   valBuffer_(maxNumerOfLabels,1.0);

   // 1 equality constraint for each variable in the graphical model
   // that all lp variables related to this gm variable summ to 1
   for(IndexType vi=0;vi<gm_.numberOfVariables();++vi){
        const LabelType numLabels=gm_.numberOfLabels(vi);
        for (LabelType l=0;l<numLabels;++l){
            const LpIndexType lpVi=this->lpNodeVi(vi,l); 
            lpVarBuffer_[l]=lpVi; 
        }
        lpSolver_.addConstraint(lpVarBuffer_.begin(),lpVarBuffer_.begin()+numLabels,valBuffer_.begin(),1.0,1.0);
   }

   // constraints on high order factorslpVi
   for(IndexType fi=0;fi<gm_.numberOfFactors();++fi){
      const FactorType & factor=gm_[fi];
      const IndexType numVar=factor.numberOfVariables();
      if(numVar>1){

         // marginalization constraints
         size_t numC=0;
         opengm::FastSequence<size_t,5> localBegin(numVar);
         for(size_t v=0;v<numVar;++v){
            localBegin[v]=numC;
            numC+=factor.numberOfLabels(v);
         }
         opengm::FastSequence<  opengm::FastSequence<  LpIndexType  ,4 >  ,10>  marginalCLpVars(numC);
         opengm::FastSequence<  opengm::FastSequence<  LpValueType  ,4 >  ,10 > marginalCVals(numC);
         for(size_t v=0;v<numVar;++v){
            const LabelType numLabels=factor.numberOfLabels(v);
            for(LabelType l=0;l<numLabels;++l){
               size_t local=localBegin[v];
               const LpIndexType lpVi=this->lpNodeVi(factor.variableIndex(v),l);
               marginalCLpVars[localBegin[v]+l].push_back(lpVi);
               marginalCVals[localBegin[v]+l].push_back(1.0);
            }
         }

         // collect for each variables state all the factors lp var where 
         // a variable has a certain label to get the marginalization
         FactorShapeWalkerType walker(factor.shapeBegin(),numVar);
         const size_t factorSize=factor.size();
         for (size_t confIndex=0;confIndex<factorSize;++confIndex,++walker){
            const LpIndexType lpVi=this->lpFactorVi(fi,confIndex);
            for( size_t v=0;v<numVar;++v){
               const LabelType gmLabel=walker.coordinateTuple()[v];
               marginalCLpVars[localBegin[v]+gmLabel].push_back(lpVi);
               marginalCVals[localBegin[v]+gmLabel].push_back(-1.0);
            }
         }
         // marginalization constraints
         // For the LP, a first order local polytope approximation of the
         // marginal polytope is used, i.e. the affine instead of the convex 
         // hull.
         for(size_t c=0;c<marginalCLpVars.size();++c){
            lpSolver_.addConstraint(marginalCLpVars[c].begin(),marginalCLpVars[c].end(),marginalCVals[c].begin(),0.0,0.0);//, "c0");
         }
      }
   }
}

template<class GM, class ACC, class LP_SOLVER>
inline InferenceTermination
LPInference<GM,ACC,LP_SOLVER>::infer()
{
   EmptyVisitorType v;
   return infer(v);
}

template<class GM, class ACC, class LP_SOLVER>
template<class VisitorType>
InferenceTermination LPInference<GM,ACC,LP_SOLVER>::infer
(
   VisitorType& visitor
)
{
   visitor.begin();
   lpSolver_.optimize();
   for(IndexType gmVi=0,lpVi=0;gmVi<gm_.numberOfVariables();++gmVi){
      const LabelType nLabels = gm_.numberOfLabels(gmVi);
      LpValueType maxVal      = -1.0;
      LabelType   maxValLabel =  0.0;
      for(LabelType l=0;l<nLabels;++l){
         const LabelType val = lpSolver_.lpArg(lpVi);
         if(val>maxVal){
            maxValLabel=l;
            maxVal=val;
         }
         ++lpVi;
      }
      gmArg_[gmVi]=maxValLabel;
   }
   visitor.end();
   return NORMAL;
}
   
template<class GM, class ACC, class LP_SOLVER>
inline void
LPInference<GM,ACC,LP_SOLVER>::reset()
{
   throw RuntimeError("LPInference::reset() is not implemented yet");
}
   
template<class GM, class ACC, class LP_SOLVER>
inline void 
LPInference<GM,ACC,LP_SOLVER>::setStartingPoint
(
   typename std::vector<typename LPInference<GM,ACC,LP_SOLVER>::LabelType>::const_iterator begin
) {
  throw RuntimeError("setStartingPoint is not implemented for LPInference");
}
   
template<class GM, class ACC, class LP_SOLVER>
inline std::string
LPInference<GM,ACC,LP_SOLVER>::name() const
{
   return "LPInference";
}

template<class GM, class ACC, class LP_SOLVER>
inline const typename LPInference<GM,ACC,LP_SOLVER>::GraphicalModelType&
LPInference<GM,ACC,LP_SOLVER>::graphicalModel() const
{
   return gm_;
}

template<class GM, class ACC, class LP_SOLVER>
inline typename GM::ValueType 
LPInference<GM,ACC,LP_SOLVER>::value() const { 
   std::vector<LabelType> states;
   arg(states);
   return gm_.evaluate(states);
}

template<class GM, class ACC, class LP_SOLVER>
inline InferenceTermination
LPInference<GM,ACC,LP_SOLVER>::arg
(
      std::vector<LabelType>& x,
      const size_t N
) const
{
   if(N==1) {
      x.resize(gm_.numberOfVariables());
      for(size_t j=0; j<x.size(); ++j) {
         x[j]=gmArg_[j];
      }
      return NORMAL;
   }
   else {
      return UNKNOWN;
   }
}

} // namespace opengm

#endif // #ifndef OPENGM_GENERIC_LP_INFERENCE_HXX
