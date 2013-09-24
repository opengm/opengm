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

   //std::cout<<"add var 1.\n";
   // ADD VARIABLES TO LP SOLVER
   lpSolver_.addVariables(this->numberOfNodeLpVariables(),
      param_.integerConstraint_ ? LpSolverType::Binary : LpSolverType::Continous, 0.0,1.0
   );
   //std::cout<<"add var ho.\n";
   lpSolver_.addVariables(this->numberOfFactorLpVariables(),
      param_.integerConstraintFactorVar_ ? LpSolverType::Binary : LpSolverType::Continous, 0.0,1.0
   );
   //std::cout<<"add var finished\n";
   lpSolver_.addVarsFinished();
   OPENGM_CHECK_OP(this->numberOfLpVariables(),==,lpSolver_.numberOfVariables(),"");

   // SET UP OBJECTIVE AND UPDATE MODEL (SINCE OBJECTIVE CHANGED)
   //std::cout<<"setupLPObjective.\n";
   this->setupLPObjective();  
   //std::cout<<"setupLPObjectiveDone.\n";
   lpSolver_.setObjectiveFinished();

   // ADD CONSTRAINTS 
   //std::cout<<"addConstraints.\n";
   this->addFirstOrderRelaxationConstraints();
   lpSolver_.updateConstraints();
   //std::cout<<"setupConstraintsDone\n";
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
LPInference<GM,ACC,LP_SOLVER>::addFirstOrderRelaxationConstraints(){

   // set constraints
   UInt64Type constraintCounter = 0;
   // \sum_i \mu_i = 1
   for(IndexType node = 0; node < gm_.numberOfVariables(); ++node) {
      lpSolver_.addConstraint(1.0,1.0);
      for(LabelType l = 0; l < gm_.numberOfLabels(node); ++l) {
         lpSolver_.addToConstraint(constraintCounter,this->lpNodeVi(node,l),1.0);
      }
      ++constraintCounter;
   }
   // \sum_i \mu_{f;i_1,...,i_n} - \mu{b;j}= 0
   for(IndexType f = 0; f < gm_.numberOfFactors(); ++f) {
      if(gm_[f].numberOfVariables() > 1) {

         marray::Marray<UInt64Type> temp(gm_[f].shapeBegin(), gm_[f].shapeEnd());
         UInt64Type counter = this->lpFactorVi(f,0);
         for(marray::Marray<UInt64Type>::iterator mit = temp.begin(); mit != temp.end(); ++mit) {
            *mit = counter++;
         }

         for(IndexType n = 0; n < gm_[f].numberOfVariables(); ++n) {
            IndexType node = gm_[f].variableIndex(n);
            for(LabelType l=0; l < gm_.numberOfLabels(node); ++l) {
               lpSolver_.addConstraint(0.0,0.0);
               lpSolver_.addToConstraint(constraintCounter,this->lpNodeVi(node,l),-1.0);
               marray::View<UInt64Type> view = temp.boundView(n, l); 
               for(marray::View<UInt64Type>::iterator vit = view.begin(); vit != view.end(); ++vit) {
                  OPENGM_CHECK_OP(*vit,>=,this->lpFactorVi(f,0)," ");
                  lpSolver_.addToConstraint(constraintCounter,*vit,1.0);
               }
               ++constraintCounter;
            }
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
   for(IndexType gmVi=0;gmVi<gm_.numberOfVariables();++gmVi){
      const LabelType nLabels = gm_.numberOfLabels(gmVi);

      LpValueType maxVal      = lpSolver_.lpArg(this->lpNodeVi(gmVi,0));
      LabelType   maxValLabel = 0;

      for(LabelType l=1;l<nLabels;++l){
         const LabelType val =lpSolver_.lpArg(this->lpNodeVi(gmVi,l));
         OPENGM_CHECK_OP(val,<=,1.0,"");
         OPENGM_CHECK_OP(val,>=,0.0,"");
         if(val>maxVal){
            maxValLabel=l;
            maxVal=val;
         }
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
