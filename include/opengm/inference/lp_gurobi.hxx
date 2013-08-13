#pragma once
#ifndef OPENGM_GUROBI_HXX
#define OPENGM_GUROBI_HXX

#include <vector>
#include <string>
#include <sstream> 
#include <iostream>

#include "opengm/opengm.hxx"
#include "opengm/inference/visitors/visitor.hxx"
#include "opengm/inference/inference.hxx"
#include "opengm/datastructures/buffer_vector.hxx"




#include "gurobi_c++.h"



namespace opengm {
  



template<class GM>
void findUnariesFi(const GM & gm , std::vector<typename GM::IndexType> & unaryFi){
   typedef typename  GM::IndexType IndexType;

   const IndexType noUnaryFactorFound=gm.numberOfFactors();

   unaryFi.resize(gm.numberOfVariables());
   std::fill(unaryFi.begin(),unaryFi.end(),noUnaryFactorFound);

   for(IndexType fi=0;fi<gm.numberOfFactors();++fi){

      if(gm[fi].numberOfVariables()==1){
         const IndexType vi = gm[fi].variableIndex(0);
         OPENGM_CHECK_OP(unaryFi[vi],==,noUnaryFactorFound, " multiple unary factor found");
         unaryFi[vi]=fi;
      }
   }
}


template<class GM>
typename GM::IndexType findMaxFactorSize(const GM & gm){
   typedef typename  GM::IndexType IndexType;
   IndexType maxSize = 0 ;
   for(IndexType fi=0;fi<gm.numberOfFactors();++fi){
      maxSize = std::max(maxSize ,gm[fi].size());
   }
   return maxSize;
}



template<class GM>
typename GM::LabelType findMaxNumberOfLabels(const GM & gm){
   typedef typename  GM::LabelType LabelType;
   typedef typename  GM::IndexType IndexType;
   LabelType maxL = 0 ;
   for(IndexType vi=0;vi<gm.numberOfVariables();++vi){
      maxL = std::max(maxL ,gm.numberOfLabels(vi));
   }
   return maxL;
}



template<class GM, class ACC,class LP_SOLVER>
class LPGurobi : public Inference<GM, ACC>
{
public:

   enum Relaxation{
      FirstOrder,
      FirstOrder2
   };

   typedef ACC AccumulationType;
   typedef GM GraphicalModelType;
   OPENGM_GM_TYPE_TYPEDEFS;

   typedef LP_SOLVER LpSolverType;
   typedef typename LpSolverType::Parameter LpSolverParameter;

   typedef Movemaker<GraphicalModelType> MovemakerType;
   typedef VerboseVisitor<LPGurobi<GM,ACC,LP_SOLVER> > VerboseVisitorType;
   typedef EmptyVisitor<LPGurobi<GM,ACC,LP_SOLVER> > EmptyVisitorType;
   typedef TimingVisitor<LPGurobi<GM,ACC,LP_SOLVER> > TimingVisitorType;
   typedef opengm::ShapeWalker<typename GM::FactorType::ShapeIteratorType> FactorShapeWalkerType;

   typedef double LpValueType;
   typedef int LpIndexType;
   typedef double LpArgType;

   class Parameter {
   public:
      Parameter(
         const LpSolverParameter & lpSolverParamter= LpSolverParameter(),
         const Relaxation  relaxation = FirstOrder
      )
      :  lpSolverParameter_(lpSolverParamter),
         relaxation_(relaxation){
      }

      LpSolverParameter lpSolverParameter_;
      Relaxation        relaxation_;
   };


   LPGurobi(const GraphicalModelType&, const Parameter& = Parameter());
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


   void setupLPObjective();


   void addFirstOrderRelaxationConstraints();
   void addFirstOrderRelaxationConstraints2();


   UInt64Type addVar(const ValueType obj);
   UInt64Type addNeutralVar();

   UInt64Type numberOfLpVariables()const;

   // get the lp variable indices
   UInt64Type lpNodeVi(const IndexType gmVi,const LabelType label)const;
   UInt64Type lpFactorVi(const IndexType gmFi,const UInt64Type labelIndex)const;
   template<class LABELING_ITERATOR>
   UInt64Type lpFactorVi(const IndexType gmFi,  LABELING_ITERATOR labelingBegin, LABELING_ITERATOR labelingEnd)const;





   template<class LPVariableIndexIterator,class CoefficientIterator>
   void addConstraint(LPVariableIndexIterator , LPVariableIndexIterator , CoefficientIterator ,const ValueType , const ValueType   , const std::string & name = std::string() );





private:



      const GraphicalModelType& gm_;
      Parameter param_;
      

      // new !!
      LpSolverType lpSolver_;

      // gurobi member vars
      std::vector<UInt64Type> nodeVarIndex_;
      std::vector<UInt64Type> factorVarIndex_;
      std::vector<IndexType> unaryFis_;

      std::vector<LabelType> gmArg_;
};




template<class GM, class ACC, class LP_SOLVER>
LPGurobi<GM,ACC,LP_SOLVER>::LPGurobi
(
      const GraphicalModelType& gm,
      const Parameter& parameter
)
:  gm_(gm),
   param_(parameter),
   lpSolver_(parameter.lpSolverParameter_), 
   nodeVarIndex_(gm.numberOfVariables()),
   factorVarIndex_(gm.numberOfFactors()),
   unaryFis_(),
   gmArg_(gm.numberOfVariables(),static_cast<LabelType>(0) )
{
   this->setupLPObjective();  
   lpSolver_.updateModel();
   if (param_.relaxation_=FirstOrder)
      this->addFirstOrderRelaxationConstraints();
   if (param_.relaxation_=FirstOrder2)
      this->addFirstOrderRelaxationConstraints2();
   lpSolver_.updateModel();
}


template<class GM, class ACC, class LP_SOLVER>
void
LPGurobi<GM,ACC,LP_SOLVER>::setupLPObjective()
{




   // count the number of lp variables
   // - from nodes 
   // - from factors

   UInt64Type numNodeLpVar   = 0;
   UInt64Type numFactorLpVar = 0;
   UInt64Type numLpVar       = 0;

   for(IndexType vi=0;vi<gm_.numberOfVariables();++vi){
      numNodeLpVar+=gm.numberOfLabels(vi);
   }
   for(IndexType fi=0;fi<gm_.numberOfFactors();++fi){
      if (gm_[fi].numberOfVariables()>1){
         numFactorLpVar+=gm_[fi].size();
      }
   }


   // allocate space for the objective
   




   // find all varible which have unaries
   const IndexType noUnaryFactorFound=gm_.numberOfFactors();
   // (will raise error if a variable has multiple unaries)
   findUnariesFi(gm_,unaryFis_);


   // max "value-table" size of factors
   const IndexType maxFactorSize = findMaxFactorSize(gm_);

   // buffer can store the "value-table" of any factor 
   ValueType * factorValBuffer = new ValueType[maxFactorSize];


   // add node variables to lp
   IndexType lpNodeVi=0;
   for(IndexType gmVi = 0 ; gmVi<gm_.numberOfVariables();++gmVi){
      // start index for lp variables for this gmVi
      nodeVarIndex_[gmVi]=lpNodeVi;
      const bool hasUnary = unaryFis_[gmVi]!=noUnaryFactorFound;

      // if there is a unary factor for this variable of the graphical model
      if(hasUnary){
         // copy value table of factor into buffer
         gm_[unaryFis_[gmVi]].copyValues(factorValBuffer);
         for(LabelType label=0;label<gm_.numberOfLabels(gmVi);++label){
            addVar(factorValBuffer[label]);
            ++lpNodeVi;
         }
      }
      // if there is no unary factor for this variable we still add a varible
      // with a neutral objective
      else{
         for(LabelType label=0;label<gm_.numberOfLabels(gmVi);++label){
            addNeutralVar();
            ++lpNodeVi;
         }
      }
   }


   // add factor variables to lp
   IndexType lpFactorVi=lpNodeVi;
   for(IndexType gmFi = 0; gmFi<gm_.numberOfFactors();++gmFi){
      const IndexType numVar = gm_[gmFi].numberOfVariables();

      if(numVar == 1){
         // if the factor is of order 1 we have already added a lp var 
         // (within the "node variables" of a factor)
         const IndexType vi0 = gm_[gmFi].variableIndex(0);
         factorVarIndex_[gmFi]=nodeVarIndex_[vi0];
      }
      else{
         // start index for lp variables for this gmVi
         factorVarIndex_[gmFi]=lpFactorVi;

         // copy value table of factor into buffer
         gm_[gmFi].copyValues(factorValBuffer);

         for(LabelType labelingIndex=0;labelingIndex<gm_[gmFi].size();++labelingIndex){   
            addVar(factorValBuffer[labelingIndex]);
            ++lpFactorVi;
         }
      }
   }

   // delete buffer which stored the "value-table" of any factor 
   delete[] factorValBuffer;
}




template<class GM, class ACC, class LP_SOLVER>
inline UInt64Type
LPGurobi<GM,ACC,LP_SOLVER>::lpNodeVi(
   const typename LPGurobi<GM,ACC,LP_SOLVER>::IndexType gmVi,
   const typename LPGurobi<GM,ACC,LP_SOLVER>::LabelType label
) const {
   return nodeVarIndex_[gmVi]+label;
}


template<class GM, class ACC, class LP_SOLVER>
inline UInt64Type
LPGurobi<GM,ACC,LP_SOLVER>::lpFactorVi(
   const typename LPGurobi<GM,ACC,LP_SOLVER>::IndexType gmFi,
   const UInt64Type labelIndex
) const {
   return factorVarIndex_[gmFi]+labelIndex;
}


template<class GM, class ACC, class LP_SOLVER>
template<class LABELING_ITERATOR>
inline UInt64Type 
LPGurobi<GM,ACC,LP_SOLVER>::lpFactorVi
(
   const typename LPGurobi<GM,ACC,LP_SOLVER>::IndexType factorIndex,
   LABELING_ITERATOR labelingBegin,
   LABELING_ITERATOR labelingEnd
)const{
   OPENGM_ASSERT(factorIndex<gm_.numberOfFactors());
   OPENGM_ASSERT(std::distance(labelingBegin,labelingEnd)==gm_[factorIndex].numberOfVariables());
   const size_t numVar=gm_[factorIndex].numberOfVariables();
   size_t labelingIndex=labelingBegin[0];
   size_t strides=gm_[factorIndex].numberOfLabels(0);
   for(size_t vi=1;vi<numVar;++vi){
      OPENGM_ASSERT(labelingBegin[vi]<gm_[factorIndex].numberOfLabels(vi));
      labelingIndex+=strides*labelingBegin[vi];
      strides*=gm_[factorIndex].numberOfLabels(vi);
   }
   return factorVarIndex_[factorIndex]+labelingIndex;
}




template<class GM, class ACC, class LP_SOLVER>
UInt64Type 
inline LPGurobi<GM,ACC,LP_SOLVER>::numberOfLpVariables()const{
   return lpSolver_.numberOfVariables();
}

template<class GM, class ACC, class LP_SOLVER>
UInt64Type
LPGurobi<GM,ACC,LP_SOLVER>::addNeutralVar(){
   lpSolver_.addVariable(0.0,1.0,0.0);
}


template<class GM, class ACC, class LP_SOLVER>
UInt64Type
LPGurobi<GM,ACC,LP_SOLVER>::addVar(
   const typename LPGurobi<GM,ACC,LP_SOLVER>::ValueType obj
){

   if(opengm::meta::Compare<OperatorType,opengm::Adder>::value){
      if(opengm::meta::Compare<ACC,opengm::Minimizer>::value)
         lpSolver_.addVariable(0.0,1.0,obj);
      else if(opengm::meta::Compare<ACC,opengm::Maximizer>::value)
         lpSolver_.addVariable(0.0,1.0,-1.0*obj);
      else
         throw RuntimeError("Wrong Accumulator");
   }
   else if(opengm::meta::Compare<OperatorType,opengm::Multiplier>::value){

      OPENGM_CHECK_OP(obj,>,0.0, "LpInterface with Multiplier as operator does not support objective<=0 ");

      if(opengm::meta::Compare<ACC,opengm::Minimizer>::value)
         lpSolver_.addVariable(0.0,1.0,std::log(obj));
      else if(opengm::meta::Compare<ACC,opengm::Maximizer>::value)
         lpSolver_.addVariable(0.0,1.0,-1.0*std::log(obj));
      else
         throw RuntimeError("Wrong Accumulator");
   }
   else
      throw RuntimeError("Wrong Operator");
}

template<class GM, class ACC, class LP_SOLVER>
inline typename GM::ValueType 
LPGurobi<GM,ACC,LP_SOLVER>::bound() const {
   if(opengm::meta::Compare<OperatorType,opengm::Adder>::value){
      if(opengm::meta::Compare<ACC,opengm::Minimizer>::value)
         return static_cast<ValueType>(lpSolver_.lpValue());
      else if(opengm::meta::Compare<ACC,opengm::Maximizer>::value)
         return -1.0*static_cast<ValueType>(lpSolver_.lpValue());
      else
         throw RuntimeError("Wrong Accumulator");
   }
   else if(opengm::meta::Compare<OperatorType,opengm::Multiplier>::value){
      if(opengm::meta::Compare<ACC,opengm::Minimizer>::value)
         return static_cast<ValueType>(std::exp(lpSolver_.lpValue()));
      else if(opengm::meta::Compare<ACC,opengm::Maximizer>::value)
         return static_cast<ValueType>(std::exp(-1.0*lpSolver_.lpValue()));
      else
         throw RuntimeError("Wrong Accumulator");
   }
   else
      throw RuntimeError("Wrong Operator");
}




template<class GM, class ACC, class LP_SOLVER>
template<class LPVariableIndexIterator,class CoefficientIterator>
void LPGurobi<GM,ACC,LP_SOLVER>::addConstraint(
      LPVariableIndexIterator lpVarBegin, 
      LPVariableIndexIterator lpVarEnd, 
      CoefficientIterator     coeffBegin,
      const LPGurobi<GM,ACC,LP_SOLVER>::ValueType   lowerBound, 
      const LPGurobi<GM,ACC,LP_SOLVER>::ValueType   upperBound, 
      const std::string & name 
){
   lpSolver_.addConstraint(lpVarBegin,lpVarEnd,coeffBegin,lowerBound,upperBound,name);
}


template<class GM, class ACC, class LP_SOLVER>
void
LPGurobi<GM,ACC,LP_SOLVER>::addFirstOrderRelaxationConstraints2(){

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
LPGurobi<GM,ACC,LP_SOLVER>::addFirstOrderRelaxationConstraints(){

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
LPGurobi<GM,ACC,LP_SOLVER>::infer()
{
   EmptyVisitorType v;
   return infer(v);
}

  
template<class GM, class ACC, class LP_SOLVER>
template<class VisitorType>
InferenceTermination LPGurobi<GM,ACC,LP_SOLVER>::infer
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
LPGurobi<GM,ACC,LP_SOLVER>::reset()
{
   throw RuntimeError("LPGurobi::reset() is not implemented yet");
}
   
template<class GM, class ACC, class LP_SOLVER>
inline void 
LPGurobi<GM,ACC,LP_SOLVER>::setStartingPoint
(
   typename std::vector<typename LPGurobi<GM,ACC,LP_SOLVER>::LabelType>::const_iterator begin
) {
  throw RuntimeError("setStartingPoint is not implemented for LPGurobi");
}
   
template<class GM, class ACC, class LP_SOLVER>
inline std::string
LPGurobi<GM,ACC,LP_SOLVER>::name() const
{
   return "LPGurobi";
}

template<class GM, class ACC, class LP_SOLVER>
inline const typename LPGurobi<GM,ACC,LP_SOLVER>::GraphicalModelType&
LPGurobi<GM,ACC,LP_SOLVER>::graphicalModel() const
{
   return gm_;
}


template<class GM, class ACC, class LP_SOLVER>
inline typename GM::ValueType 
LPGurobi<GM,ACC,LP_SOLVER>::value() const { 
   std::vector<LabelType> states;
   arg(states);
   return gm_.evaluate(states);
}




template<class GM, class ACC, class LP_SOLVER>
inline InferenceTermination
LPGurobi<GM,ACC,LP_SOLVER>::arg
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

#endif // #ifndef OPENGM_GUROBI_HXX
