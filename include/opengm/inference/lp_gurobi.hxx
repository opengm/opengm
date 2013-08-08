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



template<class GM, class ACC>
class LPGurobi : public Inference<GM, ACC>
{
public:
   enum MoveType {
      SINGLE_VARIABLE = 0,
      FACTOR = 1
   };
   typedef ACC AccumulationType;
   typedef GM GraphicalModelType;
   OPENGM_GM_TYPE_TYPEDEFS;
   typedef Movemaker<GraphicalModelType> MovemakerType;
   typedef VerboseVisitor<LPGurobi<GM,ACC> > VerboseVisitorType;
   typedef EmptyVisitor<LPGurobi<GM,ACC> > EmptyVisitorType;
   typedef TimingVisitor<LPGurobi<GM,ACC> > TimingVisitorType;
   typedef opengm::ShapeWalker<typename GM::FactorType::ShapeIteratorType> FactorShapeWalkerType;

   typedef double LpValueType;
   typedef int LpIndexType;
   typedef double LpArgType;

   class Parameter {
   public:
      Parameter(
         const bool integerConstraint = true
      )
      : integerConstraint_(integerConstraint){
      }

      bool integerConstraint_;
   };


   LPGurobi(const GraphicalModelType&, const Parameter& = Parameter());
   std::string name() const;
   const GraphicalModelType& graphicalModel() const;
   InferenceTermination infer();
   void reset();
   template<class VisitorType>
   InferenceTermination infer(VisitorType&);
   void setStartingPoint(typename std::vector<LabelType>::const_iterator);
   virtual InferenceTermination arg(std::vector<LabelType>&, const size_t = 1) const ;


   void setupLPObjective();
   void addLpFirstOrderRelaxationConstraints();


   void addNodeVar(const IndexType gmVi,const LabelType label,const ValueType obj){
      std::stringstream ss;
      ss << "nv_" << gmVi<<"_"<<label;
      if(param_.integerConstraint_){ 
         lpVars_.push_back( grbModel_.addVar(0.0, 1.0, obj, GRB_BINARY, ss.str().c_str()));
      }
      else{
         lpVars_.push_back( grbModel_.addVar(0.0, 1.0, obj, GRB_CONTINUOUS, ss.str().c_str()));
      } 
   }

   void addFactorVar(const IndexType gmFi,const LabelType labelingIndex,const ValueType obj){
      std::stringstream ss;
      ss << "fv_" << gmFi<<"_"<<labelingIndex;
      if(param_.integerConstraint_){
         lpVars_.push_back( grbModel_.addVar(0.0, 1.0, obj, GRB_BINARY, ss.str().c_str()));
      }
      else{
         lpVars_.push_back( grbModel_.addVar(0.0, 1.0, obj, GRB_CONTINUOUS, ss.str().c_str()));
      }
   }


   const IndexType variableLabelToLpVariable(const IndexType gmVi,const LabelType label)const{
      return nodeVarIndex_[gmVi]+label;
   }

   const IndexType factorLabelingToLpVariable(const IndexType gmFi,const LabelType labelIndex)const{
      return factorVarIndex_[gmFi]+labelIndex;
   }


   //GRB.LESS_EQUAL, GRB.EQUAL, or GRB.GREATER_EQUAL


   template<class LPVariableIndexIterator,class CoefficientIterator>
   void addConstraint(
         LPVariableIndexIterator lpVarBegin, 
         LPVariableIndexIterator lpVarEnd, 
         CoefficientIterator     coeffBegin,
         const ValueType   lowerBound, 
         const ValueType   upperBound, 
         const std::string & name = std::string()
   ){

      if(upperBound-lowerBound < 0.000000001){

         GRBLinExpr linExp = new GRBLinExpr();
         while(lpVarBegin!=lpVarEnd){
            const LpIndexType lpVi  = static_cast<LpIndexType>(*lpVarBegin);
            const LpValueType coeff = static_cast<LpValueType>(*coeffBegin);
            linExp.addTerms(&coeff,&lpVars_[lpVi],1);
            ++lpVarBegin;
            ++coeffBegin;
         }
         if(name.size()>0){
            grbModel_.addConstr(linExp,GRB_EQUAL,static_cast<LpValueType>(lowerBound),name.c_str());
         }
         else{
            grbModel_.addConstr(linExp,GRB_EQUAL,static_cast<LpValueType>(lowerBound));
         }
      }
      else{
         GRBLinExpr linExpLower = new GRBLinExpr();
         GRBLinExpr linExpUpper = new GRBLinExpr();
         while(lpVarBegin!=lpVarEnd){
            const LpIndexType lpVi  = static_cast<LpIndexType>(*lpVarBegin);
            const LpValueType coeff = static_cast<LpValueType>(*coeffBegin);
            linExpLower.addTerms(&coeff,&lpVars_[lpVi],1);
            linExpUpper.addTerms(&coeff,&lpVars_[lpVi],1);
            ++lpVarBegin;
            ++coeffBegin;
         }
         if(name.size()>0){
            std::string nameLower = name + std::string("_lower");
            std::string nameUpper = name + std::string("_upper");
            grbModel_.addConstr(linExpLower,GRB_GREATER_EQUAL ,static_cast<LpValueType>(lowerBound),nameLower);
            grbModel_.addConstr(linExpUpper,GRB_LESS_EQUAL    ,static_cast<LpValueType>(upperBound),nameUpper);
         }
         else{
            grbModel_.addConstr(linExpLower,GRB_GREATER_EQUAL ,static_cast<LpValueType>(lowerBound));
            grbModel_.addConstr(linExpUpper,GRB_LESS_EQUAL    ,static_cast<LpValueType>(upperBound));
         }
      }
   }


private:
      const GraphicalModelType& gm_;
      Parameter param_;
      

      GRBEnv grbEnv_;// = GRBEnv();
      GRBModel grbModel_;// = GRBModel(env);

      std::vector<GRBVar> lpVars_;
      std::vector<IndexType> nodeVarIndex_;
      std::vector<IndexType> factorVarIndex_;
      std::vector<IndexType> unaryFis_;

      std::vector<LabelType> gmArg_;
      std::vector<LpArgType> lpArg_;


};




template<class GM, class ACC>
LPGurobi<GM, ACC>::LPGurobi
(
      const GraphicalModelType& gm,
      const Parameter& parameter
)
:  gm_(gm),
   param_(parameter),
   grbEnv_(),
   grbModel_(grbEnv_),
   lpVars_(),
   nodeVarIndex_(gm.numberOfVariables()),
   factorVarIndex_(gm.numberOfFactors()),
   unaryFis_(),
   gmArg_(gm.numberOfVariables()),
   lpArg_()
{
   this->setupLPObjective();  
   grbModel_.update();
   this->addLpFirstOrderRelaxationConstraints();
  
}

//GRBLinExpr expr = new GRBLinExpr();
//expr.addTerm(1.0, x); expr.addTerm(2.0, y); expr.addTerm(3.0, z);

template<class GM, class ACC>
void
LPGurobi<GM,ACC>::addLpFirstOrderRelaxationConstraints(){
   //GRBVar* gvars = grbModel_.getVars();
   
   const double val1 = 1.0;
   const double valM1=-1.0;
   // constraints on variables 
   for(IndexType vi=0;vi<gm_.numberOfVariables();++vi){
        const LabelType numLabels=gm_.numberOfLabels(vi);
        GRBLinExpr sumStatesMustBeOne =  GRBLinExpr();
        // 1 equality constraint that summ must be 1
        for (LabelType l=0;l<numLabels;++l){
            const LpIndexType lpVi=this->variableLabelToLpVariable(vi,l);
            sumStatesMustBeOne.addTerms(&val1,&lpVars_[lpVi],1); 
            //sumStatesMustBeOne.addTerms(&val1,&(gvars[0]),1);
        }
        //equality constraint
        grbModel_.addConstr(sumStatesMustBeOne,GRB_EQUAL,1.0); //Problem with this line 
 
   }
   // constraints on high order factors
   for(IndexType fi=0;fi<gm_.numberOfFactors();++fi){
      const FactorType & factor=gm_[fi];
      const IndexType numVar=factor.numberOfVariables();
      if(numVar>1){
         FactorShapeWalkerType walker(factor.shapeBegin(),numVar);
         const size_t factorSize=factor.size();
         // 1 constraints that summ must be 1
         GRBLinExpr sumStatesMustBeOne =  GRBLinExpr();

         // marginalization constraints
         size_t numC=0;
         opengm::FastSequence<size_t,5> localBegin(numVar);
         for(size_t v=0;v<numVar;++v){
            localBegin[v]=numC;
            numC+=factor.numberOfLabels(v);
         }
         std::vector<GRBLinExpr> marginalC(numC);
         //std::vector<size_t>     termCounter(numC,0);
         for(size_t c=0;c<numC;++c){
            marginalC[c]= GRBLinExpr();
         }

         for(size_t v=0;v<numVar;++v){
            const LabelType numLabels=factor.numberOfLabels(v);
            for(LabelType l=0;l<numLabels;++l){
               size_t local=localBegin[v];
               const LpIndexType lpVi=this->variableLabelToLpVariable(factor.variableIndex(v),l);
               marginalC[localBegin[v]+l].addTerms(&val1,&lpVars_[lpVi],1);
               //termCounter[localBegin[v]+l]+=1;
            }
         }
         // collect for each variables state all the factors lp var where 
         // a variable has a certain label to get the marginalization
         for (size_t confIndex=0;confIndex<factorSize;++confIndex,++walker){
             //OPENGM_ASSERT(cIndex<numConstraints);
             const LpIndexType lpVi=this->factorLabelingToLpVariable(fi,confIndex);
             sumStatesMustBeOne.addTerms(&val1,&lpVars_[lpVi],1);
             // loop over all labels of the variables this factor:
             for( size_t v=0;v<numVar;++v){
                  const LabelType gmLabel=walker.coordinateTuple()[v];
                  size_t local=localBegin[v];
                  // double val = -1.0 * double(termCounter[localBegin[v]+gmLabel]);
                  marginalC[local+gmLabel].addTerms(&valM1,&lpVars_[lpVi],1);
             }
         }
         // marginalization constraints
         // For the LP, a first order local polytope approximation of the
         // marginal polytope is used, i.e. the affine instead of the convex 
         // hull.
         for(size_t c=0;c<marginalC.size();++c){
            grbModel_.addConstr(marginalC[c], GRB_EQUAL, 0.0);//, "c0");
         }
         // constraint that all lp var. from 
         // factor must sum to 1
         grbModel_.addConstr(sumStatesMustBeOne,GRB_EQUAL,1.0);
      }
   }
}



template<class GM, class ACC>
void
LPGurobi<GM,ACC>::setupLPObjective()
{
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
         for(LabelType label=0;label<gm_.numberOfLabels(gmVi);++lpNodeVi,++label)
            addNodeVar(gmVi,label,factorValBuffer[label]);
      }
      // if there is no unary factor for this variable we still add a varible
      // with a neutral objective
      else{
         for(LabelType label=0;label<gm_.numberOfLabels(gmVi);++lpNodeVi,++label)
            addNodeVar(gmVi,label,0.0);
      }
   }


   // add factor variables to lp
   IndexType lpFactorVi=lpNodeVi;
   for(IndexType gmFi = 0,lpFactorVi=0 ; gmFi<gm_.numberOfFactors();++gmFi){
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

         for(LabelType labelingIndex=0;labelingIndex<gm_[gmFi].size();++lpFactorVi,++labelingIndex){   
            addFactorVar(gmFi,labelingIndex,factorValBuffer[labelingIndex]);
         }
      }
   }

   // delete buffer which stored the "value-table" of any factor 
   delete[] factorValBuffer;
}



template<class GM, class ACC>
inline InferenceTermination
LPGurobi<GM,ACC>::infer()
{
   EmptyVisitorType v;
   return infer(v);
}

  
template<class GM, class ACC>
template<class VisitorType>
InferenceTermination LPGurobi<GM,ACC>::infer
(
   VisitorType& visitor
)
{
   try{
      grbModel_.optimize();

      lpArg_.resize(lpVars_.size());

      for(IndexType lpVi=0;lpVi<lpVars_.size();++lpVi){
         const LpArgType arg = lpVars_[lpVi].get(GRB_DoubleAttr_X);
         lpArg_[arg];
       
      }

      for(IndexType gmVi=0,lpVi=0;gmVi<gm_.numberOfVariables();++gmVi){
         const LabelType nLabels = gm_.numberOfLabels(gmVi);
         LpValueType maxVal=-1.0;
         LabelType   maxValLabel = 0.0;

         for(LabelType l=0;l<nLabels;++l,++lpVi){
            const LabelType val = lpVars_[lpVi].get(GRB_DoubleAttr_X);
            if(val>maxVal){
               maxValLabel=l;
               maxVal=val;
            }
         }
         gmArg_[gmVi]=maxValLabel;
      }
   } 
   catch(GRBException e) {
      std::cout << "Error code = " << e.getErrorCode() << "\n";
      std::cout << e.getMessage() <<"\n";
      throw RuntimeError("Exception during gurobi optimization");
   } 
   catch(...) {
      throw RuntimeError("Exception during gurobi optimization");
   }
   return NORMAL;
}
   

      
template<class GM, class ACC>
inline void
LPGurobi<GM, ACC>::reset()
{

}
   
template<class GM, class ACC>
inline void 
LPGurobi<GM,ACC>::setStartingPoint
(
   typename std::vector<typename LPGurobi<GM,ACC>::LabelType>::const_iterator begin
) {
  throw RuntimeError("setStartingPoint is not implemented for LPGurobi");
}
   
template<class GM, class ACC>
inline std::string
LPGurobi<GM, ACC>::name() const
{
   return "LPGurobi";
}

template<class GM, class ACC>
inline const typename LPGurobi<GM, ACC>::GraphicalModelType&
LPGurobi<GM, ACC>::graphicalModel() const
{
   return gm_;
}
  


template<class GM, class ACC>
inline InferenceTermination
LPGurobi<GM,ACC>::arg
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
