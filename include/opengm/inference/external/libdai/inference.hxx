#ifndef OPENGM_LIBDAI_HXX
#define	OPENGM_LIBDAI_HXX

#include <vector>
#include <string>
#include <iostream>
#include <typeinfo>
#include <cmath>
#include <sstream>

#include "opengm/opengm.hxx"
#include "opengm/graphicalmodel/graphicalmodel.hxx"
#include "opengm/inference/inference.hxx"
#include "opengm/inference/visitors/visitors.hxx"
#include "opengm/operations/minimizer.hxx"
#include "opengm/operations/maximizer.hxx"
#include "opengm/operations/adder.hxx"
#include "opengm/operations/multiplier.hxx"
#include "opengm/operations/integrator.hxx"
#include "opengm/datastructures/marray/marray.hxx"
#include "opengm/datastructures/sparsemarray/sparsemarray.hxx"
#include "opengm/utilities/indexing.hxx"
#include "opengm/utilities/metaprogramming.hxx"

#include <dai/alldai.h>
#include <dai/exceptions.h>

/// \cond HIDDEN_SYMBOLS

namespace opengm{
namespace external{
namespace libdai{  
   
   template<class GM, class ACC ,class SOLVER>
   class LibDaiInference 
   {
   public:
      typedef ACC AccumulationType;
	   typedef GM GraphicalModelType;
	   OPENGM_GM_TYPE_TYPEDEFS;
      typedef opengm::visitors::VerboseVisitor< SOLVER > VerboseVisitorType;
      typedef opengm::visitors::TimingVisitor<  SOLVER > TimingVisitorType;
      typedef opengm::visitors::EmptyVisitor<   SOLVER > EmptyVisitorType;
      ~LibDaiInference();
      LibDaiInference(const GM & ,const  std::string &  ); 

	   virtual const GraphicalModelType& graphicalModel_impl() const;
      virtual void reset_impl();
      virtual InferenceTermination infer_impl();
      //template<class VISITOR>
      //InferenceTermination infer(VISITOR&);
	   virtual InferenceTermination arg_impl(std::vector<LabelType>& v, const size_t= 1)const;
      virtual InferenceTermination marginal_impl(const size_t, IndependentFactorType&) const;
      virtual InferenceTermination factorMarginal_impl(const size_t, IndependentFactorType&) const;
   protected:
      ::dai::FactorGraph * convert(const GM &);
      ::dai::FactorGraph * factorGraph_;
      ::dai::InfAlg * ia_;
      const GM & gm_;
      std::string stringAlgParam_;
      size_t numberOfExtraFactors_;
   };
   
   template<class GM, class ACC ,class SOLVER>
   inline LibDaiInference<GM,ACC,SOLVER>::~LibDaiInference() {
      delete ia_;
      delete factorGraph_;
   } 
   
   template<class GM, class ACC ,class SOLVER>
   inline LibDaiInference<GM,ACC,SOLVER>::LibDaiInference
   (
      const GM & gm,
      const  std::string & string_param 
   ): gm_(gm),
      stringAlgParam_(string_param),
      numberOfExtraFactors_(0) {
      factorGraph_=convert(gm_);
      ia_=dai::newInfAlgFromString(stringAlgParam_,*factorGraph_);
      ia_->init();
   }
   
   template<class GM, class ACC ,class SOLVER>
   inline const GM & 
   LibDaiInference<GM,ACC,SOLVER>::graphicalModel_impl()const{
      return gm_;
   }
   
   template<class GM, class ACC ,class SOLVER>
   inline void 
   LibDaiInference<GM,ACC,SOLVER>::reset_impl() {
      delete ia_;
      delete factorGraph_;
      factorGraph_=convert(gm_);
      ia_=dai::newInfAlgFromString(stringAlgParam_,*factorGraph_);
      ia_->init();
   };

   template<class GM, class ACC ,class SOLVER>
   inline InferenceTermination 
   LibDaiInference<GM,ACC,SOLVER>::infer_impl() {
      try{
         ia_->run();
         return opengm::NORMAL;
      }
      catch(const dai::Exception  & e) {
         std::stringstream ss;
         ss<<"libdai Error: "<<e.message(e.getCode())<<e.getMsg()<<"\n"<<e.getDetailedMsg();
         throw ::opengm::RuntimeError(ss.str());
      }
      catch(...) {
         return opengm::INFERENCE_ERROR;
      }
      return opengm::NORMAL;
   }
   
   

   
   template<class GM, class ACC ,class SOLVER>
   inline InferenceTermination 
   LibDaiInference<GM,ACC,SOLVER>::marginal_impl
   (
      const size_t variableIndex,
      IndependentFactorType & marginalFactor
   ) const{
      try{
         OPENGM_ASSERT(variableIndex<this->gm_.numberOfVariables());
         OPENGM_ASSERT(variableIndex<this->factorGraph_->nrVars());
         ::dai::Factor mf=this->ia_->belief(this->factorGraph_->var(variableIndex));
         OPENGM_ASSERT(mf.nrStates()==gm_.numberOfLabels(variableIndex));
         const size_t varIndex[]={variableIndex};
         marginalFactor.assign(gm_,varIndex,varIndex +1);
         for(size_t i=0;i<mf.nrStates();++i) {
            marginalFactor(i)=mf.get(i);
            if(   opengm::meta::Compare<typename GM::OperatorType,opengm::Adder>::value && opengm::meta::Compare<ACC,opengm::Minimizer>::value) {
               //back-transformation of f(x)=exp(-x);
               marginalFactor(i)=static_cast<ValueType>(-1.0*std::log(mf.get(i)));
            }
            else if(   opengm::meta::Compare<typename GM::OperatorType,opengm::Adder>::value && opengm::meta::Compare<ACC,opengm::Maximizer>::value) {
               //back-transformation of f(x)=exp(x);
               marginalFactor(i)=static_cast<ValueType>(std::log(mf.get(i)));
            }
            else if( opengm::meta::Compare<typename GM::OperatorType,opengm::Multiplier>::value && opengm::meta::Compare<ACC,opengm::Maximizer>::value) {
               //back-transformation of f(x)=x;
               marginalFactor(i)=static_cast<ValueType>(mf.get(i));
            }
            else if( opengm::meta::Compare<typename GM::OperatorType,opengm::Multiplier>::value && opengm::meta::Compare<ACC,opengm::Minimizer>::value) {
               if(mf.get(i)==0.0) {
                  throw opengm::RuntimeError("zero marginal Values with OP=opengm::Multiplier with ACC=Minimizer are not supported in the opengm- libdai interface ");
               }
               //back-transformation of f(x)=1/x;
               marginalFactor(i)=static_cast<ValueType>(1.0/mf.get(i));
            }
            else{
               throw opengm::RuntimeError("OP/ACC not supported in the opengm-libdai interface ");
            }
         }
         return opengm::NORMAL;
      }
      catch(const dai::Exception  & e) {
         std::stringstream ss;
         ss<<"libdai Error: "<<e.message(e.getCode())<<" "<<e.getMsg()<<"\n"<<e.getDetailedMsg();
         throw ::opengm::RuntimeError(ss.str());
      }
      catch(...) {
         return opengm::UNKNOWN;
      }
   }
   template<class GM, class ACC ,class SOLVER>
   inline InferenceTermination 
   LibDaiInference<GM,ACC,SOLVER>::factorMarginal_impl
   (
      const size_t factorIndex,
      IndependentFactorType & marginalFactor
   ) const{
      try{
         OPENGM_ASSERT(factorIndex<this->gm_.numberOfFactors());
         OPENGM_ASSERT(factorIndex<this->factorGraph_->nrFactors()-numberOfExtraFactors_);
         
         ::dai::VarSet varset;
         for(size_t v=0;v<gm_[factorIndex].numberOfVariables();++v) {
            varset.insert( ::dai::Var(gm_[factorIndex].variableIndex(v), gm_[factorIndex].numberOfLabels(v)) );
         }
         ::dai::Factor mf=this->ia_->belief(varset);
         marginalFactor.assign(gm_,gm_[factorIndex].variableIndicesBegin(),gm_[factorIndex].variableIndicesEnd());
         OPENGM_ASSERT(mf.nrStates()==marginalFactor.size());         
         for(size_t i=0;i<mf.nrStates();++i) {
            marginalFactor(i)=mf.get(i);
            if(   opengm::meta::Compare<typename GM::OperatorType,opengm::Adder>::value && opengm::meta::Compare<ACC,opengm::Minimizer>::value) {
               //back-transformation of f(x)=exp(-x);
               marginalFactor(i)=static_cast<ValueType>(-1.0*std::log(mf.get(i)));
            }
            else if(   opengm::meta::Compare<typename GM::OperatorType,opengm::Adder>::value && opengm::meta::Compare<ACC,opengm::Maximizer>::value) {
               //back-transformation of f(x)=exp(x);
               marginalFactor(i)=static_cast<ValueType>(std::log(mf.get(i)));
            }
            else if( opengm::meta::Compare<typename GM::OperatorType,opengm::Multiplier>::value && opengm::meta::Compare<ACC,opengm::Maximizer>::value) {
               //back-transformation of f(x)=x;
               marginalFactor(i)=static_cast<ValueType>(mf.get(i));
            }
            else if( opengm::meta::Compare<typename GM::OperatorType,opengm::Multiplier>::value && opengm::meta::Compare<ACC,opengm::Integrator>::value) {
               //back-transformation of f(x)=x;
               marginalFactor(i)=static_cast<ValueType>(mf.get(i));
            }
            else if( opengm::meta::Compare<typename GM::OperatorType,opengm::Multiplier>::value && opengm::meta::Compare<ACC,opengm::Minimizer>::value) {
               if(mf.get(i)==0.0) {
                  throw opengm::RuntimeError("zero marginal Values with OP=opengm::Multiplier with ACC=Minimizer are not supported in the opengm- libdai interface ");
               }
               //back-transformation of f(x)=1/x;
               marginalFactor(i)=static_cast<ValueType>(1.0/mf.get(i));
            }
            else{
               throw opengm::RuntimeError("OP/ACC not supported in the opengm-libdai interface ");
            }
         }
         return opengm::NORMAL;
      }
      catch(const dai::Exception  & e) {
         std::stringstream ss;
         ss<<"libdai Error: "<<e.message(e.getCode())<<" "<<e.getMsg()<<"\n"<<e.getDetailedMsg();
         throw ::opengm::RuntimeError(ss.str());
      }
      catch(...) {
         return opengm::UNKNOWN;
      }
   }
   
   template<class GM, class ACC ,class SOLVER>
   inline InferenceTermination 
   LibDaiInference<GM,ACC,SOLVER>::arg_impl
   (
      std::vector<typename LibDaiInference<GM,ACC,SOLVER>::LabelType>& v,
      const size_t n
   )const{
      //std::cout <<"LIBDAI ARG"<<std::endl;
      try{
         std::vector<size_t> states=ia_->findMaximum();
         v.assign(states.begin(),states.end());
         return opengm::NORMAL;
      }
      catch(const dai::Exception  & e) {
         std::stringstream ss;
         ss<<"libdai Error: "<<e.message(e.getCode())<<" "<<e.getMsg()<<"\n"<<e.getDetailedMsg();
         throw ::opengm::RuntimeError(ss.str());
      }
      catch(...) {
         return opengm::INFERENCE_ERROR;
      }
      return opengm::NORMAL;
   }
  
   template<class GM, class ACC ,class SOLVER>
   ::dai::FactorGraph *  LibDaiInference<GM,ACC,SOLVER>::convert
   (
      const GM & gm
   ) {
      const size_t nrOfFactors=gm.numberOfFactors();
      const size_t nrOfVariables=gm.numberOfVariables();
      typedef typename GM::ValueType ValueType;
      typedef double DaiValueType;

      std::vector< ::dai::Factor > factors;
      factors.reserve(nrOfFactors);
      std::vector<dai::Var> vars(nrOfVariables);
      for (size_t i = 0; i < nrOfVariables; ++i) {
         vars[i] = ::dai::Var(i, gm.numberOfLabels(i));
      }
      size_t maxFactorSize=0;
      size_t maxNumberOfVariables=0;
      for(size_t f=0;f<nrOfFactors;++f) {
         const size_t factorSize=gm[f].size();
         const size_t numberOfVariables=gm[f].numberOfVariables();
         if(factorSize>maxFactorSize) maxFactorSize=factorSize;
         if(numberOfVariables>maxNumberOfVariables) maxNumberOfVariables=numberOfVariables;
      }
      //buffer array for factor values
       DaiValueType * factorData= new DaiValueType[maxFactorSize];
      //buffer array for variables of a factor
       ::dai::Var * varSet = new ::dai::Var[maxNumberOfVariables];
      for(size_t f=0;f<nrOfFactors;++f) {
         //factor information
         const size_t factorSize=gm[f].size();
         const size_t numberOfVariables=gm[f].numberOfVariables();
         if(numberOfVariables==0) {
            std::cout<<"\n\n WARNING \n\n";
         }
         //copy the variables of a factor into the varset
         for(size_t v=0;v<numberOfVariables;++v) {
            varSet[v]=vars[gm[f].variableIndex(v)];
         }
         dai::VarSet varset(varSet, varSet + numberOfVariables);
         //marray view to the data for easy access
         marray::View<DaiValueType,false> 
            viewToFactorData(
               gm[f].shapeBegin(),
               gm[f].shapeEnd(),
               factorData,
               marray::LastMajorOrder,
               marray::LastMajorOrder
            );
         //fill factorData array with the data from the opengm factors
         opengm::ShapeWalker<typename GM::FactorType::ShapeIteratorType> walker(gm[f].shapeBegin(),numberOfVariables);
         for(size_t i=0;i<factorSize;++i) {
            //viewToFactorData(walker.coordinateTuple().begin())=
            if(   opengm::meta::Compare<typename GM::OperatorType,opengm::Adder>::value && 
                  opengm::meta::Compare<ACC,opengm::Minimizer>::value) {
               viewToFactorData(i)=std::exp(-1.0*
                  static_cast<DaiValueType>(gm[f](walker.coordinateTuple().begin())));
            }
            else if(   opengm::meta::Compare<typename GM::OperatorType,opengm::Adder>::value && 
                  opengm::meta::Compare<ACC,opengm::Maximizer>::value) {
               viewToFactorData(i)=std::exp(1.0*
                  static_cast<DaiValueType>(gm[f](walker.coordinateTuple().begin())));
            }
            else if( opengm::meta::Compare<typename GM::OperatorType,opengm::Multiplier>::value && 
                     opengm::meta::Compare<ACC,opengm::Maximizer>::value) {
               viewToFactorData(i)=
                  static_cast<DaiValueType>(gm[f](walker.coordinateTuple().begin()));
            }
            else if( opengm::meta::Compare<typename GM::OperatorType,opengm::Multiplier>::value && 
                     opengm::meta::Compare<ACC,opengm::Integrator>::value) {
               viewToFactorData(i)=
                  static_cast<DaiValueType>(gm[f](walker.coordinateTuple().begin()));
            }
            else if( opengm::meta::Compare<typename GM::OperatorType,opengm::Multiplier>::value && 
                     opengm::meta::Compare<ACC,opengm::Minimizer>::value) {
               if(gm[f](walker.coordinateTuple().begin())==static_cast<ValueType>(0.0)) {
                  throw opengm::RuntimeError("zero Values with OP=opengm::Multiplier with ACC=Minimizer are not supported in the opengm- libdai interface ");
               }
               viewToFactorData(i)=static_cast<DaiValueType>(1.0)/
                  static_cast<DaiValueType>(gm[f](walker.coordinateTuple().begin()));
            }
            else {
               throw opengm::RuntimeError("only build in OpenGM Operators and Accumulators are supported in the opengm- libdai interface ");
            }
            ++walker;
         }
         //add factor to the factor vector
         dai::Factor factor(varset, factorData);
         OPENGM_ASSERT(factor.nrStates()==gm[f].size());
         factors.push_back(factor);
      }
       dai::FactorGraph * factorGraph = new dai::FactorGraph(factors.begin(), factors.end(), vars.begin(), vars.end());
      delete [] factorData;
      delete [] varSet;
      OPENGM_ASSERT(factorGraph->nrFactors()==gm.numberOfFactors());
      OPENGM_ASSERT(factorGraph->nrVars()==gm.numberOfVariables());
      return factorGraph;
   }
} // end namespace libdai
} // end namespace external
} //end namespace opengm

/// \endcond

#endif // OPENGM_LIBDAI_HXX 
