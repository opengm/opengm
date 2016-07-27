#pragma once
#ifndef OPENGM_DMC_HXX
#define OPENGM_DMC_HXX

#include <vector>
#include <string>
#include <iostream>

#include "opengm/opengm.hxx"
//#include "opengm/inference/visitors/visitor.hxx"
#include "opengm/inference/inference.hxx"
#include "opengm/inference/movemaker.hxx"
#include "opengm/datastructures/buffer_vector.hxx"

#include <opengm/graphicalmodel/space/simplediscretespace.hxx>
#include <opengm/functions/potts.hxx>
#include "opengm/inference/visitors/visitors.hxx"

namespace opengm {
  

template<class GM, class INF>
class DMC : public Inference<GM, typename INF::AccumulationType>
{
public:

    typedef typename INF::AccumulationType ACC;
    typedef ACC AccumulationType;
    typedef GM GraphicalModelType;
    OPENGM_GM_TYPE_TYPEDEFS;
    typedef typename INF::Parameter InfParam;
    typedef opengm::visitors::VerboseVisitor<DMC<GM,INF> > VerboseVisitorType;
    typedef opengm::visitors::EmptyVisitor<DMC<GM,INF> >  EmptyVisitorType;
    typedef opengm::visitors::TimingVisitor<DMC<GM,INF> > TimingVisitorType;

    class Parameter {
        public:

        Parameter(
            const ValueType threshold = ValueType(-0.000000001),
            const InfParam infParam = InfParam()
        )
        :   threshold_(threshold),
            infParam_(infParam){

        }

        ValueType threshold_;
        InfParam infParam_;
    };

    DMC(const GraphicalModelType&, const Parameter&);
    std::string name() const;
    const GraphicalModelType& graphicalModel() const;
    InferenceTermination infer();
    void reset();
    template<class VisitorType>
    InferenceTermination infer(VisitorType&);
    void setStartingPoint(typename std::vector<LabelType>::const_iterator);
    virtual InferenceTermination arg(std::vector<LabelType>&, const size_t = 1) const ;
    virtual ValueType value()const{
        assert(false);  // FIXME: the return of this function was missing, just added something arbitrary
        return ValueType();
    }

private:
    const GraphicalModelType& gm_;
    Parameter param_;

    ValueType value_;
    std::vector<LabelType> arg_;
};
  
template<class GM, class INF>
inline
DMC<GM, INF>::DMC
(
    const GraphicalModelType& gm,
    const Parameter& parameter
)
:   gm_(gm),
    param_(parameter),
    value_(),
    arg_(gm.numberOfVariables(), 0) {

}


      
template<class GM, class INF>
inline void
DMC<GM, INF>::reset()
{

}
   
template<class GM, class INF>
inline void 
DMC<GM,INF>::setStartingPoint
(
   typename std::vector<typename DMC<GM,INF>::LabelType>::const_iterator begin
) {
}
   
template<class GM, class INF>
inline std::string
DMC<GM, INF>::name() const
{
   return "DMC";
}

template<class GM, class INF>
inline const typename DMC<GM, INF>::GraphicalModelType&
DMC<GM, INF>::graphicalModel() const
{
   return gm_;
}
  
template<class GM, class INF>
inline InferenceTermination
DMC<GM,INF>::infer()
{
   EmptyVisitorType v;
   return infer(v);
}

  
template<class GM, class INF>
template<class VisitorType>
InferenceTermination DMC<GM,INF>::infer
(
   VisitorType& visitor
)
{
   
    visitor.begin(*this);


    LabelType lAA[2]={0, 0};
    LabelType lAB[2]={0, 1};

    // decomposition
    Partition<LabelType> ufd(gm_.numberOfVariables());
    for(size_t fi=0; fi< gm_.numberOfFactors(); ++fi){
        if(gm_[fi].numberOfVariables()==2){

            const ValueType val00  = gm_[fi](lAA);
            const ValueType val01  = gm_[fi](lAB);
            const ValueType weight = val01 - val00; 

            if(weight>param_.threshold_){
                const size_t vi0 = gm_[fi].variableIndex(0);
                const size_t vi1 = gm_[fi].variableIndex(1);
                ufd.merge(vi0, vi1);
            }
        }
        else{
            throw RuntimeError("wrong factor order for multicut");
        }
    }

    if(ufd.numberOfSets() == 1){
        //std::cout<<" all in one cc\n";
        // FALL BACK HERE!!!
        typedef typename INF:: template rebind<GM,ACC>::type OrgInf;
        typename OrgInf::Parameter orgInfParam(param_.infParam_); 
        OrgInf orgInf(gm_, orgInfParam);
        orgInf.infer();
        orgInf.arg(arg_);
        value_ = gm_.evaluate(arg_);
    }
    else {
        //std::cout<<" NOT all in one cc\n";
        std::map<LabelType, LabelType> repr;
        ufd.representativeLabeling(repr);
        //std::cout<<"gm_.numVar "<<gm_.numberOfVariables()<<"\n";
        //std::cout<<"reprs size"<<repr.size()<<"\n";
        //std::cout<<"ufd.numberOfSets() "<<ufd.numberOfSets()<<"\n";
        std::vector< std::vector< LabelType> > subVar(ufd.numberOfSets());
        // set up the sub var
        for(size_t vi=0; vi<gm_.numberOfVariables(); ++vi){
            subVar[repr[ufd.find(vi)]].push_back(vi);
        }

        const size_t nSubProb = subVar.size();

        std::vector<unsigned char> usedFactors_(gm_.numberOfFactors(),0);

        // mark all factors where weight is smaller
        // as param_.threshold_ as used
        for(size_t fi=0; fi< gm_.numberOfFactors(); ++fi){
            if(
                ufd.find(gm_[fi].variableIndex(0)) 
                !=  
                ufd.find(gm_[fi].variableIndex(1))
            )
            {
                usedFactors_[fi] = 1;
            }
        }

        std::vector<IndexType> globalToLocal(gm_.numberOfVariables(), gm_.numberOfVariables()+1);

        IndexType offset = 0;
        for(size_t subProb = 0; subProb<nSubProb; ++subProb){
            
            //std::cout<<"subProb "<<subProb<<"\n";
            const IndexType nSubVar = subVar[subProb].size();
            //std::cout<<"nSubVar "<<nSubVar<<"\n";

            typedef PottsFunction<ValueType,IndexType,IndexType> Pf;
            typedef SimpleDiscreteSpace<IndexType, IndexType> Space;
            typedef GraphicalModel<ValueType, OperatorType, Pf , Space> Model;
            Space space(nSubVar, nSubVar);
            Model subGm(space);

            for(IndexType lvi=0; lvi<nSubVar; ++lvi){
                const IndexType gvi = subVar[subProb][lvi];
                globalToLocal[gvi] = lvi;
            }


            if(nSubVar==1){
                const IndexType gvi = subVar[subProb][0];
                arg_[gvi] = offset;
            }
            else if(nSubVar==2){
                const IndexType gvi0 = subVar[subProb][0];
                const IndexType gvi1 = subVar[subProb][1];
                arg_[gvi0] =     offset;
                arg_[gvi1] =     offset;
            }
            else{

                for(IndexType lvi=0; lvi<nSubVar; ++lvi){
                    const IndexType gvi = subVar[subProb][lvi];
                    OPENGM_CHECK_OP(lvi, == , globalToLocal[gvi], ' ');
                    // number of factors
                    const size_t nf = gm_.numberOfFactors(gvi);

                    for(size_t f=0; f<nf; ++f){
                        const IndexType nfi =  gm_.factorOfVariable(gvi, f);
                        if(usedFactors_[nfi] != 1){
                            usedFactors_[nfi] = 1;

                            // add factor to graphical model
                            const ValueType val00  = gm_[nfi](lAA);
                            const ValueType val01  = gm_[nfi](lAB);
                            const ValueType weight = val01 - val00; 
                            const IndexType vi0 = gm_[nfi].variableIndex(0);
                            const IndexType vi1 = gm_[nfi].variableIndex(1);

                            if( ufd.find(vi0) !=  ufd.find(vi1)){
                                OPENGM_CHECK_OP(ufd.find(vi0),!=,ufd.find(vi1), "internal error");
                            }

                            const IndexType lvis[] = {
                                std::min(globalToLocal[vi0],globalToLocal[vi1]),
                                std::max(globalToLocal[vi0],globalToLocal[vi1])
                            };
                            const Pf pf(nSubVar, nSubVar, 0.0, weight);
                            subGm.addFactor(subGm.addFunction(pf), lvis, lvis+2);
                        }
                    }
                }

                // infer the submodel
                typedef typename INF:: template rebind<Model,ACC>::type SubInf;
                typename SubInf::Parameter subInfParam(param_.infParam_); 
                SubInf subInf(subGm, subInfParam);
                subInf.infer();

                std::vector<LabelType> subArg(subGm.numberOfVariables());
                subInf.arg(subArg);

                for(IndexType lvi=0; lvi<nSubVar; ++lvi){
                    const IndexType gvi = subVar[subProb][lvi];
                    arg_[gvi] = subArg[lvi] + offset;
                }
            }

            offset += nSubVar;
        }
        value_ = gm_.evaluate(arg_);
        visitor.end(*this);

    }
    return NORMAL;
}

template<class GM, class INF>
inline InferenceTermination
DMC<GM,INF>::arg
(
      std::vector<LabelType>& x,
      const size_t N
) const
{
   if(N==1) {
      x.resize(gm_.numberOfVariables());
      for(size_t j=0; j<x.size(); ++j) {
         x[j] =arg_[j];
      }
      return NORMAL;
   }
   else {
      return UNKNOWN;
   }
}

} // namespace opengm

#endif // #ifndef OPENGM_DMC_HXX
