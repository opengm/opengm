#pragma once
#ifndef OPENGM_LABEL_REDUCED_INFERENCE_HXX
#define OPENGM_LABEL_REDUCED_INFERENCE_HXX

#include <vector>
#include <string>
#include <iostream>

#include "opengm/opengm.hxx"
#include "opengm/inference/visitors/visitor.hxx"
#include "opengm/inference/inference.hxx"
#include "opengm/functions/function_properties_base.hxx"


namespace opengm {





template<class GM>
class LabelReductionFunction
: public FunctionBase<LabelReductionFunction<GM>, typename GM::ValueType, typename GM::IndexType, typename GM::LabelType> {
public:
    typedef typename GM::ValueType ValueType;
    typedef ValueType value_type;
    typedef typename GM::FactorType FactorType;
    typedef typename GM::OperatorType OperatorType;
    typedef typename GM::IndexType IndexType;
    typedef typename GM::LabelType LabelType;








    LabelReductionFunction();

    LabelReductionFunction(
        const FactorType & factor,
        const std::vector<LabelType> labelReduction
    )
    :   factor_(&factor),
        labelReduction_(labelReduction),
        iteratorBuffer_(factor.numberOfVariables())
    {

    }


    template<class Iterator>
    ValueType operator()(Iterator begin)const{
        for(IndexType v=0;v<factor_.dimension()){
            iteratorBuffer_[v]=labelReduction_[begin[v]];
        }
        return factor_(iteratorBuffer_.begin());
    }

    IndexType shape(const IndexType)const{
        return labelReduction_.size(); 
    }

    IndexType dimension()const{
        return factor_->numberOfVariables()
    }

    IndexType size()const{
        return std::pow(labelReduction.size(),factor_->dimension());   
    }

private:
   FactorType const* factor_;
   std::vector<LabelType> labelReduction_;  
   mutable std::vector<LabelType> iteratorBuffer_;
};






template<class INF>
class LabelReducedInference : 
    public Inference<typename INF::GraphicalModelType, typename INF::AccumulationType>
{
public:

    typedef typename INF::AccumulationType AccumulationType;
    typedef typename INF::GraphicalModelType GraphicalModelType;
    OPENGM_GM_TYPE_TYPEDEFS;
    typedef Movemaker<GraphicalModelType> MovemakerType;
    typedef LabelReducedInference<INF> SelfType;
    typedef VerboseVisitor< SelfType > VerboseVisitorType;
    typedef EmptyVisitor<   SelfType > EmptyVisitorType;
    typedef TimingVisitor<  SelfType > TimingVisitorType;


    // function types
    typedef LabelReductionFunction<GraphicalModelType> ReduceFunction;


    // sub gm
    typedef typename opengm::DiscreteSpace<IndexType, LabelType> SubSpaceType;
    typedef typename meta::TypeListGenerator< ReduceFunction >::type SubFunctionTypeList;
    typedef GraphicalModel<ValueType,OperatorType, SubFunctionTypeList,SubSpaceType> SubGmType;


    class Parameter {
    public:
        Parameter(
          const LabelType numberOfLabels
        )
        :   numberOfLabels_(numberOfLabels)
        {

        }

        LabelType numberOfLabels_;
    };
    
    LabelReducedInference(const GraphicalModelType&, const Parameter& = Parameter());
    std::string name() const;
    const GraphicalModelType& graphicalModel() const;
    InferenceTermination infer();
    template<class VisitorType>
    InferenceTermination infer(VisitorType&);
    void setStartingPoint(typename std::vector<LabelType>::const_iterator);
    virtual InferenceTermination arg(std::vector<LabelType>&, const size_t = 1) const ;

    ValueType value()const{

    }

    ValueType bound()const{

    }

private:
      const GraphicalModelType& gm_;
      Parameter param_;
};



template<class INF>
LabelReducedInference<INF>::LabelReducedInference
(
      const GraphicalModelType& gm,
      const Parameter& parameter
)
:  gm_(gm),
   param_(parameter)
{

}
      

   
template<class INF>
inline void 
LabelReducedInference<INF>::setStartingPoint
(
   typename std::vector<typename LabelReducedInference<INF>::LabelType>::const_iterator begin
) {

}
   
template<class INF>
inline std::string
LabelReducedInference<INF>::name() const
{
   return "LabelReducedInference";
}

template<class INF>
inline const typename LabelReducedInference<INF>::GraphicalModelType&
LabelReducedInference<INF>::graphicalModel() const
{
   return gm_;
}
  
template<class INF>
inline InferenceTermination
LabelReducedInference<INF>::infer()
{
   EmptyVisitorType v;
   return infer(v);
}

  
template<class INF>
template<class VisitorType>
InferenceTermination LabelReducedInference<INF>::infer
(
   VisitorType& visitor
)
{
    visitor.begin(*this,movemaker_.value(), movemaker_.value());
    /////////////////////////
    // INFERENCE CODE HERE //
    /////////////////////////



    SubGmType subGm(SubSpaceType(std::vector<LabelType>(gm_.numberOfVariables(),param_.numberOfLabels_)));

    const LabelType numberOfLabels=gm_.numberOfVariables(0);
    const float divsor = param_.numberOfLabels_ + 1 ;
    const float minLabel = float(numberOfLabels_) /divsor;

    LabelType l=0;
    std::vector<LabelType> labelReduction;
    labelReduction.reserve(param_.numberOfLabels_);
    while(l<param_.numberOfLabels_ l<numberOfLabels){
        labelReduction.push_back(LabelType(0.5f+minLabel*(l+1)));
        ++l;
    }

    for(IndexType fi=0;fi<gm_.numberOfVariables();fi){

        const ReduceFunction f(gm_[fi],labelReduction);
        subGm.addFactor(
            subGm.addFunction(f),
            subGm.variableIndicesBegin(),
            subGm.variableIndicesEnd()
        );
        

    }




    visitor.end(*this,movemaker_.value(), movemaker_.value());
    return NORMAL;
}

template<class INF>
inline InferenceTermination
LabelReducedInference<INF>::arg
(
      std::vector<LabelType>& x,
      const size_t N
) const
{

}

} // namespace opengm

#endif // #ifndef OPENGM_LABEL_REDUCED_INFERENCE_HXX
