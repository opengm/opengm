#pragma once
#ifndef OPENGM_UNARY_LOSS_FUNCTION
#define OPENGM_UNARY_LOSS_FUNCTION

#include "opengm/functions/function_properties_base.hxx"

namespace opengm {









/// \endcond

/// UnaryLossFunction convert semi-ring in a lazy fashion
///
/// \ingroup functions
template<class T,class I, class L>
class UnaryLossFunction
: public FunctionBase<UnaryLossFunction<T,I,L>, T,I,L>
{
public:

   typedef T ValueType;
   typedef T value_type;
   typedef I IndexType;
   typedef L LabelType;


   enum LossType{
        HammingLoss = 0,
        LabelVectorConf = 1,
        LabelVectorGt = 2,
        LabelMatrix = 3,
        L1Loss = 4,
        L2Loss = 5
   };

   struct SharedMultiplers{
        marray::Marray<ValueType> labelMult_;
   };




    UnaryLossFunction(
        const LabelType numberOfLabels,
        const LabelType gtLabel,
        const LossType lossType, 
        const ValueType multiplier,
        const SharedMultiplers & sharedMultiplers,
        const bool owner
    );
    template<class Iterator> ValueType operator()(Iterator begin) const;
    IndexType shape(const IndexType) const;
    IndexType dimension() const;
    IndexType size() const;

private:
   LabelType numberOfLabels_;
   LabelType gtLabel_;
   LossType lossType_;
   ValueType multiplier_;
   const SharedMultiplers * sharedMultiplers_;
   bool owner_;
};

template<class T,class I, class L>
inline
UnaryLossFunction<T,I,L>::UnaryLossFunction(
    const LabelType numberOfLabels,
    const LabelType gtLabel,
    const LossType lossType, 
    const ValueType multiplier,
    const SharedMultiplers & sharedMultiplers,
    const bool owner
)
:   numberOfLabels_(numberOfLabels),
    gtLabel_(gtLabel),
    lossType_(lossType),
    multiplier_(multiplier),
    sharedMultiplers_(&sharedMultiplers),
    owner_(owner)
{

}

template<class T,class I, class L>
template<class Iterator>
inline typename UnaryLossFunction<T,I,L>::ValueType
UnaryLossFunction<T,I,L>::operator()
(
   Iterator begin
) const {

    const LabelType l = *begin;
    const ValueType isDifferent = (l != gtLabel_ ?  1.0 : 0.0);

    switch(lossType_){
        case HammingLoss:{
            return static_cast<ValueType>(-1.0) * multiplier_ * isDifferent;
        }
        case LabelVectorConf:{
            return multiplier_ * isDifferent * sharedMultiplers_->labelMult_(l);
        }
        case LabelVectorGt:{
            return multiplier_ * isDifferent * sharedMultiplers_->labelMult_(gtLabel_);
        }
        case LabelMatrix:{
            return multiplier_ * isDifferent * sharedMultiplers_->labelMult_(l, gtLabel_);
        }
        case L1Loss:{
            return multiplier_ * static_cast<ValueType>(std::abs(int(l)-int(gtLabel_)));
        }
        case L2Loss:{
            return multiplier_ * std::pow(int(l)-int(gtLabel_),2);
        }
        default :{
            throw RuntimeError("wrong loss type");
        }
    }
}

template<class T,class I, class L>
inline typename UnaryLossFunction<T,I,L>::IndexType
UnaryLossFunction<T,I,L>::shape
(
   const typename UnaryLossFunction<T,I,L>::IndexType index
) const{
   return numberOfLabels_;
}

template<class T,class I, class L>
inline typename UnaryLossFunction<T,I,L>::IndexType
UnaryLossFunction<T,I,L>::dimension() const {
   return 1;
}

template<class T,class I, class L>
inline typename UnaryLossFunction<T,I,L>::IndexType
UnaryLossFunction<T,I,L>::size() const {
   return numberOfLabels_;
}

} // namespace opengm

#endif // #ifndef OPENGM_UNARY_LOSS_FUNCTION
