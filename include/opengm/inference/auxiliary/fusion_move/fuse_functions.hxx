#ifndef OPENGM_AUX_FUSE_FUNCTIONS_HXX
#endif  OPENGM_AUX_FUSE_FUNCTIONS_HXX

namespace opengm
{

namespace detail_fusion
{


template<class GM>
class FuseViewFunction
    : public FunctionBase<FuseViewFunction<GM>, typename GM::ValueType, typename GM::IndexType, typename GM::LabelType>
{
public:
    typedef typename GM::ValueType ValueType;
    typedef ValueType value_type;
    typedef typename GM::FactorType FactorType;
    typedef typename GM::OperatorType OperatorType;
    typedef typename GM::IndexType IndexType;
    typedef typename GM::LabelType LabelType;

    FuseViewFunction();

    FuseViewFunction(
        const FactorType &factor,
        const std::vector<LabelType> &argA,
        const std::vector<LabelType> &argB
    )
        :   factor_(&factor),
            argA_(&argA),
            argB_(&argB),
            iteratorBuffer_(factor.numberOfVariables())
    {

    }



    template<class Iterator>
    ValueType operator()(Iterator begin)const
    {
        for (IndexType i = 0; i < iteratorBuffer_.size(); ++i)
        {
            OPENGM_CHECK_OP(begin[i], < , 2, "");
            if (begin[i] == 0)
            {
                iteratorBuffer_[i] = argA_->operator[](factor_->variableIndex(i));
            }
            else
            {
                iteratorBuffer_[i] = argB_->operator[](factor_->variableIndex(i));
            }
        }
        return factor_->operator()(iteratorBuffer_.begin());
    }

    IndexType shape(const IndexType)const
    {
        return 2;
    }

    IndexType dimension()const
    {
        return iteratorBuffer_.size();
    }

    IndexType size()const
    {
        return std::pow(2, iteratorBuffer_.size());
    }

private:
    FactorType const *factor_;
    std::vector<LabelType>  const *argA_;
    std::vector<LabelType>  const *argB_;
    mutable std::vector<LabelType> iteratorBuffer_;
};


template<class GM>
class FuseViewFixFunction
    : public FunctionBase<FuseViewFixFunction<GM>, typename GM::ValueType, typename GM::IndexType, typename GM::LabelType>
{
public:
    typedef typename GM::ValueType ValueType;
    typedef ValueType value_type;
    typedef typename GM::FactorType FactorType;
    typedef typename GM::OperatorType OperatorType;
    typedef typename GM::IndexType IndexType;
    typedef typename GM::LabelType LabelType;

    FuseViewFixFunction();

    FuseViewFixFunction(
        const FactorType &factor,
        const std::vector<LabelType> &argA,
        const std::vector<LabelType> &argB
    )
        :   factor_(&factor),
            argA_(&argA),
            argB_(&argB),
            notFixedPos_(),
            iteratorBuffer_(factor.numberOfVariables())
    {
        for (IndexType v = 0; v < factor.numberOfVariables(); ++v)
        {
            const IndexType vi = factor.variableIndex(v);
            if (argA[vi] != argB[vi])
            {
                notFixedPos_.push_back(v);
            }
            else
            {
                iteratorBuffer_[v] = argA[vi];
            }
        }
    }



    template<class Iterator>
    ValueType operator()(Iterator begin)const
    {
        for (IndexType i = 0; i < notFixedPos_.size(); ++i)
        {
            const IndexType nfp = notFixedPos_[i];
            OPENGM_CHECK_OP(begin[i], < , 2, "");
            if (begin[i] == 0)
            {
                iteratorBuffer_[nfp] = argA_->operator[](factor_->variableIndex(nfp));
            }
            else
            {
                iteratorBuffer_[nfp] = argB_->operator[](factor_->variableIndex(nfp));
            }
        }
        return factor_->operator()(iteratorBuffer_.begin());
    }

    IndexType shape(const IndexType)const
    {
        return 2;
    }

    IndexType dimension()const
    {
        return notFixedPos_.size();
    }

    IndexType size()const
    {
        return std::pow(2, notFixedPos_.size());
    }

private:
    FactorType const *factor_;
    std::vector<LabelType>  const *argA_;
    std::vector<LabelType>  const *argB_;
    std::vector<IndexType> notFixedPos_;
    mutable std::vector<LabelType> iteratorBuffer_;
};


} // end namespace detail_fusion
} // end namespace opengm


#endif // OPENGM_AUX_FUSE_FUNCTIONS_HXX
