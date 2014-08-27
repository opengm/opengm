#ifndef OPENGM_AUX_FUSE_FUNCTIONS_HXX
#endif  OPENGM_AUX_FUSE_FUNCTIONS_HXX

namespace opengm
{

namespace detail_fusion
{

template<class MODEL_TYPE>
struct NativeModelProxy
{

    typedef typename MODEL_TYPE::SpaceType SpaceType;


    void createModel(const UInt64Type nVar)
    {
        model_ = new MODEL_TYPE(SpaceType(nVar, 2));
    }
    void freeModel()
    {
        delete model_;
    }

    template<class F, class ITER>
    void addFactor(const F &f, ITER viBegin, ITER viEnd)
    {
        model_->addFactor(model_->addFunction(f), viBegin, viEnd);
    }

    MODEL_TYPE *model_;
};


template<class MODEL_TYPE>
struct QpboModelProxy
{

    QpboModelProxy()
    {
        c00_[0] = 0;
        c00_[1] = 0;

        c11_[0] = 1;
        c11_[1] = 1;

        c01_[0] = 0;
        c01_[1] = 1;

        c10_[0] = 1;
        c10_[1] = 0;
    }

    void createModel(const UInt64Type nVar)
    {
        model_ = new MODEL_TYPE(nVar, 0);
        model_->AddNode(nVar);
    }
    void freeModel()
    {
        delete model_;
    }

    template<class F, class ITER>
    void addFactor(const F &f, ITER viBegin, ITER viEnd)
    {
        OPENGM_CHECK_OP(f.dimension(), <= , 2, "wrong order for QPBO");

        if (f.dimension() == 1)
        {
            model_->AddUnaryTerm(*viBegin, f(c00_), f(c11_));
        }
        else
        {
            model_->AddPairwiseTerm(
                viBegin[0], viBegin[1],
                f(c00_),
                f(c01_),
                f(c10_),
                f(c11_)
            );
        }
    }

    MODEL_TYPE *model_;
    UInt64Type c00_[2];
    UInt64Type c11_[2];
    UInt64Type c01_[2];
    UInt64Type c10_[2];
};

template<class MODEL_TYPE>
struct Ad3ModelProxy
{

    Ad3ModelProxy(const typename  MODEL_TYPE::Parameter param)
        : param_(param)
    {
    }

    void createModel(const UInt64Type nVar)
    {
        model_ = new MODEL_TYPE(nVar, 2, param_, true);
    }
    void freeModel()
    {
        delete model_;
    }

    template<class F, class ITER>
    void addFactor(const F &f, ITER viBegin, ITER viEnd)
    {
        model_->addFactor(viBegin, viEnd, f);
    }

    MODEL_TYPE *model_;
    typename  MODEL_TYPE::Parameter param_;
};

} // end namespace detail_fusion
} // end namespace opengm


#endif // OPENGM_AUX_FUSE_FUNCTIONS_HXX
