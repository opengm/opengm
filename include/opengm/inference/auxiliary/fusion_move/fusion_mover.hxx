#ifndef OPENGM_AUX_FUSION_MOVER_HXX
#define OPENGM_AUX_FUSION_MOVER_HXX




#include "opengm/opengm.hxx"
#include "opengm/graphicalmodel/graphicalmodel.hxx"
#include "opengm/graphicalmodel/space/discretespace.hxx"
#include "opengm/graphicalmodel/space/simplediscretespace.hxx"
#include "opengm/functions/view.hxx"
#include "opengm/functions/view_fix_variables_function.hxx"
#include <opengm/utilities/metaprogramming.hxx>

#include "opengm/functions/function_properties_base.hxx"
#include "opengm/inference/fix-fusion/fusion-move.hpp"

#include "opengm/inference/lazyflipper.hxx"

#ifdef WITH_CPLEX
#include "opengm/inference/lpcplex.hxx"
#endif
#ifdef WITH_QPBO
#include "QPBO.h"
#include "opengm/inference/reducedinference.hxx"
#include "opengm/inference/hqpbo.hxx"
#endif



namespace opengm
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



template<class GM, class ACC>
class FusionMover{
public:

    typedef GM GraphicalModelType;
    typedef ACC AccumulationType;
    OPENGM_GM_TYPE_TYPEDEFS;


    // function types
    typedef ViewFixVariablesFunction<GM> FixFunction;

    typedef FuseViewFunction<GM> FuseViewingFunction;
    typedef FuseViewFixFunction<GM> FuseViewingFixingFunction;

    typedef ExplicitFunction<ValueType, IndexType, LabelType> ArrayFunction;

    // sub gm
    typedef typename opengm::SimpleDiscreteSpace<IndexType, LabelType> SubSpaceType;
    typedef typename meta::TypeListGenerator< FuseViewingFunction, FuseViewingFixingFunction, ArrayFunction >::type SubFunctionTypeList;
    typedef GraphicalModel<ValueType, typename GM::OperatorType, SubFunctionTypeList, SubSpaceType> SubGmType;
public:

    FusionMover(const GM &gm);

    void setup(
        const std::vector<LabelType> &argA,
        const std::vector<LabelType> &argB,
        std::vector<LabelType> &resultArg,
        const ValueType valueA,
        const ValueType valueB
    );


    IndexType numberOfFusionMoveVariable()const
    {
        return nLocalVar_;
    }

    template<class SOLVER>
    ValueType fuse(
        const typename SOLVER::Parameter &param,
        const bool warmStart = false
    );


    template<class SOLVER>
    ValueType fuseAd3(
        const typename SOLVER::Parameter &param
    );

    template<class SOLVER>
    ValueType fuseQpbo(
    );


    template<class SOLVER>
    ValueType fuseFixQpbo(
    );

    ValueType valueResult()const
    {
        return valueResult_;
    }
    ValueType valueA()const
    {
        return valueA_;
    }
    ValueType valueB()const
    {
        return valueB_;
    }

private:
    template<class MODEL_PROXY>
    void fillSubModel(MODEL_PROXY &modelProy);

    // needed for  higher fix order reduction
    static const size_t maxOrder_ = 9;
    // graphical model to fuse states from
    const GraphicalModelType &gm_;

    std::vector<LabelType> const *argA_;
    std::vector<LabelType> const *argB_;
    std::vector<LabelType> const *argBest_;
    std::vector<LabelType> *argResult_;

    ValueType valueA_;
    ValueType valueB_;
    ValueType valueBest_;
    ValueType valueResult_;

    LabelType bestLabel_;


    std::vector<LabelType> subSpace_;
    std::vector<IndexType> localToGlobalVi_;
    std::vector<IndexType> globalToLocalVi_;
    IndexType nLocalVar_;


};

template<class GM, class ACC>
class HlFusionMover{

public:
    typedef GM GraphicalModelType;
    typedef ACC AccumulationType;
    OPENGM_GM_TYPE_TYPEDEFS;


    typedef FusionMover<GraphicalModelType,AccumulationType>    FusionMoverType ;
    typedef typename FusionMoverType::SubGmType                 SubGmType;


    #ifdef WITH_QPBO
    typedef kolmogorov::qpbo::QPBO<double>                          QpboSubInf;
    //typedef opengm::external::QPBO<SubGmType>                     QPBOSubInf;
    typedef opengm::HQPBO<SubGmType,AccumulationType>               HQPBOSubInf;
    typedef typename ReducedInferenceHelper<SubGmType>::InfGmType   ReducedGmType;
    #endif
    #ifdef WITH_CPLEX
    typedef opengm::LPCplex<SubGmType,AccumulationType>         CplexSubInf;
    #endif

    typedef opengm::LazyFlipper<SubGmType,AccumulationType>     LazyFlipperSubInf;


    typedef std::vector<LabelType> LabelVector;

    enum FusionSolver{
        DefaulFusion,
        QpboFusion,
        LazyFlipperFusion,
        CplexFuison
    };

    struct Parameter{
        Parameter(
        const FusionSolver fusionSolver=DefaulFusion,
        const size_t maxSubgraphSize = 2,
        const bool reducedInf = false,
        const bool tentacles = false,
        const bool connectedComponents = false,
        const double fusionTimeLimit = 100.0
        )
        : 
            fusionSolver_(fusionSolver),
            maxSubgraphSize_(maxSubgraphSize),
            reducedInf_(reducedInf),
            connectedComponents_(connectedComponents),
            tentacles_(tentacles),
            fusionTimeLimit_(fusionTimeLimit)
        {

        }
        FusionSolver fusionSolver_;
        size_t maxSubgraphSize_;
        bool reducedInf_;
        bool connectedComponents_;
        bool tentacles_;
        double fusionTimeLimit_;
    };

    HlFusionMover(const GM & gm, const Parameter & param) 
    :   gm_(gm),
        param_(param),
        fusionMover_(gm),
        factorOrder_(gm.factorOrder()) {

        // set default fusion mover
        if(param_.fusionSolver_==DefaulFusion){
            param_.fusionSolver_= LazyFlipperFusion;
            #ifdef  WITH_QPBO 
                param_.fusionSolver_ = QpboFusion;
            #endif
        }


        // check 
        if(param_.fusionSolver_ == QpboFusion){
            #ifndef  WITH_QPBO 
                throw RuntimeError("WITH_QPBO need to be enabled for QpboFusion");
            #endif
        }
        if(param_.fusionSolver_ == CplexFuison){
            #ifndef  WITH_CPLEX 
                throw RuntimeError("WITH_CPLEX need to be enabled for CplexFusion");
            #endif
        }
        if(param_.reducedInf_){
            #ifndef  WITH_QPBO 
                throw RuntimeError("WITH_QPBO need to be enabled for reducedInference");
            #endif
        }

    }


    bool fuse(const LabelVector & argA, const LabelVector argB, LabelVector & argRes,
                   const ValueType valA, const ValueType valB,ValueType & valRes){

        fusionMover_.setup(argA, argB, argRes, valA, valB);



        if(fusionMover_.numberOfFusionMoveVariable()>0){
            if(param_.fusionSolver_ == QpboFusion){
                #ifdef  WITH_QPBO
                if(factorOrder_<=2){
                    valRes = fusionMover_. template fuseQpbo<QpboSubInf> ();
                }
                else{
                    typename HQPBOSubInf::Parameter subInfParam;
                    valRes = fusionMover_. template fuse<HQPBOSubInf> (subInfParam,true);
                }
                #endif
            }
            else if(param_.fusionSolver_ == CplexFuison){
                #ifdef  WITH_CPLEX
                    // with reduced inference
                    if(param_.reducedInf_){
                        #ifdef WITH_QPBO
                            typedef opengm::LPCplex<ReducedGmType, AccumulationType>              _CplexSubInf;
                            typedef ReducedInference<SubGmType,AccumulationType,_CplexSubInf>     CplexReducedSubInf; 
                            typename _CplexSubInf::Parameter _subInfParam;
                            _subInfParam.integerConstraint_ = true; 
                            _subInfParam.numberOfThreads_   = 1;
                            _subInfParam.timeLimit_         = param_.fusionTimeLimit_; 
                            typename CplexReducedSubInf::Parameter subInfParam(true,param_.tentacles_,param_.connectedComponents_,_subInfParam);
                            valRes = fusionMover_. template fuse<CplexReducedSubInf> (subInfParam,true);      
                        #endif 
                    }
                    // without reduced inference
                    else{
                        typename CplexSubInf::Parameter p;
                        p.integerConstraint_ = true;
                        p.numberOfThreads_   = 1;
                        p.timeLimit_         = param_.fusionTimeLimit_;
                        valRes =  fusionMover_. template fuse<CplexSubInf> (p,true);
                    }
                #endif 
            }
            else if(param_.fusionSolver_ == LazyFlipperFusion){
                if(param_.reducedInf_){
                    #ifdef WITH_QPBO
                        typedef opengm::LazyFlipper<ReducedGmType, AccumulationType>          _LfSubInf;
                        typedef ReducedInference<SubGmType,AccumulationType,_LfSubInf>        LfReducedSubInf; 
                        typename _LfSubInf::Parameter _subInfParam;
                        _subInfParam.maxSubgraphSize_= param_.maxSubgraphSize_;
                        typename LfReducedSubInf::Parameter subInfParam(true,param_.tentacles_,param_.connectedComponents_,_subInfParam);
                        valRes = fusionMover_. template fuse<LfReducedSubInf> (subInfParam,true);      
                    #endif 
                }
                else{
                    const typename LazyFlipperSubInf::Parameter fuseInfParam(param_.maxSubgraphSize_);
                    valRes = fusionMover_. template fuse<LazyFlipperSubInf> (fuseInfParam, true);
                }
            }
            else{
               throw RuntimeError("Unknown Fusion Type! Maybe caused by missing linking!");
            }
            return true;
        }
        else{
            return false;
        }
    } 

private:
    const GraphicalModelType & gm_;
    Parameter param_;
    FusionMoverType fusionMover_;
    size_t factorOrder_;
};




/*
template<class GM, class ACC>
class MultiFusion{

public:
    typedef GM GraphicalModelType;
    typedef ACC AccumulationType;
    OPENGM_GM_TYPE_TYPEDEFS;

    typedef HlFusionMover<GraphicalModelType, AccumulationType> Fuse2;
    typedef typename Fuse2::Parameter Fuse2Parameter;
    typedef std::vector<LabelType> LabelVector;
    enum MultiFusionMode{
        Default,
        PairwiseBest
    };


    struct Parameter{
        Parameter(
            const Fuse2Parameter & fuse2Param = Fuse2Parameter()
        ) 
        :
            fuse2Param_(fuse2Param_){

        }

        Fuse2Parameter fuse2Param_;
    };

    HlFusionMover<GM, ACC> fuse2_;


    ValueType pairwiseFusion( const std::vector<LabelVector> & args
                              LabelVector & argRes){

        std::vector<LabelVector> * argsPtr = const_cast< const std::vector<LabelVector> * >(&args);
        return pairwiseFusionImpl(*argsPtr, argRes,true);

    }

private:
    ValueType pairwiseFusionImpl( std::vector<LabelVector> & args
                              LabelVector & argRes, 
                              bool first=true
    ){


        std::vector<LabelVector> improvedArgs;
        size_t nInputs = args.size();

        size_t bestInputIndex=0;
        ValueType bestInputVal = gm_.evalulate(args[0].begin(), args[0].end());
        LabelVector argRes;
        for(size_t i = 0;   i<nInputs; ++i){

            const ValueType valA = gm_.evalulate(args[i].begin(), args[i].end());
            if(ACC::bop(valA,bestInputVal)){
               bestInputIndex = i; 
               bestInputVal = valA;
            }
            for(size_t j = i+1; j<nInputs; ++j){
                const ValueType valB = gm_.evalulate(args[j].begin(), args[j].end());
                const ValueType valRes = fuse2(args[i], args[j], argRes);
                if(ACC::bop(valRes, valA) || ACC::bop(valRes, valB)){
                    improvedArgs.push_back(argRes);
                }
            }
        }
        if(improvedArgs.size()==0){
            argRes = args[bestInputIndex];
            return bestInputVal;
        }
        else if(improvedArgs.size()==1){
            argRes = improvedArgs;
            return gm_.evalulate(improvedArgs.begin(), improvedArgs.end());
        }
        else{
            if(first==false)
               args.clear();
            return this->pairwiseFusionImpl(improvedArgs, argRes, first=false)
        }
    }

    const GM & gm_;
    Fuse2 fuse2_;

};
*/

template<class GM, class ACC>
FusionMover<GM, ACC>::FusionMover(const GM &gm)
    :
    gm_(gm),
    subSpace_(gm.numberOfVariables(), 2),
    localToGlobalVi_(gm.numberOfVariables()),
    globalToLocalVi_(gm.numberOfVariables()),
    nLocalVar_(0)
{

}


template<class GM, class ACC>
void FusionMover<GM, ACC>::setup(
    const std::vector<typename FusionMover<GM, ACC>::LabelType> &argA,
    const std::vector<typename FusionMover<GM, ACC>::LabelType> &argB,
    std::vector<typename FusionMover<GM, ACC>::LabelType> &resultArg,
    const typename FusionMover<GM, ACC>::ValueType valueA,
    const typename FusionMover<GM, ACC>::ValueType valueB
)
{
    nLocalVar_ = 0;
    for (IndexType vi = 0; vi < gm_.numberOfVariables(); ++vi)
    {
        if (argA[vi] != argB[vi])
        {
            localToGlobalVi_[nLocalVar_] = vi;
            globalToLocalVi_[vi] = nLocalVar_;
            ++nLocalVar_;
        }
    }
    std::copy(argA.begin(), argA.end(), resultArg.begin());

    // store pointers
    argA_       = &argA;
    argB_       = &argB;
    argResult_  = &resultArg;

    valueA_     = valueA;
    valueB_     = valueB;

    if (ACC::bop(valueA, valueB))
    {
        argBest_ = argA_;
        valueBest_ = valueA;
        bestLabel_ = 0;
    }
    else
    {
        argBest_ = argB_;
        valueBest_ = valueB;
        bestLabel_ = 1;
    }
}


template<class GM, class ACC>
template<class MODEL_PROXY>
void FusionMover<GM, ACC>::fillSubModel(MODEL_PROXY &modelProxy)
{


    OPENGM_CHECK_OP(nLocalVar_, > , 0, "nothing to fuse");

    modelProxy.createModel(nLocalVar_);
    std::set<IndexType> addedFactors;
    for (IndexType lvi = 0; lvi < nLocalVar_; ++lvi)
    {

        const IndexType vi = localToGlobalVi_[lvi];
        const IndexType nFacVi = gm_.numberOfFactors(vi);

        for (IndexType f = 0; f < nFacVi; ++f)
        {
            const IndexType fi      = gm_.factorOfVariable(vi, f);
            const IndexType fOrder  = gm_.numberOfVariables(fi);

            // first order
            if (fOrder == 1)
            {
                OPENGM_CHECK_OP( localToGlobalVi_[lvi], == , gm_[fi].variableIndex(0), "internal error");
                OPENGM_CHECK_OP( globalToLocalVi_[gm_[fi].variableIndex(0)], == , lvi, "internal error");

                const IndexType vis[] = {lvi};
                const IndexType globalVi = localToGlobalVi_[lvi];

                ArrayFunction f(subSpace_.begin(), subSpace_.begin() + 1);


                const LabelType c[] = { (*argA_)[globalVi], (*argB_)[globalVi]  };
                f(0) = gm_[fi](c  );
                f(1) = gm_[fi](c + 1);

                //subGm_.addFactor(subGm_.addFunction(f),vis,vis+1);

                modelProxy.addFactor(f, vis, vis + 1);
            }

            // high order
            else if ( addedFactors.find(fi) == addedFactors.end() )
            {
                addedFactors.insert(fi);
                IndexType fixedVar      = 0;
                IndexType notFixedVar   = 0;

                for (IndexType vf = 0; vf < fOrder; ++vf)
                {
                    const IndexType viFactor = gm_[fi].variableIndex(vf);
                    if ((*argA_)[viFactor] != (*argB_)[viFactor])
                    {
                        notFixedVar += 1;
                    }
                    else
                    {
                        fixedVar += 1;
                    }
                }
                OPENGM_CHECK_OP(notFixedVar, > , 0, "internal error");


                if (fixedVar == 0)
                {
                    OPENGM_CHECK_OP(notFixedVar, == , fOrder, "interal error")

                    //std::cout<<"no fixations \n";

                    // get local vis
                    std::vector<IndexType> lvis(fOrder);
                    for (IndexType vf = 0; vf < fOrder; ++vf)
                    {
                        lvis[vf] = globalToLocalVi_[gm_[fi].variableIndex(vf)];
                    }

                    //std::cout<<"construct view\n";
                    FuseViewingFunction f(gm_[fi], *argA_, *argB_);



                    //std::cout<<"add  view\n";
                    //subGm_.addFactor(subGm_.addFunction(f),lvis.begin(),lvis.end());
                    modelProxy.addFactor(f, lvis.begin(), lvis.end());
                    //std::cout<<"done \n";

                }
                else
                {
                    OPENGM_CHECK_OP(notFixedVar + fixedVar, == , fOrder, "interal error")

                    //std::cout<<"fixedVar    "<<fixedVar<<"\n";
                    //std::cout<<"notFixedVar "<<notFixedVar<<"\n";

                    // get local vis
                    std::vector<IndexType> lvis;
                    lvis.reserve(notFixedVar);
                    for (IndexType vf = 0; vf < fOrder; ++vf)
                    {
                        const IndexType gvi = gm_[fi].variableIndex(vf);
                        if ((*argA_)[gvi] != (*argB_)[gvi])
                        {
                            lvis.push_back(globalToLocalVi_[gvi]);
                        }
                    }
                    OPENGM_CHECK_OP(lvis.size(), == , notFixedVar, "internal error");


                    //std::cout<<"construct fix view\n";
                    FuseViewingFixingFunction f(gm_[fi], *argA_, *argB_);
                    //std::cout<<"add  fix view\n";
                    modelProxy.addFactor(f, lvis.begin(), lvis.end());
                    //subGm_.addFactor(subGm_.addFunction(f),lvis.begin(),lvis.end());
                    //std::cout<<"done \n";

                }
            }
        }
    }
}




template<class GM, class ACC>
template<class SOLVER>
typename FusionMover<GM, ACC>::ValueType
FusionMover<GM, ACC>::fuse(
    const typename SOLVER::Parameter &param,
    const bool warmStart
)
{
    //std::cout<<"fill sub gm ... "<<std::flush; 
    NativeModelProxy<SubGmType> modelProxy;
    this->fillSubModel(modelProxy);
    //std::cout<<"done!"<<std::endl;



    //std::cout<<"solve sub problem ... "<<std::flush;
    SOLVER solver(*(modelProxy.model_), param);
    std::vector<LabelType> localArg(nLocalVar_);
    if (warmStart)
    {
        std::fill( localArg.begin(), localArg.end(),bestLabel_);
        solver.setStartingPoint(localArg.begin());
    }

    if(solver.infer()!=UNKNOWN){
       solver.arg(localArg);
       for (IndexType lvi = 0; lvi < nLocalVar_; ++lvi)
       {
          const IndexType globalVi = localToGlobalVi_[lvi];
          const LabelType l = localArg[lvi];
          (*argResult_)[globalVi] =  (l == 0 ?  (*argA_)[globalVi]  :   (*argB_)[globalVi]) ;
       }
       valueResult_ = gm_.evaluate(*argResult_);
       if (AccumulationType::bop(valueBest_, valueResult_))
       {
          valueResult_ = valueBest_;
          std::copy(argBest_->begin(), argBest_->end(), argResult_->begin());
       }
    }
    else{
       valueResult_ = valueBest_;
    }
    modelProxy.freeModel(); 
    //std::cout<<"done!"<<std::endl;
    return valueResult_;
}

template<class GM, class ACC>
template<class SOLVER>
typename FusionMover<GM, ACC>::ValueType
FusionMover<GM, ACC>::fuseAd3(
    const typename SOLVER::Parameter &param
)
{
    //std::cout<<"fill sub gm\n";
    Ad3ModelProxy<SOLVER> modelProxy(param);
    this->fillSubModel(modelProxy);



    std::vector<LabelType> localArg(nLocalVar_);
    modelProxy.model_->infer();
    modelProxy.model_->arg(localArg);

    for (IndexType lvi = 0; lvi < nLocalVar_; ++lvi)
    {
        const IndexType globalVi = localToGlobalVi_[lvi];
        const LabelType l = localArg[lvi];
        (*argResult_)[globalVi] =  (l == 0 ?  (*argA_)[globalVi]  :   (*argB_)[globalVi]) ;
    }
    valueResult_ = gm_.evaluate(*argResult_);

    if (AccumulationType::bop(valueBest_, valueResult_))
    {
        valueResult_ = valueBest_;
        std::copy(argBest_->begin(), argBest_->end(), argResult_->begin());
    }

    modelProxy.freeModel();
    return valueResult_;
}


template<class GM, class ACC>
template<class SOLVER>
typename FusionMover<GM, ACC>::ValueType
FusionMover<GM, ACC>::fuseQpbo(
)
{
    //std::cout<<"fill qbpo -2 order model\n";
    QpboModelProxy<SOLVER> modelProxy;
    this->fillSubModel(modelProxy);
    //std::cout<<"done\n";

    modelProxy.model_->MergeParallelEdges();

    //  set label for qpbo improvement
    for (IndexType lvi = 0; lvi < nLocalVar_; ++lvi)
    {
        const IndexType globalVi = localToGlobalVi_[lvi];
        modelProxy.model_->SetLabel(lvi, bestLabel_);
    }

    // do qpbo improvment
    srand( 42 );
    modelProxy.model_->Improve();

    // get result arg
    for (IndexType lvi = 0; lvi < nLocalVar_; ++lvi)
    {
        const IndexType globalVi = localToGlobalVi_[lvi];
        const LabelType l = modelProxy.model_->GetLabel(lvi);
        if (l == 0 || l == 1)
        {
            (*argResult_)[globalVi] =  (l == 0 ?  (*argA_)[globalVi]  :   (*argB_)[globalVi]) ;
        }
        else
        {
            (*argResult_)[globalVi] =  (*argBest_)[globalVi];
        }
    }
    valueResult_ = gm_.evaluate(*argResult_);
    if (AccumulationType::bop(valueBest_, valueResult_))
    {
        valueResult_ = valueBest_;
        std::copy(argBest_->begin(), argBest_->end(), argResult_->begin());
    }
    modelProxy.freeModel();
    return valueResult_;
}


template<class GM, class ACC>
template<class SOLVER>
typename FusionMover<GM, ACC>::ValueType
FusionMover<GM, ACC>::fuseFixQpbo(

)
{

    //std::cout<<"fill native for qbpo fix reduction model\n";
    NativeModelProxy<SubGmType> modelProxy;
    this->fillSubModel(modelProxy);

    const SubGmType &subGm = *(modelProxy.model_);

    // DO MOVE
    unsigned int maxNumAssignments = 1 << maxOrder_;
    std::vector<ValueType> coeffs(maxNumAssignments);
    std::vector<LabelType> cliqueLabels(maxOrder_);

    HigherOrderEnergy<ValueType, maxOrder_> hoe;
    hoe.AddVars(subGm.numberOfVariables());
    for (IndexType f = 0; f < subGm.numberOfFactors(); ++f)
    {
        IndexType size = subGm[f].numberOfVariables();
        if (size == 0)
        {
            continue;
        }
        else if (size == 1)
        {
            IndexType var = subGm[f].variableIndex(0);

            const LabelType lla[] = {0};
            const LabelType llb[] = {1};


            ValueType e0 = subGm[f](lla);
            ValueType e1 = subGm[f](llb);
            hoe.AddUnaryTerm(var, e1 - e0);
        }
        else
        {

            // unsigned int numAssignments = std::pow(2,size);
            unsigned int numAssignments = 1 << size;


            // -- // ValueType coeffs[numAssignments];
            for (unsigned int subset = 1; subset < numAssignments; ++subset)
            {
                coeffs[subset] = 0;
            }
            // For each boolean assignment, get the clique energy at the
            // corresponding labeling
            // -- // LabelType cliqueLabels[size];
            for (unsigned int assignment = 0;  assignment < numAssignments; ++assignment)
            {
                for (unsigned int i = 0; i < size; ++i)
                {
                    //if (    assignment%2 ==  (std::pow(2,i))%2  )
                    if (assignment & (1 << i))
                    {
                        cliqueLabels[i] = 0;
                    }
                    else
                    {
                        cliqueLabels[i] = 1;
                    }
                }
                ValueType energy = subGm[f](cliqueLabels.begin());
                for (unsigned int subset = 1; subset < numAssignments; ++subset)
                {
                    // if (assigment%2 != subset%2)
                    if (assignment & ~subset)
                    {
                        continue;
                    }
                    //(assigment%2 == subset%2)
                    else
                    {
                        int parity = 0;
                        for (unsigned int b = 0; b < size; ++b)
                        {
                            parity ^=  (((assignment ^ subset) & (1 << b)) != 0);
                        }
                        coeffs[subset] += parity ? -energy : energy;
                    }
                }
            }
            typename HigherOrderEnergy<ValueType, maxOrder_> ::VarId vars[maxOrder_];
            for (unsigned int subset = 1; subset < numAssignments; ++subset)
            {
                int degree = 0;
                for (unsigned int b = 0; b < size; ++b)
                {
                    if (subset & (1 << b))
                    {
                        vars[degree++] = subGm[f].variableIndex(b);
                    }
                }
                std::sort(vars, vars + degree);
                hoe.AddTerm(coeffs[subset], degree, vars);
            }
        }
    }
    SOLVER  qr(subGm.numberOfVariables(), 0);
    hoe.ToQuadratic(qr);
    qr.Solve();
    IndexType numberOfChangedVariables = 0;


    // get result arg
    for (IndexType lvi = 0; lvi < nLocalVar_; ++lvi)
    {
        const IndexType globalVi = localToGlobalVi_[lvi];
        const LabelType l = qr.GetLabel(lvi);
        if (l == 0 || l == 1)
        {
            (*argResult_)[globalVi] =  (l == 0 ?  (*argA_)[globalVi]  :   (*argB_)[globalVi]) ;
        }
        else
        {
            (*argResult_)[globalVi] =  (*argBest_)[globalVi];
        }
    }
    valueResult_ = gm_.evaluate(*argResult_);
    if (AccumulationType::bop(valueBest_, valueResult_))
    {
        valueResult_ = valueBest_;
        std::copy(argBest_->begin(), argBest_->end(), argResult_->begin());
    }
    modelProxy.freeModel();
    return valueResult_;
}



}

#endif //OPENGM_AUX_FUSION_MOVER_HXX
