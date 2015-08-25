

template<class T>
struct MaybeSetValue{
    T val_;
    bool isSet_;
    
};



template<class INDEX_TYPE>
class IntegralUnorderedMaxSizeSet{
public:
    IntegralUnorderedMaxSizeSet(const size_t maxIndex)
    :   maxSize_(maxIndex+1),
        indexInSet_(maxIndex+1,0),
        setSize_(0){
    }


    typedef INDEX_TYPE IndexType;
    void insert(const IndexType index){
        if(indexInSet_[index]==0){
            setElements_[setSize_]=index;
            indexInSet_[index]=1;
        }
    }
    bool hasIndex()const{
        return indexInSet_[index]!=0;
    }

    void removeAll()const{
        for(size_t i=0;i<setSize_;++i){
            indexInSet_[setElements_[i]]=0;
        }
    }

private:
    std::vector<unsigned  char> indexInSet_;
    std::vector<INDEX_TYPE    > setElements_;
    size_t setSize_;
};





template<class GM,class ACC>
class HighLevelFusionMover{
public:
    typedef ACC AccumulationType;
    typedef GM GmType;
    typedef GM GraphicalModelType;
    typedef INF InfType;
    typedef double QpboValueType
    OPENGM_GM_TYPE_TYPEDEFS;


    // function types
    typedef detail_fusion::ViewFixVariablesFunction<GM> FixFunction;
    typedef detail_fusion::FuseViewFunction<GM> FuseViewingFunction;
    typedef detail_fusion::FuseViewFixFunction<GM> FuseViewingFixingFunction;
    typedef ExplicitFunction<ValueType, IndexType, LabelType> ArrayFunction;




    #ifdef WITH_QPBO
    typedef kolmogorov::qpbo::QPBO<QpboValueType> QpboType;
    #endif 


    typedef std::vector<LabelType> StateVector;


    enum FusionSolver{
        QpboFusionSolver,
        CplexFusionSolver,
        BpFusionSolver,
        Ad3FusionSolver,
        LfFusionSolver,
        DefaultFusionSolver
    };


    struct ReductionParameter
    {
    public:
        bool Persistency_;
        bool Tentacle_;
        bool ConnectedComponents_;
        Parameter(
            const bool Persistency=false,
            const bool Tentacle=false,
            const bool ConnectedComponents=true
        ),
        :
            Persistency_ (Persistency),
            Tentacle_ (Tentacle),
            ConnectedComponents_ (ConnectedComponents)
        {
        };
    };

    
    struct FusionSolverParam{
        FusionSolverParam(
            const size_t subgraphSize=3,
            const size_t steps=0,
            const double damping=0.75
        ) 
        :   
            subgraphSize_(subgraphSize),
            steps_(steps),
            damping_(damping)
    {
        size_t subgraphSize_;
        size_t steps_;
        double damping_;
    };


    enum UnknownStateBehavior{
        TakeA,
        TakeB,
        TakeBest,
        Random
    };



    struct Parameter{
        Parameter(
            const FusionSolver fusionSolver                 = DefaultFusionSolver,
            const bool useReduction                         = true,
            const ReductionParameter & reductionParameter   = ReductionParameter(),
            const UnknownStateBehavior unknownStateBehavior = TakeA
        ) 
        :   fusionSolver_(fusionSolver),
            useReduction_(useReduction),
            reductionParameter_(reductionParameter),
            unknownStateBehavior_(unknownStateBehavior){
        }

        FusionSolver fusionSolver_;
        bool useReduction_;
        ReductionParameter reductionParameter_;
        UnknownStateBehavior unknownStateBehavior_;
    };


    struct FusionMoveResult{
        bool changeOrImprovement_;

    };

    HighLevelFusionMover(const GM & gm,const Parameter & parameter)
    :   gm_(gm),
        parameter_(parameter){

            this->setUpParameter();

    }


    FusionMoveResult fuse(
        const StateVector & proposalLabelsA,
        const StateVector & proposalLabelsB,
        StateVector       & resultLabels
    ){
        // mapping from local <=> global vi
        this->getForwardBackwardMapping(proposalLabelsA,proposalLabelsB);

        return fuse2Impl(proposalLabelsA,proposalLabelsB,resultLabels);
    }

    // fuse inplace
    FusionMoveResult fuse(
        const StateVector & proposalLabelsA,
        StateVector & proposalLabelsBAndResult,
    ){
        // mapping from local <=> global vi
        this->getForwardBackwardMapping(proposalLabelsA,proposalLabelsB);

        return fuse2Impl(proposalLabelsA,proposalLabelsBAndResult,proposalLabelsBAndResult);
    }

    // fuse vector
    template<class PROPOSAL_LABEL_ITERATOR>
    FusionMoveResult fuse(
        PROPOSAL_LABEL_ITERATOR proposalLabelsBegin,
        PROPOSAL_LABEL_ITERATOR proposalLabelsEnd,
        StateVector       & resultLabels
    ){
        //
        raise RuntimeError("not yet implemented");
    }

private:    
    template<class PROPOSAL_LABELS_A,class PROPOSAL_LABELS_B>
    void  getForwardBackwardMapping(
        const PROPOSAL_LABELS_A &       proposalLabelsA,
        const PROPOSAL_LABELS_B &       proposalLabelsB
    ){
        nLocalVar_ = 0;
        for (IndexType vi = 0; vi < gm_.numberOfVariables(); ++vi)
        {
            if (proposalLabelsA[vi] != proposalLabelsB[vi])
            {
                localToGlobalVi_[nLocalVar_] = vi;
                globalToLocalVi_[vi] = nLocalVar_;
                ++nLocalVar_;
            }
        }   
    }

    template<class PROPOSAL_LABELS_A,class PROPOSAL_LABELS_B,class MODEL_PROXY>
    void  setupSubmodelProxy(
        const PROPOSAL_LABELS_A & proposalLabelsA,
        const PROPOSAL_LABELS_B & proposalLabelsB
              MODEL_PROXY       & modelProxy
    )
    {
        for (IndexType lvi = 0; lvi < nLocalVar_; ++lvi)
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

                // preallocate me!!!
                const float aTwo=2;
                ArrayFunction f(&aTwo,&aTwo+1);
                const LabelType c[] = { proposalLabelsA[globalVi],proposalLabelsB[globalVi]};
                f(0) = gm_[fi](c  );
                f(1) = gm_[fi](c + 1);
                modelProxy.addFactor(f, vis, vis + 1);
            }
            // high order
            else if ( addedFactors_.hasIndex(fi) == false )
            {
                addedFactors_.insert(fi);
                IndexType fixedVar      = 0;
                IndexType notFixedVar   = 0;

                for (IndexType vf = 0; vf < fOrder; ++vf)
                {
                    const IndexType viFactor = gm_[fi].variableIndex(vf);
                    if (proposalLabelsA[viFactor] != proposalLabelsB[viFactor])
                        notFixedVar += 1;
                    else
                        fixedVar += 1;
                }
                OPENGM_CHECK_OP(notFixedVar, > , 0, "internal error");

                if (fixedVar == 0)
                {
                    OPENGM_CHECK_OP(notFixedVar, == , fOrder, "interal error");
                    // preallocate me
                    std::vector<IndexType> lvis(fOrder);
                    for (IndexType vf = 0; vf < fOrder; ++vf)
                        lvis[vf] = globalToLocalVi_[gm_[fi].variableIndex(vf)];

                    FuseViewingFunction f(gm_[fi], *argA_, *argB_);
                    modelProxy.addFactor(f, lvis.begin(), lvis.end());
                }
                else
                {
                    OPENGM_CHECK_OP(notFixedVar + fixedVar, == , fOrder, "interal error")

                    // get local vis
                    std::vector<IndexType> lvis;
                    lvis.reserve(notFixedVar);
                    for (IndexType vf = 0; vf < fOrder; ++vf)
                    {
                        const IndexType gvi = gm_[fi].variableIndex(vf);
                        if ( proposalLabelsA[gvi] != proposalLabelsB[gvi])
                            lvis.push_back(globalToLocalVi_[gvi]);
                    }
                    OPENGM_CHECK_OP(lvis.size(), == , notFixedVar, "internal error");
                    FuseViewingFixingFunction f(gm_[fi], *argA_, *argB_);
                    modelProxy.addFactor(f, lvis.begin(), lvis.end());
                }
            }
        }
        addedFactors_.removeAll();
    } 

    template<class PROPOSAL_LABELS_A,class PROPOSAL_LABELS_B,class PROPOSAL_LABELS_RESULT>
    FusionMoveResult  fuse2Impl(
        const PROPOSAL_LABELS_A &       proposalLabelsA,
        const PROPOSAL_LABELS_B &       proposalLabelsB,
              PROPOSAL_LABELS_RESULT &  proposalLabelsResult
    ){



        // do the basic fusion 
        if(param_.fusionSolver_==QpboFusionSolver)
            return this->fuse2Qpbo(proposalLabelsA,proposalLabelsB,proposalLabelsResult);

        // do some kind of improvement?

    }

    template<class PROPOSAL_LABELS_A,class PROPOSAL_LABELS_B,class PROPOSAL_LABELS_RESULT>
    FusionMoveResult  fuse2Qpbo(
        const PROPOSAL_LABELS_A &       proposalLabelsA,
        const PROPOSAL_LABELS_B &       proposalLabelsB,
              PROPOSAL_LABELS_RESULT &  proposalLabelsResult
    ){
        if(maxFactorOrder_<=2)
            return this->fuse2QpboImproment2Order(proposalLabelsA,proposalLabelsB,proposalLabelsResult);
        else
            return this->fuse2QpboAnyOrder(proposalLabelsA,proposalLabelsB,proposalLabelsResult);
    } 

    template<class PROPOSAL_LABELS_A,class PROPOSAL_LABELS_B,class PROPOSAL_LABELS_RESULT>
    FusionMoveResult  fuse2Native(
        const PROPOSAL_LABELS_A &       proposalLabelsA,
        const PROPOSAL_LABELS_B &       proposalLabelsB,
              PROPOSAL_LABELS_RESULT &  proposalLabelsResult
    ){
        
    }

    template<class PROPOSAL_LABELS_A,class PROPOSAL_LABELS_B,class PROPOSAL_LABELS_RESULT>
    FusionMoveResult  fuse2QpboImproment2Order(
        const PROPOSAL_LABELS_A &       proposalLabelsA,
        const PROPOSAL_LABELS_B &       proposalLabelsB,
              PROPOSAL_LABELS_RESULT &  proposalLabelsResult
    ){

        #ifdef WITH_QPBO
        // set up qpbo problem
        setupSubmodelProxy(proposalLabelsA,proposalLabelsB,qpboModelProxy_);

        // solve qpbo with improvement 
        srand( 42 );
        qpboModelProxy_.model_->Improve();

        // get result arg
        if(param_.unknownStateBehavior_==TakeA || param_.unknownStateBehavior_==TakeB){
            for (IndexType lvi = 0; lvi < nLocalVar_; ++lvi)
            {
                const IndexType globalVi = localToGlobalVi_[lvi];
                const LabelType l = modelProxy.model_->GetLabel(lvi);
                if (l == 0)
                    proposalLabelsResult[globalVi] = proposalLabelsA)[globalVi] ;
                if (l == 1)
                    proposalLabelsResult[globalVi] = proposalLabelsB[globalVi] ;
                else
                {
                    if(param_.unknownStateBehavior_==TakeA )
                        proposalLabelsResult[globalVi] = proposalLabelsA)[globalVi] ;
                    else
                        proposalLabelsResult[globalVi] = proposalLabelsB)[globalVi] ;
                }
            }
        }
        else{
            throw RuntimeError("not yet implemented");
        }


        //reset the qpbo solver proxy
        qpboModelProxy_.reset();

        return FusionMoveResult();

        #else 
        throw RuntimeError("fuse2QpboImproment2Order needs WITH_QPBO to be enabled");
        return FusionMoveResult();
        #endif 
    }

    template<class PROPOSAL_LABELS_A,class PROPOSAL_LABELS_B,class PROPOSAL_LABELS_RESULT>
    FusionMoveResult  fuse2QpboAnyOrder(
        const PROPOSAL_LABELS_A &       proposalLabelsA,
        const PROPOSAL_LABELS_B &       proposalLabelsB,
              PROPOSAL_LABELS_RESULT &  proposalLabelsResult
    ){
        

    }






    void setUpParameter(){

    }

    // input data
    const GM & gm_;
    const Parameter parameter_;



    // working data
    IntegralUnorderedMaxSizeSet<IndexType> addedFactors_;
    size_t maxFactorOrder_;

    std::vector<IndexType> localToGlobalVi_;
    std::vector<IndexType> globalToLocalVi_;
    IndexType nLocalVar_;


    ValueType valueA_;
    ValueType valueB_;
    bool evaluatedA_;
    bool evaluatedB_;

    // QPBO RELATED DATA
    #ifdef WITH_QPBO
    detail_fusion::QpboModelProxy<QpboType> qpboModelProxy_;
    #endif 

};