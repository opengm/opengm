
#include "opengm/opengm.hxx"
#include "opengm/graphicalmodel/graphicalmodel.hxx"
#include "opengm/graphicalmodel/space/discretespace.hxx"
#include "opengm/functions/view.hxx"
#include "opengm/functions/view_fix_variables_function.hxx"
#include "opengm/functions/constant.hxx"
#include <opengm/utilities/metaprogramming.hxx>
#include "opengm/inference/visitors/visitors.hxx"
#include "opengm/inference/dynamicprogramming.hxx"


namespace opengm{

template<class ITER>
void printIter(ITER begin , ITER end){
    while(begin!=end){
        std::cout<<*begin<<" ";
        ++begin;
    }
    std::cout<<"\n";
}


template<class GM,class GM_OUT>
void mergeFactors(
    const GM        & gm,
    GM_OUT    & gmOut
){
    typedef typename GM::IndexType  IndexType;
    typedef typename GM::LabelType  LabelType;
    typedef typename GM::ValueType  ValueType;
    typedef typename GM::FactorType FactorType;
    typedef typename GM::FunctionIdentifier Fid;
    typedef ExplicitFunction<ValueType,IndexType,LabelType> ArrayFunction;

    typedef std::map< UInt64Type , Fid > FidMap;
    FidMap firstOrderF;
    FidMap secondOrderMap;



    for(IndexType fi=0;fi<gm.numberOfFactors();++fi){
        const FactorType & fac = gm[fi];
        const IndexType order = fac.numberOfVariables();
        OPENGM_CHECK_OP(order,<=,2,"order must be <=2");


        //if(fac.numberOfVariables()>=2){
        //    std::cout<<"fi = "<<fi<<"  VARIABLES  ";
        //    printIter(fac.variableIndicesBegin(),fac.variableIndicesEnd());
        //}

        if(order==1 || order==2){

  
            UInt64Type key=fac.variableIndex(0);
            if(order==2){
                key+=(fac.variableIndex(1)*gm.numberOfVariables());
            }

            FidMap & fidMap = ( order == 1 ?  firstOrderF : secondOrderMap  );


            if(fidMap.find(key)==fidMap.end()){
                ArrayFunction function(fac.shapeBegin(),fac.shapeEnd());
                fac.copyValues(&function(0));
                const Fid fid = gmOut.addFunction(function);
                gmOut.addFactor(fid,fac.variableIndicesBegin(),fac.variableIndicesEnd());
                fidMap[key]=fid;
            }

            else{
                const Fid fid = fidMap[key];
                ArrayFunction & function = gmOut. template getFunction<ArrayFunction>(fid);
                ArrayFunction buffer(fac.shapeBegin(),fac.shapeEnd());
                fac.copyValues(&buffer(0));
                // update function
                function+=buffer;
            }

        }
        else{

        }

    }

    /*
    for(IndexType fi=0;fi<gmOut.numberOfFactors();++fi){
        const IndexType order = gmOut[fi].numberOfVariables();
        OPENGM_CHECK_OP(order,<=,2,"order must be <=2");

        if(gmOut[fi].numberOfVariables()>=2){
            std::cout<<"fi = "<<fi<<"  OUT VAR  ";
            printIter(gmOut[fi].variableIndicesBegin(),gmOut[fi].variableIndicesEnd());
        }
    }
    */ 

}





template<class GM,class ACC>
class SubmodelOptimizer{

public:
    typedef GM GraphicalModelType;
    typedef ACC AccumulationType;
    OPENGM_GM_TYPE_TYPEDEFS;



    // function types
    typedef ViewFixVariablesFunction<GM> FixFunction;
    typedef ViewFunction<GM> ViewingFunction;
    typedef PositionAndLabel<IndexType,LabelType>  PosAndLabel;
    typedef std::vector<PosAndLabel> PosAndLabelVector;

    typedef ExplicitFunction<ValueType,IndexType,LabelType> ArrayFunction;

    // sub gm
    typedef typename opengm::DiscreteSpace<IndexType, LabelType> SubSpaceType;
    typedef typename meta::TypeListGenerator< ViewingFunction,FixFunction >::type SubFunctionTypeList;
    typedef typename meta::TypeListGenerator< ArrayFunction >::type MergeSubFunctionTypeList;

    typedef GraphicalModel<ValueType, typename GM::OperatorType, SubFunctionTypeList,SubSpaceType> SubGmType;
    typedef GraphicalModel<ValueType, typename GM::OperatorType, MergeSubFunctionTypeList,SubSpaceType> MergedSubGmType;

    SubmodelOptimizer(const GM & gm)
    :   gm_(gm),
        localVariables_(gm.numberOfVariables()),
        globalToLocalVariables_(gm.numberOfVariables()),
        submodelSpace_(gm.numberOfVariables()),
        inSubmodel_(gm.numberOfVariables(),false),
        localFactorViBuffer_(),
        fixedVarPosBuffer_(),
        notFixedVarPosBuffer_(),
        nLocalVar_(0),
        handledFactor_(gm.numberOfFactors(),false),
        labels_(gm.numberOfVariables())
    {
        //std::cout<<"submodel constructor\n";
        const IndexType maxOrder = gm_.factorOrder();
        localFactorViBuffer_.resize(maxOrder);
        fixedVarPosBuffer_.resize(maxOrder);
        notFixedVarPosBuffer_.resize(maxOrder);
        //std::cout<<"submodel constructor done\n";
    }



    struct InfResult{   

    };

    struct SubmodelInfo{

    };



    // set current state
    // O( CONST )
    void setLabel(const IndexType vi,const LabelType label){
        labels_[vi]=label;
    }

    // set and unset subvariables
    // O( |SUB_VARIABLES| )
    template<class VI_ITER>
    void setVariableIndices(VI_ITER begin,VI_ITER end){
        /*
        for(IndexType vi=0;vi<gm_.numberOfVariables();++vi){
            OPENGM_CHECK(inSubmodel_[vi]==false,"");
        }
        */
        OPENGM_CHECK_OP(nLocalVar_,==,0,"internal error");
        nLocalVar_=std::distance(begin,end);
        for(IndexType localVi=0;localVi<nLocalVar_;++localVi){
            const IndexType globalVi = begin[localVi];
            OPENGM_CHECK_OP(globalVi,<,gm_.numberOfVariables(),"");
            localVariables_[localVi]=globalVi;
            submodelSpace_[localVi]=gm_.numberOfLabels(globalVi);
            globalToLocalVariables_[globalVi]=localVi;
            OPENGM_CHECK(inSubmodel_[globalVi]==false,"internal error");
            inSubmodel_[globalVi]=true;
        }
    }

    // unset all subvariables
    // O( |SUB_VARIABLES| )
    void unsetVariableIndices(){
        OPENGM_CHECK_OP(nLocalVar_,>,0,"internal error");
        for(IndexType localVi=0;localVi<nLocalVar_;++localVi){
            const IndexType globalVi = localVariables_[localVi];
            OPENGM_CHECK(inSubmodel_[globalVi]==true,"internal error");
            inSubmodel_[globalVi]=false;
        }

        nLocalVar_=0;

        /*
        for(IndexType vi=0;vi<gm_.numberOfVariables();++vi){
            OPENGM_CHECK(inSubmodel_[vi]==false,"");
        }
        */
        

    }



    // build and infer with template
    template<class SOLVER>
    bool inferSubmodelInplace(
        const typename SOLVER::Parameter & para , 
        std::vector<LabelType> & resultArg,
        const bool improving=true,
        const bool warmStart=false
    ){
        OPENGM_CHECK_OP(nLocalVar_,!=,0,"");

        if(resultArg.size()!=nLocalVar_){
            resultArg.resize(nLocalVar_);
        }

        SOLVER solver(submodelSpace_.begin(),submodelSpace_.begin()+nLocalVar_,para);
        buildModelInplace(solver);
        if(warmStart){
            for(IndexType viLocal=0;viLocal<nLocalVar_;++viLocal){
                resultArg[viLocal]=labels_[localVariables_[viLocal]];
            }
            solver.setStartingPoint(resultArg.begin());
        }
        solver.infer();
        solver.arg(resultArg);

        for(IndexType localVi=0;localVi<nLocalVar_;++localVi){
            const IndexType globalVi=localVariables_[localVi];
            if(resultArg[localVi]!=labels_[globalVi]){
                return true;
            }
        }
        return false;
    }

    template<class SOLVER>
    bool inferSubmodel(
        const typename SOLVER::Parameter & para ,
        std::vector<LabelType> & resultArg,
        const bool improving=true,
        const bool warmStart=false
    ){
        OPENGM_CHECK_OP(nLocalVar_,!=,0,"");
        if(resultArg.size()!=nLocalVar_){
            resultArg.resize(nLocalVar_);
        }
        SubGmType  subGm( SubSpaceType(submodelSpace_.begin(),submodelSpace_.begin()+nLocalVar_) );
        reserveGraphicalModel(subGm);
        buildModelOpenGm(subGm);
        SOLVER solver(subGm,para);
        if(warmStart){
            for(IndexType viLocal=0;viLocal<nLocalVar_;++viLocal){
                resultArg[viLocal]=labels_[localVariables_[viLocal]];
            }
            solver.setStartingPoint(resultArg.begin());
        }
        solver.infer();
        solver.arg(resultArg);

        for(IndexType localVi=0;localVi<nLocalVar_;++localVi){
            const IndexType globalVi=localVariables_[localVi];
            if(resultArg[localVi]!=labels_[globalVi]){
                return true;
            }
        }
        return false;
    }



    //template<class SOLVER>
    bool mergeFactorsAndInferDp(
        //const typename SOLVER::Parameter & para ,
        std::vector<LabelType> & resultArg
    ){
        OPENGM_CHECK_OP(nLocalVar_,!=,0,"");
        if(resultArg.size()!=nLocalVar_){
            resultArg.resize(nLocalVar_);
        }
        SubGmType  subGm( SubSpaceType(submodelSpace_.begin(),submodelSpace_.begin()+nLocalVar_) );
        reserveGraphicalModel(subGm);
        buildModelOpenGm(subGm);


        MergedSubGmType mergeGm( SubSpaceType(submodelSpace_.begin(),submodelSpace_.begin()+nLocalVar_) );

        mergeFactors(subGm,mergeGm);

        typedef opengm::DynamicProgramming<MergedSubGmType,AccumulationType> DpSubInf;
        
        DpSubInf solver(mergeGm);
        solver.infer();
        solver.arg(resultArg);

        for(IndexType localVi=0;localVi<nLocalVar_;++localVi){
            const IndexType globalVi=localVariables_[localVi];
            if(resultArg[localVi]!=labels_[globalVi]){
                return true;
            }
        }
        return false;
    }
    

    // build model inplace for a given solver
    void reserveGraphicalModel(SubGmType & subGm){
        OPENGM_CHECK_OP(nLocalVar_,!=,0,"");

        IndexType nFac           = 0;   // nFac
        IndexType nFullUnaries   = 0;   // (full) included unaries
        IndexType nFullHighOrder = 0;   // full included high order
        IndexType nFixeHighOrder = 0;   // high order which are lower but still high order
        IndexType nFixedUnary    = 0;   // high order which are now unaries
        IndexType facVisSpace    = 0;

        /*
        for(IndexType fi=0;fi<gm_.numberOfFactors();++fi){
            OPENGM_CHECK(handledFactor_[fi]==false,"internal error");
        }

        IndexType c=0;
        for(IndexType vi=0;vi<gm_.numberOfVariables();++vi){
            if(inSubmodel_[vi]==true){
                ++c;
            }
        }
        OPENGM_CHECK_OP(c,==,nLocalVar_,"");
        */

        // Counting all that stuff
        for(IndexType localVi=0;localVi<nLocalVar_;++localVi){
            const IndexType globalVi = localVariables_[localVi];
            OPENGM_CHECK(inSubmodel_[globalVi],"");
            OPENGM_CHECK_OP(globalToLocalVariables_[globalVi],==,localVi,"internal error...mapping invalid");
            // get all factors for variable "globalVi"
            // and iterate over them
            const IndexType nFacVar = gm_.numberOfFactors(globalVi);
            for(IndexType f=0;f<nFacVar;++f){
                const IndexType fi = gm_.factorOfVariable(globalVi,f);
                if(handledFactor_[fi]==false){
                    handledFactor_[fi]=true;

                    const FactorType & factor = gm_[fi];
                    const IndexType    order  = factor.numberOfVariables();

                    // if factor is unary :
                    if(order == 0){
                        OPENGM_CHECK(false,"order==0 is not yet supported");
                    }
                    else if(order == 1){
                        
                        const IndexType viGlobal = factor.variableIndex(0);
                        const IndexType viLocal  = globalToLocalVariables_[viGlobal];
                        OPENGM_CHECK_OP(viGlobal,==,globalVi,"");
                        OPENGM_CHECK_OP(viLocal,==,localVi,"");
                        ++nFac;
                        ++nFullUnaries;
                        ++facVisSpace;
                    }
                    // if factor is higher order we need to check 
                    // if the factor needs to be inclued completely
                    // or only partial 
                    else{
                        /*
                        bool foundOwn=false;
                        for(IndexType v=0;v<order;++v){
                            const IndexType facVi=factor.variableIndex(v);
                            const IndexType facViDGB=gm_[fi].variableIndex(v);
                            OPENGM_CHECK_OP(facVi,==,facViDGB,"");
                            if(facVi==globalVi){
                                foundOwn=true;
                                break;
                            }
                        }
                        OPENGM_CHECK(foundOwn,"");
                        OPENGM_CHECK_OP(order,>=,2,"");
                        */

                        IndexType fixedVars    = 0;
                        IndexType notFixedVars = 0;
                        for(IndexType v=0;v<order;++v){
                            const IndexType facVi=factor.variableIndex(v);
                            if(inSubmodel_[facVi]==false){
                                fixedVarPosBuffer_[fixedVars]=v;
                                ++fixedVars;
                            }
                            else{
                                notFixedVarPosBuffer_[notFixedVars]=v;
                                ++notFixedVars;
                            }   
                        }
                        const IndexType partialOrder = order - fixedVars;
                        OPENGM_CHECK_OP(notFixedVars,>,0,"");
                        OPENGM_CHECK_OP(fixedVars+notFixedVars,==,order,"internal error");
                        OPENGM_CHECK_OP(partialOrder,<=,order,"internal error");
                        OPENGM_CHECK_OP(partialOrder,>=,1,"internal error");
                        // if higher order factor is fuly included
                        if(fixedVars==0){
                            ++nFac;
                            ++nFullHighOrder;
                            facVisSpace+=order;
                        }
                        else{
                            ++nFac;
                            facVisSpace+=partialOrder;
                            if(partialOrder==1)
                                ++nFixedUnary;
                            else
                                ++nFixeHighOrder;
                        }
                    }
                }
            }
        }




        // CLEANUP
        // - clean all used factors
        for(IndexType localVi=0;localVi<nLocalVar_;++localVi){
            const IndexType globalVi = localVariables_[localVi];
            // get all factors for variable "globalVi"
            // and iterate over them
            const IndexType nFac = gm_.numberOfFactors(globalVi);
            for(IndexType f=0;f<nFac;++f){
                const IndexType fi = gm_.factorOfVariable(globalVi,f);
                handledFactor_[fi]=false;
            }
        }

        OPENGM_CHECK_OP(nFac,==,nFullUnaries+nFullHighOrder+nFixeHighOrder+nFixedUnary,"");
        subGm.reserveFactors(nFac);
        subGm.reserveFactorsVarialbeIndices(facVisSpace);
        subGm. template reserveFunctions<ViewingFunction> (nFullUnaries+nFullHighOrder);
        subGm. template reserveFunctions<FixFunction>     (nFixeHighOrder+nFixedUnary);

    }
        // build model inplace for a given solver
    void buildModelOpenGm(SubGmType & subGm){

        /*
        for(IndexType fi=0;fi<gm_.numberOfFactors();++fi){
            OPENGM_CHECK(handledFactor_[fi]==false,"internal error");
        }
        */

        for(IndexType localVi=0;localVi<nLocalVar_;++localVi){
            const IndexType globalVi = localVariables_[localVi];
            // get all factors for variable "globalVi"
            // and iterate over them
            const IndexType nFac = gm_.numberOfFactors(globalVi);
            for(IndexType f=0;f<nFac;++f){
                const IndexType fi = gm_.factorOfVariable(globalVi,f);
                if(handledFactor_[fi]==false){
                    handledFactor_[fi]=true;

                    const FactorType & factor = gm_[fi];
                    const IndexType    order  = factor.numberOfVariables();


                    // if factor is unary :
                    // - factor needs to be added to the submodel
                    if(order == 1){
                        const IndexType viGlobal = factor.variableIndex(0);
                        const IndexType viLocal  = globalToLocalVariables_[viGlobal];
                        OPENGM_CHECK_OP(viLocal,==,localVi,"");
                        OPENGM_CHECK_OP(viLocal,<,nLocalVar_,"");
                        subGm.addFactor(subGm.addFunction(ViewingFunction(factor)),&viLocal,&viLocal+1);
                    }
                    // if factor is higher order we need to check 
                    // if the factor needs to be inclued completely
                    // or only partial 
                    else{

                        IndexType fixedVars    = 0;
                        IndexType notFixedVars = 0;
                        for(IndexType v=0;v<order;++v){
                            const IndexType facVi=factor.variableIndex(v);
                            if(!inSubmodel_[facVi]){
                                fixedVarPosBuffer_[fixedVars]=v;
                                ++fixedVars;
                            }
                            else{
                                notFixedVarPosBuffer_[notFixedVars]=v;
                                ++notFixedVars;
                            }   
                        }
                        const IndexType partialOrder = order - fixedVars;
                        OPENGM_CHECK_OP(notFixedVars,>,0,"");
                        OPENGM_CHECK_OP(fixedVars+notFixedVars,==,order,"internal error");
                        OPENGM_CHECK_OP(partialOrder,<=,order,"internal error");
                        OPENGM_CHECK_OP(partialOrder,>=,1,"internal error");
                        // if higher order factor is fuly included
                        if(fixedVars==0){
                            // map factors indices from global to local
                            for(IndexType v=0;v<order;++v){
                                const IndexType facVi=factor.variableIndex(v);
                                localFactorViBuffer_[v]=globalToLocalVariables_[facVi];
                                OPENGM_CHECK_OP(localFactorViBuffer_[v],<,nLocalVar_,"");
                            }
                            // add higher order factor
                            subGm.addFactor(subGm.addFunction(ViewingFunction(factor)),
                                localFactorViBuffer_.begin(),localFactorViBuffer_.begin()+order);
                        }
                        else{
                            PosAndLabelVector positionsAndLabelsOfFixedVars(fixedVars);

                            // map factor indices from global to local
                            for(IndexType v=0;v<partialOrder;++v){
                                const IndexType facVi=factor.variableIndex(notFixedVarPosBuffer_[v]);
                                localFactorViBuffer_[v]=globalToLocalVariables_[facVi];
                                OPENGM_CHECK_OP(localFactorViBuffer_[v],<,nLocalVar_,"");
                            }

                            for(IndexType fv=0;fv<fixedVars;++fv){
                                const IndexType facVi=factor.variableIndex(fixedVarPosBuffer_[fv]);
                                positionsAndLabelsOfFixedVars[fv].position_ = fixedVarPosBuffer_[fv];
                                positionsAndLabelsOfFixedVars[fv].label_    = labels_[facVi];
                            }
                            // add partial fixed factor
                            subGm.addFactor(subGm.addFunction(FixFunction(factor,positionsAndLabelsOfFixedVars)),
                                localFactorViBuffer_.begin(),localFactorViBuffer_.begin()+order-fixedVars);
                        }
                    }
                }
            }
        }

        // CLEANUP
        // - clean all used factors
        for(IndexType localVi=0;localVi<nLocalVar_;++localVi){
            const IndexType globalVi = localVariables_[localVi];
            // get all factors for variable "globalVi"
            // and iterate over them
            const IndexType nFac = gm_.numberOfFactors(globalVi);
            for(IndexType f=0;f<nFac;++f){
                const IndexType fi = gm_.factorOfVariable(globalVi,f);
                handledFactor_[fi]=false;
            }
        }
    }

    // build model inplace for a given solver
    template<class INF_TYPE>
    void buildModelInplace(INF_TYPE & infType){

        for(IndexType localVi=0;localVi<nLocalVar_;++localVi){
            const IndexType globalVi = localVariables_[localVi];
            // get all factors for variable "globalVi"
            // and iterate over them
            const IndexType nFac = gm_.numberOfFactors(globalVi);
            for(IndexType f=0;f<nFac;++f){
                const IndexType fi = gm_.factorOfVariable(globalVi,f);
                if(handledFactor_[fi]==false){
                    handledFactor_[fi]=true;

                    const FactorType & factor = gm_[fi];
                    const IndexType    order  = factor.numberOfVariables();


                    // if factor is unary :
                    // - factor needs to be added to the submodel
                    if(order == 1){
                        const IndexType viGlobal = factor.variableIndex(0);
                        const IndexType viLocal  = globalToLocalVariables_[viGlobal];
                        OPENGM_CHECK_OP(viLocal,<,nLocalVar_,"");
                        infType.addFactor(&viLocal,&viLocal+1,factor);
                    }
                    // if factor is higher order we need to check 
                    // if the factor needs to be inclued completely
                    // or only partial 
                    else{

                        IndexType fixedVars    = 0;
                        IndexType notFixedVars = 0;
                        for(IndexType v=0;v<order;++v){
                            const IndexType facVi=factor.variableIndex(v);
                            if(!inSubmodel_[facVi]){
                                fixedVarPosBuffer_[fixedVars]=v;
                                ++fixedVars;
                            }
                            else{
                                notFixedVarPosBuffer_[notFixedVars]=v;
                                ++notFixedVars;
                            }   
                        }
                        const IndexType partialOrder = order - fixedVars;
                        OPENGM_CHECK_OP(fixedVars+notFixedVars,==,order,"internal error");
                        OPENGM_CHECK_OP(partialOrder,<=,order,"internal error");
                        OPENGM_CHECK_OP(partialOrder,>=,1,"internal error");
                        // if higher order factor is fuly included
                        if(fixedVars==0){
                            // map factors indices from global to local
                            for(IndexType v=0;v<order;++v){
                                const IndexType facVi=factor.variableIndex(v);
                                localFactorViBuffer_[v]=globalToLocalVariables_[facVi];
                                OPENGM_CHECK_OP(localFactorViBuffer_[v],<,nLocalVar_,"");
                            }
                            // add higher order factor
                            infType.addFactor(localFactorViBuffer_.begin(),localFactorViBuffer_.begin()+order,factor);
                        }
                        else{
                            PosAndLabelVector positionsAndLabelsOfFixedVars(fixedVars);

                            // map factor indices from global to local
                            for(IndexType v=0;v<partialOrder;++v){
                                const IndexType facVi=factor.variableIndex(notFixedVarPosBuffer_[v]);
                                localFactorViBuffer_[v]=globalToLocalVariables_[facVi];
                                OPENGM_CHECK_OP(localFactorViBuffer_[v],<,nLocalVar_,"");
                            }

                            for(IndexType fv=0;fv<fixedVars;++fv){
                                const IndexType facVi=factor.variableIndex(fixedVarPosBuffer_[fv]);
                                positionsAndLabelsOfFixedVars[fv].position_ = fixedVarPosBuffer_[fv];
                                positionsAndLabelsOfFixedVars[fv].label_    = labels_[facVi];
                            }
                            // add partial fixed factor
                            infType.addFactor(localFactorViBuffer_.begin(),localFactorViBuffer_.begin()+order-fixedVars, 
                                FixFunction(factor,positionsAndLabelsOfFixedVars));
                        }
                    }
                }
            }
        }

        // CLEANUP
        // - clean all used factors
        for(IndexType localVi=0;localVi<nLocalVar_;++localVi){
            const IndexType globalVi = localVariables_[localVi];
            // get all factors for variable "globalVi"
            // and iterate over them
            const IndexType nFac = gm_.numberOfFactors(globalVi);
            for(IndexType f=0;f<nFac;++f){
                const IndexType fi = gm_.factorOfVariable(globalVi,f);
                handledFactor_[fi]=false;
            }
        }
    }


    bool inSubmodel(const IndexType vi)const{
        return inSubmodel_[vi];
    }

    IndexType submodelSize()const{
        return nLocalVar_;
    }


private:
    const GM & gm_;


    // submodel 
    std::vector<IndexType> localVariables_;
    std::vector<IndexType> globalToLocalVariables_;
    std::vector<LabelType> submodelSpace_;
    std::vector<bool>      inSubmodel_;
    std::vector<IndexType> localFactorViBuffer_;
    std::vector<IndexType> fixedVarPosBuffer_;
    std::vector<IndexType> notFixedVarPosBuffer_;
    IndexType nLocalVar_;


    // factors 
    std::vector<bool> handledFactor_;

    // global labels
    std::vector<LabelType> labels_;



};

}



