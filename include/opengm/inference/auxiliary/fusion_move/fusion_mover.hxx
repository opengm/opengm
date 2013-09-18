




#include "opengm/opengm.hxx"
#include "opengm/graphicalmodel/graphicalmodel.hxx"
#include "opengm/graphicalmodel/space/discretespace.hxx"
#include "opengm/functions/view.hxx"
#include "opengm/functions/view_fix_variables_function.hxx"
#include <opengm/utilities/metaprogramming.hxx>

#include "opengm/functions/function_properties_base.hxx"


namespace opengm{




template<class GM>
class FuseViewFunction
: public FunctionBase<FuseViewFunction<GM>, typename GM::ValueType, typename GM::IndexType, typename GM::LabelType> {
public:
    typedef typename GM::ValueType ValueType;
    typedef ValueType value_type;
    typedef typename GM::FactorType FactorType;
    typedef typename GM::OperatorType OperatorType;
    typedef typename GM::IndexType IndexType;
    typedef typename GM::LabelType LabelType;

    FuseViewFunction();

    FuseViewFunction(
        const FactorType & factor,
        const std::vector<LabelType> & argA,
        const std::vector<LabelType> & argB
    )
    :   factor_(&factor),
        argA_(&argA),
        argB_(&argB),
        iteratorBuffer_(factor.numberOfVariables())
    {

    }



    template<class Iterator>
    ValueType operator()(Iterator begin)const{
        for(IndexType i=0;i<iteratorBuffer_.size();++i){
            OPENGM_CHECK_OP(begin[i],<,2,"");
            if(begin[i]==0){
                iteratorBuffer_[i]=argA_->operator[](factor_->variableIndex(i));
            }
            else{
                iteratorBuffer_[i]=argB_->operator[](factor_->variableIndex(i));
            }
        }
        return factor_->operator()(iteratorBuffer_.begin());
    }

    IndexType shape(const IndexType)const{
        return 2;
    }

    IndexType dimension()const{
        return iteratorBuffer_.size();
    }

    IndexType size()const{
        return std::pow(2,iteratorBuffer_.size());
    }

private:
    FactorType const* factor_;
    std::vector<LabelType>  const * argA_;
    std::vector<LabelType>  const * argB_;
    mutable std::vector<LabelType> iteratorBuffer_;
};







template<class GM>
class FuseViewFixFunction
: public FunctionBase<FuseViewFixFunction<GM>, typename GM::ValueType, typename GM::IndexType, typename GM::LabelType> {
public:
    typedef typename GM::ValueType ValueType;
    typedef ValueType value_type;
    typedef typename GM::FactorType FactorType;
    typedef typename GM::OperatorType OperatorType;
    typedef typename GM::IndexType IndexType;
    typedef typename GM::LabelType LabelType;

    FuseViewFixFunction();

    FuseViewFixFunction(
        const FactorType & factor,
        const std::vector<LabelType> & argA,
        const std::vector<LabelType> & argB
    )
    :   factor_(&factor),
        argA_(&argA),
        argB_(&argB),
        iteratorBuffer_(factor.numberOfVariables()),
        notFixedPos_()
    {
        for(IndexType v=0;v<factor.numberOfVariables();++v){
            const IndexType vi=factor.variableIndex(v);
            if(argA[vi]!=argB[vi]){
                notFixedPos_.push_back(v);
            }
            else{
                iteratorBuffer_[v]=argA[vi];
            }
        }   
    }



    template<class Iterator>
    ValueType operator()(Iterator begin)const{
        for(IndexType i=0;i<notFixedPos_.size();++i){
            const IndexType nfp=notFixedPos_[i];
            OPENGM_CHECK_OP(begin[i],<,2,"");
            if(begin[i]==0){
                iteratorBuffer_[nfp]=argA_->operator[](factor_->variableIndex(nfp));
            }
            else{
                iteratorBuffer_[nfp]=argB_->operator[](factor_->variableIndex(nfp));
            }
        }
        return factor_->operator()(iteratorBuffer_.begin());
    }

    IndexType shape(const IndexType)const{
        return 2;
    }

    IndexType dimension()const{
        return notFixedPos_.size();
    }

    IndexType size()const{
        return std::pow(2,notFixedPos_.size());
    }

private:
    FactorType const* factor_;
    std::vector<LabelType>  const * argA_;
    std::vector<LabelType>  const * argB_;
    std::vector<IndexType> notFixedPos_;
    mutable std::vector<LabelType> iteratorBuffer_;
};



template<class GM,class ACC>
class FusionMover{
public:

	typedef GM GraphicalModelType;
    typedef ACC AccumulationType;
    OPENGM_GM_TYPE_TYPEDEFS;


    // function types
    typedef ViewFixVariablesFunction<GM> FixFunction;

    typedef FuseViewFunction<GM> FuseViewingFunction;
    typedef FuseViewFixFunction<GM> FuseViewingFixingFunction;

    typedef ExplicitFunction<ValueType,IndexType,LabelType> ArrayFunction;

    // sub gm
    typedef typename opengm::DiscreteSpace<IndexType, LabelType> SubSpaceType;
    typedef typename meta::TypeListGenerator< FuseViewingFunction,FuseViewingFixingFunction,ArrayFunction >::type SubFunctionTypeList;
    typedef GraphicalModel<ValueType, typename GM::OperatorType, SubFunctionTypeList,SubSpaceType> SubGmType;





public:

	FusionMover(const GM & gm)
	:	
		gm_(gm),
		subSpace_(gm.numberOfVariables(),2),
		localToGlobalVi_(gm.numberOfVariables()),
		globalToLocalVi_(gm.numberOfVariables()),
		nLocalVar_(0)
	{

	}



    void setup(
        const std::vector<LabelType> & argA,
        const std::vector<LabelType> & argB,
        std::vector<LabelType> & resultArg,
        const ValueType valueA,
        const ValueType valueB
    ){
        nLocalVar_=0;
        for(IndexType vi=0;vi<gm_.numberOfVariables();++vi){
            if(argA[vi]!=argB[vi]){
                localToGlobalVi_[nLocalVar_]=vi;
                globalToLocalVi_[vi]=nLocalVar_;
                ++nLocalVar_;
            }
        }
        std::copy(argA.begin(),argA.end(),resultArg.begin());
    }



	template<class SOLVER>
	ValueType fuseInplace(
		const typename SOLVER::Parameter & param,
		const std::vector<LabelType> & argA,
		const std::vector<LabelType> & argB,
		std::vector<LabelType> & resultArg,
		const ValueType valueA,
		const ValueType valueB
	){
        this->setup(argA,argB,resultArg,valueA,valueB);
        if(nLocalVar_>0){

    		SOLVER solver(subSpace_.begin(),subSpace_.begin()+nLocalVar_,param);
    		std::set<IndexType> addedFactors;

    		for(IndexType lvi=0;lvi<nLocalVar_;++lvi){

    			const IndexType vi=localToGlobalVi_[lvi];
    			const IndexType nFacVi = gm_.numberOfFactors(vi);

    			for(IndexType f=0;f<nFacVi;++f){
    				const IndexType fi 		= gm_.factorOfVariable(vi,f);
    				const IndexType fOrder  = gm_.numberOfVariables(fi);

    				// first order
    				if(fOrder==1){
    					OPENGM_CHECK_OP( localToGlobalVi_[lvi],==,gm_[fi].variableIndex(0),"internal error");
    					OPENGM_CHECK_OP( globalToLocalVi_[gm_[fi].variableIndex(0)],==,lvi,"internal error");

    					const IndexType vis[]={lvi};
    					const IndexType globalVi=localToGlobalVi_[lvi];

    					ArrayFunction f(subSpace_.begin(),subSpace_.begin()+1);


    					const LabelType c[]={ argA[globalVi],argB[globalVi]  };
    					f(0)=gm_[fi](c  );
    					f(1)=gm_[fi](c+1);

    					solver.addFactor(vis,vis+1,f);
    				}

    				// high order
    				else if( addedFactors.find(fi)==addedFactors.end() ){
    					addedFactors.insert(fi);
    					IndexType fixedVar 		=0;
    					IndexType notFixedVar 	=0;

    					for(IndexType vf=0;vf<fOrder;++vf){
    						const IndexType viFactor = gm_[fi].variableIndex(vf);
    						if(argA[viFactor]!=argB[viFactor]){
    							notFixedVar+=1;
    						}
    						else{
    							fixedVar+=1;
    						}
    					}
    					OPENGM_CHECK_OP(notFixedVar,>,0,"internal error");


    					if(fixedVar==0){
                            OPENGM_CHECK_OP(notFixedVar,==,fOrder,"interal error");
                            std::vector<IndexType> lvis(fOrder);
                            for(IndexType vf=0;vf<fOrder;++vf){
                                lvis[vf]=globalToLocalVi_[gm_[fi].variableIndex(vf)];
                            }

                            FuseViewingFunction f(gm_[fi],argA,argB);
                            solver.addFactor(lvis.begin(),lvis.end(),f);
    					}
    					else{
                            OPENGM_CHECK_OP(notFixedVar+notFixedVar,==,fOrder,"interal error");
                            std::vector<IndexType> lvis;
                            lvis.reserve(notFixedVar);
                            for(IndexType vf=0;vf<fOrder;++vf){
                                const IndexType gvi=gm_[fi].variableIndex(vf);
                                if(argA[gvi]!=argB[gvi]){
                                    lvis.push_back(globalToLocalVi_[gvi]);
                                }
                            }
                            OPENGM_CHECK_OP(lvis.size(),==,notFixedVar,"internal error");
                            FuseViewingFixingFunction f(gm_[fi],argA,argB);
                            solver.addFactor(lvis.begin(),lvis.end(),f);
    					}
    				}
    			}
    		}

            solver.infer();
            std::vector<LabelType> localArg(nLocalVar_);
            solver.arg(localArg);

            

            for(IndexType lvi=0;lvi<nLocalVar_;++lvi){
                const IndexType globalVi=localToGlobalVi_[lvi];
                const LabelType l = localArg[lvi];
                if(l==0){
                    resultArg[globalVi]=argA[globalVi];
                }
                else{
                    resultArg[globalVi]=argB[globalVi];
                }

            }
        }
        return gm_.evaluate(resultArg);
	}



    template<class SOLVER>
    ValueType fuse(
        const typename SOLVER::Parameter & param,
        const std::vector<LabelType> & argA,
        const std::vector<LabelType> & argB,
        std::vector<LabelType> & resultArg,
        const ValueType valueA,
        const ValueType valueB,
        const bool warmStart=false
    ){
        this->setup(argA,argB,resultArg,valueA,valueB);
        if(nLocalVar_>0){


            SubGmType subGm(SubSpaceType(subSpace_.begin(),subSpace_.begin()+nLocalVar_));
            std::set<IndexType> addedFactors;

            for(IndexType lvi=0;lvi<nLocalVar_;++lvi){

                const IndexType vi=localToGlobalVi_[lvi];
                const IndexType nFacVi = gm_.numberOfFactors(vi);

                for(IndexType f=0;f<nFacVi;++f){
                    const IndexType fi      = gm_.factorOfVariable(vi,f);
                    const IndexType fOrder  = gm_.numberOfVariables(fi);

                    // first order
                    if(fOrder==1){
                        OPENGM_CHECK_OP( localToGlobalVi_[lvi],==,gm_[fi].variableIndex(0),"internal error");
                        OPENGM_CHECK_OP( globalToLocalVi_[gm_[fi].variableIndex(0)],==,lvi,"internal error");

                        const IndexType vis[]={lvi};
                        const IndexType globalVi=localToGlobalVi_[lvi];

                        ArrayFunction f(subSpace_.begin(),subSpace_.begin()+1);


                        const LabelType c[]={ argA[globalVi],argB[globalVi]  };
                        f(0)=gm_[fi](c  );
                        f(1)=gm_[fi](c+1);

                        subGm.addFactor(subGm.addFunction(f),vis,vis+1);
                    }

                    // high order
                    else if( addedFactors.find(fi)==addedFactors.end() ){
                        addedFactors.insert(fi);
                        IndexType fixedVar      =0;
                        IndexType notFixedVar   =0;

                        for(IndexType vf=0;vf<fOrder;++vf){
                            const IndexType viFactor = gm_[fi].variableIndex(vf);
                            if(argA[viFactor]!=argB[viFactor]){
                                notFixedVar+=1;
                            }
                            else{
                                fixedVar+=1;
                            }
                        }
                        OPENGM_CHECK_OP(notFixedVar,>,0,"internal error");


                        if(fixedVar==0){
                            OPENGM_CHECK_OP(notFixedVar,==,fOrder,"interal error")

                            //std::cout<<"no fixations \n";

                            // get local vis
                            std::vector<IndexType> lvis(fOrder);
                            for(IndexType vf=0;vf<fOrder;++vf){
                                lvis[vf]=globalToLocalVi_[gm_[fi].variableIndex(vf)];
                            }

                            //std::cout<<"construct view\n";
                            FuseViewingFunction f(gm_[fi],argA,argB);

               

                            //std::cout<<"add  view\n";
                            subGm.addFactor(subGm.addFunction(f),lvis.begin(),lvis.end());
                            //std::cout<<"done \n";

                        }
                        else{
                            OPENGM_CHECK_OP(notFixedVar+notFixedVar,==,fOrder,"interal error")

                            //std::cout<<"fixedVar    "<<fixedVar<<"\n";
                            //std::cout<<"notFixedVar "<<notFixedVar<<"\n";

                            // get local vis
                            std::vector<IndexType> lvis;
                            lvis.reserve(notFixedVar);
                            for(IndexType vf=0;vf<fOrder;++vf){
                                const IndexType gvi=gm_[fi].variableIndex(vf);
                                if(argA[gvi]!=argB[gvi]){
                                    lvis.push_back(globalToLocalVi_[gvi]);
                                }
                            }
                            OPENGM_CHECK_OP(lvis.size(),==,notFixedVar,"internal error");


                            //std::cout<<"construct fix view\n";
                            FuseViewingFixingFunction f(gm_[fi],argA,argB);
                            //std::cout<<"add  fix view\n";
                            subGm.addFactor(subGm.addFunction(f),lvis.begin(),lvis.end());
                            //std::cout<<"done \n";

                        }
                    }
                }
            }

            SOLVER solver(subGm,param);
            std::vector<LabelType> localArg(nLocalVar_);
            if(warmStart){
                if(AccumulationType::bop(valueA,valueB)){
                    for(IndexType lvi=0;lvi<nLocalVar_;++lvi){
                        const IndexType globalVi=localToGlobalVi_[lvi];
                        localArg[lvi]=0;
                    } 
                }
                else{
                    for(IndexType lvi=0;lvi<nLocalVar_;++lvi){
                        const IndexType globalVi=localToGlobalVi_[lvi];
                        localArg[lvi]=1;
                    } 
                }
                solver.setStartingPoint(localArg.begin());
            }

            solver.infer();
            
            solver.arg(localArg);

            

            for(IndexType lvi=0;lvi<nLocalVar_;++lvi){
                const IndexType globalVi=localToGlobalVi_[lvi];
                const LabelType l = localArg[lvi];
                if(l==0){
                    resultArg[globalVi]=argA[globalVi];
                }
                else{
                    resultArg[globalVi]=argB[globalVi];
                }

            }
        }
        return gm_.evaluate(resultArg);
    }





private:
	const GraphicalModelType & gm_;
	

	std::vector<LabelType> subSpace_;
	std::vector<IndexType> localToGlobalVi_;
	std::vector<IndexType> globalToLocalVi_;
	IndexType nLocalVar_;


};

}