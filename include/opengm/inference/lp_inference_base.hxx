#include <opengm/opengm.hxx>

namespace opengm{

    template<class GM>
    typename GM::IndexType findMaxFactorSize(const GM & gm){
       typedef typename  GM::IndexType IndexType;
       IndexType maxSize = 0 ;
       for(IndexType fi=0;fi<gm.numberOfFactors();++fi){
          maxSize = std::max(maxSize ,gm[fi].size());
       }
       return maxSize;
    }



    template<class GM>
    typename GM::LabelType findMaxNumberOfLabels(const GM & gm){
       typedef typename  GM::LabelType LabelType;
       typedef typename  GM::IndexType IndexType;
       LabelType maxL = 0 ;
       for(IndexType vi=0;vi<gm.numberOfVariables();++vi){
          maxL = std::max(maxL ,gm.numberOfLabels(vi));
       }
       return maxL;
    }



	template<class GM>
	void findUnariesFi(const GM & gm , std::vector<typename GM::IndexType> & unaryFi){
	   typedef typename  GM::IndexType IndexType;

	   const IndexType noUnaryFactorFound=gm.numberOfFactors();

	   unaryFi.resize(gm.numberOfVariables());
	   std::fill(unaryFi.begin(),unaryFi.end(),noUnaryFactorFound);

	   for(IndexType fi=0;fi<gm.numberOfFactors();++fi){

	      if(gm[fi].numberOfVariables()==1){
	         const IndexType vi = gm[fi].variableIndex(0);
	         OPENGM_CHECK_OP(unaryFi[vi],==,noUnaryFactorFound, " multiple unary factor found");
	         unaryFi[vi]=fi;
	      }
	   }
	}



    template<class GM,class ACC>
    class LpInferenceBase{
    public:

        typedef ACC AccumulationType;
        typedef GM GraphicalModelType;
        OPENGM_GM_TYPE_TYPEDEFS;

        LpInferenceBase(const GraphicalModelType & gm);


        // get the lp variable indices
        UInt64Type lpNodeVi(const IndexType gmVi,const LabelType label)const;
        UInt64Type lpFactorVi(const IndexType gmFi,const UInt64Type labelIndex)const;
        template<class LABELING_ITERATOR>
        UInt64Type lpFactorVi(const IndexType gmFi,  LABELING_ITERATOR labelingBegin, LABELING_ITERATOR labelingEnd)const;
        UInt64Type numberOfLpVariables()const;

        UInt64Type numberOfNodeLpVariables()const{return numNodeVar_;}
        UInt64Type numberOfFactorLpVariables()const{return numFactorVar_;}

    protected:
        ValueType valueToMinSumValue(const ValueType val)const;
        ValueType valueFromMinSumValue(const ValueType val)const;

        bool hasUnary(const IndexType vi)const;
        IndexType unaryFactorIndex(const IndexType vi)const;

	private:

		const GraphicalModelType & 	gm_;
		std::vector<UInt64Type> 	nodeVarIndex_;
		std::vector<UInt64Type> 	factorVarIndex_;
		std::vector<IndexType>   	unaryFis_;

		UInt64Type 					numNodeVar_;
		UInt64Type					numFactorVar_;


	};


    template<class GM,class ACC>
    inline UInt64Type
    LpInferenceBase<GM,ACC>::numberOfLpVariables()const{
        return numNodeVar_+numFactorVar_;
    }

    template<class GM,class ACC>
    inline typename LpInferenceBase<GM,ACC>::ValueType 
    LpInferenceBase<GM,ACC>::valueToMinSumValue
    (
        const typename LpInferenceBase<GM,ACC>::ValueType val
    )const{
        if(opengm::meta::Compare<OperatorType,opengm::Adder>::value){
            if(opengm::meta::Compare<ACC,opengm::Minimizer>::value)
                return val;
            else if(opengm::meta::Compare<ACC,opengm::Maximizer>::value)
                return -1.0*val;
            else
                throw RuntimeError("Wrong Accumulator");
        }
        else if(opengm::meta::Compare<OperatorType,opengm::Multiplier>::value){
            OPENGM_CHECK_OP(val,>,0.0, "LpInterface with Multiplier as operator does not support objective<=0 ");
            if(opengm::meta::Compare<ACC,opengm::Minimizer>::value)
                return std::log(val);
            else if(opengm::meta::Compare<ACC,opengm::Maximizer>::value)
                return -1.0*std::log(val);
            else
                throw RuntimeError("Wrong Accumulator");
        }
        else
            throw RuntimeError("Wrong Operator");
    }

    template<class GM,class ACC>
    inline typename LpInferenceBase<GM,ACC>::ValueType 
    LpInferenceBase<GM,ACC>::valueFromMinSumValue
    (
        const typename LpInferenceBase<GM,ACC>::ValueType val
    )const{
        if(opengm::meta::Compare<OperatorType,opengm::Adder>::value){
            if(opengm::meta::Compare<ACC,opengm::Minimizer>::value)
                return val;
            else if(opengm::meta::Compare<ACC,opengm::Maximizer>::value)
                return -1.0*val;
            else
                throw RuntimeError("Wrong Accumulator");
        }
        else if(opengm::meta::Compare<OperatorType,opengm::Multiplier>::value){
            if(opengm::meta::Compare<ACC,opengm::Minimizer>::value)
                return std::exp(val);
            else if(opengm::meta::Compare<ACC,opengm::Maximizer>::value)
                return std::exp(-1.0*val);
            else
                throw RuntimeError("Wrong Accumulator");
        }
        else
            throw RuntimeError("Wrong Operator");
    }

    template<class GM,class ACC>
    LpInferenceBase<GM,ACC>::LpInferenceBase(
        const typename LpInferenceBase<GM,ACC>::GraphicalModelType & gm
    )
    :
        gm_(gm),
        nodeVarIndex_(gm.numberOfVariables()),
        factorVarIndex_(gm.numberOfFactors()),
        unaryFis_(),
        numNodeVar_(0),
        numFactorVar_(0)
    {
        // find unary vis
        findUnariesFi(gm_,unaryFis_);

        // count the number of lp variables
        // - from nodes 
        // - from factors
        // remember the "start" lpIndex for each 
        // - gm variable
        // - gm factor 
        UInt64Type lpVarIndex=0;
        for(IndexType vi=0;vi<gm_.numberOfVariables();++vi){
            nodeVarIndex_[vi] = lpVarIndex;
            numNodeVar_ += gm_.numberOfLabels(vi);
            lpVarIndex    += gm_.numberOfLabels(vi);
        }
        for(IndexType fi=0;fi<gm_.numberOfFactors();++fi){
            if (gm_[fi].numberOfVariables()>1){
                factorVarIndex_[fi]=lpVarIndex;
                numFactorVar_   += gm_[fi].size();
                lpVarIndex      += gm_[fi].size();
            }
            else if (gm_[fi].numberOfVariables()==1){
                const IndexType vi0 = gm_[fi].variableIndex(0);
                factorVarIndex_[fi]=nodeVarIndex_[vi0];
            }
            else{
                throw RuntimeError("const factors within lp interface are not yet allowed");
            }
        }
    }




    template<class GM, class ACC>
    inline UInt64Type
    LpInferenceBase<GM,ACC>::lpNodeVi(
       const typename LpInferenceBase<GM,ACC>::IndexType gmVi,
       const typename LpInferenceBase<GM,ACC>::LabelType label
    ) const {
       return nodeVarIndex_[gmVi]+label;
    }


    template<class GM, class ACC>
    inline UInt64Type
    LpInferenceBase<GM,ACC>::lpFactorVi(
       const typename LpInferenceBase<GM,ACC>::IndexType gmFi,
       const UInt64Type labelIndex
    ) const {
       return factorVarIndex_[gmFi]+labelIndex;
    }


    template<class GM, class ACC>
    template<class LABELING_ITERATOR>
    inline UInt64Type 
    LpInferenceBase<GM,ACC>::lpFactorVi
    (
       const typename LpInferenceBase<GM,ACC>::IndexType factorIndex,
       LABELING_ITERATOR labelingBegin,
       LABELING_ITERATOR labelingEnd
    )const{
       OPENGM_ASSERT(factorIndex<gm_.numberOfFactors());
       OPENGM_ASSERT(std::distance(labelingBegin,labelingEnd)==gm_[factorIndex].numberOfVariables());
       const size_t numVar=gm_[factorIndex].numberOfVariables();
       size_t labelingIndex=labelingBegin[0];
       size_t strides=gm_[factorIndex].numberOfLabels(0);
       for(size_t vi=1;vi<numVar;++vi){
          OPENGM_ASSERT(labelingBegin[vi]<gm_[factorIndex].numberOfLabels(vi));
          labelingIndex+=strides*labelingBegin[vi];
          strides*=gm_[factorIndex].numberOfLabels(vi);
       }
       return factorVarIndex_[factorIndex]+labelingIndex;
    }


    template<class GM, class ACC>
    inline bool 
    LpInferenceBase<GM,ACC>::hasUnary(
        const typename LpInferenceBase<GM,ACC>::IndexType vi
    )const{
        return unaryFis_[vi]!=gm_.numberOfFactors();
    }

    template<class GM, class ACC>
    inline typename LpInferenceBase<GM,ACC>::IndexType 
    LpInferenceBase<GM,ACC>::unaryFactorIndex(
        const typename LpInferenceBase<GM,ACC>::IndexType vi
    )const{
        return unaryFis_[vi];
    }






}






