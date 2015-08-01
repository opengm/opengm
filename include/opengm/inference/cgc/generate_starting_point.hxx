#include <vector>

#include <opengm/opengm.hxx>
#include <opengm/datastructures/partition.hxx>

namespace opengm{

    /**
     * If approxSize > 0, first runs splitBFS to partition nodes into clusters.
     * 
     * Then, merge neighboring nodes with lambda > threshold.
     * 
     * The resulting partitioning of the nodes is written into resultArg.
     */
    template<class GM>
    void startFromThreshold(
        const GM & gm,
        std::vector<typename GM::ValueType> & lambdas,
        std::vector<typename GM::LabelType> & resultArg,
        const typename GM::ValueType threshold=0.0
    ){
        typedef typename GM::IndexType IndexType;
        typedef typename GM::LabelType LabelType;
        typedef typename GM::ValueType ValueType;

        resultArg.resize(gm.numberOfVariables());



        Partition<IndexType> ufd(gm.numberOfVariables());

        for(IndexType fi=0;fi<gm.numberOfFactors();++fi){
            OPENGM_CHECK_OP(gm[fi].numberOfVariables(),==,2,"");

            const IndexType vi0 =gm[fi].variableIndex(0);
            const IndexType vi1 =gm[fi].variableIndex(1);
            if(lambdas[fi]>threshold){
                ufd.merge(vi0,vi1);
            }
        }

        std::map<IndexType,IndexType> representativeLabeling;

        ufd.representativeLabeling(representativeLabeling);

        for(IndexType vi=0;vi<gm.numberOfVariables();++vi){
            const IndexType find=ufd.find(vi);
            const IndexType dense=representativeLabeling[find];
            resultArg[vi]=dense;
        }
        
    }

}

// kate: space-indent on; indent-width 4; replace-tabs on; indent-mode cstyle; remove-trailing-space; replace-trailing-spaces-save; 
