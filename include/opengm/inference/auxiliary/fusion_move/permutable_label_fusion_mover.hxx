#ifndef OPENGM_PERMUTABLE_LABEL_FUSION_MOVER_HXX
#define OPENGM_PERMUTABLE_LABEL_FUSION_MOVER_HXX


#include <opengm/inference/inference.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>



namespace opengm{


template<class GM, class ACC>
class PermutableLabelFusionMove{

public:

    typedef GM GraphicalModelType;
    typedef ACC AccumulationType;
    OPENGM_GM_TYPE_TYPEDEFS;


    PermutableLabelFusionMove(const GraphicalModelType & gm)
    :   
        gm_(gm)
    {

    }



    size_t intersect(
        const std::vector<LabelType> & a,
        const std::vector<LabelType> & b,
        std::vector<LabelType> & res
    ){
        Partition<LabelType> ufd(gm_.numberOfVariables());
        for(size_t fi=0; fi< gm_.numberOfFactors(); ++fi){
            if(gm_[fi].numberOfVariables()==2){

                const size_t vi0 = gm_[fi].variableIndex(0);
                const size_t vi1 = gm_[fi].variableIndex(1);



                if(a[vi0] == a[vi1] && b[vi0] == b[vi1]){
                    ufd.merge(vi0, vi1);
                }
            }   
            else{
                throw RuntimeError("only implemented for second order");
            }
        }
        std::map<LabelType, LabelType> repr;
        ufd.representativeLabeling(repr);
        for(size_t vi=0; vi<gm_.numberOfVariables(); ++vi){
            res[vi]=repr[ufd.find(vi)];
        }
        return ufd.numberOfSets();
    }

    void fuse(
        std::vector<LabelType> & a,
        std::vector<LabelType> & b,
        std::vector<LabelType> & res
    ){

        std::vector<LabelType> ab(gm_.numberOfVariables());
        size_t numNewVar = this->intersect(a, b, ab);
        std::cout<<"numNewVar "<<numNewVar<<"\n";


        // get the new smaller graph
        
    }


private:
    const GM & gm_;
};


}


#endif /* OPENGM_PERMUTABLE_LABEL_FUSION_MOVER_HXX */
