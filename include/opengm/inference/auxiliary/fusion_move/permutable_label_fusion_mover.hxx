#ifndef OPENGM_PERMUTABLE_LABEL_FUSION_MOVER_HXX
#define OPENGM_PERMUTABLE_LABEL_FUSION_MOVER_HXX


#include <opengm/inference/inference.hxx>
#include <opengm/inference/multicut.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/graphicalmodel/space/simplediscretespace.hxx>
#include <opengm/functions/potts.hxx>


namespace opengm{


template<class GM, class ACC>
class PermutableLabelFusionMove{

public:

    typedef GM GraphicalModelType;
    typedef ACC AccumulationType;
    OPENGM_GM_TYPE_TYPEDEFS;


    typedef SimpleDiscreteSpace<IndexType, LabelType>       SubSpace;
    typedef PottsFunction<ValueType, IndexType, LabelType>  PFunction;
    typedef GraphicalModel<ValueType, Adder, PFunction , SubSpace> SubModel;


    PermutableLabelFusionMove(const GraphicalModelType & gm)
    :   
        gm_(gm)
    {

    }



    void printArg(const std::vector<LabelType> arg) {
         const size_t nx = 3; // width of the grid
        const size_t ny = 3; // height of the grid
        const size_t numberOfLabels = nx*ny;

        size_t i=0;
        for(size_t y = 0; y < ny; ++y){
            
            for(size_t x = 0; x < nx; ++x) {
                std::cout<<arg[i]<<"  ";
            }
            std::cout<<"\n";
            ++i;
        }
        
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
        //std::cout<<" A\n";
        //printArg(a);
        //std::cout<<" B\n";
        //printArg(b);
        //std::cout<<" RES\n";
        //printArg(res);

        return ufd.numberOfSets();
    }

    bool fuse(
        const std::vector<LabelType> & a,
        const std::vector<LabelType> & b,
        std::vector<LabelType> & res,
        const ValueType valA,
        const ValueType valB,
        ValueType & valRes
    ){

        std::vector<LabelType> ab(gm_.numberOfVariables());
        size_t numNewVar = this->intersect(a, b, ab);
        //std::cout<<"numNewVar "<<numNewVar<<"\n";

        if(numNewVar==1){
            return false;
        }

        const ValueType intersectedVal = gm_.evaluate(ab);



        // get the new smaller graph
        typedef std::map<UInt64Type, float> MapType;
        MapType accWeights;
        typedef typename MapType::iterator MapIter;
        typedef typename MapType::const_iterator MapCIter;
        size_t erasedEdges = 0;
        size_t notErasedEdges = 0;


        LabelType lAA[2]={0, 0};
        LabelType lAB[2]={0, 1};




        for(size_t fi=0; fi< gm_.numberOfFactors(); ++fi){
            if(gm_[fi].numberOfVariables()==2){
                const size_t vi0 = gm_[fi].variableIndex(0);
                const size_t vi1 = gm_[fi].variableIndex(1);

                const size_t cVi0 = ab[vi0] < ab[vi1] ? ab[vi0] : ab[vi1];
                const size_t cVi1 = ab[vi0] < ab[vi1] ? ab[vi1] : ab[vi0];

                OPENGM_CHECK_OP(cVi0,<,gm_.numberOfVariables(),"");
                OPENGM_CHECK_OP(cVi1,<,gm_.numberOfVariables(),"");


                if(cVi0 == cVi1){
                    ++erasedEdges;
                }
                else{
                    ++notErasedEdges;

                    // get the weight
                    ValueType val00  = gm_[fi](lAA);
                    ValueType val01  = gm_[fi](lAB);
                    ValueType weight = val01 - val00; 

                    //std::cout<<"vAA"<<val00<<" vAB "<<val01<<" w "<<weight<<"\n";

                    // compute key
                    const UInt64Type key = cVi0 + numNewVar*cVi1;

                    // check if key is in map
                    MapIter iter = accWeights.find(key);

                    // key not yet in map
                    if(iter == accWeights.end()){
                        accWeights[key] = weight;
                    }
                    // key is in map 
                    else{
                        iter->second += weight;
                    }

                }

            }
        }
        OPENGM_CHECK_OP(erasedEdges+notErasedEdges, == , gm_.numberOfFactors(),"something wrong");
        //std::cout<<"erased edges      "<<erasedEdges<<"\n";
        //std::cout<<"not erased edges  "<<notErasedEdges<<"\n";
        //std::cout<<"LEFT OVER FACTORS "<<accWeights.size()<<"\n";


        // make the actual sub graphical model
        SubSpace subSpace(numNewVar, numNewVar);
        SubModel subGm(subSpace);

        for(MapCIter i = accWeights.begin(); i!=accWeights.end(); ++i){
            const UInt64Type key    = i->first;
            const ValueType weight = i->second;

            const UInt64Type cVi1 = key/numNewVar;
            const UInt64Type cVi0 = key - cVi1*numNewVar;
            const UInt64Type vis[2] = {cVi0, cVi1};

            //OPENGM_CHECK_OP(cVi0, < , cVi1, "internal error");
            //std::cout<<"vi0 "<<cVi0<<"vi1 "<<cVi1<<" w "<< weight<<"\n";

            PFunction pf(numNewVar, numNewVar, 0.0, weight);

            subGm.addFactor(subGm.addFunction(pf), vis, vis+2);
        }

        //std::cout<<"solve with multicuts\n";
        typedef Multicut<SubModel, Minimizer> Inf;
        typedef  typename  Inf::Parameter Param;
        Param p(0,0.0);
        //p.workFlow_ = "(MTC)(CC)";
        //p.integerConstraint_ = true;
        Multicut<SubModel, Minimizer> multicut(subGm,p);
        multicut.infer();

        std::vector<LabelType> subArg;
        multicut.arg(subArg);

        const ValueType resultVal = subGm.evaluate(subArg);

        

        // translate sub arg to arg

        for(IndexType vi=0; vi<gm_.numberOfVariables(); ++vi){
            res[vi] = subArg[ab[vi]];
        }

        const ValueType projectedResultVal = gm_.evaluate(res);
        valRes = projectedResultVal;
        //std::cout<<"valA "<<valA<<" valB "<<valB<<" valRes "<<resultVal<<" pResVal "<<projectedResultVal<<" intersectedVal "<<intersectedVal<<"\n";

        return true;
    }


private:
    const GM & gm_;
};





}


#endif /* OPENGM_PERMUTABLE_LABEL_FUSION_MOVER_HXX */
