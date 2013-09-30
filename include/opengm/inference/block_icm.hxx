#ifndef OPENGM_BLOCKED_SOLVER_HXX
#define OPENGM_BLOCKED_SOLVER_HXX

#include <vector>
#include <string>
#include <iostream>

#include "opengm/opengm.hxx"
#include "opengm/inference/visitors/visitor.hxx"
#include "opengm/inference/inference.hxx"




#include <opengm/inference/messagepassing/messagepassing.hxx>
#include "opengm/inference/auxiliary/submodel/submodel_builder.hxx"



#include <opengm/inference/self_fusion.hxx>
#include <opengm/inference/infandflip.hxx>

#include "opengm/algorithms/split_gm.hxx"

namespace opengm {
  


template<class GM,class ACC>
struct BlockedSolverHelper{
    typedef SubmodelOptimizer<GM,ACC> SubOptimizer;
    typedef typename SubOptimizer::SubGmType SubGmType;
};


template<class GM,class INFERENCE>
class BlockedSolver : public Inference<GM, typename INFERENCE::AccumulationType>
{
public:

    typedef typename INFERENCE::AccumulationType AccumulationType;
    typedef GM GraphicalModelType;
    OPENGM_GM_TYPE_TYPEDEFS;
    typedef VerboseVisitor<BlockedSolver<GM,INFERENCE> > VerboseVisitorType;
    typedef EmptyVisitor<BlockedSolver<GM,INFERENCE> > EmptyVisitorType;
    typedef TimingVisitor<BlockedSolver<GM,INFERENCE> > TimingVisitorType;

    typedef typename INFERENCE::Parameter BlockInfParamType;


    typedef SubmodelOptimizer<GraphicalModelType,AccumulationType> SubOptimizer;
    typedef typename SubOptimizer::SubGmType SubGmType;
    // turn the base inference into a "self fusionated - lazy flipper improved solver"


    typedef SelfFusion<INFERENCE>  SelfFusedInf;
    typedef InfAndFlip<SubGmType,AccumulationType, SelfFusedInf>  InfAndFlipInf;





    class Parameter {
    public:
    Parameter(
        const size_t bisectionsLevels           = 1,
        const size_t blockMaxSubgraphSize       = 2,
        const bool optimalBlockSover            = false,
        const bool warmStartBlockSolver         = false,
        const bool fuseBlockSolverResult        = false,
        const BlockInfParamType & blockInfParam = BlockInfParamType()

    ) : 
        bisectionsLevels_(bisectionsLevels),
        blockMaxSubgraphSize_(blockMaxSubgraphSize),
        optimalBlockSover_(optimalBlockSover),
        warmStartBlockSolver_(warmStartBlockSolver),
        fuseBlockSolverResultr_(fuseBlockSolverResult),
        blockInfParam_(blockInfParam)
    {

    }
        size_t bisectionsLevels_;
        size_t blockMaxSubgraphSize_;
        bool optimalBlockSover_;
        bool warmStartBlockSolver_;
        bool fuseBlockSolverResultr_;
        BlockInfParamType blockInfParam_;
    };

   BlockedSolver(const GraphicalModelType&, const Parameter& = Parameter() );
   std::string name() const;
   const GraphicalModelType& graphicalModel() const;
   InferenceTermination infer();
   template<class VisitorType>
   InferenceTermination infer(VisitorType&);
   void setStartingPoint(typename std::vector<LabelType>::const_iterator);
   virtual InferenceTermination arg(std::vector<LabelType>&, const size_t = 1) const ;

    ValueType value()const{
        return value_;
    }



private:

    const GraphicalModelType& gm_;
    Parameter param_;

    // submodel
    std::vector<std::vector<IndexType> > blockModelVis_;

    SubOptimizer subOptimizer_;
    std::vector<IndexType> argBest_;
    std::vector<IndexType> argTest_;
    ValueType value_;
    std::vector< RandomAccessSet<IndexType> > blockAdj_;


};



template<class GM,class INFERENCE>
BlockedSolver<GM,INFERENCE>::BlockedSolver
(
      const GraphicalModelType& gm,
      const Parameter& parameter
)
:  gm_(gm),
   param_(parameter),
   subOptimizer_(gm),
   argBest_(gm.numberOfVariables(),0),
   argTest_(gm.numberOfVariables(),0),
   value_(),
   blockAdj_()
{
    AccumulationType::ineutral(value_);

    std::cout<<"graph bisection\n";

    std::vector<IndexType>  visToSplit(gm_.numberOfVariables());
    std::vector<IndexType>  varState;

    for(IndexType vi=0;vi<gm_.numberOfVariables();++vi){
        visToSplit[vi]=vi;
    }  

    std::cout<<"graph bisection start \n";
    const size_t nBlocks=recuriveGraphBisection(
        gm_,
        0 , //viA
        gm_.numberOfVariables()-1, // viB,
        visToSplit,
        param_.bisectionsLevels_,   // levels (4 partitions)
        varState
    );
    blockModelVis_.resize(nBlocks);

    std::cout<<"graph bisection end \n";

    for(IndexType vi=0;vi<gm_.numberOfVariables();++vi){
        //std::cout<<"vi "<<vi<<" "<<int (varState[vi])<<"\n";
        blockModelVis_[varState[vi]].push_back(vi);
    }


    for(IndexType vi=0;vi<gm_.numberOfVariables();++vi){
        subOptimizer_.setLabel(vi,0);
    }

    for(IndexType bi=0;bi<blockModelVis_.size();++bi){
           std::sort(blockModelVis_[bi].begin(),blockModelVis_[bi].end());
    }


    std::cout<<"compute block adj.";
    blockAdj_.resize(blockModelVis_.size());

    for(IndexType fi=0;fi<gm_.numberOfFactors();++fi){

        const IndexType order=gm_[fi].numberOfVariables();

        if(order>=2){
            for(IndexType va=0;va<order-1;++va){
                for(IndexType vb=va+1;vb<order;++vb){



                    const IndexType ba=varState[gm_[fi].variableIndex(va)];
                    const IndexType bb=varState[gm_[fi].variableIndex(vb)];
                    if(ba!=bb){
                        blockAdj_[ba].insert(bb);
                        blockAdj_[bb].insert(ba);
                    }
                }
            }
        }

    }
}
      

   
template<class GM,class INFERENCE>
inline void 
BlockedSolver<GM,INFERENCE>::setStartingPoint
(
   typename std::vector<typename BlockedSolver<GM,INFERENCE>::LabelType>::const_iterator begin
) {

}
   
template<class GM,class INFERENCE>
inline std::string
BlockedSolver<GM,INFERENCE>::name() const
{
   return "BlockedSolver";
}

template<class GM,class INFERENCE>
inline const typename BlockedSolver<GM,INFERENCE>::GraphicalModelType&
BlockedSolver<GM,INFERENCE>::graphicalModel() const
{
   return gm_;
}
  
template<class GM,class INFERENCE>
inline InferenceTermination
BlockedSolver<GM,INFERENCE>::infer()
{
   EmptyVisitorType v;
	//VerboseVisitorType v;
   return infer(v);
}

  
template<class GM,class INFERENCE>
template<class VisitorType>
InferenceTermination BlockedSolver<GM,INFERENCE>::infer
(
  VisitorType& visitor
)
{
    visitor.begin(*this);

    std::vector<bool> isClean(blockModelVis_.size(),false);


    typedef typename INFERENCE::Parameter     InfParam;
    typedef typename SelfFusedInf::Parameter  SelfFusedInfParameter;
    typedef typename InfAndFlipInf::Parameter InfAndFlipParameter;



    // setup self fused parameter
    SelfFusedInfParameter selfFusedParam;
    selfFusedParam.fuseNth_=1;
    selfFusedParam.fusionSolver_=SelfFusedInf::LazyFlipperFusion;
    selfFusedParam.infParam_=param_.blockInfParam_;
    selfFusedParam.maxSubgraphSize_=2;
    selfFusedParam.damping_=0.5;
    selfFusedParam.bpSteps_=10;


    // set up final parameter (InfAndFlip)
    InfAndFlipParameter infAndFlipParam;
    infAndFlipParam.maxSubgraphSize_=param_.blockMaxSubgraphSize_;
    infAndFlipParam.subPara_ = selfFusedParam;



    bool changes=true;
    while(changes){
        changes=false;


        std::cout<<" --- "<<blockModelVis_.size()<<"\n";

        for(IndexType bi=0;bi<blockModelVis_.size();++bi){

            if(!isClean[bi]){

                std::vector<LabelType> states;

                subOptimizer_.setVariableIndices(blockModelVis_[bi].begin(), blockModelVis_[bi].end());

                const ValueType valBevoreMove = gm_.evaluate(argBest_.begin());

                bool c = subOptimizer_. template inferSubmodel<InfAndFlipInf>(infAndFlipParam ,states,true,true);

                // write result into test arg
                for(IndexType svi=0;svi<blockModelVis_[bi].size();++svi){
                    argTest_[blockModelVis_[bi][svi]]=states[svi];
                }
                const ValueType valAfterMove = gm_.evaluate(argTest_.begin());

                std::cout<<blockModelVis_[bi].size()<<" value bevore "<<valBevoreMove<<" value After "<<valAfterMove << " d "<<valBevoreMove- valAfterMove<<"\n";

                bool improvment;
                OPENGM_CHECK(AccumulationType::bop(valBevoreMove,valAfterMove)==false,"infandflip error");
                if(AccumulationType::bop(valAfterMove,valBevoreMove)){
                    // write result into best  arg
                    for(IndexType svi=0;svi<blockModelVis_[bi].size();++svi){
                        argBest_[blockModelVis_[bi][svi]]=states[svi];
                        const IndexType s=states[svi];
                        const IndexType nl=gm_.numberOfLabels(blockModelVis_[bi][svi]);
                        OPENGM_CHECK(s<nl ," ");
                        subOptimizer_.setLabel(blockModelVis_[bi][svi],s);
                    }
                    changes=true;
                    improvment=true;
                }
                else{
                    // write result into best  arg
                    for(IndexType svi=0;svi<blockModelVis_[bi].size();++svi){
                        argTest_[blockModelVis_[bi][svi]]=argBest_[blockModelVis_[bi][svi]];
                    }
                    improvment=false;
                }
                subOptimizer_.unsetVariableIndices();

                // set own block to clean
                isClean[bi]=true;
                // changes (only if improved)
                if(improvment){
                    const IndexType nABlocks=blockAdj_[bi].size();
                    for(IndexType ab=0;ab<nABlocks;++ab){
                        isClean[blockAdj_[bi][ab]]=false;
                    }
                }
            }
            else{
                std::cout<<"skip clean block\n";
            }

        }
    }

    visitor.end(*this);
    return NORMAL;
}

template<class GM,class INFERENCE>
inline InferenceTermination
BlockedSolver<GM,INFERENCE>::arg
(
    std::vector<LabelType>& x,
    const size_t N
) const
{
    x.resize(gm_.numberOfVariables());
    for(IndexType vi=0;vi<gm_.numberOfVariables();++vi){
        x[vi]=argBest_[vi];
    }
    return NORMAL;
}

} // namespace opengm

#endif // #ifndef OPENGM_BLOCKED_SOLVER_HXX
