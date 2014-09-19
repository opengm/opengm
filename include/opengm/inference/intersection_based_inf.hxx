#pragma once
#ifndef OPENGM_FUSION_BASED_INF_HXX
#define OPENGM_FUSION_BASED_INF_HXX

#include <vector>
#include <string>
#include <iostream>

#include "opengm/opengm.hxx"
#include "opengm/inference/inference.hxx"
#include "opengm/inference/visitors/visitors.hxx"
#include "opengm/utilities/random.hxx"

// Fusion Move Solver (they solve binary problems)
#include "opengm/inference/astar.hxx"
#include "opengm/inference/lazyflipper.hxx"
#include "opengm/inference/infandflip.hxx"
#include "opengm/inference/messagepassing/messagepassing.hxx"

#ifdef WITH_CPLEX
#include "opengm/inference/lpcplex.hxx"
#include "opengm/inference/auxiliary/fusion_move/permutable_label_fusion_mover.hxx"
#endif



#include <stdlib.h>     /* srand, rand */


#include "opengm/inference/lazyflipper.hxx"

// fusion move model generator
#include "opengm/inference/auxiliary/fusion_move/fusion_mover.hxx"







#ifndef NOVIGRA
#include <vigra/adjacency_list_graph.hxx>
#include <vigra/merge_graph_adaptor.hxx>
#include <vigra/hierarchical_clustering.hxx>
#include <vigra/priority_queue.hxx>
#include <vigra/random.hxx>
#include <vigra/graph_algorithms.hxx>
#endif 





namespace opengm
{











namespace proposal_gen{

    #ifndef NOVIGRA


    template<class VALUE_TYPE>
    class WeightRandomization{
    public:
        typedef VALUE_TYPE ValueType;



        struct Parameter{

            enum NoiseType{
                NormalAdd = 0 ,
                UniformAdd = 1,
                NormalMult = 2,
                None = 3
            };

            Parameter(
                const NoiseType noiseType = NormalAdd,
                const float     noiseParam = 1.0,
                const UInt32    seed = 42,
                const bool      ignoreSeed = true
            )
            : 
            noiseType_(noiseType),
            noiseParam_(noiseParam_),
            seed_(seed),
            ignoreSeed_(ignoreSeed){

            }

            NoiseType noiseType_;
            float noiseParam_;
            UInt32 seed_;
            bool ignoreSeed_;
        };

        WeightRandomization(const Parameter & param = Parameter())
        : 
            param_(param),
            randGen_(param.seed_, param.ignoreSeed_){

        }

        void randomize(const std::vector<ValueType> & weights, std::vector<ValueType> & rweights){
            // PERTUBATION 


            if (param_.noiseType_ == Parameter::NormalAdd){
                for(size_t i=0; i<rweights.size(); ++i){
                    rweights[i] = weights[i] + randGen_.normal()*param_.noiseParam_;
                }
            }
            else if(param_.noiseType_ == Parameter::UniformAdd){
                for(size_t i=0; i<rweights.size(); ++i){
                    rweights[i] = weights[i] + randGen_.uniform(-1.0*param_.noiseParam_, param_.noiseParam_);
                }
            }
            else if(param_.noiseType_ == Parameter::NormalMult){
                for(size_t i=0; i<rweights.size(); ++i){
                    rweights[i] = weights[i]*randGen_.normal(1.0, param_.noiseParam_);
                }
            }
            else if(param_.noiseType_ == Parameter::None){
                rweights = weights;
            }
            else{
                throw RuntimeError("wrong noise type");
            }

        }
        vigra::RandomNumberGenerator< > & randGen(){
            return randGen_;
        }

    private:
        Parameter param_;
        vigra::RandomNumberGenerator< > randGen_;

    };



    template<class GM, class ACC >
    class McClusterOp{
    public:
        typedef ACC AccumulationType;
        typedef GM GraphicalModelType;
        OPENGM_GM_TYPE_TYPEDEFS;

        typedef vigra::AdjacencyListGraph Graph;
        typedef vigra::MergeGraphAdaptor< Graph > MergeGraph;


        typedef WeightRandomization<ValueType> WeightRand;
        typedef typename  WeightRand::Parameter WeightRandomizationParam;

        typedef typename MergeGraph::Edge Edge;
        typedef ValueType WeightType;
        typedef IndexType index_type;
        struct Parameter
        {



            Parameter(
                const WeightRandomizationParam & randomizer  = WeightRandomizationParam(),
                const float stopWeight = 0.0,
                const float reduction = -1.0
            )
            :
                randomizer_(randomizer),
                stopWeight_(stopWeight),
                reduction_(reduction)
            {

            }
            WeightRandomizationParam  randomizer_;
            float stopWeight_;
            float reduction_;
        };


        McClusterOp(const Graph & graph , MergeGraph & mergegraph, 
                    const Parameter & param)
        :
            graph_(graph),
            mergeGraph_(mergegraph),
            pq_(graph.edgeNum()),
            param_(param),
            nodeStopNum_(0),
            rWeights_(graph.edgeNum()),
            wRandomizer_(param_.randomizer_){

            if(param_.reduction_>0.000001){
                float keep = 1.0 - param_.reduction_;
                keep = std::max(0.0f, keep);
                keep = std::min(1.0f, keep);
                nodeStopNum_ = IndexType(float(graph_.nodeNum())*keep);
            }

        }




        void reset(){
            pq_.reset();
        }


        void setWeights(const std::vector<ValueType> & weights){
            
            wRandomizer_.randomize(weights, rWeights_);

            for(size_t i=0; i<graph_.edgeNum(); ++i){
                pq_.push(i, rWeights_[i]);
            }
        }

        Edge contractionEdge(){
            index_type minLabel = pq_.top();
            while(mergeGraph_.hasEdgeId(minLabel)==false){
                pq_.deleteItem(minLabel);
                minLabel = pq_.top();
            }
            return Edge(minLabel);
        }

        /// \brief get the edge weight of the edge which should be contracted next
        WeightType contractionWeight(){
            index_type minLabel = pq_.top();
            while(mergeGraph_.hasEdgeId(minLabel)==false){
                pq_.deleteItem(minLabel);
                minLabel = pq_.top();
            }
            return pq_.topPriority();

        }

        /// \brief get a reference to the merge
        MergeGraph & mergeGraph(){
            return mergeGraph_;
        }

        bool done()const{
            const bool doneByWeight = pq_.topPriority()<=ValueType(param_.stopWeight_);
            const bool doneByNodeNum = mergeGraph_.nodeNum()<nodeStopNum_;
            return doneByWeight || doneByNodeNum;
        }

        void mergeEdges(const Edge & a,const Edge & b){
            rWeights_[a.id()]+=rWeights_[b.id()];
            pq_.push(a.id(), rWeights_[a.id()]);
            pq_.deleteItem(b.id());
        }

        void eraseEdge(const Edge & edge){
            pq_.deleteItem(edge.id());
        }

        const Graph & graph_;
        MergeGraph & mergeGraph_;
        vigra::ChangeablePriorityQueue< ValueType ,std::greater<ValueType> > pq_;
        Parameter param_;
        size_t nodeStopNum_;
        std::vector<ValueType> rWeights_;
        WeightRand wRandomizer_;
    };




    template<class GM, class ACC>
    class RandomizedHierarchicalClustering{
    public:
        typedef ACC AccumulationType;
        typedef GM GraphicalModelType;
        OPENGM_GM_TYPE_TYPEDEFS;

        typedef vigra::AdjacencyListGraph Graph;
        typedef vigra::MergeGraphAdaptor< Graph > MGraph;
        
        typedef WeightRandomization<ValueType> WeightRand;
        typedef typename  WeightRand::Parameter WeightRandomizationParam;



        typedef typename  Graph::Edge GraphEdge;

        struct Parameter
        {



            Parameter(
                const WeightRandomizationParam & randomizer = WeightRandomizationParam(),
                const float stopWeight = 0.0,
                const float reduction = -1.0
            )
            :
                randomizer_(randomizer),
                stopWeight_(stopWeight),
                reduction_(reduction)
            {

            }
            WeightRandomizationParam  randomizer_;
            float stopWeight_;
            float reduction_;
        };

        typedef McClusterOp<GM, ACC > Cop;
        typedef typename Cop::Parameter CopParam;
        typedef vigra::HierarchicalClustering< Cop > HC;
        typedef typename HC::Parameter HcParam;




        RandomizedHierarchicalClustering(const GM & gm, const Parameter & param = Parameter())
        : 
            gm_(gm),
            weights_(gm.numberOfFactors(),ValueType(0.0)),
            param_(param),
            graph_(),
            mgraph_(NULL),
            clusterOp_(NULL)
        {


            LabelType lAA[2]={0, 0};
            LabelType lAB[2]={0, 1};

            //std::cout<<"add nodes\n";
            for(size_t i=0; i<gm_.numberOfVariables();++i){
                graph_.addNode(i);
            }

            //std::cout<<"add edges\n";
            for(size_t i=0; i<gm_.numberOfFactors(); ++i){
                if(gm_[i].numberOfVariables()==2){
                    ValueType val00  = gm_[i](lAA);
                    ValueType val01  = gm_[i](lAB);
                    ValueType weight = val01 - val00; 
                    const GraphEdge gEdge = graph_.addEdge(gm_[i].variableIndex(0),gm_[i].variableIndex(1));
                    weights_[gEdge.id()]+=weight;
                }
            }
            
        }

        ~RandomizedHierarchicalClustering(){
            if(mgraph_ != NULL){
                delete mgraph_;
                delete clusterOp_;
            }
        }
        size_t defaultNumStopIt() {return 100;}

        void reset(){

        }
        void getProposal(const std::vector<LabelType> &current , std::vector<LabelType> &proposal){




            if(mgraph_ == NULL){
                CopParam copParam(param_.randomizer_, param_.stopWeight_,
                                  param_.reduction_);
                mgraph_ = new MGraph(graph_);
                clusterOp_ = new Cop(graph_, *mgraph_ , copParam);
            }
            else{
                mgraph_->reset();
                clusterOp_->reset();
            }


            HcParam p;
            p.verbose_=false;
            p.buildMergeTreeEncoding_=false;


        
            //std::cout<<"alloc cluster op\n";
          

            //std::cout<<"set weights \n";
            clusterOp_->setWeights(weights_);

            //std::cout<<"alloc hc\n";
            HC hc(*clusterOp_,p);

            //std::cout<<"start\n";
            hc.cluster();

            //std::cout<<"get reps.\n";
            for(size_t i=0; i< gm_.numberOfVariables(); ++i){
                proposal[i] =   hc.reprNodeId(i);
            }
            //std::cout<<"get reps.done \n";

        }
    private:
        const GM & gm_;
        Parameter param_;
        std::vector<ValueType> weights_;
        vigra::AdjacencyListGraph graph_;
        MGraph * mgraph_;
        Cop * clusterOp_;
    };


    template<class G, class VEC>
    struct VectorViewEdgeMap{
    public:
        typedef typename G::Edge  Key;
        typedef typename VEC::value_type Value;
        typedef typename VEC::reference Reference;
        typedef typename VEC::const_reference ConstReference;

        VectorViewEdgeMap(const G & g, VEC & vec)
        :
            graph_(g),
            vec_(vec){
        }

        Reference operator[](const Key & key){
            return vec_[graph_.id(key)];
        }
        ConstReference operator[](const Key & key)const{
            return vec_[graph_.id(key)];
        }
    private:
        const G & graph_;
        VEC & vec_;
    };

    template<class G, class VEC>
    struct VectorViewNodeMap{
    public:
        typedef typename G::Node  Key;
        typedef typename VEC::value_type Value;
        typedef typename VEC::reference Reference;
        typedef typename VEC::const_reference ConstReference;

        VectorViewNodeMap(const G & g, VEC & vec)
        :
            graph_(g),
            vec_(vec){
        }

        Reference operator[](const Key & key){
            return vec_[graph_.id(key)];
        }
        ConstReference operator[](const Key & key)const{
            return vec_[graph_.id(key)];
        }
    private:
        const G & graph_;
        VEC & vec_;
    };




    template<class GM, class ACC>
    class RandomizedWatershed{
    public:
        typedef ACC AccumulationType;
        typedef GM GraphicalModelType;
        OPENGM_GM_TYPE_TYPEDEFS;

        typedef vigra::AdjacencyListGraph Graph;


        typedef typename  Graph::Edge GraphEdge;

        typedef typename  Graph:: template EdgeMap<ValueType> WeightMap;
        typedef typename  Graph:: template EdgeMap<UInt32>    LabelMap;

        typedef WeightRandomization<ValueType> WeightRand;
        typedef typename  WeightRand::Parameter WeightRandomizationParam;

        struct Parameter
        {




            Parameter(
                const float seedFraction = 0.01,
                const WeightRandomizationParam randomizer = WeightRandomizationParam()

            )
            :
                seedFraction_(seedFraction),
                randomizer_(randomizer)
            {

            }

            float seedFraction_;
            WeightRandomizationParam randomizer_;
        };




        RandomizedWatershed(const GM & gm, const Parameter & param = Parameter())
        : 
            gm_(gm),
            param_(param),
            wRandomizer_(param.randomizer_),
            graph_(),
            weights_(gm_.numberOfFactors()),
            rWeights_(),
            seeds_()
        {

            LabelType lAA[2]={0, 0};
            LabelType lAB[2]={0, 1};

            //std::cout<<"add nodes\n";
            for(size_t i=0; i<gm_.numberOfVariables();++i){
                graph_.addNode(i);
            }

            //std::cout<<"add edges\n";
            for(size_t i=0; i<gm_.numberOfFactors(); ++i){
                if(gm_[i].numberOfVariables()==2){
                    ValueType val00  = gm_[i](lAA);
                    ValueType val01  = gm_[i](lAB);
                    ValueType weight = val01 - val00; 
                    const GraphEdge gEdge = graph_.addEdge(gm_[i].variableIndex(0),gm_[i].variableIndex(1));
                    weights_[gEdge.id()]+=weight;
                }
            }
            weights_.resize(graph_.edgeNum());
            rWeights_.resize(graph_.edgeNum());
            seeds_.resize(graph_.maxNodeId()+1);
        }


        size_t defaultNumStopIt() {return 100;}

        void reset(){

        }
        void getProposal(const std::vector<LabelType> &current , std::vector<LabelType> &proposal){


            const size_t nSeeds = size_t(float(graph_.nodeNum())*param_.seedFraction_+0.5f);
            std::fill(seeds_.begin(), seeds_.end(), 0);


            // randomize weights
            wRandomizer_.randomize(weights_, rWeights_);

            // vectorViewEdgeMap
            VectorViewEdgeMap<Graph, std::vector<ValueType> > wMap(graph_, rWeights_);
            VectorViewNodeMap<Graph, std::vector<LabelType> > sMap(graph_, seeds_);
            VectorViewNodeMap<Graph, std::vector<LabelType> > lMap(graph_, proposal);

            for(size_t i=0; i<nSeeds; ++i){
                const int randId = wRandomizer_.randGen().uniformInt(graph_.nodeNum());
                seeds_[randId] = i+1;
            }

            // negate
            for(size_t i=0; i<graph_.edgeNum(); ++i){
                rWeights_[i] *= -1.0;
            }

            vigra::edgeWeightedWatershedsSegmentation(graph_, wMap, sMap, lMap);
        }
    private:
        const GM & gm_;
        Parameter param_;
        WeightRand wRandomizer_;
        Graph graph_;
        std::vector<ValueType>  weights_;
        std::vector<ValueType>  rWeights_;
        std::vector<LabelType>  seeds_;
    };



    #endif

}


template<class GM, class PROPOSAL_GEN>
class IntersectionBasedInf : public Inference<GM, typename  PROPOSAL_GEN::AccumulationType>
{
public:
    typedef PROPOSAL_GEN ProposalGen;
    typedef typename ProposalGen::AccumulationType AccumulationType;
    typedef AccumulationType ACC;
    typedef GM GraphicalModelType;
    OPENGM_GM_TYPE_TYPEDEFS;

    typedef opengm::visitors::VerboseVisitor<IntersectionBasedInf<GM, PROPOSAL_GEN> > VerboseVisitorType;
    typedef opengm::visitors::EmptyVisitor<IntersectionBasedInf<GM, PROPOSAL_GEN> >  EmptyVisitorType;
    typedef opengm::visitors::TimingVisitor<IntersectionBasedInf<GM, PROPOSAL_GEN> > TimingVisitorType;

    typedef PermutableLabelFusionMove<GraphicalModelType, AccumulationType>  FusionMoverType;

    typedef typename ProposalGen::Parameter ProposalParameter;
    typedef typename FusionMoverType::Parameter FusionParameter;



    class Parameter
    {
    public:
        Parameter(
            const ProposalParameter & proposalParam = ProposalParameter(),
            const FusionParameter   & fusionParam = FusionParameter(),
            const size_t numIt=1000,
            const size_t numStopIt = 0
        )
            :   proposalParam_(proposalParam),
                fusionParam_(fusionParam),
                numIt_(numIt),
                numStopIt_(numStopIt)
        {

        }
        ProposalParameter proposalParam_;
        FusionParameter fusionParam_;
        size_t numIt_;
        size_t numStopIt_;

    };


    IntersectionBasedInf(const GraphicalModelType &, const Parameter & = Parameter() );
    ~IntersectionBasedInf();
    std::string name() const;
    const GraphicalModelType &graphicalModel() const;
    InferenceTermination infer();
    void reset();
    template<class VisitorType>
    InferenceTermination infer(VisitorType &);
    void setStartingPoint(typename std::vector<LabelType>::const_iterator);
    virtual InferenceTermination arg(std::vector<LabelType> &, const size_t = 1) const ;
    virtual ValueType value()const {return bestValue_;}
private:


    const GraphicalModelType &gm_;
    Parameter param_;


    FusionMoverType * fusionMover_;


    PROPOSAL_GEN proposalGen_;
    ValueType bestValue_;
    std::vector<LabelType> bestArg_;
    size_t maxOrder_;
};




template<class GM, class PROPOSAL_GEN>
IntersectionBasedInf<GM, PROPOSAL_GEN>::IntersectionBasedInf
(
    const GraphicalModelType &gm,
    const Parameter &parameter
)
    :  gm_(gm),
       param_(parameter),
       fusionMover_(NULL),
       proposalGen_(gm_, parameter.proposalParam_),
       bestValue_(),
       bestArg_(gm_.numberOfVariables(), 0),
       maxOrder_(gm.factorOrder())
{
    ACC::neutral(bestValue_);   
    fusionMover_ = new FusionMoverType(gm_,parameter.fusionParam_);


    //set default starting point
    std::vector<LabelType> conf(gm_.numberOfVariables(),0);
    for (size_t i=0; i<gm_.numberOfVariables(); ++i){
        for(typename GM::ConstFactorIterator it=gm_.factorsOfVariableBegin(i); it!=gm_.factorsOfVariableEnd(i);++it){
            if(gm_[*it].numberOfVariables() == 1){
                ValueType v;
                ACC::neutral(v);
                for(LabelType l=0; l<gm_.numberOfLabels(i); ++l){
                    if(ACC::bop(gm_[*it](&l),v)){
                        v=gm_[*it](&l);
                        conf[i]=l;
                    }
                }
                continue;
            }
        } 
    }
    setStartingPoint(conf.begin());
}


template<class GM, class PROPOSAL_GEN>
IntersectionBasedInf<GM, PROPOSAL_GEN>::~IntersectionBasedInf()
{

    delete fusionMover_;
}


template<class GM, class PROPOSAL_GEN>
inline void
IntersectionBasedInf<GM, PROPOSAL_GEN>::reset()
{
    throw RuntimeError("not implemented yet");
}

template<class GM, class PROPOSAL_GEN>
inline void
IntersectionBasedInf<GM, PROPOSAL_GEN>::setStartingPoint
(
    typename std::vector<typename IntersectionBasedInf<GM, PROPOSAL_GEN>::LabelType>::const_iterator begin
)
{
    std::copy(begin, begin + gm_.numberOfVariables(), bestArg_.begin());
    bestValue_ = gm_.evaluate(bestArg_.begin());
}

template<class GM, class PROPOSAL_GEN>
inline std::string
IntersectionBasedInf<GM, PROPOSAL_GEN>::name() const
{
    return "IntersectionBasedInf";
}

template<class GM, class PROPOSAL_GEN>
inline const typename IntersectionBasedInf<GM, PROPOSAL_GEN>::GraphicalModelType &
IntersectionBasedInf<GM, PROPOSAL_GEN>::graphicalModel() const
{
    return gm_;
}

template<class GM, class PROPOSAL_GEN>
inline InferenceTermination
IntersectionBasedInf<GM, PROPOSAL_GEN>::infer()
{
    EmptyVisitorType v;
    return infer(v);
}


template<class GM, class PROPOSAL_GEN>
template<class VisitorType>
InferenceTermination IntersectionBasedInf<GM, PROPOSAL_GEN>::infer
(
    VisitorType &visitor
)
{
    // evaluate the current best state
    bestValue_ = gm_.evaluate(bestArg_.begin());

    visitor.begin(*this);


    if(param_.numStopIt_ == 0){
        param_.numStopIt_ = proposalGen_.defaultNumStopIt();
    }

    std::vector<LabelType> proposedState(gm_.numberOfVariables());
    std::vector<LabelType> fusedState(gm_.numberOfVariables());

    size_t countRoundsWithNoImprovement = 0;

    for(size_t iteration=0; iteration<param_.numIt_; ++iteration){
        // store initial value before one proposal  round
        const ValueType valueBeforeRound = bestValue_;

        proposalGen_.getProposal(bestArg_,proposedState);

        // this might be to expensive
        ValueType proposalValue = gm_.evaluate(proposedState);
        //ValueType proposalValue = 100000000000000000000000.0;

   

        const bool anyVar = fusionMover_->fuse(bestArg_,proposedState, fusedState, 
                                                 bestValue_, proposalValue, bestValue_);
        if(anyVar){
            if( !ACC::bop(bestValue_, valueBeforeRound)){
                ++countRoundsWithNoImprovement;
            }
            else{
                // Improvement
                countRoundsWithNoImprovement = 0;
                bestArg_ = fusedState;
            }
            if(visitor(*this)!=0){
                break;
            }
        }
        else{
            ++countRoundsWithNoImprovement;
        }
        // check if converged or done
        if(countRoundsWithNoImprovement==param_.numStopIt_ && param_.numStopIt_ !=0 )
            break;
    }
    visitor.end(*this);
    return NORMAL;
}




template<class GM, class PROPOSAL_GEN>
inline InferenceTermination
IntersectionBasedInf<GM, PROPOSAL_GEN>::arg
(
    std::vector<LabelType> &x,
    const size_t N
) const
{
    if (N == 1)
    {
        x.resize(gm_.numberOfVariables());
        for (size_t j = 0; j < x.size(); ++j)
        {
            x[j] = bestArg_[j];
        }
        return NORMAL;
    }
    else
    {
        return UNKNOWN;
    }
}

} // namespace opengm

#endif // #ifndef OPENGM_FUSION_BASED_INF_HXX
