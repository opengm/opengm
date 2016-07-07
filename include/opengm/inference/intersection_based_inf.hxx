#pragma once
#ifndef OPENGM_INTERSECTION_BASED_INF_HXX
#define OPENGM_INTERSECTION_BASED_INF_HXX

#include <vector>
#include <string>
#include <iostream>

#include <omp.h>

#include "opengm/opengm.hxx"
#include "opengm/inference/inference.hxx"
#include "opengm/inference/visitors/visitors.hxx"
#include "opengm/utilities/random.hxx"

// Fusion Move Solver (they solve binary problems)
#include "opengm/inference/astar.hxx"
#include "opengm/inference/lazyflipper.hxx"
#include "opengm/inference/infandflip.hxx"
#include "opengm/inference/messagepassing/messagepassing.hxx"

#if defined(WITH_CPLEX) || defined(WITH_QPBO) || (defined(WITH_PLANARITY) && defined(WITH_BLOSSOM5)) 
#include "opengm/inference/auxiliary/fusion_move/permutable_label_fusion_mover.hxx"
#endif

#ifdef WITH_QPBO
#include <QPBO.h>
#endif

#include <stdlib.h>     /* srand, rand */


#include "opengm/inference/lazyflipper.hxx"


#include <fstream>
#include <iterator>
#include <string>
#include <vector>

// FIXME
//using namespace std;
//#define Isinf Isinf2
#include <opengm/inference/cgc.hxx>


#include <vigra/adjacency_list_graph.hxx>
#include <vigra/merge_graph_adaptor.hxx>
#include <vigra/hierarchical_clustering.hxx>
#include <vigra/priority_queue.hxx>
#include <vigra/random.hxx>
#include <vigra/graph_algorithms.hxx>



namespace opengm
{




template<class INFERENCE, class INTERSECTION_BASED, class INTERSECTION_BASED_VISITOR>
class CgcRedirectingVisitor{

public:

    CgcRedirectingVisitor(INTERSECTION_BASED & intersectionBased, INTERSECTION_BASED_VISITOR & otherVisitor)
    :   intersectionBased_(intersectionBased),
        otherVisitor_(otherVisitor){

    }


    void begin(INFERENCE & inf){

    }

    size_t operator()(INFERENCE & inf){
        inf.arg(intersectionBased_._getArgRef());
        intersectionBased_._setBestVal(inf.value());
        return otherVisitor_(intersectionBased_);
    }

    void end(INFERENCE & inf){


    }

    void addLog(const std::string & logName){



    }

    void log(const std::string & logName,const double logValue){



    }

private:
    INTERSECTION_BASED & intersectionBased_;
    INTERSECTION_BASED_VISITOR & otherVisitor_;
};






namespace proposal_gen{

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
                const size_t    seed = 42,
                const bool      ignoreSeed = true,
                const bool      autoScale = false,
                const float     permuteN = -1.0
            )
            : 
            noiseType_(noiseType),
            noiseParam_(noiseParam),
            seed_(seed),
            ignoreSeed_(ignoreSeed),
            autoScale_(autoScale),
            permuteN_(permuteN)
            {

            }

            NoiseType noiseType_;
            float noiseParam_;
            size_t seed_;
            bool ignoreSeed_;
            bool autoScale_;
            float permuteN_;
        };

        WeightRandomization(const Parameter & param = Parameter())
        : 
            param_(param),
            randGen_(vigra::UInt32(param.seed_), param.ignoreSeed_),
            calclulatedMinMax_(false){

        }

        void randomize(const std::vector<ValueType> & weights, std::vector<ValueType> & rweights){
 
            if(param_.autoScale_ && !calclulatedMinMax_){
                // find the min max 
                ValueType wmin = std::numeric_limits<ValueType>::infinity();
                ValueType wmax = static_cast<ValueType>(-1.0)*std::numeric_limits<ValueType>::infinity();
                for(size_t i=0; i<rweights.size(); ++i){
                    wmin = std::min(weights[i], wmin);
                    wmax = std::max(weights[i], wmax);
                }
                ValueType range = wmax - wmin;
                calclulatedMinMax_ = true;
                param_.noiseParam_  *= range;
            }


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
                std::copy(weights.begin(), weights.begin()+rweights.size(), rweights.begin());
            }
            else{
                throw RuntimeError("wrong noise type");
            }



            // DO PERMUTATION
            if(param_.permuteN_ > 0.0){
                size_t nP = param_.permuteN_ > 1.0 ? 
                    size_t(param_.permuteN_) : 
                    size_t(param_.permuteN_ * float(rweights.size()));

                for(size_t p=0; p< nP; ++p){
                    size_t fi0 = randGen_.uniformInt(rweights.size());
                    size_t fi1 = randGen_.uniformInt(rweights.size());
                    if(fi0!=fi1){
                        std::swap(rweights[fi0], rweights[fi1]);
                    }
                }
            }

        }
        vigra::RandomNumberGenerator< > & randGen(){
            return randGen_;
        }

    private:
        Parameter param_;
        vigra::RandomNumberGenerator< > randGen_;

        bool calclulatedMinMax_;
    };


    #ifndef NOVIGRA
    template<class GM, class ACC >
    class RandMcClusterOp{
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
                const float nodeNum = -1.0,
                const float ignoreNegativeWeights = false,
                const bool setCutToZero = false
            )
            :
                randomizer_(randomizer),
                stopWeight_(stopWeight),
                nodeStopNum_(nodeNum),
                setCutToZero_(setCutToZero)
            {

            }
            WeightRandomizationParam  randomizer_;
            float stopWeight_;
            float nodeStopNum_;
            float ignoreNegativeWeights_;
            bool setCutToZero_;
        };


        RandMcClusterOp(const Graph & graph , MergeGraph & mergegraph, 
                    const Parameter & param)
        :
            graph_(graph),
            mergeGraph_(mergegraph),
            pq_(graph.edgeNum()),
            param_(param),
            nodeStopNum_(0),
            rWeights_(graph.edgeNum()),
            wRandomizer_(param.randomizer_){

            if(param_.nodeStopNum_>0.000001 && param_.nodeStopNum_<=1.0 ){
                float keep = param_.nodeStopNum_;
                keep = std::max(0.0f, keep);
                keep = std::min(1.0f, keep);
                nodeStopNum_ = IndexType(float(graph_.nodeNum())*keep);
            }
            else if(param_.nodeStopNum_ >= 1.0){
                nodeStopNum_ = param_.nodeStopNum_ ;
            }
            else{
                nodeStopNum_ = 2;
            }

        }




        void reset(){
            pq_.reset();
        }


        void setWeights(const std::vector<ValueType> & weights,
                        const std::vector<LabelType> & labels){
            
            wRandomizer_.randomize(weights, rWeights_);


            if(param_.setCutToZero_ == false){
                for(size_t i=0; i<graph_.edgeNum(); ++i){
                    pq_.push(i, rWeights_[i]);
                }
            }
            else{
                for(size_t i=0; i<graph_.edgeNum(); ++i){
                    size_t u = graph_.id(graph_.u(graph_.edgeFromId(i)));
                    size_t v = graph_.id(graph_.v(graph_.edgeFromId(i)));
                    if(labels[u] == labels[v])
                        pq_.push(i, rWeights_[i]);
                    else
                        pq_.push(i, 0.0);
                }
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

        typedef RandMcClusterOp<GM, ACC > Cop;
        typedef typename Cop::Parameter CopParam;
        typedef vigra::HierarchicalClusteringImpl< Cop > HC;
        typedef typename HC::Parameter HcParam;



        typedef typename  Graph::Edge GraphEdge;

        struct Parameter : public  CopParam
        {



            Parameter(
                const WeightRandomizationParam & randomizer = WeightRandomizationParam(),
                const float stopWeight = 0.0,
                const float nodeStopNum = -1.0,
                const bool ignoreNegativeWeights = false,
                const bool setCutToZero = true
            )
            :  CopParam(randomizer, stopWeight, nodeStopNum,setCutToZero)
            {

            }
        };





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
                    const ValueType val00  = gm_[i](lAA);
                    const ValueType val01  = gm_[i](lAB);
                    const ValueType weight = val01 - val00; 
                    if(!param_.ignoreNegativeWeights_ || weight >= 0.0){
                        const GraphEdge gEdge = graph_.addEdge(gm_[i].variableIndex(0),gm_[i].variableIndex(1));
                        weights_[gEdge.id()]+=weight;
                    }
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
                mgraph_ = new MGraph(graph_);
                clusterOp_ = new Cop(graph_, *mgraph_ , param_);
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
            clusterOp_->setWeights(weights_, current);

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


    template<class GM, class ACC>
    class RandomizedWatershed{
    public:
        typedef ACC AccumulationType;
        typedef GM GraphicalModelType;
        OPENGM_GM_TYPE_TYPEDEFS;

        typedef vigra::AdjacencyListGraph Graph;


        typedef typename  Graph::Edge GraphEdge;

        typedef typename  Graph:: template EdgeMap<ValueType> WeightMap;
        typedef typename  Graph:: template EdgeMap<vigra::UInt32>    LabelMap;

        typedef WeightRandomization<ValueType> WeightRand;
        typedef typename  WeightRand::Parameter WeightRandomizationParam;

        struct Parameter
        {



            Parameter(
                const float seedFraction = 0.01,
                const bool ignoreNegativeWeights = false,
                const bool seedFromNegativeEdges = true,
                const WeightRandomizationParam randomizer = WeightRandomizationParam()

            )
            :
                seedFraction_(seedFraction),
                ignoreNegativeWeights_(ignoreNegativeWeights),
                seedFromNegativeEdges_(seedFromNegativeEdges),
                randomizer_(randomizer)
            {

            }

            float seedFraction_;
            bool ignoreNegativeWeights_;
            bool seedFromNegativeEdges_;
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
            seeds_(),
            negativeFactors_()
        {

            LabelType lAA[2]={0, 0};
            LabelType lAB[2]={0, 1};

            //std::cout<<"add nodes\n";
            for(size_t i=0; i<gm_.numberOfVariables();++i){
                graph_.addNode(i);
            }

            for(size_t i=0; i<gm_.numberOfFactors(); ++i){
                if(gm_[i].numberOfVariables()==2){
                    ValueType val00  = gm_[i](lAA);
                    ValueType val01  = gm_[i](lAB);
                    ValueType weight = val01 - val00; 
                    if(!param_.ignoreNegativeWeights_ || weight >= 0.0){
                        const GraphEdge gEdge = graph_.addEdge(gm_[i].variableIndex(0),gm_[i].variableIndex(1));
                        weights_[gEdge.id()]+=weight;
                    }
                    if(param_.seedFromNegativeEdges_ && weight < 0.0){
                        negativeFactors_.push_back(i);
                    }
                }
            }

            //weights_.resize(graph_.edgeNum());
            rWeights_.resize(graph_.edgeNum());
            seeds_.resize(graph_.maxNodeId()+1);
        }


        size_t defaultNumStopIt() {return 100;}

        void reset(){

        }
        void getProposal(const std::vector<LabelType> &current , std::vector<LabelType> &proposal){


            const size_t nSeeds = param_.seedFraction_ <=1.0 ? 
                size_t(float(graph_.nodeNum())*param_.seedFraction_+0.5f) :
                size_t(param_.seedFraction_ + 0.5);
            std::fill(seeds_.begin(), seeds_.end(), 0);


            // randomize weights
            wRandomizer_.randomize(weights_, rWeights_);

            // vectorViewEdgeMap
            VectorViewEdgeMap<Graph, std::vector<ValueType> > wMap(graph_, rWeights_);
            VectorViewNodeMap<Graph, std::vector<LabelType> > sMap(graph_, seeds_);
            VectorViewNodeMap<Graph, std::vector<LabelType> > lMap(graph_, proposal);


            if(!param_.seedFromNegativeEdges_){
                for(size_t i=0; i<nSeeds; ++i){
                    const int randId = wRandomizer_.randGen().uniformInt(graph_.nodeNum());
                    seeds_[randId] = i+1;
                }
            }
            else{
                //std::cout<<"using n seeds = "<<nSeeds<<"\n";


                for(size_t i=0; i<nSeeds/2; ++i){
                    const int randId = wRandomizer_.randGen().uniformInt(negativeFactors_.size());
                    const IndexType fi  = negativeFactors_[randId];

                    //std::cout<<" fi "<<fi<<" ";
                    const IndexType vi0 = gm_[fi].variableIndex(0);
                    const IndexType vi1 = gm_[fi].variableIndex(1);

                    seeds_[vi0] = (2*i)+1;
                    seeds_[vi1] = (2*i+1)+1;
                }
                //std::cout<<"\n";
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
        std::vector<IndexType> negativeFactors_;
    };
    #endif


    #ifdef WITH_QPBO
    template<class GM, class ACC>
    class QpboBased{
    public:
        typedef ACC AccumulationType;
        typedef GM GraphicalModelType;
        OPENGM_GM_TYPE_TYPEDEFS;

        typedef WeightRandomization<ValueType> WeightRand;
        typedef typename  WeightRand::Parameter WeightRandomizationParam;

        class Parameter
        {
        public:
            Parameter(
                const WeightRandomizationParam & randomizer = WeightRandomizationParam()
            )
            : randomizer_(randomizer)
            {

            }
            WeightRandomizationParam randomizer_;
        };




        QpboBased(const GM & gm, const Parameter & param = Parameter())
        : 
            gm_(gm),
            param_(param),
            qpbo_(NULL),
            //qpbo_(int(gm.numberOfVariables()), int(gm.numberOfFactors()), NULL),
            iteration_(0),
            weights_(gm.numberOfFactors()),
            rweights_(gm.numberOfFactors()),
            wRandomizer_(param_.randomizer_)
        {
            //srand(42);
            qpbo_ = new kolmogorov::qpbo::QPBO<ValueType>(int(gm.numberOfVariables()), 
                                                          int(gm.numberOfFactors()), NULL);

            LabelType lAA[2]={0, 0};
            LabelType lAB[2]={0, 1};


            for(size_t i=0; i<gm_.numberOfFactors(); ++i){
                if(gm_[i].numberOfVariables()==2){
                    ValueType val00  = gm_[i](lAA);
                    ValueType val01  = gm_[i](lAB);
                    ValueType weight = val01 - val00; 
                    weights_[i]=weight;
                }
            }

        }

        ~QpboBased(){
            delete qpbo_;
        }

        size_t defaultNumStopIt() {
            return 0;
        }

        void reset(){

        }
        void getProposal(const std::vector<LabelType> &current , std::vector<LabelType> &proposal){

            LabelType lAA[2]={0, 0};
            LabelType lAB[2]={0, 1};


            if(iteration_>0){
               qpbo_->Reset();
            }

            // randomize
            // randomize weights
            wRandomizer_.randomize(weights_, rweights_);

            // add nodes
            qpbo_->AddNode(gm_.numberOfVariables());

            // add edges
            for(size_t i=0; i<gm_.numberOfFactors(); ++i){
               if(gm_[i].numberOfVariables()==2){


                   // check if current edge is cut

                   const IndexType vi0 = gm_[i].variableIndex(0);
                   const IndexType vi1 = gm_[i].variableIndex(1);

                   if(current[vi0] == current[vi1]){
                       const ValueType weight = rweights_[i];
                       qpbo_->AddPairwiseTerm( vi0, vi1, 0.0, weight, weight, 0.0);
                   }
                   else{
                       qpbo_->AddPairwiseTerm( vi0, vi1, 0.0, 0.0, 0.0, 0.0);
                   }
               }
            }

            // merge parallel edges
            qpbo_->MergeParallelEdges();

            // set label for one variable
            qpbo_->SetLabel(0, 0);

            // run optimization
            qpbo_->Improve();

            for(IndexType vi=0; vi<gm_.numberOfVariables(); ++vi){
               const int l = qpbo_->GetLabel(vi);
               proposal[vi]  = l ==-1 ? 0 : l ;
            }

            ++iteration_;
        }
    private:
        const GM & gm_;
        Parameter param_;
        kolmogorov::qpbo::QPBO<ValueType> * qpbo_;
        size_t iteration_; 
        std::vector<ValueType> weights_;
        std::vector<ValueType> rweights_;
        WeightRand wRandomizer_; 
    };

    #endif 

    /*
    template<class GM, class ACC>
    class BlockGen{
    public:
        typedef ACC AccumulationType;
        typedef GM GraphicalModelType;
        OPENGM_GM_TYPE_TYPEDEFS;

        typedef WeightRandomization<ValueType> WeightRand;
        typedef typename  WeightRand::Parameter WeightRandomizationParam;

        class Parameter
        {
        public:
            Parameter(
                const double numVar = 0.1,
                const bool seedFromNegativeEdges = true,
                const WeightRandomizationParam & randomizer = WeightRandomizationParam()
            )
            :   
            numVar_(numVar),
            randomizer_(randomizer)
            {

            }
            WeightRandomizationParam randomizer_;
            double numVar_;
        };




        BlockGen(const GM & gm, const Parameter & param = Parameter())
        : 
            gm_(gm),
            param_(param),
            wRandomizer_(param_.randomizer_),
            viAdjacency_(gm.numberOfVariables()),
            vis_(gm.numberOfVariables()),
            usedVi_(gm.numberOfVariables(), 0)
        {

            // compute variable adjacency
            gm.variableAdjacencyList(viAdjacency_);

            LabelType lAA[2]={0, 0};
            LabelType lAB[2]={0, 1};

            if(param_.seedFromNegativeEdges_){
                for(size_t i=0; i<gm_.numberOfFactors(); ++i){
                    if(gm_[i].numberOfVariables()==2){
                        ValueType val00  = gm_[i](lAA);
                        ValueType val01  = gm_[i](lAB);
                        ValueType weight = val01 - val00; 
                        if(weight < 0.0){
                            negativeFactors_.push_back(i);
                        }
                    }
                }
            }

        }

        ~BlockGen(){
   
        }

        size_t defaultNumStopIt() {
            return 0;
        }

        void reset(){

        }
        void getProposal(const std::vector<LabelType> &current , std::vector<LabelType> &proposal){

            std::fill(proposal.begin(), proposal.end(), LabelType(0));

            // how many variable should the subgraph contain
            size_t sgSize  = size_t( param_.numVar_ > 1.0f ? 
                                    param_.numVar_ :  
                                    float(gm_.numberOfVariables())*param_.numVar_);

            size_t rVar;
            // get a random negative edge factor 
            if(param_.seedFromNegativeEdges_){
                const size_t rFactor = wRandomizer_.uniformInt(negativeFactors_.size());
                rVar = gm_[rFactor].variableIndex(wRandomizer_.uniformInt(2));
            }
            else{
                rVar = wRandomizer_.uniformInt(gm_.numberOfVariables());
            }


            // grow subgraph SG until |SG| == param_.numVar_;
            std::fill(usedVi_.begin(),usedVi_.end(),false);
            vis.clear();
            vis.push_back(startVi);
            usedVi_[startVi]=true;
            std::queue<size_t> viQueue;
            viQueue.push(startVi);

            std::fill(distance_.begin(),distance_.begin()+gm_.numberOfVariables(),0);

            const size_t maxSgSize = (param_.maxBlockSize_==0? gm_.numberOfVariables() :param_.maxBlockSize_);
            while(viQueue.size()!=0  &&  vis.size()<=maxSgSize) {
                size_t cvi=viQueue.front();
                viQueue.pop();
                // for each neigbour of cvi
                for(size_t vni=0;vni<viAdjacency_[cvi].size();++vni) {
                    // if neighbour has not been visited
                    const size_t vn=viAdjacency_[cvi][vni];
                    if(usedVi_[vn]==false) {
                        // set as visited
                        usedVi_[vn]=true;
                        // insert into the subgraph vis
                        distance_[vn]=distance_[cvi]+1;
                        if(distance_[vn]<=radius){
                            if(vis.size()<maxSgSize){
                                vis.push_back(vn);
                                viQueue.push(vn);
                            }
                            else{
                                break;
                            }
                        }
                    }
                }
            }

        }
    private:
        const GM & gm_;
        Parameter param_;
        WeightRand wRandomizer_; 
        std::vector<IndexType> negativeFactors_;
        std::vector< RandomAccessSet<IndexType> > viAdjacency_;
        std::vector<IndexType>  sgVis_;
        std::vector<unsigned char> usedVi_;
    };
    */

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
            const size_t numStopIt = 0,
            const size_t parallelProposals = 1,
            const bool cgcFinalization = false,
            const bool planar = false,
            const bool doCutMove = false,
            const bool acceptFirst = true,
            const bool warmStart = true,
            const std::vector<bool> & allowCutsWithin = std::vector<bool> ()
        )
            :   proposalParam_(proposalParam),
                fusionParam_(fusionParam),
                numIt_(numIt),
                numStopIt_(numStopIt),
                parallelProposals_(parallelProposals),
                cgcFinalization_(cgcFinalization),
                planar_(planar),
                doCutMove_(doCutMove),
                acceptFirst_(acceptFirst),
                warmStart_(warmStart),
                allowCutsWithin_(allowCutsWithin)
        {
            storagePrefix_ = std::string("");
        }
        ProposalParameter proposalParam_;
        FusionParameter fusionParam_;
        size_t numIt_;
        size_t numStopIt_;
        size_t parallelProposals_;
        bool cgcFinalization_;
        bool planar_;
        bool doCutMove_;
        bool acceptFirst_;
        std::vector<bool> allowCutsWithin_;
        bool warmStart_;
        std::string storagePrefix_;

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


    std::vector<LabelType> & _getArgRef(){
        return bestArg_;
    }

    void _setBestVal(const ValueType value){
        bestValue_ = value;
    }
private:

    template<class VisitorType>
    InferenceTermination inferIntersectionBased(VisitorType &);


    typedef FusionMoverType * FusionMoverTypePtr;
    typedef PROPOSAL_GEN *    ProposalGenTypePtr;

    const GraphicalModelType &gm_;
    Parameter param_;


    FusionMoverType * fusionMover_;
    FusionMoverTypePtr * fusionMoverArray_;


    PROPOSAL_GEN * proposalGen_;
    ProposalGenTypePtr * proposalGenArray_; 

    ValueType bestValue_;
    std::vector<LabelType> bestArg_;
    size_t maxOrder_;

    typedef CGC<GM, ACC> CgcInf;
    typedef  typename  CgcInf::Parameter CgcParam;

    CgcInf * cgcInf_;
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
       fusionMoverArray_(NULL),
       proposalGen_(NULL),
       proposalGenArray_(NULL),
       //proposalGen_(gm_, parameter.proposalParam_),
       bestValue_(),
       bestArg_(gm_.numberOfVariables(), 0),
       maxOrder_(gm.factorOrder()),
       cgcInf_(NULL)
{
    ACC::neutral(bestValue_);   


    //param_.fusionParam_.allowCutsWithin_ = param_.allowCutsWithin_;
    //fusionMover_ = 


    size_t nFuser  = param_.parallelProposals_;
    fusionMoverArray_ = new FusionMoverTypePtr[nFuser];
    proposalGenArray_ = new ProposalGenTypePtr[nFuser];

    for(size_t f=0; f<nFuser; ++f){
        fusionMoverArray_[f] = new FusionMoverType(gm_,param_.fusionParam_);
        proposalGenArray_[f] = new PROPOSAL_GEN(gm_, param_.proposalParam_);
    }

    fusionMover_ = fusionMoverArray_[0];
    proposalGen_ = proposalGenArray_[0];

    if(!param_.warmStart_){
        //set default starting point
        std::vector<LabelType> conf(gm_.numberOfVariables(),0);
        setStartingPoint(conf.begin());
    }
    else{

        LabelType lAA[2]={0, 0};
        LabelType lAB[2]={0, 1};
        Partition<LabelType> ufd(gm_.numberOfVariables());
        for(size_t fi=0; fi< gm_.numberOfFactors(); ++fi){
            if(gm_[fi].numberOfVariables()==2){

                const ValueType val00  = gm_[fi](lAA);
                const ValueType val01  = gm_[fi](lAB);
                const ValueType weight = val01 - val00; 
                if(weight>0.0){
                    const size_t vi0 = gm_[fi].variableIndex(0);
                    const size_t vi1 = gm_[fi].variableIndex(1);
                    ufd.merge(vi0, vi1);
                }
            }
            else{
                throw RuntimeError("wrong factor order for multicut");
            }
        }
        std::vector<LabelType> conf(gm_.numberOfVariables(),0);
        for(IndexType vi=0; vi<gm_.numberOfVariables(); ++vi){
            conf[vi] = ufd.find(vi);            
        }
        setStartingPoint(conf.begin());
    }


    

    if(param_.cgcFinalization_){
        CgcParam cgcParam;
        cgcParam.planar_ = param_.planar_;
        cgcParam.doCutMove_ = param_.doCutMove_;
        cgcParam.startFromThreshold_ = false;
        cgcInf_ = new CgcInf(gm_, cgcParam);
    }
}


template<class GM, class PROPOSAL_GEN>
IntersectionBasedInf<GM, PROPOSAL_GEN>::~IntersectionBasedInf()
{
    for(size_t f=0; f<param_.parallelProposals_; ++f){
        delete fusionMoverArray_[f];// = new FusionMoverType(gm_,parameter.fusionParam_);
        delete proposalGenArray_[f];
    }

    delete[] fusionMoverArray_;
    delete[] proposalGenArray_;

    if(param_.cgcFinalization_){
        delete cgcInf_;
    }
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
){
    visitor.begin(*this);
    InferenceTermination infTerm = this->inferIntersectionBased(visitor);
    if(param_.cgcFinalization_){


        CgcRedirectingVisitor<CgcInf, IntersectionBasedInf<GM, PROPOSAL_GEN> , VisitorType>  redirectingVisitor(*this, visitor);

        cgcInf_->setStartingPoint(bestArg_.begin());
        cgcInf_->infer(redirectingVisitor);
        cgcInf_->arg(bestArg_);
        bestValue_ = gm_.evaluate(bestArg_);
    }
    visitor.end(*this);
    return infTerm;
}


template<class GM, class PROPOSAL_GEN>
template<class VisitorType>
InferenceTermination IntersectionBasedInf<GM, PROPOSAL_GEN>::inferIntersectionBased
(
    VisitorType &visitor
)
{
    // evaluate the current best state
    bestValue_ = gm_.evaluate(bestArg_.begin());

    


    if(param_.numStopIt_ == 0){
        param_.numStopIt_ = proposalGen_->defaultNumStopIt();
    }

    std::vector<LabelType> proposedState(gm_.numberOfVariables());
    std::vector<LabelType> fusedState(gm_.numberOfVariables());

    size_t countRoundsWithNoImprovement = 0;


    size_t nFuser  = param_.parallelProposals_;

    std::vector< std::vector<LabelType> > pVec;
    std::vector< std::vector<LabelType> > rVec;

    std::vector<ValueType> vVec;
    std::vector<bool> dVec;
    if(nFuser>1){
        pVec.resize(nFuser);
        rVec.resize(nFuser);
        vVec.resize(nFuser);
        dVec.resize(nFuser);
        for(size_t i=0; i<nFuser; ++i){
            pVec[i].resize(gm_.numberOfVariables());
            rVec[i].resize(gm_.numberOfVariables());
            dVec[i]=false;
        }
    }

    const bool mmcv  = param_.allowCutsWithin_.size()>0;

    for(size_t iteration=0; iteration<param_.numIt_; ++iteration){

        if(mmcv && iteration == 0){
            ACC::neutral(bestValue_);
            continue;
        }
        else if(!mmcv && iteration == 0 && param_.acceptFirst_ && !param_.warmStart_){
            proposalGen_->getProposal(bestArg_,proposedState);
            std::copy(proposedState.begin(),  proposedState.end(), bestArg_.begin());
            bestValue_ = gm_.evaluate(bestArg_);
            if(visitor(*this)!=0){
                break;
            }
            continue;
        }

        // store initial value before one proposal  round
        const ValueType valueBeforeRound = bestValue_;


        bool anyVar=true;

        if(nFuser == 1){
            proposalGen_->getProposal(bestArg_,proposedState);
            ValueType proposalValue = gm_.evaluate(proposedState);
            if(mmcv)
                ACC::neutral(proposalValue);
            //std::cout<<"best val "<<bestValue_<<" pval "<<proposalValue<<"\n";


            if(!mmcv){
                anyVar = fusionMover_->fuse(bestArg_,proposedState, fusedState, 
                                            bestValue_, proposalValue, bestValue_);
            }
            else{
                //std::cout<<"do cuts within\n";
                //anyVar = fusionMover_->fuseMmwc(bestArg_,proposedState, fusedState, 
                //                            bestValue_, proposalValue, bestValue_);
                ////std::cout<<"bV "<<bestValue_<<" pV "<<proposalValue<<"\n";
            }


            if(!param_.storagePrefix_.empty()){

                {
                    std::stringstream ss;
                    ss<<param_.storagePrefix_<<iteration<<"proposal.txt";
                    std::ofstream f(ss.str().c_str());
                    for(size_t i=0; i<gm_.numberOfVariables(); ++i) {
                        f << proposedState[i] << '\n';
                    }
                }
                {
                    std::stringstream ss;
                    ss<<param_.storagePrefix_<<iteration<<"cbest.txt";
                    std::ofstream f(ss.str().c_str());
                    for(size_t i=0; i<gm_.numberOfVariables(); ++i) {
                        f << bestArg_[i] << '\n';
                    }
                }
                {
                    std::stringstream ss;
                    ss<<param_.storagePrefix_<<iteration<<"nbest.txt";
                    std::ofstream f(ss.str().c_str());
                    for(size_t i=0; i<gm_.numberOfVariables(); ++i) {
                        f << fusedState[i] << '\n';
                    }
                }
            }
        }
        else{

            // get proposals (so far not in parallel)
            //std::cout<<"generate proposas\n";
            //for(size_t i=0; i<nFuser; ++i){
            //    dVec[i]=false;
            //    proposalGen_.getProposal(bestArg_,pVec[i]);
            //}

            #pragma omp parallel for
            for(size_t i=0; i<nFuser; ++i){
                 //#pragma omp critical(printstuff)
                 //{
                   //std::cout<<"fuse i"<<i<<"\n";
                 //}
                proposalGenArray_[i]->getProposal(bestArg_,pVec[i]);
                bool tmp = fusionMoverArray_[i]->fuse(bestArg_,pVec[i], rVec[i], 
                                        bestValue_, gm_.evaluate(pVec[i]), vVec[i]);
                if(bestValue_ < vVec[i]){
                    dVec[i] = true;
                }
            }
            bool done = false;
            size_t total = nFuser;
            size_t c = 0;
            while(!done){
                //std::cout<<"TOTAL "<<total<<"\n";
                size_t left = 0;
                for(size_t i=0; i<total; ++i){
                    if(dVec[i]==false){
                        pVec[left] = rVec[i];
                        ++left;
                    }
                }
                if(left == 0 && c == 0){
                    break;
                }
                else if(left==0 || left == 1){
                   fusedState = rVec[0];
                   bestValue_ = vVec[0];
                   break;
                }
                ++c;
                // fuse all pairs
                #pragma omp parallel for
                for(size_t i=0; i<left; i+=2){
                    if(i==left-1){
                        continue;
                    }
                    //std::cout<<"fuse ii"<<i<<"\n";
                    bool tmp = fusionMoverArray_[i]->fuse(
                        pVec[i],pVec[i+1],rVec[i], 
                        gm_.evaluate(pVec[i]),
                        gm_.evaluate(pVec[i+1]),
                        vVec[i]
                    );
                    dVec[i+1] = true;
                    --left;
                }
                total = left;
            }
        }






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
            if(visitor(*this)!=0){
                break;
            }
            ++countRoundsWithNoImprovement;
        }
        // check if converged or done
        if(countRoundsWithNoImprovement==param_.numStopIt_ && param_.numStopIt_ !=0 ){
            break;
        }
    }
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

#endif // #ifndef OPENGM_INTERSECTION_BASED_INF_HXX
