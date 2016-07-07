#ifndef OPENGM_PERMUTABLE_LABEL_FUSION_MOVER_HXX
#define OPENGM_PERMUTABLE_LABEL_FUSION_MOVER_HXX


#include <opengm/inference/inference.hxx>
#ifdef WITH_CPLEX
#include <opengm/inference/multicut.hxx>
#include <opengm/inference/dmc.hxx>
#endif
#include "opengm/inference/auxiliary/fusion_move/fusion_mover.hxx"

#if defined(WITH_QPBO) || (defined(WITH_PLANARITY) && defined(WITH_BLOSSOM5)) 
#include <opengm/inference/cgc.hxx>
#endif 

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/graphicalmodel/space/simplediscretespace.hxx>
#include <opengm/functions/potts.hxx>


#ifndef NOVIGRA

#ifdef WITH_BOOST
    #ifndef WITH_BOOST_GRAPH
        #define WITH_BOOST_GRAPH
    #endif
#endif

#include <vigra/adjacency_list_graph.hxx>
#include <vigra/merge_graph_adaptor.hxx>
#include <vigra/hierarchical_clustering.hxx>
#include <vigra/priority_queue.hxx>
#include <vigra/random.hxx>
#include <vigra/graph_algorithms.hxx>

#endif

namespace opengm{







    #ifndef NOVIGRA
    template<class GM, class ACC >
    class McClusterOp{
    public:
        typedef ACC AccumulationType;
        typedef GM GraphicalModelType;
        OPENGM_GM_TYPE_TYPEDEFS;

        typedef vigra::AdjacencyListGraph Graph;
        typedef vigra::MergeGraphAdaptor< Graph > MergeGraph;


        typedef typename MergeGraph::Edge Edge;
        typedef ValueType WeightType;
        typedef IndexType index_type;
        struct Parameter
        {



            Parameter(
                const float stopWeight = 0.0
            )
            :
                stopWeight_(stopWeight){
            }
            float stopWeight_;
        };


        McClusterOp(const Graph & graph , 
                    MergeGraph & mergegraph, 
                    const Parameter & param,
                    std::vector<ValueType> & weights
                   )
        :
            graph_(graph),
            mergeGraph_(mergegraph),
            pq_(graph.edgeNum()),
            param_(param),
            weights_(weights){

            for(size_t i=0; i<graph_.edgeNum(); ++i){
                size_t u = graph_.id(graph_.u(graph_.edgeFromId(i)));
                size_t v = graph_.id(graph_.v(graph_.edgeFromId(i)));
                pq_.push(i, weights_[i]);
            }
        }




        void reset(){
            pq_.reset();
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
            return pq_.topPriority()<=ValueType(param_.stopWeight_);
        }

        void mergeEdges(const Edge & a,const Edge & b){
            weights_[a.id()]+=weights_[b.id()];
            pq_.push(a.id(), weights_[a.id()]);
            pq_.deleteItem(b.id());
        }

        void eraseEdge(const Edge & edge){
            pq_.deleteItem(edge.id());
        }

        const Graph & graph_;
        MergeGraph & mergeGraph_;
        vigra::ChangeablePriorityQueue< ValueType ,std::greater<ValueType> > pq_;
        Parameter param_;
        std::vector<ValueType> & weights_;
    };


    #endif





template<class GM, class ACC>
class PermutableLabelFusionMove{

public:

    typedef GM GraphicalModelType;
    typedef ACC AccumulationType;
    OPENGM_GM_TYPE_TYPEDEFS;
    typedef std::map<UInt64Type, ValueType> MapType;
    typedef typename MapType::iterator MapIter;
    typedef typename MapType::const_iterator MapCIter;


    typedef PermutableLabelFusionMove<GM, ACC> SelfType;

    enum FusionSolver{
        DefaultSolver,
        MulticutSolver,
        MulticutNativeSolver,
        CgcSolver,
        HierachicalClusteringSolver,
        BaseSolver,
        ClassicFusion
    };

    struct Parameter{
        Parameter(
            const FusionSolver fusionSolver = SelfType::DefaultSolver,
            const bool planar = false,
            const std::string  workflow = std::string(),
            const int nThreads = 1,
            const bool decompose = false,
            const std::vector<bool> & allowCutsWithin  = std::vector<bool>()
        )
        : 
            fusionSolver_(fusionSolver),
            planar_(planar),
            workflow_(workflow),
            nThreads_(nThreads),
            decompose_(decompose),
            allowCutsWithin_(allowCutsWithin)
        {

        }
        FusionSolver fusionSolver_;
        bool planar_;
        std::string workflow_;
        int nThreads_;
        bool decompose_;
        std::vector<bool> allowCutsWithin_;

    };

    typedef SimpleDiscreteSpace<IndexType, LabelType>       SubSpace;
    typedef PottsFunction<ValueType, IndexType, LabelType>  PFunction;
    typedef ExplicitFunction<ValueType, IndexType, LabelType>  EFunction;
    typedef GraphicalModel<ValueType, Adder, OPENGM_TYPELIST_2(EFunction,PFunction) , SubSpace> SubModel;


    PermutableLabelFusionMove(const GraphicalModelType & gm, const Parameter & param = Parameter())
    :   
        gm_(gm),
        param_(param)
    {
        if(param_.fusionSolver_ == DefaultSolver){

            #ifdef WITH_CPLEX
                param_.fusionSolver_ = MulticutSolver;
            #endif
            
            if(param_.fusionSolver_ == DefaultSolver){
                #if defined(WITH_QPBO) || (defined(WITH_PLANARITY) && defined(WITH_BLOSSOM5))  
                    param_.fusionSolver_ = CgcSolver;
                #endif
            }
            if(param_.fusionSolver_ == DefaultSolver){
                #ifndef NOVIGRA
                    if(param_.planar_){
                        param_.fusionSolver_ = HierachicalClusteringSolver;
                    }
                #endif
            }
            if(param_.fusionSolver_ == DefaultSolver){
                throw RuntimeError("WITH_CLEX || defined(WITH_QPBO) || (defined(WITH_PLANARITY) && defined(WITH_BLOSSOM5)) must be enabled");
            }
        }
        else if(param_.fusionSolver_ == MulticutSolver){
            #ifndef WITH_CPLEX
                throw RuntimeError("WITH_CLEX must be enabled for this fusionSolver");
            #endif
        }
        else if(param_.fusionSolver_ == CgcSolver){
            #if ! (defined(WITH_QPBO) || (defined(WITH_PLANARITY) && defined(WITH_BLOSSOM5)) )
                throw RuntimeError("defined(WITH_QPBO) || (defined(WITH_PLANARITY) && defined(WITH_BLOSSOM5))  must be enabled for this fusionSolver");
            #endif
        }
        else if(param_.fusionSolver_ == HierachicalClusteringSolver){
            #ifndef WITH_VIGRA
                throw RuntimeError("WITH_VIGRA  must be enabled for this fusionSolver");
            #endif
        }
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
            else if(gm_[fi].numberOfVariables()==1){

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

        if(param_.fusionSolver_ == ClassicFusion)
            return fuseClassic(a,b,res,valA,valB,valRes);

        std::vector<LabelType> ab(gm_.numberOfVariables());
        IndexType numNewVar = this->intersect(a, b, ab);
        //std::cout<<"numNewVar "<<numNewVar<<"\n";

        if(numNewVar==1){
            return false;
        }

        const ValueType intersectedVal = gm_.evaluate(ab);



        // get the new smaller graph


        MapType accWeights;
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



        //std::cout<<"INTERSECTED SIZE "<<numNewVar<<"\n";

        if(param_.fusionSolver_ == CgcSolver){
            return doMoveCgc(accWeights,ab,numNewVar, a, b, res, valA, valB, valRes);
        }
        else if(param_.fusionSolver_ == MulticutSolver){
            return doMoveMulticut(accWeights,ab,numNewVar, a, b, res, valA, valB, valRes);
        }
        else if(param_.fusionSolver_ == MulticutNativeSolver){
            return doMoveMulticutNative(accWeights,ab,numNewVar, a, b, res, valA, valB, valRes);
        }
        else if(param_.fusionSolver_ == HierachicalClusteringSolver){
            return doMoveHierachicalClustering(accWeights,ab,numNewVar, a, b, res, valA, valB, valRes);
        }
        else if(param_.fusionSolver_ == BaseSolver){
            return doMoveBase(accWeights,ab,numNewVar, a, b, res, valA, valB, valRes);
        }
        else{
            throw RuntimeError("unknown fusionSolver");
            return false;
        }

    }


    bool fuseClassic(
        const std::vector<LabelType> & a,
        const std::vector<LabelType> & b,
        std::vector<LabelType> & res,
        const ValueType valA,
        const ValueType valB,
        ValueType & valRes
    ){
        LabelType maxL = *std::max_element(a.begin(), a.end());
        std::vector<LabelType> bb = b;
        for(size_t i=0; i<bb.size(); ++i){
            bb[i] += maxL;
        }
        typename HlFusionMover<GM, ACC>::Parameter p;
        HlFusionMover<GM, ACC> hlf(gm_,p);
        hlf.fuse(a,b,res, valA, valB, valRes);
        // make dense
        std::map<LabelType, LabelType> mdense;

        LabelType dl=0;
        for(size_t i=0;i<res.size(); ++i){
            const LabelType resL = res[i];
            if(mdense.find(resL) == mdense.end()){
                res[i] = dl;
                ++dl;
            }
            else{
                res[i] = mdense[res[i]];
            }
        }  
        valRes  = gm_.evaluate(res);
        if(valRes< std::min(valA,valB)){
            // make dense
            std::map<LabelType, LabelType> mdense;

            LabelType dl=0;
            for(size_t i=0;i<res.size(); ++i){
                const LabelType resL = res[i];
                if(mdense.find(resL) == mdense.end()){
                    res[i] = dl;
                    ++dl;
                }
                else{
                    res[i] = mdense[res[i]];
                }
            }
        }
        else if(valA<valRes){
            valRes=valA;
            res = a;
        }
        else{
            valRes=valB;
            res = b;
        }
        assert(false);  // FIXME: the return of this function was missing, just added something arbitrary
        return false;
    }



    bool doMoveCgc(
        const MapType & accWeights,
        const std::vector<LabelType> & ab,
        const IndexType numNewVar,
        const std::vector<LabelType> & a,
        const std::vector<LabelType> & b,
        std::vector<LabelType> & res,
        const ValueType valA,
        const ValueType valB,
        ValueType & valRes
    ){
        #if defined(WITH_QPBO) || (defined(WITH_PLANARITY) && defined(WITH_BLOSSOM5)) 

        // make the actual sub graphical model
        SubSpace subSpace(numNewVar, numNewVar);
        SubModel subGm(subSpace);

        // reserve space
        subGm. template reserveFunctions<PFunction>(accWeights.size());
        subGm.reserveFactors(accWeights.size());
        subGm.reserveFactorsVarialbeIndices(accWeights.size()*2);

        for(MapCIter i = accWeights.begin(); i!=accWeights.end(); ++i){
            const UInt64Type key    = i->first;
            const ValueType weight = i->second;

            const UInt64Type cVi1 = key/numNewVar;
            const UInt64Type cVi0 = key - cVi1*numNewVar;
            const UInt64Type vis[2] = {cVi0, cVi1};

            PFunction pf(numNewVar, numNewVar, 0.0, weight);
            subGm.addFactor(subGm.addFunction(pf), vis, vis+2);
        }

        std::vector<LabelType> subArg;

        //::cout<<"WITH MC\n";
        typedef CGC<SubModel, Minimizer> Inf;
        typedef  typename  Inf::Parameter Param;

        Param p;
        p.planar_ = param_.planar_;

        Inf inf(subGm,p);
        inf.infer();
        inf.arg(subArg);

        for(IndexType vi=0; vi<gm_.numberOfVariables(); ++vi){
            res[vi] = subArg[ab[vi]];
        }
        const ValueType resultVal = subGm.evaluate(subArg);
        const ValueType projectedResultVal = gm_.evaluate(res);
        const std::vector<LabelType> & bestArg = valA < valB ? a : b;
        const ValueType bestProposalVal  =  valA < valB ? valA : valB;

        valRes = bestProposalVal < projectedResultVal ? bestProposalVal : projectedResultVal;
        if(projectedResultVal < bestProposalVal){
            //for(IndexType vi=0; vi<gm_.numberOfVariables(); ++vi){
            //    res[vi] = subArg[ab[vi]];
            //}
        }
        else{
            for(IndexType vi=0; vi<gm_.numberOfVariables(); ++vi){
                res[vi] = bestArg[vi];
            }
        }
        return true;
        #else
            throw RuntimeError("defined(WITH_QPBO) || (defined(WITH_PLANARITY) && defined(WITH_BLOSSOM5))");
            return false;
        #endif
    }

    bool doMoveBase(
        const MapType & accWeights,
        const std::vector<LabelType> & ab,
        const IndexType numNewVar,
        const std::vector<LabelType> & a,
        const std::vector<LabelType> & b,
        std::vector<LabelType> & res,
        const ValueType valA,
        const ValueType valB,
        ValueType & valRes
    ){
        const std::vector<LabelType> & bestArg = valA < valB ? a : b;
        valRes =  valA < valB ? valA : valB;
        for(IndexType vi=0; vi<gm_.numberOfVariables(); ++vi){
            res[vi] = bestArg[vi];
        }
        return true;
    }

    bool doMoveMulticut(
        const MapType & accWeights,
        const std::vector<LabelType> & ab,
        const IndexType numNewVar,
        const std::vector<LabelType> & a,
        const std::vector<LabelType> & b,
        std::vector<LabelType> & res,
        const ValueType valA,
        const ValueType valB,
        ValueType & valRes
    ){
        #ifdef WITH_CPLEX
        // make the actual sub graphical model
        SubSpace subSpace(numNewVar, numNewVar);
        SubModel subGm(subSpace);

        // reserve space
        subGm. template reserveFunctions<PFunction>(accWeights.size());
        subGm.reserveFactors(accWeights.size());
        subGm.reserveFactorsVarialbeIndices(accWeights.size()*2);

        for(MapCIter i = accWeights.begin(); i!=accWeights.end(); ++i){
            const UInt64Type key    = i->first;
            const ValueType weight = i->second;

            const UInt64Type cVi1 = key/numNewVar;
            const UInt64Type cVi0 = key - cVi1*numNewVar;
            const UInt64Type vis[2] = {cVi0, cVi1};

            PFunction pf(numNewVar, numNewVar, 0.0, weight);
            subGm.addFactor(subGm.addFunction(pf), vis, vis+2);
        }

        std::vector<LabelType> subArg;

        //try{
            //::cout<<"WITH MC\n";
            typedef Multicut<SubModel, Minimizer> McInf;
            typedef  typename  McInf::Parameter McParam;
            McParam pmc(0,0.0);

            if(param_.nThreads_ <= 0){
                pmc.numThreads_ = 0;
            }
            else{
                pmc.numThreads_ = param_.nThreads_;
            }
            pmc.workFlow_ = param_.workflow_;


            if(param_.decompose_ == false){
                McInf inf(subGm,pmc);
                inf.infer();
                inf.arg(subArg);
            }
            else{
                typedef DMC<SubModel, McInf> DmcInf;
                typedef  typename  DmcInf::Parameter DmcParam;
                typedef  typename DmcInf::InfParam DmcInfParam;
                DmcParam dmcParam;
                DmcInfParam dmcInfParam(pmc);

                dmcParam.infParam_  = dmcInfParam;

                DmcInf inf(subGm, dmcParam);
                inf.infer();
                inf.arg(subArg);
            }
        //}
        //catch(...){
        //     std::cout<<"error from cplex\n....\n";
        //}

        for(IndexType vi=0; vi<gm_.numberOfVariables(); ++vi){
            res[vi] = subArg[ab[vi]];
        }
        const ValueType resultVal = subGm.evaluate(subArg);
        const ValueType projectedResultVal = gm_.evaluate(res);
        const std::vector<LabelType> & bestArg = valA < valB ? a : b;
        const ValueType bestProposalVal  =  valA < valB ? valA : valB;

        valRes = bestProposalVal < projectedResultVal ? bestProposalVal : projectedResultVal;
        if(projectedResultVal < bestProposalVal){
            //for(IndexType vi=0; vi<gm_.numberOfVariables(); ++vi){
            //    res[vi] = subArg[ab[vi]];
            //}
        }
        else{
            for(IndexType vi=0; vi<gm_.numberOfVariables(); ++vi){
                res[vi] = bestArg[vi];
            }
        }
        return true;
        #else
            throw RuntimeError("needs WITH_CPLEX");
            return false;
        #endif
    }


    bool doMoveMulticutNative(
        const MapType & accWeights,
        const std::vector<LabelType> & ab,
        const IndexType numNewVar,
        const std::vector<LabelType> & a,
        const std::vector<LabelType> & b,
        std::vector<LabelType> & res,
        const ValueType valA,
        const ValueType valB,
        ValueType & valRes
    ){
        #ifdef WITH_CPLEX
        std::vector<LabelType> subArg;

        //::cout<<"WITH MC\n";
        typedef Multicut<SubModel, Minimizer> Inf;
        typedef  typename  Inf::Parameter Param;
        Param p(0,0.0);

        if(param_.nThreads_ <= 0){
            p.numThreads_ = 0;
        }
        else{
            p.numThreads_ = param_.nThreads_;
        }
        p.workFlow_ = param_.workflow_;

        Inf inf(numNewVar, accWeights, p);
        inf.infer();
        inf.arg(subArg);
        


        for(IndexType vi=0; vi<gm_.numberOfVariables(); ++vi){
            res[vi] = subArg[ab[vi]];
        }

        const ValueType projectedResultVal = gm_.evaluate(res);
        const std::vector<LabelType> & bestArg = valA < valB ? a : b;
        const ValueType bestProposalVal  =  valA < valB ? valA : valB;

        valRes = bestProposalVal < projectedResultVal ? bestProposalVal : projectedResultVal;
        if(projectedResultVal < bestProposalVal){
            //for(IndexType vi=0; vi<gm_.numberOfVariables(); ++vi){
            //    res[vi] = subArg[ab[vi]];
            //}
        }
        else{
            for(IndexType vi=0; vi<gm_.numberOfVariables(); ++vi){
                res[vi] = bestArg[vi];
            }
        }
        return true;
        #else
            throw RuntimeError("needs WITH_CPLEX");
            return false;
        #endif
    }

    bool doMoveHierachicalClustering(
        const MapType & accWeights,
        const std::vector<LabelType> & ab,
        const IndexType numNewVar,
        const std::vector<LabelType> & a,
        const std::vector<LabelType> & b,
        std::vector<LabelType> & res,
        const ValueType valA,
        const ValueType valB,
        ValueType & valRes
    ){
        #ifndef NOVIGRA
        typedef vigra::AdjacencyListGraph Graph;
        typedef typename Graph::Edge Edge;
        typedef vigra::MergeGraphAdaptor< Graph > MergeGraph;
        typedef McClusterOp<GM,ACC> ClusterOp;
        typedef typename ClusterOp::Parameter ClusterOpParam;
        typedef vigra::HierarchicalClusteringImpl< ClusterOp > HC;
        typedef typename HC::Parameter HcParam;
        
        std::vector<ValueType> weights(accWeights.size(),0.0);

        Graph graph(numNewVar, accWeights.size());
        for(MapCIter i = accWeights.begin(); i!=accWeights.end(); ++i){
            const UInt64Type key    = i->first;
            const ValueType weight = i->second;

            const UInt64Type cVi1 = key/numNewVar;
            const UInt64Type cVi0 = key - cVi1*numNewVar;
            
            const Edge e = graph.addEdge(cVi0, cVi1);
            weights[graph.id(e)] = weight;
        }

        MergeGraph mg(graph);




        const ClusterOpParam clusterOpParam;
        ClusterOp clusterOp(graph, mg, clusterOpParam, weights);




        HcParam p;
        HC hc(clusterOp,p);

        //std::cout<<"start\n";
        hc.cluster();



        for(IndexType vi=0; vi<gm_.numberOfVariables(); ++vi){
            res[vi] = hc.reprNodeId(ab[vi]);
        }

        const ValueType projectedResultVal = gm_.evaluate(res);
        const std::vector<LabelType> & bestArg = valA < valB ? a : b;
        const ValueType bestProposalVal  =  valA < valB ? valA : valB;

        valRes = bestProposalVal < projectedResultVal ? bestProposalVal : projectedResultVal;
        if(projectedResultVal < bestProposalVal){
            //for(IndexType vi=0; vi<gm_.numberOfVariables(); ++vi){
            //    res[vi] = subArg[ab[vi]];
            //}
        }
        else{
            for(IndexType vi=0; vi<gm_.numberOfVariables(); ++vi){
                res[vi] = bestArg[vi];
            }
        }
        return true;
        #else   
            throw RuntimeError("needs VIGRA");
            return false;
        #endif
    }

private:
    const GM & gm_;
    Parameter param_;
};





}


#endif /* OPENGM_PERMUTABLE_LABEL_FUSION_MOVER_HXX */
