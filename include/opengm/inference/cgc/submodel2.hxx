#ifndef OPENGM_HMC_SUBMODEL2
#define OPENGM_HMC_SUBMODEL2

#include <deque>

#include <boost/format.hpp>
#include <boost/unordered_map.hpp>

#ifdef WITH_CPLEX
#include <opengm/inference/multicut.hxx>
#endif
#ifdef WITH_QPBO
#include "QPBO.h"
#endif

#if defined(WITH_BLOSSOM5) && defined(WITH_PLANARITY)
#include <opengm/inference/auxiliary/planar_maxcut.hxx>
#endif
#include <opengm/opengm.hxx>
#include <opengm/utilities/timer.hxx>


#undef OPENGM_CHECK_OP
#define OPENGM_CHECK_OP(A,OP,B,TXT) 


#undef OPENGM_CHECK
#define OPENGM_CHECK(B,TXT) 


template<class GM>
class SubmodelCGC{
public:    
    typedef typename GM::ValueType      ValueType;
    typedef typename GM::IndexType      IndexType;
    typedef typename GM::LabelType      LabelType;

    typedef std::vector<ValueType>      ValueVector;
    typedef std::vector<double >        DoubleVector;
    typedef std::vector<IndexType>      IndexVector;
    typedef std::vector<LabelType>      LabelVector;
    typedef std::vector<bool>           BoolVector;
    typedef std::vector<unsigned char>  PseudoBoolVector;

    typedef std::pair<int,ValueType> IVPairType;

    // set up map from node pair to edge index
    typedef std::pair<size_t,size_t>          Edge;
    typedef std::pair<Edge, size_t>           EdgeMapEntry;
    typedef boost::unordered_map<Edge,size_t> EdgeMap;
    
    enum Mode {
        SingleSubset,
        TwoSubsets
    };

    // constructor from graphical model
    SubmodelCGC(const GM & gm,const IndexType maxBruteForceSize2,const IndexType maxBruteForceSize4,const bool useBfs);


    //////////////////////////////////
    // function to infer submodel   //
    //////////////////////////////////
    ValueType inferMulticut(Mode mode);
    ValueType inferQPBOI(Mode mode);
    ValueType inferPlanarMaxCut();
    ValueType inferBruteForce2();
    ValueType inferBruteForce4();
   
    /**
     * globalArg: length #nodes of full model
     * colorCC:   color of CC for which we want to run inference
     * offset:    needed to disambiguate colors w.r.t. rest of the nodes
     * deque:     After running infer, the region decomposes into N connected components.
     *            Then, deque.size() += N, and deque[prevSize()+i]
     *            is a representative node label of connected component i
     */
    template<class ARG>
    IVPairType inferSubset(
        ARG & globalArg,
        const LabelType colorCC,
        const IndexType viCC,
        IndexType offset,
        std::deque<IndexType> & deque,
        const bool planar,
        bool verbose = false
    );

    /**
     * globalArg: length #nodes of full model
     * colorCC0,  color of two neighboring CCs for which we want to run inference again
     * colorCC1
     * offset:    needed to disambiguate colors w.r.t. rest of the nodes
     */
    template<class ARG>
    IVPairType infer2Subsets(
        ARG & globalArg,
        const LabelType colorCC0,
        const LabelType colorCC1,
        const IndexType viCC0,
        const IndexType viCC1,
        IndexType offset,
        const bool planar=true
    );
    



    /**
     * changes: whether, in the optimization for the current submodel, there were any changes
     *          w.r.t to the previous solution stored for the global model
     *          (this is checked by energy only! FIXME: degenerate solutions?
     * 
     * dirtyFactors: current dirtyness per factor (global)
     * 
     * This function updates dirtyFactors according to this:
     * 
     * - if a factor's variables are both in the subgraph, mark this factor as clean 
     * - if only a subset of the factor's variables are in the subgraph:
     *   - if changes, mark this factor as dirty
     *   - if no changes, do not change dirtyness
     * 
     */
    void updateDirtyness(std::vector<unsigned char>& dirtyFactors, const bool changes){
        OPENGM_CHECK_OP(dirtyFactors.size(),==,gm_.numberOfFactors(), " ");

        OPENGM_CHECK_OP(nInsideFactors_,>,0, " ");
        OPENGM_CHECK_OP(nBorderFactors_,>,0, " ");

        // make inside undirty if there are no changes
        if(!changes){
            for(IndexType f=0;f<nInsideFactors_;++f){
                if(dirtyFactors[insideFactors_[f]] == 2) {
                    OPENGM_CHECK(false, "shouldn't happen");
                }
                dirtyFactors[insideFactors_[f]] = 0;
            }
        }
        // if there are improvements 
        else{

            // mark inside as clean and border as dirty
            if(lastNCC_<=2){

                // inside to clean
                for(IndexType f=0;f<nInsideFactors_;++f){
                    if(dirtyFactors[insideFactors_[f]] == 2) {
                        OPENGM_CHECK(false, "shouldn't happen");
                    }
                    dirtyFactors[insideFactors_[f]] = 0;
                }
                // border to dirty
                for(IndexType f=0;f<nBorderFactors_;++f){
                    if(dirtyFactors[borderFactor_[f]] == 2) {
                        continue;
                    }
                    dirtyFactors[borderFactor_[f]] = 1;
                }

            }

            // mark inside and border as diry
            else{
                //std::cout<<"\n\n\n\n\n    JOOOO \n\n\n";
                for(IndexType f=0;f<nInsideFactors_;++f){
                    if(dirtyFactors[insideFactors_[f]] == 2) {
                        OPENGM_CHECK(false, "shouldn't happen");
                    }
                    dirtyFactors[insideFactors_[f]] = 1;
                }
                for(IndexType f=0;f<nBorderFactors_;++f){
                    if(dirtyFactors[borderFactor_[f]] == 2) {
                        continue;
                    }
                    dirtyFactors[borderFactor_[f]] = 1;
                }
            }
        }


        /*
        for(IndexType f=0;f<nInsideFactors_;++f){
            if(dirtyFactors[insideFactors_[f]] == 2) {
                OPENGM_CHECK(false, "shouldn't happen");
            }
            dirtyFactors[insideFactors_[f]] = 0;
        }
        if(changes){
            for(IndexType f=0;f<nBorderFactors_;++f){
                if(dirtyFactors[borderFactor_[f]] == 2) {
                    continue;
                }
                dirtyFactors[borderFactor_[f]] = 1;
            }
        }
        */
    }

    void cleanInsideAndBorder(){
        nInsideFactors_=0;
        nBorderFactors_=0;
    }

private:
    
    /**
     * arg: node labeling (node coloring)
     * color: the color of the connected component of nodes for which we want to build the submodel 
     */
    template<class ARG>
    void setSubVarImplicit(const ARG & arg, const LabelType color,const IndexType vi);
    template<class ARG>
    void setSubVarImplicitBfs(const ARG & arg, const LabelType color,const IndexType vi);
    /**
     * arg: node labeling (node coloring)
     * color0, color1: colors of neighboring connected components. The submodel includes all nodes labeled
     *                 with theses colors
     */
    template<class ARG>
    void setSubVarImplicit(const ARG & arg,
        const LabelType color0,const LabelType color1,
        const IndexType vi0,const IndexType vi1
    );
    template<class ARG>
    void setSubVarImplicitBfs(const ARG & arg,
        const LabelType color0,const LabelType color1,
        const IndexType vi0,const IndexType vi1
    );

    
    /**
     * number of connected components of the current submodel
     */
    IndexType ccFromLocalArg(const IndexType offset);

    void getEmbeddingGraph();
    void freeEmbeddingGraph();
    
    // set up variables of submodel from explicit var
    void setUpSubFactors();
    // unset variables
    void unsetSubVar();
    
    // global graphical model
    const GM & gm_;

    // local<->global var mapping
    IndexVector  subVarToGlobal_;
    IndexVector  globalVarToLocal_;

    // inclued factors and variables in submodel
    PseudoBoolVector incluedGlobalVar_;
    BoolVector       incluedGlobalFactors_;
    BoolVector       isABorderFactor_;


    // local arg buffer
    LabelVector localArg_;
    LabelVector localArgTest_;
    
    // local factor - global factor mapping
    marray::Marray<IndexType> localFactorVis_;

    // local and global lambdas 
    ValueVector  globalLambdas_;
    DoubleVector localLambdas_;

    // inside factor (buffer)
    // outside factor (buffer)
    IndexVector insideFactors_;
    IndexVector borderFactor_;
    IndexType   nInsideFactors_;
    IndexType   nBorderFactors_;

    // number of Xxx for easy reading
    IndexType numSubVar_;
    IndexType numSubFactors_;

    // sub edge map
    EdgeMap localEdgemap_;

    // parameters
    IndexType maxBruteForceSize2_;
    IndexType maxBruteForceSize4_;
    bool greedyMode_;

    ValueType oldCutValue_;
    bool isOptCut_;

    bool useBfs_;

    std::vector<opengm::RandomAccessSet<IndexType> > visAdj_;

    std::vector<IndexType> stack_;

    IndexType lastNCC_;
};

//------------------------------------------------------------------------------------------------------------//
// IMPLEMENTATION
//------------------------------------------------------------------------------------------------------------//

template<class GM>
template<class ARG>
typename SubmodelCGC<GM>::IVPairType 
SubmodelCGC<GM>::infer2Subsets(
    ARG & globalArg,
    typename SubmodelCGC<GM>::LabelType colorCC0,
    typename SubmodelCGC<GM>::LabelType colorCC1,
    const typename SubmodelCGC<GM>::IndexType  viCC0,
    const typename SubmodelCGC<GM>::IndexType  viCC1,
    typename SubmodelCGC<GM>::IndexType offset,
    const bool planar
){

    OPENGM_CHECK_OP(colorCC0,!=,colorCC1,"must be different colors");

    //std::cout<<"set up sub var \n";
    if(useBfs_){
        this->setSubVarImplicitBfs(globalArg,colorCC0,colorCC1,viCC0,viCC1);
    }
    else{
        this->setSubVarImplicit(globalArg,colorCC0,colorCC1,viCC0,viCC1);
    }
    if(numSubVar_<=1){
        this->unsetSubVar();
        return IVPairType(-1,0.0);
    }
    //std::cout<<"set up sub factors \n";
    this->setUpSubFactors();

    if(false && isOptCut_==true){
        this->unsetSubVar();
        return IVPairType(-3,oldCutValue_);
    }

    ValueType valueOfArg=0.0;
    if(numSubVar_<= maxBruteForceSize2_ && planar==true){
        //std::cout<<"do bruteforce on "<<numSubVar_<<" vars "<<numSubFactors_<<"facs \n";
        if(numSubVar_<maxBruteForceSize4_ && numSubVar_>=3)
            valueOfArg=this->inferBruteForce4();
        else
            valueOfArg=this->inferBruteForce2();
    }
    else if(planar==true){
        OPENGM_CHECK(planar,"");
        //this->getEmbeddingGraph();
        //valueOfArg=this->inferIsInf();
        //this->freeEmbeddingGraph();
        valueOfArg = this->inferPlanarMaxCut();
    }
    else{
       //std::cout<<"do multicut on "<<numSubVar_<<" vars "<<numSubFactors_<<"facs \n";
       //valueOfArg = this->inferMulticut(TwoSubsets); 
        valueOfArg = this->inferQPBOI(TwoSubsets);
    }

    OPENGM_CHECK(greedyMode_," ");
    if(oldCutValue_<valueOfArg){
        //std::cout<<"we are worse or equal same..."<<valueOfArg <<" old "<<oldCutValue_<<"\n";

        //std::cout<<"no improvement \n";
        this->unsetSubVar();
        return IVPairType(-2,oldCutValue_);
    }
    else if(valueOfArg+0.0000001<oldCutValue_){
        //std::cout<<"better....\n";
        //std::cout<<" old "<<oldCutValue_<<"\n";
        //std::cout<<" new "<<valueOfArg<<"\n";
 
    }
    else{
        //std::cout<<"same....\n";
        //std::cout<<" old "<<oldCutValue_<<"\n";
        //std::cout<<" new "<<valueOfArg<<"\n";
        this->unsetSubVar();
        return IVPairType(-2,oldCutValue_);
    }

    
    //std::cout<<"get ccs \n";
    // infer local probel and get cc's
    IndexType numCC = this->ccFromLocalArg(offset);
    lastNCC_ = numCC; //remember for later
    //std::cout<<"write back -- offset is "<<offset<<"\n";

    std::vector<IndexType> exampleForCC(numCC);

    // write back to global arg
    for(IndexType localVi=0;localVi<numSubVar_;++localVi){
        const LabelType ccLabel  = localArg_[localVi];
        const IndexType globalVi = subVarToGlobal_[localVi];
        OPENGM_CHECK_OP(ccLabel-offset,<,numCC," ");
        exampleForCC[ccLabel-offset]=globalVi;
        globalArg[globalVi]=ccLabel;
    }

    // free stuff and unset variables
    this->unsetSubVar();
    return IVPairType(numCC,valueOfArg-oldCutValue_);
    //return boost::python::make_tuple(numCC,valueOfArg-oldCutValue_);
}


template<class GM>
template<class ARG>
typename SubmodelCGC<GM>::IVPairType
SubmodelCGC<GM>::inferSubset(
    ARG                                          &  globalArg,
    typename SubmodelCGC<GM>::LabelType                colorCC,
    typename SubmodelCGC<GM>::IndexType                 viCC,
    typename SubmodelCGC<GM>::IndexType                 offset,
    std::deque<typename SubmodelCGC<GM>::IndexType> &     deque,
    const bool planar,
    bool verbose
){
    using boost::format;

    // set up local problem
    if(useBfs_){
        this->setSubVarImplicitBfs(globalArg,colorCC,viCC);
    }
    else{
        this->setSubVarImplicit(globalArg,colorCC,viCC);
    }
    
    if(verbose) {
        std::cout << format("    inferSubset with %d variables") % numSubVar_ << std::endl;
    }
    
    if(numSubVar_<=1){
        this->unsetSubVar();
        return IVPairType(-1.0,0.0);
    }
    //std::cout<<"set up sub factors \n";
    this->setUpSubFactors();
    
    ValueType valueOfArg=0.0;
    if(planar){
        if(numSubVar_<= maxBruteForceSize2_){
            if(numSubVar_<maxBruteForceSize4_ && numSubVar_>=3)
                valueOfArg=this->inferBruteForce4();
            else
                valueOfArg=this->inferBruteForce2();
        }
        else{
            //std::cout<<"get embedding graph \n";
            //this->getEmbeddingGraph();
            //valueOfArg=this->inferIsInf();
            //this->freeEmbeddingGraph();
            valueOfArg = this->inferPlanarMaxCut();
        }
    }
    else{
       //valueOfArg=this->inferMulticut(SingleSubset);
        valueOfArg=this->inferQPBOI(SingleSubset);
    }
    
    //std::cout<<"get ccs \n";
    // infer local probel and get cc's
    IndexType numCC = this->ccFromLocalArg(offset);

    //std::cout<<"write back -- offset is "<<offset<<"\n";

    std::vector<IndexType> exampleForCC(numCC);

    if(numCC>1){
        // write back to global arg
        for(IndexType localVi=0;localVi<numSubVar_;++localVi){
            const LabelType ccLabel  = localArg_[localVi];
            const IndexType globalVi = subVarToGlobal_[localVi];
            OPENGM_CHECK_OP(ccLabel-offset,<,numCC," ");
            exampleForCC[ccLabel-offset]=globalVi;
            globalArg[globalVi]=ccLabel;
        }
        
        if(true){
            for(size_t i=0;i<exampleForCC.size();++i){
                deque.push_back(exampleForCC[i]);
            }
        }
    }

    // free stuff and unset variables
    this->unsetSubVar();
    return IVPairType(numCC,valueOfArg);

}

template<class GM>
template<class ARG>
void SubmodelCGC<GM>::setSubVarImplicit( 
    const ARG & arg,
    const typename SubmodelCGC<GM>::LabelType ccColor,
    const typename SubmodelCGC<GM>::IndexType vi
){
    greedyMode_=false;
    IndexType viLocal=0;
    for(IndexType viGlobal=0;viGlobal<gm_.numberOfVariables();++viGlobal){
        if(arg[viGlobal]==ccColor){
            subVarToGlobal_[viLocal]=viGlobal;
            globalVarToLocal_[viGlobal]=viLocal;
            OPENGM_CHECK_OP(incluedGlobalVar_[viGlobal],==,0,"internal inconsistency");
            incluedGlobalVar_[viGlobal]=1;
            ++viLocal;
        }
    }
    numSubVar_=viLocal;
}

template<class GM>
template<class ARG>
void SubmodelCGC<GM>::setSubVarImplicitBfs( 
    const ARG & arg,
    const typename SubmodelCGC<GM>::LabelType ccColor,
    const typename SubmodelCGC<GM>::IndexType vi
){
    greedyMode_=false;
    IndexType viLocal=0;
    for(IndexType viGlobal=0;viGlobal<gm_.numberOfVariables();++viGlobal){
        if(arg[viGlobal]==ccColor){
            subVarToGlobal_[viLocal]=viGlobal;
            globalVarToLocal_[viGlobal]=viLocal;
            OPENGM_CHECK_OP(incluedGlobalVar_[viGlobal],==,0,"internal inconsistency");
            incluedGlobalVar_[viGlobal]=1;
            ++viLocal;
        }
    }
    numSubVar_=viLocal;
}



template<class GM>
template<class ARG>
void SubmodelCGC<GM>::setSubVarImplicit( 
    const ARG & arg,
    const typename SubmodelCGC<GM>::LabelType     ccColor0,
    const typename SubmodelCGC<GM>::LabelType     ccColor1,
    const typename SubmodelCGC<GM>::IndexType vi0,
    const typename SubmodelCGC<GM>::IndexType vi1
){
    greedyMode_=true;
    IndexType viLocal=0;
    for(IndexType viGlobal=0;viGlobal<gm_.numberOfVariables();++viGlobal){

        const LabelType cVi=arg[viGlobal];
        if(cVi==ccColor0 || cVi==ccColor1){
            subVarToGlobal_[viLocal]=viGlobal;
            globalVarToLocal_[viGlobal]=viLocal;
            OPENGM_CHECK_OP(incluedGlobalVar_[viGlobal],==,0,"internal inconsistency");
            incluedGlobalVar_[viGlobal] = cVi==ccColor0 ? 1 : 2;
            ++viLocal;
        }
    }
    numSubVar_=viLocal;
}

template<class GM>
template<class ARG>
void SubmodelCGC<GM>::setSubVarImplicitBfs( 
    const ARG & arg,
    const typename SubmodelCGC<GM>::LabelType     ccColor0,
    const typename SubmodelCGC<GM>::LabelType     ccColor1,
    const typename SubmodelCGC<GM>::IndexType vi0,
    const typename SubmodelCGC<GM>::IndexType vi1
){
    greedyMode_=true;


    //std::cout<<"fill bfs\n";
    //std::cout<<" v0 "<< vi0 << " c0 "<< ccColor0<<"\n";
    //std::cout<<" v1 "<< vi1 << " c1 "<< ccColor1<<"\n";


    //std::queue<IndexType> cQueue;

    IndexType ssize = 2;
    stack_[0]=vi0;
    stack_[1]=vi1;

    //cQueue.push(vi0);
    //cQueue.push(vi1);
    subVarToGlobal_[0]=vi0;
    subVarToGlobal_[1]=vi1;
    incluedGlobalVar_[vi0]=1;
    incluedGlobalVar_[vi1]=2;
    numSubVar_=2;

    while(ssize>0){

        const IndexType vi=stack_[ssize-1];
        --ssize;

        if(incluedGlobalVar_[vi]==0){
            incluedGlobalVar_[vi]= (arg[vi]==ccColor0 ? 1 : 2);
            // need to be sorted later
            subVarToGlobal_[numSubVar_]=vi;
            ++numSubVar_;
        }

        for(IndexType n=0;n<visAdj_[vi].size();++n){
            const IndexType nvi =  visAdj_[vi][n];
            const LabelType cvi = arg[nvi];

            if(incluedGlobalVar_[nvi]==0 && (cvi==ccColor0 || cvi==ccColor1)){
                
                incluedGlobalVar_[nvi]= (arg[nvi]==ccColor0 ? 1 : 2);
                // need to be sorted later
                subVarToGlobal_[numSubVar_]=nvi;

                ++numSubVar_;

                stack_[ssize]=nvi;
                ++ssize;
                OPENGM_CHECK_OP(ssize,<=,gm_.numberOfVariables()*2,"");
            }
        }
    }

    //(std::cout<<" n local var bfs "<<numSubVar_<<"\n";
    //std::cout<<"sort stuff\n";

    std::sort(subVarToGlobal_.begin(),subVarToGlobal_.begin()+numSubVar_);

    //std::cout<<"global to local\n";

    for(IndexType lvi=0;lvi<numSubVar_;++lvi){
        const IndexType gvi=subVarToGlobal_[lvi];
        globalVarToLocal_[gvi]=lvi;
    }


    //std::cout<<"bfs done\n";
}

template<class GM>
inline SubmodelCGC<GM>::SubmodelCGC(
    const GM & gm,
    const IndexType maxBruteForceSize2,
    const IndexType maxBruteForceSize4,
    const bool useBfs
)
:     gm_(gm),
    subVarToGlobal_(gm.numberOfVariables()),
    globalVarToLocal_(gm.numberOfVariables()),
    incluedGlobalVar_(gm.numberOfVariables(),false),
    incluedGlobalFactors_(gm.numberOfFactors(),false),
    isABorderFactor_(gm.numberOfFactors(),false),
    localArg_(gm.numberOfVariables()),
    localArgTest_(gm.numberOfVariables()),
    localFactorVis_(),
    globalLambdas_(gm.numberOfFactors()),
    localLambdas_(gm.numberOfFactors()),
    insideFactors_(gm.numberOfFactors()),
    borderFactor_(gm.numberOfFactors()),
    nInsideFactors_(0),
    nBorderFactors_(0),
    numSubVar_(0),
    numSubFactors_(0),
    localEdgemap_(),
    maxBruteForceSize2_(0),//maxBruteForceSize2<4 ? 4 : maxBruteForceSize2),
    maxBruteForceSize4_(0)//maxBruteForceSize4<4 ? 4 : maxBruteForceSize4)
{ 
    gm_.variableAdjacencyList(visAdj_);
    stack_.resize(gm_.numberOfVariables()*2);
    useBfs_=useBfs;
    std::fill(incluedGlobalVar_.begin(),incluedGlobalVar_.end(),false);
    // resize factor Vis
    IndexType shape[2]={gm.numberOfFactors(),3};
    localFactorVis_.resize(shape,shape+2);
    //
    LabelType lAA[2] = { 0, 0};
    LabelType lAB[2] = { 0, 1};
    for(IndexType fi=0;fi<gm_.numberOfFactors();++fi){
        globalLambdas_[fi]=gm[fi].operator()(lAB)-gm[fi].operator()(lAA);
    }
}


template<class GM>
typename SubmodelCGC<GM>::IndexType 
SubmodelCGC<GM>::ccFromLocalArg
(
    const IndexType offset
){
    // merge with UFD (and primal to dual arg)
    opengm::Partition<IndexType> ufd(numSubVar_);
    for(IndexType f=0;f<numSubFactors_;++f){
        const IndexType sv0=localFactorVis_(f,0);
        const IndexType sv1=localFactorVis_(f,1);
        if( localArg_[sv0]  == localArg_[sv1]){
            ufd.merge(sv0,sv1);
        }
    }

    // relabel with UFD MAP
    // and write to final result
    const IndexType numberOfCCs=ufd.numberOfSets();
    std::map<IndexType,IndexType> repLabeling;
    ufd.representativeLabeling(repLabeling);

    for(IndexType subVi=0;subVi<numSubVar_;++subVi){
        const IndexType findSubVi  = ufd.find(subVi);
        const IndexType denseLabel = repLabeling[findSubVi];
        localArg_[subVi]=denseLabel + offset;
    }
    return numberOfCCs;
}




template<class GM>
typename SubmodelCGC<GM>::ValueType SubmodelCGC<GM>::inferMulticut(
    Mode mode
){
    #ifdef WITH_CPLEX
    typedef opengm::Multicut<GM,opengm::Minimizer> Multicut;
    typename Multicut::Parameter para;
    para.workFlow_="(IC)(CC-I)";
    Multicut mc(numSubVar_,numSubFactors_,localFactorVis_,localLambdas_,para);

    if(mode == SingleSubset) {
        std::cout << "    [SS] MC with #var=" << numSubVar_ << ", #factors=" << numSubFactors_ << std::flush;
    }
    else {
        std::cout << "    [TS] MC with #var=" << numSubVar_ << ", #factors=" << numSubFactors_ << std::flush;
    }

    
    std::vector<IndexType> mcarg;
    //std::cout<<"run multicut \n";
    //McVerboseVisitor visitor;
    opengm::Timer t; t.tic();
    mc.infer();
    t.toc();
    double e = t.elapsedTime(); 
    std::cout << " ... " << std::fixed << 1000.0*e << " msec." << std::endl;
    
    //std::cout<<"get multicut arg\n";
    mc.arg(mcarg);

    std::copy(mcarg.begin(),mcarg.end(),localArg_.begin());


    //std::cout<<"get multicut value\n";
    ValueType value=0;
    
    for(IndexType fiLocal=0;fiLocal<numSubFactors_;++fiLocal){
        const IndexType localVi0 = localFactorVis_(fiLocal,0);
        const IndexType localVi1 = localFactorVis_(fiLocal,1);
        if(localArg_[localVi0]!=localArg_[localVi1])
            value+=localLambdas_[fiLocal];
    }
    //std::cout<<"return value\n";
    return value;
    #else
        throw opengm::RuntimeError("inferMulticut needs WITH_CPLEX");
    #endif
}

template<class GM>
typename SubmodelCGC<GM>::ValueType SubmodelCGC<GM>::inferQPBOI(
    Mode mode
){  
#ifdef WITH_QPBO
   typedef double REAL;
   typedef typename  kolmogorov::qpbo::QPBO<REAL>::NodeId NodeId;
   typedef typename  kolmogorov::qpbo::QPBO<REAL>::EdgeId EdgeId;
   typedef typename  kolmogorov::qpbo::QPBO<REAL>::ProbeOptions ProbeOptions;
                 
   kolmogorov::qpbo::QPBO<REAL>* qpbo = new kolmogorov::qpbo::QPBO<REAL>(numSubVar_, numSubFactors_); // construct with an error message function
   qpbo->AddNode(numSubVar_);
   qpbo->AddUnaryTerm(0, 0.0, 10000000.0);
   for(size_t i=0; i<numSubFactors_; ++i){
      qpbo->AddPairwiseTerm( (NodeId)localFactorVis_(i,0), (NodeId)localFactorVis_(i,1),    (REAL)0.0, (REAL)localLambdas_[i],(REAL)localLambdas_[i],(REAL)0.0 );
   }
   qpbo->MergeParallelEdges(); 
   for(size_t i=0; i < numSubVar_ ; ++i)
      qpbo->SetLabel(i, 0);

   srand( 42 );
   qpbo->Improve();
   
   // get the labels
   for ( size_t i=0; i < numSubVar_ ; ++i ) {
      localArg_[i] = qpbo->GetLabel(i);
   }            

   ValueType value=0;
   for(IndexType fiLocal=0;fiLocal<numSubFactors_;++fiLocal){
      const IndexType localVi0 = localFactorVis_(fiLocal,0);
      const IndexType localVi1 = localFactorVis_(fiLocal,1);
      if(localArg_[localVi0]!=localArg_[localVi1])
         value+=localLambdas_[fiLocal];
   }
   delete qpbo;
   return value;
#else
   throw opengm::RuntimeError("inferQPBOI needs WITH_QPBO");
   return 0.0;
#endif
  
  
}


template<class GM>
typename SubmodelCGC<GM>::ValueType SubmodelCGC<GM>::inferPlanarMaxCut(

){
#if defined(WITH_BLOSSOM5) && defined(WITH_PLANARITY)  
    opengm::external::PlanarMaxCut solver(numSubVar_, numSubFactors_);
    for(size_t i=0; i<numSubFactors_; ++i){
        solver.addEdge(localFactorVis_(i,0),localFactorVis_(i,1), -1.0*localLambdas_[i]);
    }
    solver.calculateCut();
    solver.getLabeling(localArg_);


    ValueType value=0;
    for(IndexType i=0;i<numSubFactors_;++i){
        const IndexType localVi0 = localFactorVis_(i,0);
        const IndexType localVi1 = localFactorVis_(i,1);
        if(localArg_[localVi0]!=localArg_[localVi1])
            value+=localLambdas_[i];
    }
    return value;
#else
   throw opengm::RuntimeError("inferPlanarMaxCut needs WITH_BLOSSOM5 and WITH_PLANARITY");
   return 0.0;
#endif
}



template<class GM>
typename GM::ValueType SubmodelCGC<GM>::inferBruteForce2(){
    OPENGM_CHECK_OP(numSubVar_,<=,64,"too many variables for brute force");
    // INFER
    ValueType bestVal=0.0;
    const opengm::UInt64Type numFlipVar=numSubVar_-1;
    const opengm::UInt64Type numConfig=2^numFlipVar;
    localArgTest_[0]=0;
    std::fill(localArgTest_.begin(),localArgTest_.begin()+numSubVar_,0);
    std::fill(localArg_.begin()    ,localArg_.begin()+numSubVar_,0);
    for ( opengm::UInt64Type i = 0 ; i < numConfig ; i++ ){

        for ( opengm::UInt64Type j = 0 ; j < numFlipVar ; j++ ){
            localArgTest_[j+1] = static_cast<opengm::UInt64Type>(bool( i & (1 << j) ));
        }
        // EVALUATE
        ValueType newVal = 0.0;
        for(IndexType f=0;f<numSubFactors_;++f){

            if(localArgTest_[localFactorVis_(f,0)]!=localArgTest_[localFactorVis_(f,1)])
                newVal+=localLambdas_[f];
        }
        // CHECK WHICH IS BETTER
        if(newVal<bestVal){
            bestVal=newVal;
            std::copy(localArgTest_.begin(),localArgTest_.begin()+numSubVar_,localArg_.begin());
        }
    }
       return bestVal;
}

template<class GM>
typename GM::ValueType SubmodelCGC<GM>::inferBruteForce4(){
    OPENGM_CHECK_OP(numSubVar_,<=,15,"to many variables for brute force 4");
    // INFER
    ValueType bestVal=0.0;
    const opengm::UInt64Type numFlipVar=numSubVar_-1;
    const opengm::UInt64Type numConfig=std::pow(4,numFlipVar);
    localArgTest_[0]=0;
    std::fill(localArgTest_.begin(),localArgTest_.begin()+numSubVar_,0);
    std::fill(localArg_.begin()    ,localArg_.begin()+numSubVar_,0);
    for ( opengm::UInt64Type i = 0 ; i < numConfig ; i++ ){

        for ( opengm::UInt64Type j = 0 ; j < numFlipVar ; j++ ){
            const opengm::UInt64Type ba = static_cast<bool>(i & ( 1 << (j*2)    ) );
            opengm::UInt64Type bb = static_cast<bool>(i & ( 1 << (j*2 +1) ) );
            bb=bb<<1;
            const opengm::UInt64Type  c = ba | bb;
            //std::cout<<"c "<<c<<"\n";
            localArgTest_[j+1] =  c;
        }
        // EVALUATE
        ValueType newVal = 0.0;
        for(IndexType f=0;f<numSubFactors_;++f){

            if(localArgTest_[localFactorVis_(f,0)]!=localArgTest_[localFactorVis_(f,1)])
                newVal+=localLambdas_[f];
        }
        // CHECK WHICH IS BETTER
        if(newVal<bestVal){
            bestVal=newVal;
            std::copy(localArgTest_.begin(),localArgTest_.begin()+numSubVar_,localArg_.begin());
        }
    }
       return bestVal;
}





template<class GM>
inline void SubmodelCGC<GM>::setUpSubFactors(){

    OPENGM_CHECK_OP(numSubFactors_,==,0,"internal inconsistency");
    OPENGM_CHECK_OP(numSubVar_,>,0,"internal inconsistency");
    //OPENGM_CHECK_OP(nInsideFactors_,==,0,"internal inconsistency");
    //OPENGM_CHECK_OP(nBorderFactors_,==,0,"internal inconsistency");
    nBorderFactors_=0;
    nInsideFactors_=0;
    oldCutValue_=0.0;
    isOptCut_=true;
    for(IndexType viLocal=0;viLocal<numSubVar_;++viLocal){
        const IndexType viGlobal=subVarToGlobal_[viLocal];    
        const IndexType numFacVar = gm_.numberOfFactors(viGlobal);
        for(IndexType f=0; f<numFacVar; ++f){
            const IndexType fiGlobal=gm_.factorOfVariable(viGlobal,f);

            if(incluedGlobalFactors_[fiGlobal]==false){
                const IndexType viA = gm_.variableOfFactor(fiGlobal,0);
                const IndexType viB = gm_.variableOfFactor(fiGlobal,1);
                const IndexType viGlobalOther = viA==viGlobal ? viB : viA;

                const IndexType viGlobal0 = viGlobal<viGlobalOther ? viGlobal : viGlobalOther;
                const IndexType viGlobal1 = viGlobal<viGlobalOther ? viGlobalOther : viGlobal;

                // check if the other variable of the factor is 
                // also in the sub-problem
                if(incluedGlobalVar_[viGlobalOther]!=0){
                    // this is a inside factors
                    OPENGM_CHECK_OP(nInsideFactors_,<,gm_.numberOfFactors(),"");
                    insideFactors_[nInsideFactors_]=fiGlobal;
                    ++nInsideFactors_;



                    incluedGlobalFactors_[fiGlobal]=true;
                    // subgraph
                    localFactorVis_(numSubFactors_,0)=globalVarToLocal_[viGlobal0];
                    localFactorVis_(numSubFactors_,1)=globalVarToLocal_[viGlobal1];
                    localFactorVis_(numSubFactors_,2)=fiGlobal;
                    // local lambda
                    const ValueType lamb=globalLambdas_[fiGlobal];
                    if(incluedGlobalVar_[viGlobalOther]!=incluedGlobalVar_[viGlobal]){
                        oldCutValue_+=lamb;
                        if(lamb>=0.0){
                            isOptCut_=false;
                        }
                    }
                    
                    localLambdas_[numSubFactors_]=lamb;
                    ++numSubFactors_;
                }
                else{
                    // this is a border factor
                    OPENGM_CHECK_OP(nBorderFactors_,<,gm_.numberOfFactors(),"");
                    borderFactor_[nBorderFactors_]=fiGlobal;
                    ++nBorderFactors_;
                }
            }
        }
    }

    //std::cout<<"inside factor "<<nInsideFactors_<<"\n";
    //std::cout<<"border factor "<<nBorderFactors_<<"\n";


}

template<class GM>
inline void SubmodelCGC<GM>::unsetSubVar(){
    OPENGM_CHECK_OP(numSubVar_,>,0,"internal inconsistency");
    for(IndexType viLocal=0;viLocal<numSubVar_;++viLocal){
        const IndexType viGlobal=subVarToGlobal_[viLocal];
        OPENGM_CHECK(incluedGlobalVar_[viGlobal]>0,"internal inconsistency");
        incluedGlobalVar_[viGlobal]=0;
    }
    for(IndexType fiLocal=0;fiLocal<numSubFactors_;++fiLocal){
        OPENGM_CHECK(incluedGlobalFactors_[localFactorVis_(fiLocal,2)],"internal inconsistency");
        incluedGlobalFactors_[localFactorVis_(fiLocal,2)]=false;
    }
    numSubVar_=0;
    numSubFactors_=0;
    //nInsideFactors_=0;
    //nBorderFactors_=0;


    localEdgemap_.clear();
}

#endif /* OPENGM_HMC_SUBMODEL2 */
