#ifndef HELPER_HXX
#define HELPER_HXX

#include <boost/python.hpp>
#include <boost/python/module.hpp>
#include <opengm/python/opengmpython.hxx>
#include <opengm/python/converter.hxx>
#include <opengm/python/numpyview.hxx>

#include <opengm/inference/icm.hxx>
#include <opengm/inference/lazyflipper.hxx>
#include <opengm/inference/self_fusion.hxx>
#include <opengm/learning/gridsearch-learning.hxx>
#include <opengm/inference/messagepassing/messagepassing.hxx>

#ifdef WITH_CPLEX
#include <opengm/inference/lpcplex.hxx>
#include <opengm/inference/multicut.hxx>
#endif

#ifdef WITH_QPBO
#include <opengm/inference/external/qpbo.hxx>
#include <opengm/inference/reducedinference.hxx>
#endif

#ifdef WITH_TRWS
#include <opengm/inference/external/trws.hxx>
#endif


namespace opengm{

template<class LEARNER>
class LearnerInferenceSuite: public boost::python::def_visitor<LearnerInferenceSuite<LEARNER> >{
public:
    friend class boost::python::def_visitor_access;

    LearnerInferenceSuite(){

    }

    template<class INF>
    static void pyLearn_Inf(LEARNER & learner, const typename INF::Parameter & param)
    {
        learner. template learn<INF>(param);
    }

    #ifdef WITH_QPBO
    template<class INF>
    static void pyLearn_ReducedInf(
        LEARNER & learner, 
        const typename INF::Parameter & param,
        const bool persistency,
        const bool tentacles,
        const bool connectedComponents
    )
    {

        typedef typename INF::GraphicalModelType GmType;
        typedef typename opengm::ReducedInferenceHelper<GmType>::InfGmType RedInfGm;

        // rebind the inference to the RedInfGm
        typedef typename INF:: template RebindGm<RedInfGm>::type RedInfRebindInf;


        typedef typename RedInfRebindInf::Parameter RedInfRebindInfParam;
        typedef opengm::ReducedInference<GmType, opengm::Minimizer, RedInfRebindInf> RedInf;
        typedef typename RedInf::Parameter RedInfParam;

        RedInfRebindInfParam redInfRebindInfParam(param);

        RedInfParam redInfPara;
        redInfPara.subParameter_ = redInfRebindInfParam;
        redInfPara.Persistency_ = persistency;
        redInfPara.Tentacle_ = tentacles;
        redInfPara.ConnectedComponents_ = connectedComponents;

        learner. template learn<RedInf>(redInfPara);
    }
    #endif


    #ifdef WITH_QPBO
    template<class INF>
    static void pyLearn_ReducedInfSelfFusion(
        LEARNER & learner, 
        const typename INF::Parameter & param,
        const bool persistency,
        const bool tentacles,
        const bool connectedComponents
    )
    {

        typedef typename INF::GraphicalModelType GmType;
        typedef typename opengm::ReducedInferenceHelper<GmType>::InfGmType RedInfGm;

        // rebind the inference to the RedInfGm
        typedef typename INF:: template RebindGm<RedInfGm>::type RedInfRebindInf;


        typedef typename RedInfRebindInf::Parameter RedInfRebindInfParam;
        typedef opengm::ReducedInference<GmType, opengm::Minimizer, RedInfRebindInf> RedInf;
        typedef typename RedInf::Parameter RedInfParam;

        RedInfRebindInfParam redInfRebindInfParam(param);

        RedInfParam redInfPara;
        redInfPara.subParameter_ = redInfRebindInfParam;
        redInfPara.Persistency_ = persistency;
        redInfPara.Tentacle_ = tentacles;
        redInfPara.ConnectedComponents_ = connectedComponents;


        typedef opengm::SelfFusion<RedInf> SelfFusionInf;
        typedef typename SelfFusionInf::Parameter SelfFusionInfParam;
        SelfFusionInfParam sfParam;

        sfParam.infParam_ = redInfPara;
        sfParam.fuseNth_ = 10;
        sfParam.maxSubgraphSize_ = 2;
        sfParam.reducedInf_ = true;
        sfParam.tentacles_ = false;
        sfParam.connectedComponents_ = true;
        sfParam.fusionTimeLimit_ = 100.0;
        sfParam.numStopIt_ = 10.0;
        sfParam.fusionSolver_ = SelfFusionInf::QpboFusion;

        learner. template learn<SelfFusionInf>(sfParam);
    }
    #endif


    template<class INF>
    static void pyLearn_SelfFusion(
        LEARNER & learner, 
        const typename INF::Parameter & param,
        const size_t fuseNth,
        const std::string & fusionSolver,
        const UInt64Type maxSubgraphSize,
        const bool reducedInf,
        const bool connectedComponents,
        const double fusionTimeLimit,
        const size_t numStopIt
    )
    {

        typedef typename INF::GraphicalModelType GmType;
        
        typedef opengm::SelfFusion<INF> SelfFusionInf;
        typedef typename SelfFusionInf::Parameter SelfFusionInfParam;


        SelfFusionInfParam sfParam;

        if(fusionSolver ==std::string("qpbo")){
            sfParam.fusionSolver_ = SelfFusionInf::QpboFusion;
        }
        else if(fusionSolver ==std::string("cplex")){
            sfParam.fusionSolver_ = SelfFusionInf::CplexFusion;
        }
        else if(fusionSolver ==std::string("lf")){
            sfParam.fusionSolver_ = SelfFusionInf::LazyFlipperFusion;
        }

        sfParam.infParam_ = param;
        sfParam.fuseNth_ = fuseNth;
        sfParam.maxSubgraphSize_ = maxSubgraphSize;
        sfParam.reducedInf_ = reducedInf;
        sfParam.tentacles_ = false;
        sfParam.connectedComponents_ = connectedComponents;
        sfParam.fusionTimeLimit_ = fusionTimeLimit;
        sfParam.numStopIt_ = numStopIt;

        learner. template learn<SelfFusionInf>(sfParam);
    }







    template <class classT>
    void visit(classT& c) const{
        // SOME INFERENCE METHODS
        typedef typename LEARNER::GMType GMType;
        typedef typename LEARNER::Parameter PyLearnerParam;
        typedef typename LEARNER::DatasetType DatasetType;
        typedef opengm::Minimizer ACC;

        typedef opengm::ICM<GMType, ACC> IcmInf;
        typedef opengm::LazyFlipper<GMType, ACC> LazyFlipperInf;
        typedef opengm::BeliefPropagationUpdateRules<GMType, ACC> UpdateRulesType;
        typedef opengm::MessagePassing<GMType, ACC, UpdateRulesType, opengm::MaxDistance> BpInf;

        #ifdef WITH_CPLEX
            typedef opengm::LPCplex<GMType, ACC> Cplex;
            typedef opengm::Multicut<GMType, ACC> Multicut;
        #endif

        #ifdef WITH_QPBO
            typedef opengm::external::QPBO<GMType>  QpboExternal;
        #endif

        #ifdef WITH_TRWS
            typedef opengm::external::TRWS<GMType>  TrwsExternal;
        #endif

        c
            //.def("_learn",&pyLearn_Inf<IcmInf>)
            //.def("_learn",&pyLearn_Inf<LazyFlipperInf>)
            //.def("_learn",&pyLearn_Inf<BpInf>)
            #ifdef WITH_CPLEX
            //.def("_learn",&pyLearn_Inf<Cplex>) 
            .def("_learn",&pyLearn_Inf<Multicut>)
            #endif
            #ifdef WITH_QPBO
            .def("_learn",&pyLearn_Inf<QpboExternal>)
            #endif
            #ifdef WITH_TRWS
            .def("_learn",&pyLearn_Inf<TrwsExternal>)
            #endif

            #if 0
            // REDUCED INFERENCE
            #ifdef WITH_QPBO
                .def("_learnReducedInf",&pyLearn_ReducedInf<LazyFlipperInf>)
                #ifdef WITH_TRWS
                .def("_learnReducedInf",&pyLearn_ReducedInf<TrwsExternal>)
                #endif
                #ifdef WITH_CPLEX
                .def("_learnReducedInf",&pyLearn_ReducedInf<Cplex>)
                #endif
            #endif

            // SELF FUSION
            #ifdef WITH_TRWS
            .def("_learnSelfFusion",&pyLearn_SelfFusion<TrwsExternal>)
            #endif

            // REDUCED INFERNCE SELF FUSION
            #if defined(WITH_TRWS) && defined(WITH_QPBO)
            .def("_learnReducedInfSelfFusion",&pyLearn_ReducedInfSelfFusion<TrwsExternal>)
            #endif
            #endif
        ;
    }
};



template<class DS>
class DatasetInferenceSuite: public boost::python::def_visitor<DatasetInferenceSuite<DS> >{
public:
   friend class boost::python::def_visitor_access;

   DatasetInferenceSuite(){

   }

   template<class INF>
   static typename DS::ValueType pyGetLossWithInf(DS & ds, const typename INF::Parameter & param, const size_t i)
   {
       return ds. template getLoss<INF>(param, i);
   }

   template<class INF>
   static typename DS::ValueType pyGetTotalLossWithInf(DS & ds, const typename INF::Parameter & param)
   {
       return ds. template getTotalLoss<INF>(param);
   }

   template <class classT>
   void visit(classT& c) const{
       // SOME INFERENCE METHODS
       typedef typename DS::GMType GMType;
       typedef opengm::Minimizer ACC;

       typedef opengm::ICM<GMType, ACC> IcmInf;
       typedef opengm::LazyFlipper<GMType, ACC> LazyFlipperInf;
       typedef opengm::BeliefPropagationUpdateRules<GMType, ACC> UpdateRulesType;
       typedef opengm::MessagePassing<GMType, ACC, UpdateRulesType, opengm::MaxDistance> BpInf;

#ifdef WITH_CPLEX
       typedef opengm::LPCplex<GMType, ACC> Cplex;
       typedef opengm::Multicut<GMType, ACC> Multicut;
#endif
#ifdef WITH_QPBO
       typedef opengm::external::QPBO<GMType>  QpboExternal;
#endif
#ifdef WITH_TRWS
       typedef opengm::external::TRWS<GMType>  TrwsExternal;
#endif






      c
          .def("_getLoss",&pyGetLossWithInf<IcmInf>)
          .def("_getTotalLoss",&pyGetTotalLossWithInf<IcmInf>)
          .def("_getLoss",&pyGetLossWithInf<LazyFlipperInf>)
          .def("_getTotalLoss",&pyGetTotalLossWithInf<LazyFlipperInf>)
          .def("_getLoss",&pyGetLossWithInf<BpInf>)
          .def("_getTotalLoss",&pyGetTotalLossWithInf<BpInf>)
#ifdef WITH_CPLEX
          .def("_getLoss",&pyGetLossWithInf<Cplex>)
          .def("_getTotalLoss",&pyGetTotalLossWithInf<Cplex>)
          .def("_getLoss",&pyGetLossWithInf<Multicut>)
          .def("_getTotalLoss",&pyGetTotalLossWithInf<Multicut>)
#endif
#ifdef WITH_QPBO
          .def("_getLoss",&pyGetLossWithInf<QpboExternal>)
          .def("_getTotalLoss",&pyGetTotalLossWithInf<QpboExternal>)
#endif
#ifdef WITH_TRWS
          .def("_getLoss",&pyGetLossWithInf<TrwsExternal>)
          .def("_getTotalLoss",&pyGetTotalLossWithInf<TrwsExternal>)
#endif
      ;
   }
};



} // namespace opengm

#endif // HELPER_HXX

