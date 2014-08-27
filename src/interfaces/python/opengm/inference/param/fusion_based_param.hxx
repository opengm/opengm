#ifndef LAZY_FLIPPER_PARAM
#define LAZY_FLIPPER_PARAM

#include "param_exporter_base.hxx"
//solver specific
#include <opengm/inference/fusion_based_inf.hxx>

using namespace boost::python;

template<class INFERENCE>
class InfParamExporterFusionBased{

public:
    typedef typename INFERENCE::ValueType ValueType;
    typedef typename INFERENCE::Parameter Parameter;
    typedef InfParamExporterFusionBased<INFERENCE> SelfType;

    inline static void set 
    (
        Parameter & p,
        const typename INFERENCE::ProposalGen  proposalGen,
        const typename INFERENCE::FusionSolver fusionSolver,
        const size_t                numIt,
        const size_t                numStopIt,
        const double                fusionTimeLimit,
        const opengm::UInt64Type    maxSubgraphSize,
        const opengm::UInt64Type    solverSteps,
        const bool                  useDirectInterface,
        const bool                  reducedInf,
        const bool                  connectedComponents,
        const bool                  tentacles,
        const float                 temperature,
        const double                sigma,
        const bool                  useEstimatedMarginals
    ) {
        p.proposalGen_=proposalGen;
        p.fusionSolver_=fusionSolver;
    } 

    void static exportInfParam(const std::string & className){
    class_<Parameter > ( className.c_str() , init< > ())
        .def_readwrite("generator", &Parameter::proposalGen_,
            "proposal label generator (default = 'AlphaExpansion')  : \n\n"
            "  * 'AlphaExpansion'\n\n"
            "  * 'AlphaBetaSwap'\n\n"
            "  * 'Random'\n\n"
            "  * 'RandomLF'\n\n"
            "  * 'NonUniformRandom'\n\n"
            "  * 'Blur'\n\n"
            "  * 'EnergyBlur'\n\n"
        )
        .def_readwrite("fusionSolver",&Parameter::fusionSolver_,
            "type of proposal generator inference  : \n\n"
            "  * 'qpbo'\n\n"
            "  * 'cplex'\n\n"
            "  * 'lf'\n\n"
        )

        .def_readwrite("steps",&Parameter::numIt_,"total iterations")
        .def_readwrite("numStopIt",&Parameter::numStopIt_,"stop after n not successful steps")
        .def_readwrite("fusionTimeLimit",&Parameter::fusionTimeLimit_, "time limit for a single fusion move")
        .def_readwrite("maxSubgraphSize",&Parameter::maxSubgraphSize_, "max subgraph size if lf is used as fusion mover")
        .def_readwrite("solverSteps",&Parameter::solverSteps_, "solver steps")
        .def_readwrite("useDirectInterface",&Parameter::useDirectInterface_,"use direct interface (experimental)")
        .def_readwrite("reducedInf",&Parameter::reducedInf_,"use reduced inf")
        .def_readwrite("connectedComponents",&Parameter::connectedComponents_,"if reduced inf is used, use connectedComponents?")
        .def_readwrite("tentacles",&Parameter::tentacles_,"if reduced inf is used, use tentacle elimination")
        .def_readwrite("temperature",&Parameter::temperature_, "temperature for marginal estimation")
        .def_readwrite("sigma",&Parameter::sigma_,"blur sigma")
        .def_readwrite("useEstimatedMarginals",&Parameter::useEstimatedMarginals_,"used estimated marginals (experimental)")

        .def ("set", &SelfType::set, 
            (
                boost::python::arg("generator")=INFERENCE::AlphaExpansion,
                boost::python::arg("fusionSolver")=INFERENCE::QpboFusion,
                boost::python::arg("steps")=100, 
                boost::python::arg("numStopIt")=0,   
                boost::python::arg("fusionTimeLimit")=100.0, 
                boost::python::arg("maxSubgraphSize")=3,   
                boost::python::arg("solverSteps")=10, 
                boost::python::arg("useDirectInterface")=false,
                boost::python::arg("reducedInf")=false,
                boost::python::arg("connectedComponents")=false,
                boost::python::arg("tentacles")=false,
                boost::python::arg("temperature")=1.0,
                boost::python::arg("sigma")=20.0,
                boost::python::arg("useEstimatedMarginals")=false
            ) 
        )
    ;
    }
};

template<class GM,class ACC>
class InfParamExporter<opengm::FusionBasedInf<GM,ACC> >  : public  InfParamExporterFusionBased<opengm::FusionBasedInf< GM,ACC> > {

};

#endif
