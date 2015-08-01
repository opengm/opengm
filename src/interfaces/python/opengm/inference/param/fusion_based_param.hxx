#ifndef FUSION_BASED_PARAM
#define FUSION_BASED_PARAM

#include "empty_param.hxx"
#include "param_exporter_base.hxx"
//solver specific
#include <opengm/inference/fusion_based_inf.hxx>

using namespace boost::python;

template<class INFERENCE>
class InfParamExporterFusionBased{

public:
    typedef typename INFERENCE::ValueType ValueType;
    typedef typename INFERENCE::Parameter Parameter;
    typedef typename INFERENCE::ProposalGen Gen;
    typedef typename Gen::Parameter GenParameter;
    typedef InfParamExporterFusionBased<INFERENCE> SelfType;

    inline static void set 
    (
        Parameter & p,
        const GenParameter & proposalParam,
        const size_t         numIt,
        const size_t         numStopIt
    ) {
        p.proposalParam_ = proposalParam;
        p.numIt_ = numIt;
        p.numStopIt_ = numStopIt;
    } 

    void static exportInfParam(const std::string & className){
    class_<Parameter > ( className.c_str() , init< > ())
        .def_readwrite("proposalParam",&Parameter::proposalParam_,"parameters of the proposal generator")
        .def_readwrite("numIt",&Parameter::numIt_,"total number of iterations")
        .def_readwrite("numStopIt",&Parameter::numStopIt_,"stop after n not successful steps")
        .def ("set", &SelfType::set, 
            (
                boost::python::arg("proposalParam")=GenParameter(),
                boost::python::arg("numIt")=1000, 
                boost::python::arg("numStopIt")=0
            ) 
        )
    ;
    }
};


/*

template<class GM, class ACC>
class InfParamExporter<
    opengm::proposal_gen::RandomizedHierarchicalClustering<GM, ACC>
>{

public:
    typedef opengm::proposal_gen::RandomizedHierarchicalClustering<GM, ACC> GEN;
    typedef typename GEN::ValueType ValueType;
    typedef typename GEN::Parameter Parameter;
    typedef InfParamExporter< GEN > SelfType;

    inline static void set 
    (
        Parameter & p,
        const float         noise,
        const float         stopWeight,
        const float         reduction
    ) {
        p.noise_ = noise;
        p.stopWeight_ = stopWeight;
    } 

    void static exportInfParam(const std::string & className){
    class_<Parameter > ( className.c_str() , init< > ())
        .def_readwrite("noise",&Parameter::noise_,"noise level")
        .def_readwrite("stopWeight",&Parameter::stopWeight_,"stopWeight")
        .def_readwrite("reduction",&Parameter::reduction_,"reduction")
        .def ("set", &SelfType::set, 
            (
                boost::python::arg("noise")=1.0,
                boost::python::arg("stopWeight")=0.0,
                boost::python::arg("reduction")=-1.0
            ) 
        )
    ;
    }
};
*/





/*
typedef opengm::proposal_gen::AlphaExpansionGen<GM, ACC>    AEGen;
typedef opengm::proposal_gen::AlphaBetaSwapGen<GM, ACC>     ABGen;
typedef opengm::proposal_gen::UpDownGen<GM, ACC>            UDGen;
typedef opengm::proposal_gen::RandomGen<GM, ACC>            RGen;
typedef opengm::proposal_gen::RandomLFGen<GM, ACC>          RLFGen;
typedef opengm::proposal_gen::NonUniformRandomGen<GM, ACC>  NURGen;
typedef opengm::proposal_gen::BlurGen<GM, ACC>              BlurGen;
typedef opengm::proposal_gen::EnergyBlurGen<GM, ACC>        EBlurGen;
*/


template<class INFERENCE>
class InfParamExporterEmpty;


#define _EMPTY_PROPOSAL_PARAM(clsName)                        \
template<class GM,class ACC>                                  \
class InfParamExporter<          clsName <GM,ACC>     >       \
: public  InfParamExporterEmpty< clsName < GM,ACC>    > {     \
};
_EMPTY_PROPOSAL_PARAM(opengm::proposal_gen::AlphaExpansionGen)
_EMPTY_PROPOSAL_PARAM(opengm::proposal_gen::AlphaBetaSwapGen)
_EMPTY_PROPOSAL_PARAM(opengm::proposal_gen::UpDownGen)
_EMPTY_PROPOSAL_PARAM(opengm::proposal_gen::RandomGen)
//_EMPTY_PROPOSAL_PARAM(opengm::proposal_gen::RandomLFGen)
//_EMPTY_PROPOSAL_PARAM(opengm::proposal_gen::Random2Gen)
#undef _EMPTY_PROPOSAL_PARAM



template<class GM,class ACC>
class InfParamExporter<opengm::FusionBasedInf<GM,ACC> >  : public  InfParamExporterFusionBased<opengm::FusionBasedInf< GM,ACC> > {

};

#endif
