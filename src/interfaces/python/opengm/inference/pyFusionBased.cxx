#include <boost/python.hpp>
#include <string>
#include "inf_def_visitor.hxx"

#include <opengm/inference/fusion_based_inf.hxx>
#include <param/fusion_based_param.hxx>


template<class GM,class ACC>
void export_fusion_based(){
    using namespace boost::python;
    import_array();
    append_subnamespace("solver");

    // setup 
    InfSetup setup;
    setup.cite       = "";
    setup.algType    = "fusion-moves";
    setup.guarantees = "";
    setup.examples   = "";

    typedef opengm::FusionBasedInf<GM, ACC>  PyFusionBasedInf;
    enum_<typename PyFusionBasedInf::FusionSolver> ("_FusionBased_FusionSolver")
        .value("qpbo",     PyFusionBasedInf::QpboFusion)
        .value("cplex",   PyFusionBasedInf::CplexFusion)
        .value("lf",    PyFusionBasedInf::LazyFlipperFusion)
    ;

    enum_<typename PyFusionBasedInf::ProposalGen> ("_FusionBased_ProposalGen")
        .value("AlphaExpansion",     PyFusionBasedInf::AlphaExpansion)
        .value("AlphaBetaSwap",     PyFusionBasedInf::AlphaBetaSwap)
        .value("Random",     PyFusionBasedInf::Random)
        .value("RandomLF",     PyFusionBasedInf::RandomLF)
        .value("NonUniformRandom",     PyFusionBasedInf::NonUniformRandom)
        .value("Blur",     PyFusionBasedInf::Blur)
        .value("EnergyBlur",     PyFusionBasedInf::EnergyBlur)
    ;


    // export parameter

    exportInfParam<PyFusionBasedInf>("_FusionBased");
    // export inferencePyFusionBasedInf
    class_< PyFusionBasedInf>("_FusionBased",init<const GM & >())  
    .def(InfSuite<PyFusionBasedInf>(std::string("FusionBased"),setup))
    ;
}

template void export_fusion_based<opengm::python::GmAdder,opengm::Minimizer>();
