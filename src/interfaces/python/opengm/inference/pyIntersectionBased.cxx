#ifndef NOVIGRA
#include <boost/python.hpp>
#include <string>
#include "inf_def_visitor.hxx"

#include <opengm/inference/intersection_based_inf.hxx>
#include <param/intersection_based_param.hxx>



#ifdef WITH_CPLEX
#include "opengm/inference/auxiliary/fusion_move/permutable_label_fusion_mover.hxx"
#endif

template<class GEN>
void export_intersection_based_t( InfSetup & setup, const std::string & genName){

    typedef opengm::IntersectionBasedInf<typename GEN::GraphicalModelType, GEN> INF; 
    setup.hyperParameters= StringVector(1,genName);
    const std::string baseName("IntersectionBased");
    const std::string tBaseName = baseName +  std::string("_") + genName;
    const std::string name = std::string("_")+tBaseName; 
    exportInfParam<INF>(name.c_str()); // "IntersectionBased"
    // export inferencePyIntersectionBasedInf
    class_< INF>(name.c_str(),init<const typename GEN::GraphicalModelType & >())  
    .def(InfSuite<INF>(baseName,setup))
    ;
}

template<class GEN>
void export_intersection_based_proposal_param( InfSetup & setup, const std::string & genName){

    setup.hyperParameters= StringVector(1,genName);
    const std::string baseName("FusionBased");
    const std::string tBaseName = baseName +  std::string("_") + genName;
    const std::string name = std::string("_")+tBaseName+std::string("_ProposalParam"); 
    exportInfParam<GEN>(name.c_str()); // "IntersectionBased"
}




template<class GM,class ACC>
void export_intersection_based(){
    using namespace boost::python;
    import_array();
    append_subnamespace("solver");

    // documentation 
    InfSetup setup;
    setup.cite       = "";
    setup.algType    = "fusion-moves";
    setup.hyperParameterKeyWords        = StringVector(1,std::string("generator"));
    setup.hyperParametersDoc            = StringVector(1,std::string("proposal generator"));
    // parameter of inference will change if hyper parameter changes
    setup.hasInterchangeableParameter   = false;



    #ifndef NOVIGRA
    typedef opengm::proposal_gen::RandomizedHierarchicalClustering<GM, opengm::Minimizer>   RHCGen;
    typedef opengm::proposal_gen::RandomizedWatershed<GM, opengm::Minimizer>                RWSGen;
    #endif


    typedef opengm::proposal_gen::QpboBased<GM, opengm::Minimizer>                QpboGen;


    typedef opengm::proposal_gen::WeightRandomization<typename GM::ValueType> WeightRand;

    typedef typename  WeightRand::Parameter PyWeightRand;







    enum_<typename PyWeightRand::NoiseType> ("_WeightRandomization_NoiseType_")
        .value("normalAdd",    PyWeightRand::NormalAdd)
        .value("uniformAdd",   PyWeightRand::UniformAdd)
        .value("normalMult",  PyWeightRand::NormalMult)
        .value("none",  PyWeightRand::None)
    ;

    class_<PyWeightRand>("_WeightRandomizerParameter_", init<>())
        .def_readwrite("noiseType",&PyWeightRand::noiseType_)
        .def_readwrite("noiseParam",&PyWeightRand::noiseParam_)
        .def_readwrite("seed",&PyWeightRand::seed_)
        .def_readwrite("ignoreSeed",&PyWeightRand::ignoreSeed_)
    ;


    exportInfParam<  opengm::PermutableLabelFusionMove<GM, ACC> >("_PermutableLabelFusionMove"); // "IntersectionBased"

    


    #ifndef NOVIGRA
    // RandomizedHierarchicalClustering
    {   
        setup.isDefault=true;
        const std::string genName("randomizedHierarchicalClustering");
        typedef RHCGen GEN;

        export_intersection_based_proposal_param<GEN>(setup, genName);
        export_intersection_based_t<GEN>(setup, genName);
    }
    // RandomizedHierarchicalClustering
    {   
        setup.isDefault=false;
        const std::string genName("randomizedWatershed");
        typedef RWSGen GEN;

        export_intersection_based_proposal_param<GEN>(setup, genName);
        export_intersection_based_t<GEN>(setup, genName);
    }
    #endif
     // Qpbo Based
    #ifdef WITH_QPBO
    {   
        setup.isDefault=false;
        const std::string genName("qpboBased");
        typedef QpboGen GEN;

        export_intersection_based_proposal_param<GEN>(setup, genName);
        export_intersection_based_t<GEN>(setup, genName);
    }
    #endif
    
}

template void export_intersection_based<opengm::python::GmAdder,opengm::Minimizer>();
#endif