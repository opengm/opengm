#include <boost/python.hpp>
#include <string>
#include "inf_def_visitor.hxx"

#include <opengm/inference/fusion_based_inf.hxx>
#include <param/fusion_based_param.hxx>



template<class GEN>
void export_fusion_based_t( InfSetup & setup, const std::string & genName){

    typedef opengm::FusionBasedInf<typename GEN::GraphicalModelType, GEN> INF; 
    setup.hyperParameters= StringVector(1,genName);
    const std::string baseName("FusionBased");
    const std::string tBaseName = baseName +  std::string("_") + genName;
    const std::string name = std::string("_")+tBaseName; 
    exportInfParam<INF>(name.c_str()); // "_FusionBased"
    // export inferencePyFusionBasedInf
    class_< INF>(name.c_str(),init<const typename GEN::GraphicalModelType & >())  
    .def(InfSuite<INF>(baseName,setup))
    ;
}

template<class GEN>
void export_proposal_param( InfSetup & setup, const std::string & genName){

    setup.hyperParameters= StringVector(1,genName);
    const std::string baseName("FusionBased");
    const std::string tBaseName = baseName +  std::string("_") + genName;
    const std::string name = std::string("_")+tBaseName+std::string("_ProposalParam"); 
    exportInfParam<GEN>(name.c_str()); // "_FusionBased"
}




template<class GM,class ACC>
void export_fusion_based(){
    using namespace boost::python;
    import_array();
    append_subnamespace("solver");

    // documentation 
    InfSetup setup;
    setup.cite       = "";
    setup.algType    = "fusion-moves";
    setup.hyperParameterKeyWords        = StringVector(1,std::string("generator"));
    setup.hyperParametersDoc            = StringVector(1,std::string("inference based proposal generator"));
    // parameter of inference will change if hyper parameter changes
    setup.hasInterchangeableParameter   = false;



   typedef opengm::proposal_gen::AlphaExpansionGen<GM, opengm::Minimizer>   AEGen;
   typedef opengm::proposal_gen::AlphaBetaSwapGen<GM, opengm::Minimizer>    ABGen;
   typedef opengm::proposal_gen::JumpUpDownGen<GM, opengm::Minimizer>       JUDGen;
   typedef opengm::proposal_gen::MJumpUpDownGen<GM, opengm::Minimizer>      MJUDGen;
   typedef opengm::proposal_gen::UpDownGen<GM, opengm::Minimizer>           UDGen;
   typedef opengm::proposal_gen::RandomGen<GM, opengm::Minimizer>           RGen;
   typedef opengm::proposal_gen::RandomLFGen<GM, opengm::Minimizer>         RLFGen;
   typedef opengm::proposal_gen::NonUniformRandomGen<GM, opengm::Minimizer> NURGen;
   typedef opengm::proposal_gen::BlurGen<GM, opengm::Minimizer>             BlurGen;
   typedef opengm::proposal_gen::EnergyBlurGen<GM, opengm::Minimizer>       EBlurGen;



    // A-EXP
    {   
        setup.isDefault=true;
        const std::string genName("alphaExpansion");
        typedef AEGen GEN;

        export_proposal_param<GEN>(setup, genName);
        export_fusion_based_t<GEN>(setup, genName);
    }
    // AB-ABGen
    {   
        setup.isDefault=false;
        const std::string genName("alphaBetaSwap");
        typedef ABGen GEN;

        export_proposal_param<GEN>(setup, genName);
        export_fusion_based_t<GEN>(setup, genName);
    }
    // MJUDGen
    {   
        setup.isDefault=false;
        const std::string genName("mJumpUpDown");
        typedef MJUDGen GEN;

        export_proposal_param<GEN>(setup, genName);
        export_fusion_based_t<GEN>(setup, genName);
    }
    // JUDGen
    {   
        setup.isDefault=false;
        const std::string genName("jumpUpDown");
        typedef JUDGen GEN;

        export_proposal_param<GEN>(setup, genName);
        export_fusion_based_t<GEN>(setup, genName);
    }
    // UDGen
    {   
        setup.isDefault=false;
        const std::string genName("upDown");
        typedef UDGen GEN;

        export_proposal_param<GEN>(setup, genName);
        export_fusion_based_t<GEN>(setup, genName);
    }
    // RGen
    {   
        setup.isDefault=false;
        const std::string genName("random");
        typedef RGen GEN;

        export_proposal_param<GEN>(setup, genName);
        export_fusion_based_t<GEN>(setup, genName);
    }
    // RLFGen
    {   
        setup.isDefault=false;
        const std::string genName("randomLf");
        typedef RLFGen GEN;

        export_proposal_param<GEN>(setup, genName);
        export_fusion_based_t<GEN>(setup, genName);
    }
}

template void export_fusion_based<opengm::python::GmAdder,opengm::Minimizer>();
