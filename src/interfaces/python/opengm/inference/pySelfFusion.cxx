//#define GraphicalModelDecomposition DualDecompostionSubgradientInference_GraphicalModelDecomposition

#include <boost/python.hpp>
#include <string>
#include "inf_def_visitor.hxx"

#include <opengm/inference/messagepassing/messagepassing.hxx>
#include <opengm/inference/self_fusion.hxx>

#include <param/self_fusion_param.hxx>


#include <opengm/python/opengmpython.hxx>
#include <opengm/python/converter.hxx>
#include <opengm/python/numpyview.hxx>
#include <opengm/python/pythonfunction.hxx>

#ifdef WITH_TRWS
#include <opengm/inference/external/trws.hxx>
#endif //WITH_TRWS

using namespace boost::python;


template<class GM,class ACC>
void export_self_fusion(){

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



    // BP
    {
        typedef opengm::BeliefPropagationUpdateRules<GM,ACC> UpdateRulesType;
        typedef opengm::MessagePassing<GM, ACC,UpdateRulesType, opengm::MaxDistance> InfType;
        typedef opengm::SelfFusion<InfType> SelfFusionInf;

        enum_<typename SelfFusionInf::FusionSolver> ("_SelfFusionBp_FusionSolver")
        .value("qpbo",     SelfFusionInf::QpboFusion)
        .value("cplex",   SelfFusionInf::CplexFusion)
        .value("lf",    SelfFusionInf::LazyFlipperFusion)
        ;


        // set up hyper parameter name for this template
        setup.isDefault=true;

        setup.hyperParameters= StringVector(1,std::string("bp"));
        // export parameter
        //exportInfParam<SelfFusionInf>("_SubParameter_SelfFusion_Bp");
        exportInfParam<SelfFusionInf>("_SelfFusion_Bp");
        // export inferences
        class_< SelfFusionInf>("_SelfFusion_Bp",init<const GM & >())  
        .def(InfSuite<SelfFusionInf,false>(std::string("SelfFusion"),setup))
        ;
    }

    // Trws
    {
        typedef opengm::TrbpUpdateRules<GM,ACC> UpdateRulesType;
        typedef opengm::MessagePassing<GM, ACC,UpdateRulesType, opengm::MaxDistance> InfType;
        typedef opengm::SelfFusion<InfType> SelfFusionInf;

        enum_<typename SelfFusionInf::FusionSolver> ("_SelfFusionTrwBp_FusionSolver")
        .value("qpbo",     SelfFusionInf::QpboFusion)
        .value("cplex",   SelfFusionInf::CplexFusion)
        .value("lf",    SelfFusionInf::LazyFlipperFusion)
        ;


        // set up hyper parameter name for this template
        setup.isDefault=false;

        setup.hyperParameters= StringVector(1,std::string("trwbp"));
        // export parameter
        //exportInfParam<SelfFusionInf>("_SubParameter_SelfFusion_TrwBp");
        exportInfParam<SelfFusionInf>("_SelfFusion_TrwBp");
        // export inferences
        class_< SelfFusionInf>("_SelfFusion_TrwBp",init<const GM & >())  
        .def(InfSuite<SelfFusionInf,false>(std::string("SelfFusion"),setup))
        ;
    }

    #ifdef WITH_TRWS
    // trws
    {
        typedef opengm::external::TRWS<GM> InfType;
        typedef opengm::SelfFusion<InfType> SelfFusionInf;

        enum_<typename SelfFusionInf::FusionSolver> ("_SelfFusionTrws_FusionSolver")
        .value("qpbo",     SelfFusionInf::QpboFusion)
        .value("cplex",   SelfFusionInf::CplexFusion)
        .value("lf",    SelfFusionInf::LazyFlipperFusion)
        ;


        // set up hyper parameter name for this template
        setup.isDefault=false;

        setup.hyperParameters= StringVector(1,std::string("trws"));
        // export parameter
        //exportInfParam<SelfFusionInf>("_SubParameter_SelfFusion_Trws");
        exportInfParam<SelfFusionInf>("_SelfFusion_Trws");
        // export inferences
        class_< SelfFusionInf>("_SelfFusion_Trws",init<const GM & >())  
        .def(InfSuite<SelfFusionInf,false>(std::string("SelfFusion"),setup))
        ;
    }
    #endif //WITH_TRWS
   
}

template void export_self_fusion<opengm::python::GmAdder,opengm::Minimizer>();
