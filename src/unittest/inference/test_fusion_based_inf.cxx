#include <stdlib.h>
#include <vector>
#include <set>
#include <functional>

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/operations/multiplier.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/operations/maximizer.hxx>
#include <opengm/graphicalmodel/space/simplediscretespace.hxx>
#include <opengm/functions/potts.hxx>
#include <opengm/inference/fusion_based_inf.hxx>

#include <opengm/unittests/blackboxtester.hxx>
#include <opengm/unittests/blackboxtests/blackboxtestgrid.hxx>
#include <opengm/unittests/blackboxtests/blackboxtestfull.hxx>
#include <opengm/unittests/blackboxtests/blackboxteststar.hxx>




void test_ae_generator()
{
    typedef opengm::SimpleDiscreteSpace<size_t, size_t> Space;
    Space space(100, 5);
    typedef opengm::GraphicalModel < double, opengm::Adder,
            OPENGM_TYPELIST_2(opengm::ExplicitFunction<double> , opengm::PottsFunction<double> ) , Space > GmType;
    GmType gm(space);


    {
        typedef opengm::proposal_gen::AlphaExpansionGen<GmType, opengm::Minimizer> AeGen;
        AeGen::Parameter param;
        AeGen aeGen(gm, param);
        
        std::vector<size_t> current(gm.numberOfVariables(),0);
        std::vector<size_t> proposal(gm.numberOfVariables(),0);


        aeGen.getProposal(current,proposal);
        OPENGM_ASSERT_OP(proposal[0],==,0);
       
        aeGen.getProposal(current,proposal);
        OPENGM_ASSERT_OP(proposal[0],==,1);

        aeGen.getProposal(current,proposal);
        OPENGM_ASSERT_OP(proposal[0],==,2);

        aeGen.getProposal(current,proposal);
        OPENGM_ASSERT_OP(proposal[0],==,3);

        aeGen.getProposal(current,proposal);
        OPENGM_ASSERT_OP(proposal[0],==,4);

        aeGen.getProposal(current,proposal);
        OPENGM_ASSERT_OP(proposal[0],==,0);

        aeGen.getProposal(current,proposal);
        OPENGM_ASSERT_OP(proposal[0],==,1);

        aeGen.getProposal(current,proposal);
        OPENGM_ASSERT_OP(proposal[0],==,2);

        aeGen.getProposal(current,proposal);
        OPENGM_ASSERT_OP(proposal[0],==,3);

        aeGen.getProposal(current,proposal);
        OPENGM_ASSERT_OP(proposal[0],==,4);

        aeGen.getProposal(current,proposal);

    }
}   

void test_ab_swap_generator()
{
    typedef opengm::SimpleDiscreteSpace<size_t, size_t> Space;
    Space space(100, 4);
    typedef opengm::GraphicalModel < double, opengm::Adder,
            OPENGM_TYPELIST_2(opengm::ExplicitFunction<double> , opengm::PottsFunction<double> ) , Space > GmType;
    GmType gm(space);


    {
        typedef opengm::proposal_gen::AlphaBetaSwapGen<GmType, opengm::Minimizer> Gen;
        Gen::Parameter param;
        Gen gen(gm, param);
        
        std::vector<size_t> current(gm.numberOfVariables(),0);
        std::vector<size_t> proposal(gm.numberOfVariables(),0);
 
        gen.getProposal(current,proposal);
        OPENGM_ASSERT_OP(gen.currentAlpha(),==,1);
        OPENGM_ASSERT_OP(gen.currentBeta(),==,0);

        gen.getProposal(current,proposal);
        OPENGM_ASSERT_OP(gen.currentAlpha(),==,2);
        OPENGM_ASSERT_OP(gen.currentBeta(),==,0);

        gen.getProposal(current,proposal);
        OPENGM_ASSERT_OP(gen.currentAlpha(),==,3);
        OPENGM_ASSERT_OP(gen.currentBeta(),==,0);

        gen.getProposal(current,proposal);
        OPENGM_ASSERT_OP(gen.currentAlpha(),==,0);
        OPENGM_ASSERT_OP(gen.currentBeta(),==,1);

        gen.getProposal(current,proposal);        
        OPENGM_ASSERT_OP(gen.currentAlpha(),==,2);
        OPENGM_ASSERT_OP(gen.currentBeta(),==,1);

        gen.getProposal(current,proposal);
        OPENGM_ASSERT_OP(gen.currentAlpha(),==,3);
        OPENGM_ASSERT_OP(gen.currentBeta(),==,1);

        gen.getProposal(current,proposal);
        OPENGM_ASSERT_OP(gen.currentAlpha(),==,0);
        OPENGM_ASSERT_OP(gen.currentBeta(),==,2);

        gen.getProposal(current,proposal);        
        OPENGM_ASSERT_OP(gen.currentAlpha(),==,1);
        OPENGM_ASSERT_OP(gen.currentBeta(),==,2);

        gen.getProposal(current,proposal);
        OPENGM_ASSERT_OP(gen.currentAlpha(),==,3);
        OPENGM_ASSERT_OP(gen.currentBeta(),==,2);

        gen.getProposal(current,proposal);
        OPENGM_ASSERT_OP(gen.currentAlpha(),==,0);
        OPENGM_ASSERT_OP(gen.currentBeta(),==,3);

        gen.getProposal(current,proposal);        
        OPENGM_ASSERT_OP(gen.currentAlpha(),==,1);
        OPENGM_ASSERT_OP(gen.currentBeta(),==,3);

        gen.getProposal(current,proposal);
        OPENGM_ASSERT_OP(gen.currentAlpha(),==,2);
        OPENGM_ASSERT_OP(gen.currentBeta(),==,3);

        gen.getProposal(current,proposal);
        std::cout <<gen.currentAlpha()<<gen.currentBeta()<<std::endl;
        OPENGM_ASSERT_OP(gen.currentAlpha(),==,1);
        OPENGM_ASSERT_OP(gen.currentBeta(),==,0);

        gen.getProposal(current,proposal);
        OPENGM_ASSERT_OP(gen.currentAlpha(),==,2);
        OPENGM_ASSERT_OP(gen.currentBeta(),==,0);

        gen.getProposal(current,proposal);
    }
}   


int main()
{
    typedef opengm::GraphicalModel<double, opengm::Adder> SumGmType;


    typedef opengm::BlackBoxTestGrid<SumGmType> SumGridTest;
    typedef opengm::BlackBoxTestFull<SumGmType> SumFullTest;
    typedef opengm::BlackBoxTestStar<SumGmType> SumStarTest;

    opengm::InferenceBlackBoxTester<SumGmType> sumTester;
    sumTester.addTest(new SumGridTest(10, 10, 10, false, true, SumGridTest::RANDOM, opengm::PASS, 1));
    //sumTester.addTest(new SumFullTest(6,    3, false,    3, SumFullTest::POTTS, opengm::PASS, 1));



    test_ae_generator();
    test_ab_swap_generator();


    typedef opengm::proposal_gen::AlphaBetaSwapGen<SumGmType, opengm::Minimizer> ABGen;
    typedef opengm::proposal_gen::AlphaBetaSwapGen<SumGmType, opengm::Minimizer> AEGen;


    std::cout << "FusionBasedInf AlphaExpansion Tests ..." << std::endl;
    {
       std::cout << "  * Minimization/Adder  ..." << std::endl;
       typedef opengm::FusionBasedInf<SumGmType, ABGen> InfType;
       InfType::Parameter para;
       sumTester.test<InfType>(para);
       std::cout << " OK!"<<std::endl;
    }
    std::cout << "FusionBasedInf AlphaBetaSwap Tests ..." << std::endl;
    {
       std::cout << "  * Minimization/Adder  ..." << std::endl;
       typedef opengm::FusionBasedInf<SumGmType, AEGen> InfType;
       InfType::Parameter para;
       sumTester.test<InfType>(para);
       std::cout << " OK!"<<std::endl;
    }


}



