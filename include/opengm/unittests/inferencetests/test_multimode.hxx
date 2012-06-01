#pragma once
#ifndef OPENGM_TEST_TEST_MULTIMODE_HXX
#define OPENGM_TEST_TEST_MULTIMODE_HXX


/// \cond HIDDEN_SYMBOLS
namespace opengm {
   namespace test{
      /// \brief TestMultiMode<INF> 
      /// Build a second order chain-model with several modes and check if the 
      /// algorithm finds an optimal solution. This is important for solvers based on
      /// relaxations in which also fractional optimal solutions exists and a mapping 
      /// to the integral set has to be done. 
      template <class INF>
      class TestMultiMode : public TestBase<INF>
      {
      public:
         TestMultiMode(const size_t, const size_t);
         virtual void test(typename INF::Parameter);
      private:
         const size_t numVar_;
         const size_t numStates_;
      };
      
      /// \brief TestMultiMode Constructor
      /// \param numVar number of variables to add
      /// \param numStates number of states per variables
      template <class INF>
      TestMultiMode<INF>::TestMultiMode(const size_t numVar, const size_t numStates) : numVar_(numVar), numStates_(numStates)
      {;}
        
      /// \brief test<INF> start test with algorithm INF
      /// \param para parameters of algorithm
      template <class INF>
      void TestMultiMode<INF>::test(typename INF::Parameter para) {
         typedef typename INF::GraphicalModelType GraphicalModelType;
         
         std::cout << "  - MultiModeTest ... " << std::flush;

         std::vector<size_t> numberOfLabels(numVar_, numStates_);
         GraphicalModelType gm(numberOfLabels.begin(), numberOfLabels.end());
         
         typename GraphicalModelType::ExplicitFunctionType f(numberOfLabels.begin(), numberOfLabels.begin() + 2, 1);
         for (int i = 0; i < numStates_ - 1; i++) {
            f(i, i + 1) = 0;
         }
         f(numStates_ - 1, 0) = 0;
         
         typename GraphicalModelType::FunctionIdentifier fId = gm.addFunction(f);
        
         size_t var[2];
         for (int i = 0; i < numVar_ - 1; i++) {
            var[0] = i;
            var[1] = i + 1;
            gm.addFactor(fId, var, var + 2);
         }
         
      
         INF inf(gm);
         inf.infer(para);
   
         std::vector<size_t> sol;
         inf.arg(sol);
   
         bool test = true;
         for (size_t i = 0; i < numVar_ - 1; i++) {
            test = test && ( (sol[i]+1) % numStates_ == sol[i+1] );
         }
         OPENGM_ASSERT(test);  
         
         std::cout <<"done!"<<std::endl;
      }
   }
}
/// \endcond
#endif
