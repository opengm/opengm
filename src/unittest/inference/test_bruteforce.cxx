#include <vector>

#include <opengm/unittests/test.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/operations/multiplier.hxx>
#include <opengm/inference/bruteforce.hxx>

template<class T>
struct BruteforceTest
{
    void run()
    {
		typedef T ValueType;
        typedef opengm::GraphicalModel<
             ValueType,									//value type (should be float double or long double)
             opengm::Multiplier							//operator (something like Adder or Multiplier)
        >
        GraphicalModelType;
        // 2 ) factor types:
        // type of the factors and factor functions:
        typedef opengm::ExplicitFunction<ValueType> ExplicitFunctionType ;		//explicit Factorfunction(=dense marray)
        //typedef typename  GraphicalModelType::ImplicitFunctionType ImplicitFunctionType ;
        //typedef typename  GraphicalModelType::SparseFunctionType SparseFunctionType ;
        typedef typename   GraphicalModelType::FunctionIdentifier FunctionIdentifier;
        typedef opengm::Bruteforce<GraphicalModelType,opengm::Minimizer> Bruteforce;
       //function evaluate()
        size_t nos[] = {2,2,2};
        GraphicalModelType gm(opengm::DiscreteSpace<size_t,size_t>(&nos[0], &nos[3]));
        size_t vis1[] = {0,1};
        size_t vis2[] = {2};
        ExplicitFunctionType f1(nos,nos+2);
        ExplicitFunctionType f2(nos,nos+1);
        f1(0,0) = 1.0;
        f1(0,1) = 2.0;
        f1(1,0) = 0.5;
        f1(1,1) = 4.0;
        f2(0)   = 9.0;
        f2(1)   = 7.0;
        FunctionIdentifier i1=gm.addFunction(f1);
        FunctionIdentifier i2 = gm.addFunction(f2);
        gm.addFactor(i1,vis1,vis1+2);
        gm.addFactor(i2,vis2,vis2+1);
        {
            Bruteforce bruteforce(gm);
            bruteforce.infer();
            std::vector<size_t> sol;
            OPENGM_TEST(bruteforce.arg(sol) == opengm::NORMAL);
            OPENGM_TEST(sol[0]==1 && sol[1]==0 && sol[2]==1);
        }
        size_t nos3[] = {3,3,3};
        GraphicalModelType gmExplicit(opengm::DiscreteSpace<size_t,size_t>(nos3,nos3+3));
        {
            Bruteforce bruteforceE(gmExplicit);
            bruteforceE.infer();
            std::vector<size_t> solE;
            OPENGM_TEST(bruteforceE.arg(solE) == opengm::NORMAL);
        }
    }
};

int main() {
   std::cout << "Bruteforce Tests ..." << std::endl;
   {
      {BruteforceTest<float> t; t.run();}
      {BruteforceTest<double> t; t.run();}
   }
   return 0;
}
