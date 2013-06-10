#include <vector>
#include <set>
#include <functional>

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/operations/multiplier.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/operations/maximizer.hxx>
#include <opengm/inference/gibbs.hxx>
#include <opengm/unittests/blackboxtester.hxx>
#include <opengm/unittests/blackboxtests/blackboxtestgrid.hxx>
#include <opengm/unittests/blackboxtests/blackboxtestfull.hxx>
#include <opengm/unittests/blackboxtests/blackboxteststar.hxx>

template<class OP, class ACC>
class GibbsTest {
public:
    typedef OP Operation;
    typedef ACC Accumulation;
    typedef opengm::GraphicalModel<double, Operation> GraphicalModel;
    typedef opengm::Gibbs<GraphicalModel, Accumulation> Gibbs;

    GibbsTest();
    void run();

private:
    double valueEqual() const;
    double valueUnequal() const;
    void buildModel();
    void sample();
    void testMarginals();

    size_t numberOfVariables;
    size_t numberOfLabels;
    typename Gibbs::Parameter parameter;
    double relativeTolerance;

    GraphicalModel gm;
    opengm::ExplicitFunction<typename GraphicalModel::ValueType> f2;
    typename GraphicalModel::FunctionIdentifier fid1;
    typename GraphicalModel::FunctionIdentifier fid2;
    opengm::GibbsMarginalVisitor<Gibbs> visitor;
};

template<class OP, class ACC>
inline
GibbsTest<OP, ACC>::GibbsTest()
:   numberOfVariables(10),
    numberOfLabels(2),
    parameter(),
    relativeTolerance(0.3)
{}

#define VALUE_EQUAL(op,acc,ve,vu) \
    template<> inline double GibbsTest<op, acc>::valueEqual() const { return ve; } \
    template<> inline double GibbsTest<op, acc>::valueUnequal() const { return vu; }

VALUE_EQUAL(opengm::Multiplier, opengm::Maximizer, 0.3, 0.2)
VALUE_EQUAL(opengm::Adder, opengm::Minimizer, -std::log(0.3), -std::log(0.2))

#undef VALUE_EQUAL

template<class OP, class ACC>
void GibbsTest<OP, ACC>::buildModel()
{
    std::vector<size_t> numbersOfStates(numberOfVariables, numberOfLabels);
    gm = GraphicalModel(opengm::DiscreteSpace<size_t,size_t>(numbersOfStates.begin(), numbersOfStates.end()));

    // add 2nd order function
    size_t shape2[] = {numberOfLabels, numberOfLabels};
    f2 = opengm::ExplicitFunction<typename GraphicalModel::ValueType>(shape2, shape2 + 2);
    f2(0, 0) = valueEqual();
    f2(0, 1) = valueUnequal();
    f2(1, 0) = valueUnequal();
    f2(1, 1) = valueEqual();
    fid2 = gm.addFunction(f2);

    // add 2nd order factors
    for(size_t j=0; j<numberOfVariables-1; ++j) {
       size_t variableIndices[] = {j, j+1};
       gm.addFactor(fid2, variableIndices, variableIndices + 2);
    }
}

template<class OP, class ACC>
void GibbsTest<OP, ACC>::sample()
{
    Gibbs gibbs(gm, parameter);
    visitor = opengm::GibbsMarginalVisitor<Gibbs>(gm);
    for(size_t j = 0; j < gm.numberOfVariables(); ++j) {
       visitor.addMarginal(j);
    }
    for(size_t j = 0; j < gm.numberOfVariables() - 1; ++j) {
       size_t variableIndices[] = {j, j + 1};
       visitor.addMarginal(variableIndices, variableIndices + 2);
    }
    gibbs.infer(visitor);
}

template<class OP, class ACC>
void GibbsTest<OP, ACC>::testMarginals()
{
    for(size_t j = 0; j < gm.numberOfVariables(); ++j) {
       double tolerance = 0.5 * relativeTolerance;
       // std::cout << j << ": ";
       for(size_t k = 0; k < 2; ++k) {
          const double p = static_cast<double>(visitor.marginal(j)(k)) / visitor.numberOfSamples();
          // std::cout << p << "/0.5 ";
          OPENGM_TEST(p > 0.5 - tolerance && p < 0.5 + tolerance);
       }
       // std::cout << std::endl;
    }
    for(size_t j = gm.numberOfVariables(); j < visitor.numberOfMarginals(); ++j) {
       // std::cout << j << ": ";
       for(size_t x = 0; x < 2; ++x)
       for(size_t y = 0; y < 2; ++y) {
          const double p = static_cast<double>(visitor.marginal(j)(x, y)) / visitor.numberOfSamples();
          double pTrue;
          if(x == y) {
              pTrue = 0.3;
          }
          else {
              pTrue = 0.2;
          }
          // std::cout << p << "/" << pTrue << " ";
          double tolerance = pTrue * relativeTolerance;
          OPENGM_TEST(p > pTrue - tolerance && p < pTrue + tolerance);
       }
       // std::cout << std::endl;
    }
}

template<class OP, class ACC>
void GibbsTest<OP, ACC>::run()
{
    buildModel();
    sample();
    testMarginals();
}

void standardTests() {
    typedef opengm::GraphicalModel<double, opengm::Adder> SumGmType;
    typedef opengm::GraphicalModel<double, opengm::Multiplier > ProdGmType;
    typedef opengm::BlackBoxTestGrid<SumGmType> SumGridTest;
    typedef opengm::BlackBoxTestFull<SumGmType> SumFullTest;
    typedef opengm::BlackBoxTestStar<SumGmType> SumStarTest;
    typedef opengm::BlackBoxTestGrid<ProdGmType> ProdGridTest;
    typedef opengm::BlackBoxTestFull<ProdGmType> ProdFullTest;
    typedef opengm::BlackBoxTestStar<ProdGmType> ProdStarTest;

    opengm::InferenceBlackBoxTester<SumGmType> sumTester;
    sumTester.addTest(new SumGridTest(3, 3, 2, false, true, SumGridTest::POTTS, opengm::PASS, 1));
    sumTester.addTest(new SumFullTest(4,    3, false,    3, SumFullTest::POTTS, opengm::PASS, 1));

    opengm::InferenceBlackBoxTester<ProdGmType> prodTester;
    prodTester.addTest(new ProdGridTest(3, 3, 2, false, true, ProdGridTest::RANDOM, opengm::PASS, 1));
    prodTester.addTest(new ProdFullTest(4,    3, false,    3, ProdFullTest::RANDOM, opengm::PASS, 1));

    std::cout << "Gibbs Tests ..." << std::endl;
    {
       std::cout << "  * Minimization/Adder..." << std::endl;
       typedef opengm::Gibbs<SumGmType, opengm::Minimizer> Gibbs;
       Gibbs::Parameter para;
       sumTester.test<Gibbs>(para);
       std::cout << " OK!"<<std::endl;
    }
    {
       std::cout << "  * Maximization/Multiplier..." << std::endl;
       typedef opengm::Gibbs<ProdGmType, opengm::Maximizer> Gibbs;
       Gibbs::Parameter para;
       prodTester.test<Gibbs>(para);
       std::cout << " OK!"<<std::endl;
    }
}

int main() {
    { GibbsTest<opengm::Multiplier, opengm::Maximizer> test; test.run(); }
    { GibbsTest<opengm::Adder, opengm::Minimizer> test; test.run(); }

    standardTests();
    return 0;
}
