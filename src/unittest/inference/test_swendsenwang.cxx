#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/operations/multiplier.hxx>
#include <opengm/operations/integrator.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/operations/maximizer.hxx>
#include <opengm/operations/normalize.hxx>
#include <opengm/inference/swendsenwang.hxx>
#include <opengm/inference/gibbs.hxx>
#include <opengm/inference/messagepassing/messagepassing.hxx>
#include <opengm/unittests/blackboxtester.hxx>
#include <opengm/unittests/blackboxtests/blackboxtestgrid.hxx>
#include <opengm/unittests/blackboxtests/blackboxtestfull.hxx>
#include <opengm/unittests/blackboxtests/blackboxteststar.hxx>

template<class OP, class ACC>
class SwendsenWangTest {
public:
    typedef OP Operation;
    typedef ACC Accumulation;
    typedef opengm::GraphicalModel<double, Operation> GraphicalModel;
    typedef opengm::SwendsenWang<GraphicalModel, Accumulation> SwendsenWang;

    SwendsenWangTest();
    void run();

private:
    double valueEqual() const;
    double valueUnequal() const;
    void buildModel();
    void sample();
    void testMarginals();

    size_t numberOfVariables;
    size_t numberOfStates;
    typename SwendsenWang::Parameter parameter;
    double relativeTolerance;
    GraphicalModel gm;
    typename opengm::ExplicitFunction<double> f2;
    typename GraphicalModel::FunctionIdentifier fid1;
    typename GraphicalModel::FunctionIdentifier fid2;
    opengm::SwendsenWangMarginalVisitor<SwendsenWang> visitor;
};

template<class OP, class ACC>
inline
SwendsenWangTest<OP, ACC>::SwendsenWangTest()
:   numberOfVariables(10),
    numberOfStates(2),
    parameter(1e5, 1e5),
    relativeTolerance(0.3)
{}

#define VALUE_EQUAL(op,acc,ve,vu) \
    template<> inline double SwendsenWangTest<op, acc>::valueEqual() const { return ve; } \
    template<> inline double SwendsenWangTest<op, acc>::valueUnequal() const { return vu; }
VALUE_EQUAL(opengm::Multiplier, opengm::Maximizer, 0.3, 0.2)
VALUE_EQUAL(opengm::Adder, opengm::Minimizer, -std::log(0.3), -std::log(0.2))
#undef VALUE_EQUAL

template<class OP, class ACC>
void SwendsenWangTest<OP, ACC>::buildModel()
{
    std::vector<size_t> numbersOfStates(numberOfVariables, numberOfStates);
    gm = GraphicalModel(opengm::DiscreteSpace<size_t,size_t>(numbersOfStates.begin(), numbersOfStates.end()));
    // add 2nd order function
    size_t shape2[] = {numberOfStates, numberOfStates};
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
void SwendsenWangTest<OP, ACC>::sample()
{
    SwendsenWang sw(gm, parameter);
    visitor = opengm::SwendsenWangMarginalVisitor<SwendsenWang>(gm);
    for(size_t j = 0; j < gm.numberOfVariables(); ++j) {
       visitor.addMarginal(j);
    }
    for(size_t j = 0; j < gm.numberOfVariables() - 1; ++j) {
       size_t variableIndices[] = {j, j + 1};
       visitor.addMarginal(variableIndices, variableIndices + 2);
    }
    sw.infer(visitor);
}

template<class OP, class ACC>
void SwendsenWangTest<OP, ACC>::testMarginals()
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
void SwendsenWangTest<OP, ACC>::run()
{
    buildModel();
    sample();
    testMarginals();
}

// This black box test compares
// - 1st order marginals sampled using Swendsen-Wang
//   with true 1st order marginals computed by BP
// - 2nd order marginals sampled using Swendsen-Wang
//   with 2nd order marginals sampled using Gibbs
void biasedModelTest() {
   typedef opengm::GraphicalModel<double, opengm::Multiplier> GraphicalModel;
   typedef opengm::SwendsenWang<GraphicalModel, opengm::Maximizer> SwendsenWang;
   typedef opengm::Gibbs<GraphicalModel, opengm::Maximizer> Gibbs;
   typedef opengm::BeliefPropagationUpdateRules<GraphicalModel, opengm::Integrator> BpUpdateRules;
   typedef opengm::MessagePassing<GraphicalModel, opengm::Integrator, BpUpdateRules> BeliefPropagation;

   const size_t numberOfSamplingSteps = 1e5;
   const size_t numberOfBurnInSteps = 1e5;
   const double relativeTolerance = 0.3;

   // build graphical model
   size_t numberOfVariables = 10;
   size_t numberOfStates = 2;
   std::vector<size_t> numbersOfStates(numberOfVariables, numberOfStates);
   GraphicalModel gm(opengm::DiscreteSpace<size_t,size_t>(numbersOfStates.begin(), numbersOfStates.end()));

   // add 2nd order function
   size_t shape2[] = {numberOfStates, numberOfStates};
   opengm::ExplicitFunction<double> f2(shape2, shape2 + 2);
   f2(0, 0) = 0.4;
   f2(0, 1) = 0.1;
   f2(1, 0) = 0.1;
   f2(1, 1) = 0.4;
   GraphicalModel::FunctionIdentifier fid2 = gm.addFunction(f2);

   // add 2nd order factors
   for(size_t j=0; j<numberOfVariables-1; ++j) {
      size_t variableIndices[] = {j, j+1};
      gm.addFactor(fid2, variableIndices, variableIndices + 2);
   }

   // add 1st order function
   size_t shape1[] = {numberOfStates};
   opengm::ExplicitFunction<double>  f1(shape1, shape1 + 1);
   f1(0) = 0.2;
   f1(1) = 0.8;
   GraphicalModel::FunctionIdentifier fid1 = gm.addFunction(f1);

   // add 1st order factor
   {
      size_t variableIndices[] = {0};
      gm.addFactor(fid1, variableIndices, variableIndices + 1);
   }

   // compute exact 1st order marginals
   BeliefPropagation bp(gm);
   BeliefPropagation::EmptyVisitorType bpVisitor;
   bp.infer(bpVisitor);

   // sample 2nd order marginals using a Gibbs sampler
   Gibbs gibbs(gm);
   opengm::GibbsMarginalVisitor<Gibbs> gibbsVisitor(gm);
   for(size_t j = 0; j < gm.numberOfVariables() - 1; ++j) {
      size_t variableIndices[] = {j, j + 1};
      gibbsVisitor.addMarginal(variableIndices, variableIndices + 2);
   }
   gibbs.infer(gibbsVisitor);

   // sample 1st and 2nd order marginals using Swendsen Wang
   SwendsenWang::Parameter swParameter(numberOfSamplingSteps,
      numberOfBurnInSteps);
   SwendsenWang sw(gm, swParameter);
   opengm::SwendsenWangMarginalVisitor<SwendsenWang> visitor(gm);
   for(size_t j = 0; j < gm.numberOfVariables(); ++j) {
      visitor.addMarginal(j);
   }
   for(size_t j = 0; j < gm.numberOfVariables() - 1; ++j) {
      size_t variableIndices[] = {j, j + 1};
      visitor.addMarginal(variableIndices, variableIndices + 2);
   }
   sw.infer(visitor);

   // print marginals
   /*
   std::cout << "SW marginals:" << std::endl;
   for(size_t j = 0; j < gm.numberOfVariables(); ++j) {
      std::cout << j << ": ";
      for(size_t k = 0; k < 2; ++k) {
         const double p = static_cast<double>(visitor.marginal(j)(k)) / visitor.numberOfSamples();
         std::cout << p << " ";
      }
      std::cout << std::endl;
   }

   std::cout << "BP marginals:" << std::endl;
   for(size_t j = 0; j < gm.numberOfVariables(); ++j) {
      GraphicalModel::IndependentFactorType trueMarginal;
      bp.marginal(j, trueMarginal);
      // normalize BP marginal
      {
         GraphicalModel::ValueType sum = trueMarginal(0) + trueMarginal(1);
         trueMarginal(0) /= sum;
         trueMarginal(1) /= sum;
      }
      std::cout << j << ": ";
      for(size_t k = 0; k < 2; ++k) {
         std::cout << trueMarginal(k) << " ";
      }
      std::cout << std::endl;
   }
   */

   // test marginals
   for(size_t j = 0; j < gm.numberOfVariables(); ++j) {
      GraphicalModel::IndependentFactorType trueMarginal;
      bp.marginal(j, trueMarginal);
      // normalize BP marginal
      GraphicalModel::ValueType sum = trueMarginal(0) + trueMarginal(1);
      trueMarginal(0) /= sum;
      trueMarginal(1) /= sum;
      for(size_t k = 0; k < 2; ++k) {
         double tolerance = trueMarginal(k) * relativeTolerance;
         const double p = static_cast<double>(visitor.marginal(j)(k)) / visitor.numberOfSamples();
         OPENGM_TEST(p > trueMarginal(k) - tolerance && p < trueMarginal(k) + tolerance);
      }
   }
   for(size_t j = gm.numberOfVariables(); j < visitor.numberOfMarginals(); ++j) {
      for(size_t x = 0; x < 2; ++x)
      for(size_t y = 0; y < 2; ++y) {
         const double p = static_cast<double>(visitor.marginal(j)(x, y)) / visitor.numberOfSamples();
         const double pGibbs = static_cast<double>(gibbsVisitor.marginal(j - gm.numberOfVariables())(x, y)) / gibbsVisitor.numberOfSamples();
         OPENGM_TEST(2.0 * opengm::abs(p - pGibbs) / (p + pGibbs) < relativeTolerance);
      }
   }
}

int main() {
   { SwendsenWangTest<opengm::Multiplier, opengm::Maximizer> test; test.run(); }
   { SwendsenWangTest<opengm::Adder, opengm::Minimizer> test; test.run(); }
   biasedModelTest();
   return 0;
}
