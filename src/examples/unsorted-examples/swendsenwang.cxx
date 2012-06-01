#include <vector>
#include <iostream>

#include <opengm/graphicalmodel/space/simplediscretespace.hxx>
#include <opengm/functions/explicit_function.hxx>
#include <opengm/functions/potts.hxx>
#include <opengm/operations/multiplier.hxx>
#include <opengm/operations/maximizer.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/inference/swendsenwang.hxx>

typedef opengm::SimpleDiscreteSpace<> Space;
typedef opengm::ExplicitFunction<double> ExplicitFunction;
typedef opengm::PottsFunction<double> PottsFunction;
typedef opengm::meta::TypeListGenerator<ExplicitFunction, PottsFunction>::type FunctionTypes;
typedef opengm::GraphicalModel<double, opengm::Multiplier, FunctionTypes, Space> GraphicalModel;
typedef opengm::SwendsenWang<GraphicalModel, opengm::Maximizer> SwendsenWang;
typedef opengm::SwendsenWangMarginalVisitor<SwendsenWang> MarginalVisitor;

// build a Markov Chain with 10 binary variables in which
// - the first variable is more likely to be labeled 1 than 0
// - neighboring variables are more likely to have similar labels than dissimilar
void buildGraphicalModel(GraphicalModel& gm) {
   const size_t numberOfVariables = 10;
   const size_t numberOfLabels = 2;
   Space space(numberOfVariables, numberOfLabels);
   gm = GraphicalModel(space);

   // add 1st order function
   GraphicalModel::FunctionIdentifier fid1;
   {
      ExplicitFunction f(&numberOfLabels, &numberOfLabels + 1);
      f(0) = 0.2;
      f(1) = 0.8;
      fid1 = gm.addFunction(f);
   }

   // add 2nd order function
   GraphicalModel::FunctionIdentifier fid2;
   {
      const double probEqual = 0.7;
      const double probUnequal = 0.3;
      PottsFunction f(2, 2, probEqual, probUnequal);
      fid2 = gm.addFunction(f);
   }

   // add 1st order factor (at first variable)
   {
      size_t variableIndices[] = {0};
      gm.addFactor(fid1, variableIndices, variableIndices + 1);
   }

   // add 2nd order factors
   for(size_t j = 0; j < numberOfVariables - 1; ++j) {
      size_t variableIndices[] = {j, j+1};
      gm.addFactor(fid2, variableIndices, variableIndices + 2);
   }
}

// use SwendsenWang sampling for the purpose of finding the most probable labeling
void swendsenWangSamplingForOptimization(const GraphicalModel& gm) {
   const size_t numberOfSamplingSteps = 1e4;
   const size_t numberOfBurnInSteps = 1e4;
   SwendsenWang::Parameter parameter(numberOfSamplingSteps, numberOfBurnInSteps);
   SwendsenWang swendsenWang(gm, parameter);
   swendsenWang.infer();
   std::vector<size_t> argmax;
   swendsenWang.arg(argmax);
   
   std::cout << "most probable labeling sampled: (";
   for(size_t j = 0; j < argmax.size(); ++j) {
      std::cout << argmax[j] << ", ";
   }
   std::cout << "\b\b)" << std::endl;
}

// use SwendsenWang sampling to estimate marginals
void swendsenWangSamplingForMarginalEstimation(const GraphicalModel& gm) {
   MarginalVisitor visitor(gm);

   // extend the visitor to sample first order marginals
   for(size_t j = 0; j < gm.numberOfVariables(); ++j) {
      visitor.addMarginal(j);
   }

   // extend the visitor to sample certain second order marginals
   for(size_t j = 0; j < gm.numberOfVariables() - 1; ++j) {
      size_t variableIndices[] = {j, j + 1};
      visitor.addMarginal(variableIndices, variableIndices + 2);
   }

   // sample
   SwendsenWang swendsenWang(gm);
   swendsenWang.infer(visitor);

   // output sampled first order marginals
   std::cout << "sampled first order marginals:" << std::endl;
   for(size_t j = 0; j < gm.numberOfVariables(); ++j) {
      std::cout << "x" << j << ": ";
      for(size_t k = 0; k < 2; ++k) {
         const double p = static_cast<double>(visitor.marginal(j)(k)) / visitor.numberOfSamples();
         std::cout << p << ' ';
      }
      std::cout << std::endl;
   }

   // output sampled second order marginals
   std::cout << "sampled second order marginals:" << std::endl;
   for(size_t j = gm.numberOfVariables(); j < visitor.numberOfMarginals(); ++j) {
      std::cout << "x" << visitor.marginal(j).variableIndex(0) 
         << ", x" << visitor.marginal(j).variableIndex(1) 
         << ": ";
      for(size_t x = 0; x < 2; ++x)
         for(size_t y = 0; y < 2; ++y) {
            const double p = static_cast<double>(visitor.marginal(j)(x, y)) / visitor.numberOfSamples();
            std::cout << p << ' ';
         }
         std::cout << std::endl;
   }
}

int main() {
   GraphicalModel gm;

   buildGraphicalModel(gm);
   swendsenWangSamplingForOptimization(gm);
   swendsenWangSamplingForMarginalEstimation(gm);

   return 0;
}
