#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/graphicalmodel/space/simplediscretespace.hxx>
#include <opengm/functions/potts.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/inference/messagepassing/messagepassing.hxx>

using namespace std; // 'using' is used only in example code
using namespace opengm;

int main() {
   // construct a label space with numberOfVariables many variables,
   // each having numberOfLabels many labels
   const size_t numberOfVariables = 40; 
   const size_t numberOfLabels = 5;
   typedef SimpleDiscreteSpace<size_t, size_t> Space;
   Space space(numberOfVariables, numberOfLabels);

   // construct a graphical model with 
   // - addition as the operation (template parameter Adder)
   // - support for Potts functions (template parameter PottsFunction<double>)
   typedef OPENGM_TYPELIST_2(ExplicitFunction<double>, PottsFunction<double>) FunctionTypelist;
   typedef GraphicalModel<double, Adder, FunctionTypelist, Space> Model;
   Model gm(space);
   
   // for each variable, add one 1st order functions and one 1st order factor
   for(size_t v = 0; v < numberOfVariables; ++v) {
      const size_t shape[] = {numberOfLabels};
      ExplicitFunction<double> f(shape, shape + 1);
      for(size_t s = 0; s < numberOfLabels; ++s) {
         f(s) = static_cast<double>(rand()) / RAND_MAX;
      }
      Model::FunctionIdentifier fid = gm.addFunction(f);

      size_t variableIndices[] = {v};
      gm.addFactor(fid, variableIndices, variableIndices + 1);
   }

   // add one (!) 2nd order Potts function
   PottsFunction<double> f(numberOfLabels, numberOfLabels, 0.0, 0.3);
   Model::FunctionIdentifier fid = gm.addFunction(f);

   // for each pair of consecutive variables,
   // add one factor that refers to the Potts function 
   for(size_t v = 0; v < numberOfVariables - 1; ++v) {
      size_t variableIndices[] = {v, v + 1};
      gm.addFactor(fid, variableIndices, variableIndices + 2);
   }    

   // set up the optimizer (loopy belief propagation)
   typedef BeliefPropagationUpdateRules<Model, Minimizer> UpdateRules;
   typedef MessagePassing<Model, Minimizer, UpdateRules, MaxDistance> BeliefPropagation;
   const size_t maxNumberOfIterations = numberOfVariables * 2;
   const double convergenceBound = 1e-7;
   const double damping = 0.0;
   BeliefPropagation::Parameter parameter(maxNumberOfIterations, convergenceBound, damping);
   BeliefPropagation bp(gm, parameter);

   // optimize (approximately)
   BeliefPropagation::VerboseVisitorType visitor;
   bp.infer(visitor);

   // obtain the (approximate) argmin
   vector<size_t> labeling(numberOfVariables);
   bp.arg(labeling);
}
