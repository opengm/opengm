#include <iostream>

#include <opengm/opengm.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/inference/icm.hxx>

using namespace std; // 'using' is used only in example code

int main(int argc, char** argv) {
   // construct a graphical model with 
   // - 5 variables with {5, 5, 2, 2, 10} labels
   // - addition as the operation (template parameter Adder)
   typedef opengm::GraphicalModel<float, opengm::Adder> Model;  
   size_t numbersOfLabels[] = {5, 5, 2, 2, 10};
   Model gm(opengm::DiscreteSpace<>(numbersOfLabels, numbersOfLabels + 5));

   // add 1st order functions and factors to the model
   typedef opengm::ExplicitFunction<float> ExplicitFunction;
   typedef Model::FunctionIdentifier FunctionIdentifier;
   for(size_t variable = 0; variable < gm.numberOfVariables(); ++variable) {
      // construct 1st order function
      const size_t shape[] = {gm.numberOfLabels(variable)};
      ExplicitFunction f(shape, shape + 1);
      for(size_t state = 0; state < gm.numberOfLabels(variable); ++state) {
         f(state) = float(rand()) / RAND_MAX; // random toy data
      }
      // add function
      FunctionIdentifier id = gm.addFunction(f);
      // add factor
      size_t variableIndex[] = {variable};
      gm.addFactor(id, variableIndex, variableIndex + 1);
   }

   // add 3rd order functions and factors to the model
   for(size_t variable1 = 0; variable1 < gm.numberOfVariables(); ++variable1)
   for(size_t variable2 = variable1 + 1; variable2 < gm.numberOfVariables(); ++variable2)
   for(size_t variable3 = variable2 + 1; variable3 < gm.numberOfVariables(); ++variable3) {
      const size_t shape[] = {
         gm.numberOfLabels(variable1),
         gm.numberOfLabels(variable2),
         gm.numberOfLabels(variable3)
      };
      // construct 3rd order function
      ExplicitFunction f(shape, shape + 3);
      for(size_t state1 = 0; state1 < gm.numberOfLabels(variable1); ++state1)
      for(size_t state2 = 0; state2 < gm.numberOfLabels(variable2); ++state2)
      for(size_t state3 = 0; state3 < gm.numberOfLabels(variable3); ++state3) {         
         f(state1, state2, state3) = float(rand()) / RAND_MAX; // random toy data
      }
      FunctionIdentifier id = gm.addFunction(f);
      // sequences of variable indices need to be (and in this case are) sorted
      size_t variableIndexSequence[] = {variable1, variable2, variable3};
      gm.addFactor(id, variableIndexSequence, variableIndexSequence + 3);
   }

   // set up the optimizer (ICM)
   typedef opengm::ICM<Model, opengm::Minimizer> IcmType;
   typedef IcmType::VerboseVisitorType VerboseVisitorType;
   IcmType icm(gm);

   // obtain the (approximate) argmin
   VerboseVisitorType verboseVisitor;
   icm.infer(verboseVisitor);

   // output the (approximate) argmin
   vector<size_t> argmin;
   icm.arg(argmin);
   for(size_t variable = 0; variable < gm.numberOfVariables(); ++variable) {
      cout << "x" << variable << "=" << argmin[variable] << "\n";
   }
}
