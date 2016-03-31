#include <vector>

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/operations/maximizer.hxx>
#include <opengm/inference/gibbs.hxx>
#include <opengm/inference/icm.hxx>
#include <opengm/inference/astar.hxx>

using namespace std; // 'using' is used only in example code
using namespace opengm;

int main() {  
   typedef GraphicalModel<double, Adder > Model;
   Model gm;

   // Let us assume we have added some variables
   // factors and functions to the graphical model gm
   // ( see other exampels for building a model )

   // ICM
   {
      // typedefs to a ICM minimizer and maximizer
      typedef opengm::ICM<Model,Minimizer> OptimizerMinimizerType;
      typedef opengm::ICM<Model,Maximizer> OptimizerMaximizerType;
      typedef OptimizerMinimizerType::Parameter OptimizerMinimizerParameterType;
      typedef OptimizerMaximizerType::Parameter OptimizerMaximizerParameterType;
      
      // construct solver parameters (all parameters have default values)
      //
      vector<Model::LabelType> startingPoint(gm.numberOfVariables());
      //fill starting point
      // ....
      // assume starting point is filled with labels
      OptimizerMinimizerParameterType minimizerParameter(
         OptimizerMinimizerType::SINGLE_VARIABLE,  // flip a single variable (FACTOR for flip all var. a factor depends on)
         startingPoint
      );
      // without starting point
      OptimizerMaximizerParameterType maximizerParameter(
         OptimizerMaximizerType::FACTOR,  // flip a single variable (FACTOR for flip all var. a factor depends on)
         startingPoint
      );
      
      // construct optimizers ( minimizer and maximizer )
      OptimizerMinimizerType optimizerMinimizer(gm,minimizerParameter);
      OptimizerMaximizerType optimizerMaximizer(gm,maximizerParameter);
      
      // optimize the models ( minimizer and maximize )
      optimizerMinimizer.infer();
      optimizerMaximizer.infer();
      
      // get the argmin / argmax 
      vector<Model::LabelType> argmin,argmax;
      optimizerMinimizer.arg(argmin);
      optimizerMaximizer.arg(argmax);
   }

   return 0;
}
