#include <vector>

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/graphicalmodel/space/discretespace.hxx>
#include <opengm/graphicalmodel/space/simplediscretespace.hxx>
#include <opengm/graphicalmodel/space/grid_space.hxx>

using namespace std; // 'using' is used only in example code

int main() {
   typedef float ValueType;
   typedef opengm::UInt32Type IndexType;
   typedef opengm::UInt8Type LabelType;
   typedef opengm::Adder OperationType;
   typedef opengm::ExplicitFunction<ValueType> FunctionType;

   // dense space where all variables can have
   // a different number of variables
   {
      typedef opengm::DiscreteSpace<IndexType,LabelType> SpaceType;
      typedef opengm::GraphicalModel<
         ValueType,
         OperationType,
         FunctionType,
         SpaceType
      > GraphicalModelType;
      const LabelType numbersOfLabels[] = {2,4,6,4,3};

      // graphical model with 5 variables with 2 4 6 4 and 3 labels
      GraphicalModelType gm(SpaceType(numbersOfLabels, numbersOfLabels + 4));
   }

   // simple space where all variables have
   // the same number of labels
   {
      typedef opengm::SimpleDiscreteSpace<IndexType,LabelType> SpaceType;
      typedef opengm::GraphicalModel<
         ValueType,
         OperationType,
         FunctionType,
         SpaceType
      > GraphicalModelType;

      // graphical model with 5 variables, each having 2 labels 
      const IndexType numberOfVariables = 5;
      const LabelType numberOfLabels = 4;
      GraphicalModelType gm(SpaceType(numberOfVariables, numberOfLabels));
   }

   return 0;
}