/***********************************************************************
 * Tutorial:     Saving and Loading Models 
 * Author:       Joerg Hendrik Kappes
 * Date:         07.07.2014
 * Dependencies: HDF5
 *
 * Description:
 * ------------
 * This Example construct a model store it and load it.
 *
 ************************************************************************/

#include <iostream>

#include <opengm/opengm.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/graphicalmodel/graphicalmodel_hdf5.hxx>
#include <opengm/operations/adder.hxx>

int main(int argc, char** argv) {
   //*******************
   //** Typedefs
   //*******************
   typedef double                                                               ValueType;          // type used for values
   typedef size_t                                                               IndexType;          // type used for indexing nodes and factors (default : size_t)
   typedef size_t                                                               LabelType;          // type used for labels (default : size_t)
   typedef opengm::Adder                                                        OpType;             // operation used to combine terms
   typedef opengm::ExplicitFunction<ValueType,IndexType,LabelType>              ExplicitFunction;   // shortcut for explicite function
   typedef opengm::meta::TypeListGenerator<ExplicitFunction>::type              FunctionTypeList;   // list of all function the model cal use (this trick avoids virtual methods) - here only one
   typedef opengm::DiscreteSpace<IndexType, LabelType>                          SpaceType;          // type used to define the feasible statespace
   typedef opengm::GraphicalModel<ValueType,OpType,FunctionTypeList,SpaceType>  Model;              // type of the model
   typedef Model::FunctionIdentifier                                            FunctionIdentifier; // type of the function identifier


   //*******************
   //** Code
   //*******************

   std::cout << "Start building the model ... "<<std::flush;
   // Build empty Model
   LabelType numbersOfLabels[] = {5, 5, 2, 2, 10};
   Model gm(SpaceType(numbersOfLabels, numbersOfLabels + 5));

   // Add 1st order functions and factors to the model
   for(IndexType variable = 0; variable < gm.numberOfVariables(); ++variable) {
      // construct 1st order function
      const LabelType shape[] = {gm.numberOfLabels(variable)};
      ExplicitFunction f(shape, shape + 1);
      for(LabelType state = 0; state < gm.numberOfLabels(variable); ++state) { 
         f(&state) = ValueType(rand()) / RAND_MAX; // general function interface
         //f(state) = ValueType(rand()) / RAND_MAX; // only works for ExpliciteFunction
      }
      // add function
      FunctionIdentifier id = gm.addFunction(f);
      // add factor
      IndexType variableIndex[] = {variable};
      gm.addFactor(id, variableIndex, variableIndex + 1);
   }
   // add 2nd order function and factors to the model
   {
      IndexType vars[]  = {0,1}; 
      LabelType shape[] = {numbersOfLabels[0],numbersOfLabels[1]};
      LabelType state[] = {0,0};
      ExplicitFunction f(shape, shape + 2);
      for(state[0] = 0; state[0] < gm.numberOfLabels(0); ++state[0]){
         for(state[1] = 0; state[1] < gm.numberOfLabels(1); ++state[1]) {
            f(state)              = ValueType(rand()) / RAND_MAX; // general function interface
            //f(state[0], state[1]) = ValueType(rand()) / RAND_MAX; // only works for ExpliciteFunction
         }
      }
      // add function
      FunctionIdentifier fid = gm.addFunction(f);
      // add factor
      gm.addFactor(fid, vars, vars + 2);
   }

   // add 3rd order function and factors to the model
   {
      IndexType vars[]  = {2,3,4}; 
      LabelType shape[] = {numbersOfLabels[2],numbersOfLabels[3],numbersOfLabels[4]};
      LabelType state[] = {0,0,0};
      ExplicitFunction f(shape, shape + 3);
      for(state[0] = 0; state[0] < gm.numberOfLabels(2); ++state[0]){
         for(state[1] = 0; state[1] < gm.numberOfLabels(3); ++state[1]) { 
            for(state[2] = 0; state[2] < gm.numberOfLabels(4); ++state[2]) {
               f(state)              = ValueType(rand()) / RAND_MAX; // general function interface
               //f(state[0], state[1], state[2]) = ValueType(rand()) / RAND_MAX; // only works for ExpliciteFunction
            }
         }
      }
      // add function
      FunctionIdentifier fid = gm.addFunction(f);
      // add factor
      gm.addFactor(fid, vars, vars + 3);
   }
   std::cout << "done"<<std::endl;

   std::cout << "Store model ... "<<std::flush;
   //Store the model into HDF5 file
   opengm::hdf5::save(gm,"./model.h5","modelname");
   std::cout << "done"<<std::endl;
   
   std::cout << "Load model ... "<<std::flush;
   //Load the model into HDF5 file
   Model gm2;
   opengm::hdf5::load(gm2,"./model.h5","modelname");
  std::cout << "done"<<std::endl;
  

   std::cout << "Compare models ... "<<std::flush;
   bool equal=true;
   std::vector<LabelType> l(5,0);
   for(l[0]=0; l[0]<5; ++l[0])
      for(l[1]=0; l[1]<5; ++l[1])
         for(l[2]=0; l[2]<2; ++l[2])
            for(l[3]=0; l[3]<2; ++l[3])
               for(l[4]=0; l[4]<10; ++l[4])
                  if(gm.evaluate(l) != gm2.evaluate(l))
                     equal=false;

   if(equal)
      std::cout << "Equal "<<std::endl;
   else  
      std::cout << "Not-Equal "<<std::endl;

   
}
