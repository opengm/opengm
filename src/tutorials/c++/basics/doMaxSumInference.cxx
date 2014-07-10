/***********************************************************************
 * Tutorial:     Inference on the Max-Sum-Semiring (maximizing a energy function)
 * Author:       Joerg Hendrik Kappes
 * Date:         09.07.2014
 * Dependencies: None
 *
 * Description:
 * ------------
 * This Example construct a model and find the labeling with the lowest energy 
 *
 ************************************************************************/

#include <iostream>

#include <opengm/opengm.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/functions/potts.hxx>

#include <opengm/inference/icm.hxx>
#include <opengm/inference/messagepassing/messagepassing.hxx>
#include <opengm/inference/trws/trws_trws.hxx>


int main(int argc, char** argv) {
   std::cout <<"Max-Sum-Semiring"<<std::endl;
   //*******************
   //** Typedefs
   //*******************
   typedef double                                                                 ValueType;          // type used for values
   typedef size_t                                                                 IndexType;          // type used for indexing nodes and factors (default : size_t)
   typedef size_t                                                                 LabelType;          // type used for labels (default : size_t)
   typedef opengm::Adder                                                          OpType;             // operation used to combine terms
   typedef opengm::ExplicitFunction<ValueType,IndexType,LabelType>                ExplicitFunction;   // shortcut for explicite function 
   typedef opengm::PottsFunction<ValueType,IndexType,LabelType>                   PottsFunction;      // shortcut for Potts function
   typedef opengm::meta::TypeListGenerator<ExplicitFunction,PottsFunction>::type  FunctionTypeList;   // list of all function the model cal use (this trick avoids virtual methods) - here only one
   typedef opengm::DiscreteSpace<IndexType, LabelType>                            SpaceType;          // type used to define the feasible statespace
   typedef opengm::GraphicalModel<ValueType,OpType,FunctionTypeList,SpaceType>    Model;              // type of the model
   typedef Model::FunctionIdentifier                                              FunctionIdentifier; // type of the function identifier

   //******************
   //** DATA
   //******************
   IndexType N = 6;
   IndexType M = 4;  
   int data[] = { 4, 4, 4, 4, 6, 0,
                  0, 7, 2, 4, 4, 0,
                  9, 9, 4, 4, 9, 9,
                  2, 2, 9, 9, 9, 9 };

   std::cout << "Start building the model ... "<<std::endl;
   // Build empty Model
   LabelType numLabel = 2;
   std::vector<LabelType> numbersOfLabels(N*M,numLabel);
   Model gm(SpaceType(numbersOfLabels.begin(), numbersOfLabels.end()));

   // Add 1st order functions and factors to the model
   for(IndexType variable = 0; variable < gm.numberOfVariables(); ++variable) {
      // construct 1st order function
      const LabelType shape[] = {gm.numberOfLabels(variable)};
      ExplicitFunction f(shape, shape + 1);
      f(0) = std::fabs(data[variable] - 2.0);
      f(1) = std::fabs(data[variable] - 8.0);
      // add function
      FunctionIdentifier id = gm.addFunction(f);
      // add factor
      IndexType variableIndex[] = {variable};
      gm.addFactor(id, variableIndex, variableIndex + 1);
   }
   // add 2nd order functions for all variables neighbored on the grid
   {
      // add a potts function to the model
      PottsFunction potts(numLabel, numLabel, 0.0, 2.0);
      FunctionIdentifier pottsid = gm.addFunction(potts);

      IndexType vars[]  = {0,1}; 
      for(IndexType n=0; n<N;++n){
         for(IndexType m=0; m<M;++m){
            vars[0] = n + m*N;
            if(n+1<N){ //check for right neighbor
               vars[1] =  (n+1) + (m  )*N;
               OPENGM_ASSERT(vars[0] < vars[1]); // variables need to be ordered!
               gm.addFactor(pottsid, vars, vars + 2);
            } 
            if(m+1<M){ //check for lower neighbor
               vars[1] =  (n  ) + (m+1)*N; 
               OPENGM_ASSERT(vars[0] < vars[1]); // variables need to be ordered!
               gm.addFactor(pottsid, vars, vars + 2);
            }
         }
      }
   }

   // Infer with LBP
   std::cout << "Start LBP inference ... " <<std::endl;
   typedef opengm::BeliefPropagationUpdateRules<Model, opengm::Maximizer> UpdateRules;
   typedef opengm::MessagePassing<Model, opengm::Maximizer, UpdateRules, opengm::MaxDistance>  LBP; 
  
   LBP::Parameter parameter(100, 0.01, 0.8); //maximal number of iterations=0, minimal message distance=0.01, damping=0.8
   LBP lbp(gm, parameter); 
  
   lbp.infer();
   std::vector<LabelType> l_LBP;
   lbp.arg(l_LBP);
   std::cout << "Energy :  "<<lbp.value() << std::endl; 
   std::cout << "State :  ";
   for (size_t i=0; i<l_LBP.size(); ++i) std::cout <<l_LBP[i]<<" ";
   std::cout  << std::endl; 


   // Infer with ICM
   std::cout << "Start ICM inference ... " <<std::endl;
   typedef opengm::ICM<Model,opengm::Maximizer> ICM;
   ICM icm (gm);
   icm.infer();
   std::vector<LabelType> l_ICM;
   icm.arg(l_ICM);
   std::cout << "Energy :  "<<icm.value() << std::endl; 
   std::cout << "State :  ";
   for (size_t i=0; i<l_ICM.size(); ++i) std::cout <<l_ICM[i]<<" ";
   std::cout  << std::endl; 

   // Infer with TRWSI 
   std::cout << "Start TRWSi inference ... " <<std::endl;
   typedef opengm::TRWSi<Model,opengm::Maximizer> TRWSi;
   TRWSi::Parameter para(100);
   para.precision_=1e-12;
   TRWSi trws(gm,para);
   trws.infer();
   std::vector<LabelType> l_TRWS;
   trws.arg(l_TRWS);
   std::cout << "Energy :  "<<trws.value() << std::endl; 
   std::cout << "State :  ";
   for (size_t i=0; i<l_TRWS.size(); ++i) std::cout <<l_TRWS[i]<<" ";
   std::cout  << std::endl;


   // The list of solvers above is of cause not complete.
   // For example with other solvers and more used parameters see:
   // - opengm/src/unittests/inference/* 

 
   return 0;
};
