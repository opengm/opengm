/***********************************************************************
 * Tutorial:     Inference on the Min-Sum-Semiring (minimizing a energy function)
 * Author:       Joerg Hendrik Kappes
 * Date:         04.07.2014
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
#include <opengm/operations/and.hxx>
#include <opengm/operations/or.hxx>

#include <opengm/inference/messagepassing/messagepassing.hxx>
#include <opengm/inference/bruteforce.hxx>

#ifdef WITH_BOOST
#include "opengm/inference/sat.hxx"
#endif


int main(int argc, char** argv) {
   std::cout <<"Or-And-Semiring"<<std::endl;
   {
      //*******************
      //** Typedefs
      //*******************
      typedef bool                                                                   ValueType;          // type used for values
      typedef size_t                                                                 IndexType;          // type used for indexing nodes and factors (default : size_t)
      typedef size_t                                                                 LabelType;          // type used for labels (default : size_t)
      typedef opengm::And                                                            OpType;             // operation used to combine terms
      typedef opengm::ExplicitFunction<ValueType,IndexType,LabelType>                ExplicitFunction;   // shortcut for explicite function 
      typedef opengm::meta::TypeListGenerator<ExplicitFunction>::type                FunctionTypeList;   // list of all function the model cal use (this trick avoids virtual methods) - here only one
      typedef opengm::DiscreteSpace<IndexType, LabelType>                            SpaceType;          // type used to define the feasible statespace
      typedef opengm::GraphicalModel<ValueType,OpType,FunctionTypeList,SpaceType>    Model;              // type of the model
      typedef Model::FunctionIdentifier                                              FunctionIdentifier; // type of the function identifier

      //******************
      //** DATA
      //******************

      std::vector<LabelType> numStates(5,2);
      Model gmChain(opengm::DiscreteSpace<IndexType,LabelType>(numStates.begin(),numStates.end()));
      Model gmRing(opengm::DiscreteSpace<IndexType,LabelType>(numStates.begin(),numStates.end()));
      std::vector<LabelType> shape(2,2);
      ExplicitFunction function(shape.begin(), shape.end());
      function(0,0)=true;
      function(1,0)=false;
      function(0,1)=false; 
      function(1,1)=true;
      FunctionIdentifier eqChain = gmChain.addFunction(function);  
      FunctionIdentifier eqRing  = gmRing.addFunction(function);  
      function(0,0)=false;
      function(1,0)=true;
      function(0,1)=true; 
      function(1,1)=false;
      FunctionIdentifier neqChain = gmChain.addFunction(function);
      FunctionIdentifier neqRing  = gmRing.addFunction(function);

      std::vector<IndexType> var(2);
      for(IndexType i=0;i<4;++i){
         var[0]=i; var[1]=i+1;
         gmChain.addFactor(neqChain, var.begin(), var.end());
         gmRing.addFactor(eqRing, var.begin(), var.end());
      }
      var[0]=0; var[1]=(4);
      gmRing.addFactor(neqRing, var.begin(), var.end());

#ifdef WITH_BOOST
      opengm::SAT<Model> satChain(gmChain); 
      opengm::SAT<Model> satRing(gmRing);
      satChain.infer(); 
      satRing.infer();
      std::cout << "Chain = "<<satChain.value();
      std::cout << std::endl;
      std::cout << "Ring = "<<satRing.value();
      std::cout << std::endl;
#endif
   
      // Most other methods can not be directly applied since they require a inverse operation which is not defined for "And"
      // We will check if we can generalize our BP implementation for OR-AND (SAT) - Problems without slowing down the
      // implementation in general.
      // 
      // Since OpenGM supports this one can use OpenGM and implement a specific solver.
      // Alternatively one can emulate this with other semirings as shown below.


   }
 std::cout <<"Emulate Or-And-Semiring by Min-Sum-Semiring"<<std::endl;
   {
      //*******************
      //** Typedefs
      //*******************
      typedef double                                                                 ValueType;          // type used for values
      typedef size_t                                                                 IndexType;          // type used for indexing nodes and factors (default : size_t)
      typedef size_t                                                                 LabelType;          // type used for labels (default : size_t)
      typedef opengm::Adder                                                          OpType;             // operation used to combine terms
      typedef opengm::ExplicitFunction<ValueType,IndexType,LabelType>                ExplicitFunction;   // shortcut for explicite function 
      typedef opengm::meta::TypeListGenerator<ExplicitFunction>::type                FunctionTypeList;   // list of all function the model cal use (this trick avoids virtual methods) - here only one
      typedef opengm::DiscreteSpace<IndexType, LabelType>                            SpaceType;          // type used to define the feasible statespace
      typedef opengm::GraphicalModel<ValueType,OpType,FunctionTypeList,SpaceType>    Model;              // type of the model
      typedef Model::FunctionIdentifier                                              FunctionIdentifier; // type of the function identifier

      //******************
      //** DATA
      //******************

      
      std::vector<LabelType> numStates(5,2);
      Model gmChain(opengm::DiscreteSpace<IndexType,LabelType>(numStates.begin(),numStates.end()));
      Model gmRing(opengm::DiscreteSpace<IndexType,LabelType>(numStates.begin(),numStates.end()));
      std::vector<LabelType> shape(2,2);
      ExplicitFunction function(shape.begin(), shape.end());
      function(0,0)=0;
      function(1,0)=1;
      function(0,1)=1; 
      function(1,1)=0;
      FunctionIdentifier eqChain = gmChain.addFunction(function);  
      FunctionIdentifier eqRing  = gmRing.addFunction(function);  
      function(0,0)=1;
      function(1,0)=0;
      function(0,1)=0; 
      function(1,1)=1;
      FunctionIdentifier neqChain = gmChain.addFunction(function);
      FunctionIdentifier neqRing  = gmRing.addFunction(function);

      std::vector<IndexType> var(2);
      for(IndexType i=0;i<4;++i){
         var[0]=i; var[1]=i+1;
         gmChain.addFactor(neqChain, var.begin(), var.end());
         gmRing.addFactor(eqRing, var.begin(), var.end());
      }
      var[0]=0; var[1]=(4);
      gmRing.addFactor(neqRing, var.begin(), var.end());

      typedef opengm::BeliefPropagationUpdateRules<Model,opengm::Minimizer> UpdateRulesType;
      typedef opengm::MessagePassing<Model, opengm::Minimizer,UpdateRulesType, opengm::MaxDistance>   BP;

      BP bpChain(gmChain); 
      BP bpRing(gmRing);
      bpChain.infer(); 
      bpRing.infer();
      std::vector<LabelType> lChain(5);
      std::vector<LabelType> lRing(5);
      bpChain.arg(lChain);
      bpRing.arg(lRing);
      std::cout << "Chain = "<<(bpChain.value()<0.5)<<" | ";
      for (size_t i=0; i<5; ++i)  std::cout << lChain[i]<<" ";
      std::cout << std::endl;
      std::cout << "Ring = "<<(bpRing.value()<0.5)<<" | ";
      for (size_t i=0; i<5; ++i)  std::cout << lRing[i]<<" ";
      std::cout << std::endl;
   }


   return 0;
};
