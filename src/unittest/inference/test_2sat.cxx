#include <stdlib.h>
#include <iostream>
#include <vector>

#include "opengm/opengm.hxx"
#include "opengm/unittests/test.hxx"
#include "opengm/graphicalmodel/graphicalmodel.hxx"

#ifdef WITH_BOOST
#include "opengm/inference/sat.hxx"
#endif

template<class SAT>
void test_empty(){
   typedef SAT SatType;
   typedef typename SAT::GraphicalModelType GraphicalModelType;
   std::cout << "    - Test empty model ... "<<std::flush;
   std::vector<size_t> numStates(5,2);
   GraphicalModelType gm(opengm::DiscreteSpace<size_t,size_t>(numStates.begin(),numStates.end()));
   SatType sat(gm);
   sat.infer();
   OPENGM_TEST(sat.value() == true);
   std::cout << "OK!"<<std::endl;  
}

template<class SAT>
void test_chain(){ 
   typedef SAT SatType;
   typedef typename SAT::GraphicalModelType GraphicalModelType;
   typedef opengm::ExplicitFunction<typename GraphicalModelType::ValueType> ExplicitFunctionType;
    
   std::cout << "    - Test chain model ... "<<std::flush;
   std::vector<size_t> numStates(5,2);
   GraphicalModelType gm1(opengm::DiscreteSpace<size_t,size_t>(numStates.begin(),numStates.end()));
   GraphicalModelType gm2(opengm::DiscreteSpace<size_t,size_t>(numStates.begin(),numStates.end()));
   std::vector<size_t> shape(2,2);
   ExplicitFunctionType function(shape.begin(), shape.end());
   function(0,0)=true;
   function(1,0)=false;
   function(0,1)=false; 
   function(1,1)=true;
   typename GraphicalModelType::FunctionIdentifier eq1 = gm1.addFunction(function); 
   typename GraphicalModelType::FunctionIdentifier eq2 = gm2.addFunction(function); 
   function(0,0)=false;
   function(1,0)=true;
   function(0,1)=true; 
   function(1,1)=false;
   typename GraphicalModelType::FunctionIdentifier neq1 = gm1.addFunction(function);
   typename GraphicalModelType::FunctionIdentifier neq2 = gm2.addFunction(function);

   std::vector<size_t> var(2);
   for(size_t i=0;i<4;++i){
      var[0]=i; var[1]=i+1;
      gm1.addFactor(neq1, var.begin(), var.end());
      gm2.addFactor(i%2 ? eq2 : neq2, var.begin(), var.end());
   }

   SatType sat1(gm1); 
   SatType sat2(gm2);
   sat1.infer();
   sat2.infer(); 

   OPENGM_TEST(sat1.value() == true);
   OPENGM_TEST(sat2.value() == true);
   std::cout << "OK!"<<std::endl; 
}

template<class SAT>
void test_ring(){
   typedef SAT SatType;
   typedef typename SAT::GraphicalModelType GraphicalModelType;
   typedef opengm::ExplicitFunction<typename GraphicalModelType::ValueType> ExplicitFunctionType;
   
   std::cout << "    - Test chain model ... "<<std::flush;
   std::vector<size_t> numStates(5,2);
   GraphicalModelType gm1(opengm::DiscreteSpace<size_t,size_t>(numStates.begin(),numStates.end()));
   GraphicalModelType gm2(opengm::DiscreteSpace<size_t,size_t>(numStates.begin(),numStates.end()));
   std::vector<size_t> shape(2,2);
   ExplicitFunctionType function(shape.begin(), shape.end());
   function(0,0)=true;
   function(1,0)=false;
   function(0,1)=false; 
   function(1,1)=true;
   typename GraphicalModelType::FunctionIdentifier eq1 = gm1.addFunction(function); 
   typename GraphicalModelType::FunctionIdentifier eq2 = gm2.addFunction(function); 
   function(0,0)=false;
   function(1,0)=true;
   function(0,1)=true; 
   function(1,1)=false;
   typename GraphicalModelType::FunctionIdentifier neq1 = gm1.addFunction(function);
   typename GraphicalModelType::FunctionIdentifier neq2 = gm2.addFunction(function);

   std::vector<size_t> var(2);
   for(size_t i=0;i<4;++i){
      var[0]=i; var[1]=(i+1);
      gm1.addFactor(eq1, var.begin(), var.end());
      gm2.addFactor(neq2, var.begin(), var.end());
   }
   var[0]=0; var[1]=(4);
   gm1.addFactor(eq1, var.begin(), var.end());
   gm2.addFactor(neq2, var.begin(), var.end());

   SatType sat1(gm1); 
   SatType sat2(gm2);
   sat1.infer();
   sat2.infer(); 

   OPENGM_TEST(sat1.value() == true);
   OPENGM_TEST(sat2.value() == false);
   std::cout << "OK!"<<std::endl; 
}

int main(int argc, char** argv) { 
   std::cout << "Test 2SAT ... " <<std::endl;
#ifdef WITH_BOOST
   typedef opengm::GraphicalModel<bool, opengm::And> GraphicalModelType;
   typedef opengm::SAT<GraphicalModelType> SatType; 
   std::cout << "  * Implementation using BOOST ..." <<std::endl;
   test_empty<SatType>();
   test_chain<SatType>(); 
   test_ring<SatType>();
#endif
   std::cout << " done!" << std::endl;
   return (EXIT_SUCCESS);
}
