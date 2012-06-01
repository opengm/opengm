#include <stdlib.h>
#include <opengm/operations/adder.hxx>
#include <opengm/unittests/test.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/operations/maximizer.hxx>
#include <opengm/operations/integrator.hxx>

int main() {
   {
      typedef double ValueType;
      typedef size_t IndexType;
      typedef size_t LabelType;
      std::vector<size_t> var(2,0); var[1]=1;
      std::vector<size_t> shape(2,3);
      opengm::IndependentFactor<ValueType,IndexType,LabelType> temp(var.begin(),var.end(),shape.begin(),shape.end());
      temp(0,0)=6;
      temp(1,0)=2;
      temp(2,0)=3;
      temp(0,1)=4;
      temp(1,1)=5;
      temp(2,1)=1;
      temp(0,2)=7;
      temp(1,2)=9;
      temp(2,2)=8;


      ValueType v;
      std::vector<size_t> state;

      OPENGM_TEST(temp.numberOfVariables()==2);
      temp.accumulate<opengm::Minimizer>(v);
      OPENGM_TEST(v==1);
      OPENGM_TEST(temp.numberOfVariables()==2);
      v=0;
      temp.accumulate<opengm::Minimizer>(v,state);
      OPENGM_TEST(v==1);
      OPENGM_TEST(state.size()==2);
      OPENGM_TEST(state[0]==2);
      OPENGM_TEST(state[1]==1);

      temp.accumulate<opengm::Maximizer>(v);
      OPENGM_TEST(v==9);
      v=0;
      temp.accumulate<opengm::Maximizer>(v,state);
      OPENGM_TEST(v==9);
      OPENGM_TEST(state.size()==2);
      OPENGM_TEST(state[0]==1);
      OPENGM_TEST(state[1]==2);


      temp.accumulate<opengm::Integrator>(v);
      OPENGM_TEST(v==45);
   }

   {
      typedef double ValueType;
      typedef size_t IndexType;
      typedef size_t LabelType;
      size_t var[]={0,1,2};
      size_t shape[]={3,3,3};
      opengm::IndependentFactor<ValueType,IndexType,LabelType> temp(var,var+3,shape,shape+3);
      temp(0,0,0)=0;
      temp(0,0,1)=1;
      temp(0,0,2)=2;

      temp(0,1,0)=3;
      temp(0,1,1)=4;
      temp(0,1,2)=5;

      temp(0,2,0)=6;
      temp(0,2,1)=7;
      temp(0,2,2)=8;

      temp(1,0,0)=9;
      temp(1,0,1)=10;
      temp(1,0,2)=11;

      temp(1,1,0)=12;
      temp(1,1,1)=13;
      temp(1,1,2)=14;

      temp(1,2,0)=15;
      temp(1,2,1)=16;
      temp(1,2,2)=17;

      temp(2,0,0)=18;
      temp(2,0,1)=19;
      temp(2,0,2)=20;

      temp(2,1,0)=21;
      temp(2,1,1)=22;
      temp(2,1,2)=23;

      temp(2,2,0)=24;
      temp(2,2,1)=25;
      temp(2,2,2)=26;

      //accumulat over var 0:
      {
         size_t accvar[]={0};
         //minimizer
         {
            opengm::IndependentFactor<ValueType,IndexType,LabelType> res;
            opengm::IndependentFactor<ValueType,IndexType,LabelType> res2=temp;
            temp.accumulate<opengm::Minimizer>(accvar,accvar+1,res);
            OPENGM_TEST(res.numberOfVariables()==2);
            OPENGM_TEST(res.variableIndex(0)==1);
            OPENGM_TEST(res.variableIndex(1)==2);
            OPENGM_TEST(res.numberOfLabels(0)==3);
            OPENGM_TEST(res.numberOfLabels(1)==3);

            OPENGM_TEST(res(0,0)==0);
            OPENGM_TEST(res(0,1)==1);
            OPENGM_TEST(res(0,2)==2);

            OPENGM_TEST(res(1,0)==3);
            OPENGM_TEST(res(1,1)==4);
            OPENGM_TEST(res(1,2)==5);

            OPENGM_TEST(res(2,0)==6);
            OPENGM_TEST(res(2,1)==7);
            OPENGM_TEST(res(2,2)==8);

            res2.accumulate<opengm::Minimizer>(accvar,accvar+1);

            OPENGM_TEST(res2.numberOfVariables()==2);
            OPENGM_TEST(res2.variableIndex(0)==1);
            OPENGM_TEST(res2.variableIndex(1)==2);
            OPENGM_TEST(res2.numberOfLabels(0)==3);
            OPENGM_TEST(res2.numberOfLabels(1)==3);

            OPENGM_TEST(res2(0,0)==0);
            OPENGM_TEST(res2(0,1)==1);
            OPENGM_TEST(res2(0,2)==2);

            OPENGM_TEST(res2(1,0)==3);
            OPENGM_TEST(res2(1,1)==4);
            OPENGM_TEST(res2(1,2)==5);

            OPENGM_TEST(res2(2,0)==6);
            OPENGM_TEST(res2(2,1)==7);
            OPENGM_TEST(res2(2,2)==8);
         }
      }
   }
   return 0;
}
