#include <stdlib.h>
#include <vector>
#include <set>
#include <functional>

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/operations/multiplier.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/operations/maximizer.hxx>
#include <opengm/operations/integrator.hxx>
#include <opengm/inference/external/libdai/junction_tree.hxx>

//#include <opengm/unittests/blackboxtester.hxx>
//#include <opengm/unittests/blackboxtests/blackboxtestgrid.hxx>
//#include <opengm/unittests/blackboxtests/blackboxtestfull.hxx>
//#include <opengm/unittests/blackboxtests/blackboxteststar.hxx>


void testSumProd(){
   typedef opengm::GraphicalModel<double,opengm::Multiplier,opengm::ExplicitFunction<double, size_t, size_t>,opengm::DiscreteSpace<size_t, size_t> > Model;
   typedef Model::IndependentFactorType IndependentFactor; 
   typedef opengm::external::libdai::JunctionTree<Model, opengm::Integrator> JTT;

   size_t n_var=2;
   int n_stats[]={2,2};

   opengm::DiscreteSpace<size_t,size_t> space;
   for(int i=0;i<n_var;i++)
   {
      space.addVariable(n_stats[i]);   
   }
   Model model(space);

   opengm::ExplicitFunction<double> f1(n_stats,n_stats+2,0);
   f1(0,0)=0.2;
   f1(0,1)=0.1;
   f1(1,0)=0.3;
   f1(1,1)=0.4;
 
   double marg[4] = {0.3,0.7,0.5,0.5};

   size_t vars1[]={0,1};
   Model::FunctionIdentifier fid1=model.addFunction(f1);
   size_t facid1=model.addFactor(fid1,vars1,vars1+2);
   size_t vars[]={0,1};

   JTT::UpdateRule updateRule_jt;// = JTT::UpdateRule::HUGIN;
   JTT::Heuristic heuristic;//      = JTT::Heuristic::MINFILL;
   JTT::Parameter parameter_jt(updateRule_jt, heuristic,0);
   JTT jt(model, parameter_jt);

   Model::IndependentFactorType IF;
   jt.infer();
   
   for(size_t i=0;i<n_var;i++){
      jt.marginal(i,IF);
      std::cout<<"X_"<<i<<": ";
      for(size_t j=0;j<n_stats[i];j++){
         std::cout<<"state: "<<j<<" marginal value: "<<IF(j)<<"; ";
         OPENGM_ASSERT(std::fabs(IF(j)-marg[i*2+j])<0.000001);
      }
      std::cout<<"\n";
   }   
}

int main() {
   testSumProd();
}
