#include <vector>
#include <iostream>

#include <opengm/learning/loss/generalized-hammingloss.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/graphicalmodel/graphicalmodel_factor.hxx>

//*************************************
typedef double ValueType;
typedef size_t IndexType;
typedef size_t LabelType;
typedef opengm::meta::TypeListGenerator<opengm::ExplicitFunction<ValueType,IndexType,LabelType> >::type FunctionListType;
typedef opengm::GraphicalModel<ValueType,opengm::Adder, FunctionListType, opengm::DiscreteSpace<IndexType,LabelType> > GM;

//*************************************


int main() {

   opengm::learning::GeneralizedHammingLoss::Parameter param;
   param.labelLossMultiplier_.push_back(2.0);
   param.labelLossMultiplier_.push_back(1.0);
   param.labelLossMultiplier_.push_back(0.5);

   param.nodeLossMultiplier_.push_back(5.0);
   param.nodeLossMultiplier_.push_back(6.0);
   param.nodeLossMultiplier_.push_back(7.0);
   param.nodeLossMultiplier_.push_back(8.0);

   // create loss
   opengm::learning::GeneralizedHammingLoss loss(param);

   // evaluate for a test point
   std::vector<size_t> labels;
   labels.push_back(0);
   labels.push_back(1);
   labels.push_back(2);
   labels.push_back(2);

   std::vector<size_t> ground_truth;
   ground_truth.push_back(1);
   ground_truth.push_back(1);
   ground_truth.push_back(1);
   ground_truth.push_back(1);


   // add loss to a model and evaluate for a given labeling
   GM gm;
   size_t numberOfLabels = 3;
   gm.addVariable(numberOfLabels);
   gm.addVariable(numberOfLabels);
   gm.addVariable(numberOfLabels);
   gm.addVariable(numberOfLabels);
   OPENGM_ASSERT_OP(loss.loss(gm, labels.begin(), labels.end(), ground_truth.begin(), ground_truth.end()), ==, 17.5);

   // add a unary to node 2 (if indexed from 1)
   opengm::ExplicitFunction<GM::ValueType,GM::IndexType,GM::LabelType> f(&numberOfLabels, &(numberOfLabels)+1, 2.0);
   size_t variableIndex = 1;
   gm.addFactor(gm.addFunction(f), &variableIndex, &variableIndex+1);
   OPENGM_ASSERT_OP(gm.evaluate(labels.begin()), ==, 2.0);

   // loss augmented model:
   loss.addLoss(gm, ground_truth.begin());
   OPENGM_ASSERT_OP(gm.evaluate(labels.begin()), ==, -15.5);
}
