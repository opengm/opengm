#include <iostream>

#include <opengm/unittests/test.hxx>
#include <opengm/inference/auxiliary/lp_functiontransfer.hxx>
#include <opengm/utilities/random.hxx>

#include <opengm/functions/explicit_function.hxx>
#include <opengm/functions/soft_constraint_functions/sum_constraint_function.hxx>
#include <opengm/functions/soft_constraint_functions/label_cost_function.hxx>

void testExplicitFunction();
void testSumConstraintFunction();
void testLabelCostFunction();

int main(int argc, char** argv){
   std::cout << "LP Function Transformation test...  " << std::endl;

   std::cout << "Test explicit function" << std::endl;
   testExplicitFunction();
   std::cout << "Test sum constraint function" << std::endl;
   testSumConstraintFunction();

   std::cout << "Test label cost function" << std::endl;
   testLabelCostFunction();

   std::cout << "done..." << std::endl;
   return 0;
}

void testExplicitFunction() {
   typedef double ValueType;
   typedef size_t IndexType;
   typedef size_t LabelType;

   typedef opengm::LPFunctionTransfer<ValueType, IndexType, LabelType> LPFunctionTransformationType;
   typedef opengm::ExplicitFunction<ValueType, IndexType, LabelType>         ExplicitFunctionType;

   OPENGM_TEST_EQUAL(LPFunctionTransformationType::isTransferable<ExplicitFunctionType>(), false);

   ExplicitFunctionType f;

   bool catchNumSlackVariablesError = false;
   try {
      LPFunctionTransformationType::numSlackVariables(f);
   } catch(opengm::RuntimeError& error) {
      catchNumSlackVariablesError = true;
   }
   OPENGM_TEST_EQUAL(catchNumSlackVariablesError, true);

   bool catchGetSlackVariablesOrderError = false;
   try {
      LPFunctionTransformationType::IndicatorVariablesContainerType order;
      LPFunctionTransformationType::getSlackVariablesOrder(f, order);
   } catch(opengm::RuntimeError& error) {
      catchGetSlackVariablesOrderError = true;
   }
   OPENGM_TEST_EQUAL(catchGetSlackVariablesOrderError, true);

   bool catchGetSlackVariablesObjectiveCoefficientsError = false;
   try {
      LPFunctionTransformationType::SlackVariablesObjectiveCoefficientsContainerType coefficients;
      LPFunctionTransformationType::getSlackVariablesObjectiveCoefficients(f, coefficients);
   } catch(opengm::RuntimeError& error) {
      catchGetSlackVariablesObjectiveCoefficientsError = true;
   }
   OPENGM_TEST_EQUAL(catchGetSlackVariablesObjectiveCoefficientsError, true);

   bool catchGetIndicatorVariablesError = false;
   try {
      LPFunctionTransformationType::IndicatorVariablesContainerType variables;
      LPFunctionTransformationType::getIndicatorVariables(f, variables);
   } catch(opengm::RuntimeError& error) {
      catchGetIndicatorVariablesError = true;
   }
   OPENGM_TEST_EQUAL(catchGetIndicatorVariablesError, true);

   bool catchGetLinearConstraintsError = false;
   try {
      LPFunctionTransformationType::LinearConstraintsContainerType constraints;
      LPFunctionTransformationType::getLinearConstraints(f, constraints);
   } catch(opengm::RuntimeError& error) {
      catchGetLinearConstraintsError = true;
   }
   OPENGM_TEST_EQUAL(catchGetLinearConstraintsError, true);
}

void testSumConstraintFunction() {
   typedef double ValueType;
   typedef size_t IndexType;
   typedef size_t LabelType;

   typedef opengm::LPFunctionTransfer<ValueType, IndexType, LabelType> LPFunctionTransformationType;
   typedef opengm::SumConstraintFunction<ValueType, IndexType, LabelType>    SumConstraintFunctionType;

   const IndexType numVariables = 15;
   const LabelType minNumLabels = 1;
   const LabelType maxNumLabels = 10;
   const ValueType minCoefficientsValue = -2.0;
   const ValueType maxCoefficientsValue = 2.0;
   const ValueType minLambda = 1.0;
   const ValueType maxLambda = 2.0;

   typedef opengm::RandomUniformInteger<LabelType> RandomUniformLabelType;
   RandomUniformLabelType labelGenerator(0, maxNumLabels);

   typedef opengm::RandomUniformFloatingPoint<double> RandomUniformValueType;
   RandomUniformValueType coefficientsGenerator(minCoefficientsValue, maxCoefficientsValue);
   RandomUniformValueType lambdaGenerator(minLambda, maxLambda);

   // create function
   size_t numCoefficients = 0;
   std::vector<LabelType> shape(numVariables);
   LabelType currentMaxNumLabels = 0;
   for(IndexType i = 0; i < numVariables; ++i) {
      const LabelType numLabels = labelGenerator() + 1;
      if(numLabels > currentMaxNumLabels) {
         currentMaxNumLabels = numLabels;
      }
      shape[i] = numLabels;
      numCoefficients += numLabels;
   }

   std::vector<ValueType> coefficients(std::max(numCoefficients, numVariables * shape[0]));
   std::vector<ValueType> coefficientsInverted(coefficients.size());
   ValueType coefficientsAbsoluteSum = 0.0;
   for(IndexType i = 0; i < coefficients.size(); ++i) {
      const ValueType coefficient = coefficientsGenerator();
      coefficients[i] = coefficient;
      coefficientsInverted[i] = -coefficient;
      coefficientsAbsoluteSum += std::abs(coefficient);
   }

   const ValueType lambda = lambdaGenerator();
   const ValueType bound = (2 * coefficientsAbsoluteSum) / (static_cast<ValueType>(maxNumLabels - minNumLabels) * numVariables);

   SumConstraintFunctionType SumConstraintFunctionDifferentNumLabels(shape.begin(), shape.end(), coefficients.begin(), coefficients.begin() + numCoefficients, false, lambda, bound);
   SumConstraintFunctionType SumConstraintFunctionDifferentNumLabelsSharedCoefficients(shape.begin(), shape.end(), coefficients.begin(), coefficients.begin() + currentMaxNumLabels, true, lambda, bound);
   SumConstraintFunctionType SumConstraintFunctionSameNumLabels(numVariables, shape[0], coefficients.begin(), coefficients.begin() + (numVariables * shape[0]), false, lambda, bound);
   SumConstraintFunctionType SumConstraintFunctionSameNumLabelsSharedCoefficients(numVariables, shape[0], coefficients.begin(), coefficients.begin() + shape[0], true, lambda, bound);

   OPENGM_TEST_EQUAL(LPFunctionTransformationType::isTransferable<SumConstraintFunctionType>(), true);

   bool catchNumSlackVariablesError = false;
   try {
      OPENGM_TEST_EQUAL(LPFunctionTransformationType::numSlackVariables(SumConstraintFunctionDifferentNumLabels), 1);
      OPENGM_TEST_EQUAL(LPFunctionTransformationType::numSlackVariables(SumConstraintFunctionDifferentNumLabelsSharedCoefficients), 1);
      OPENGM_TEST_EQUAL(LPFunctionTransformationType::numSlackVariables(SumConstraintFunctionSameNumLabels), 1);
      OPENGM_TEST_EQUAL(LPFunctionTransformationType::numSlackVariables(SumConstraintFunctionSameNumLabelsSharedCoefficients), 1);
   } catch(opengm::RuntimeError& error) {
      catchNumSlackVariablesError = true;
   }
   OPENGM_TEST_EQUAL(catchNumSlackVariablesError, false);

   bool catchGetSlackVariablesOrderError = false;
   try {
      LPFunctionTransformationType::IndicatorVariablesContainerType order1;
      LPFunctionTransformationType::IndicatorVariablesContainerType order2;
      LPFunctionTransformationType::IndicatorVariablesContainerType order3;
      LPFunctionTransformationType::IndicatorVariablesContainerType order4;
      LPFunctionTransformationType::getSlackVariablesOrder(SumConstraintFunctionDifferentNumLabels, order1);
      LPFunctionTransformationType::getSlackVariablesOrder(SumConstraintFunctionDifferentNumLabelsSharedCoefficients, order2);
      LPFunctionTransformationType::getSlackVariablesOrder(SumConstraintFunctionSameNumLabels, order3);
      LPFunctionTransformationType::getSlackVariablesOrder(SumConstraintFunctionSameNumLabelsSharedCoefficients, order4);
      OPENGM_TEST_EQUAL(order1.size(), 1);
      OPENGM_TEST_EQUAL(std::distance(order1[0].begin(), order1[0].end()), 1);
      OPENGM_TEST_EQUAL(order1[0].begin()->first, numVariables);
      OPENGM_TEST_EQUAL(order1[0].begin()->second, 0);
      OPENGM_TEST_EQUAL(order1[0].getLogicalOperatorType(), LPFunctionTransformationType::IndicatorVariableType::And);
      OPENGM_TEST_EQUAL(order2.size(), 1);
      OPENGM_TEST_EQUAL(std::distance(order2[0].begin(), order2[0].end()), 1);
      OPENGM_TEST_EQUAL(order2[0].begin()->first, numVariables);
      OPENGM_TEST_EQUAL(order2[0].begin()->second, 0);
      OPENGM_TEST_EQUAL(order2[0].getLogicalOperatorType(), LPFunctionTransformationType::IndicatorVariableType::And);
      OPENGM_TEST_EQUAL(order3.size(), 1);
      OPENGM_TEST_EQUAL(std::distance(order3[0].begin(), order3[0].end()), 1);
      OPENGM_TEST_EQUAL(order3[0].begin()->first, numVariables);
      OPENGM_TEST_EQUAL(order3[0].begin()->second, 0);
      OPENGM_TEST_EQUAL(order3[0].getLogicalOperatorType(), LPFunctionTransformationType::IndicatorVariableType::And);
      OPENGM_TEST_EQUAL(order4.size(), 1);
      OPENGM_TEST_EQUAL(std::distance(order4[0].begin(), order4[0].end()), 1);
      OPENGM_TEST_EQUAL(order4[0].begin()->first, numVariables);
      OPENGM_TEST_EQUAL(order4[0].begin()->second, 0);
      OPENGM_TEST_EQUAL(order4[0].getLogicalOperatorType(), LPFunctionTransformationType::IndicatorVariableType::And);
   } catch(opengm::RuntimeError& error) {
      catchGetSlackVariablesOrderError = true;
   }
   OPENGM_TEST_EQUAL(catchGetSlackVariablesOrderError, false);

   bool catchGetSlackVariablesObjectiveCoefficientsError = false;
   try {
      LPFunctionTransformationType::SlackVariablesObjectiveCoefficientsContainerType slackVariablesObjectiveCoefficients1;
      LPFunctionTransformationType::SlackVariablesObjectiveCoefficientsContainerType slackVariablesObjectiveCoefficients2;
      LPFunctionTransformationType::SlackVariablesObjectiveCoefficientsContainerType slackVariablesObjectiveCoefficients3;
      LPFunctionTransformationType::SlackVariablesObjectiveCoefficientsContainerType slackVariablesObjectiveCoefficients4;
      LPFunctionTransformationType::getSlackVariablesObjectiveCoefficients(SumConstraintFunctionDifferentNumLabels, slackVariablesObjectiveCoefficients1);
      LPFunctionTransformationType::getSlackVariablesObjectiveCoefficients(SumConstraintFunctionDifferentNumLabelsSharedCoefficients, slackVariablesObjectiveCoefficients2);
      LPFunctionTransformationType::getSlackVariablesObjectiveCoefficients(SumConstraintFunctionSameNumLabels, slackVariablesObjectiveCoefficients3);
      LPFunctionTransformationType::getSlackVariablesObjectiveCoefficients(SumConstraintFunctionSameNumLabelsSharedCoefficients, slackVariablesObjectiveCoefficients4);
      OPENGM_TEST_EQUAL(slackVariablesObjectiveCoefficients1.size(), 1);
      OPENGM_TEST_EQUAL(slackVariablesObjectiveCoefficients1[0], lambda);
      OPENGM_TEST_EQUAL(slackVariablesObjectiveCoefficients2.size(), 1);
      OPENGM_TEST_EQUAL(slackVariablesObjectiveCoefficients2[0], lambda);
      OPENGM_TEST_EQUAL(slackVariablesObjectiveCoefficients3.size(), 1);
      OPENGM_TEST_EQUAL(slackVariablesObjectiveCoefficients3[0], lambda);
      OPENGM_TEST_EQUAL(slackVariablesObjectiveCoefficients4.size(), 1);
      OPENGM_TEST_EQUAL(slackVariablesObjectiveCoefficients4[0], lambda);
   } catch(opengm::RuntimeError& error) {
      catchGetSlackVariablesObjectiveCoefficientsError = true;
   }
   OPENGM_TEST_EQUAL(catchGetSlackVariablesObjectiveCoefficientsError, false);

   bool catchGetIndicatorVariablesError = false;
   try {
      LPFunctionTransformationType::IndicatorVariablesContainerType variables1;
      LPFunctionTransformationType::IndicatorVariablesContainerType variables2;
      LPFunctionTransformationType::IndicatorVariablesContainerType variables3;
      LPFunctionTransformationType::IndicatorVariablesContainerType variables4;
      LPFunctionTransformationType::getIndicatorVariables(SumConstraintFunctionDifferentNumLabels, variables1);
      LPFunctionTransformationType::getIndicatorVariables(SumConstraintFunctionDifferentNumLabelsSharedCoefficients, variables2);
      LPFunctionTransformationType::getIndicatorVariables(SumConstraintFunctionSameNumLabels, variables3);
      LPFunctionTransformationType::getIndicatorVariables(SumConstraintFunctionSameNumLabelsSharedCoefficients, variables4);
      OPENGM_TEST_EQUAL(variables1.size(), numCoefficients + 1);
      OPENGM_TEST_EQUAL(variables2.size(), numCoefficients + 1);
      OPENGM_TEST_EQUAL(variables3.size(), (shape[0] * numVariables) + 1);
      OPENGM_TEST_EQUAL(variables4.size(), (shape[0] * numVariables) + 1);
      IndexType currentVariable = 0;
      LabelType currentLabel = 0;

      for(size_t variablesIter = 0; variablesIter < variables1.size() - 1; ++variablesIter) {
         OPENGM_TEST_EQUAL(std::distance(variables1[variablesIter].begin(), variables1[variablesIter].end()), 1);
         OPENGM_TEST_EQUAL(variables1[variablesIter].begin()->first, currentVariable);
         OPENGM_TEST_EQUAL(variables1[variablesIter].begin()->second, currentLabel);
         OPENGM_TEST_EQUAL(variables1[variablesIter].getLogicalOperatorType(), LPFunctionTransformationType::IndicatorVariableType::And);
         OPENGM_TEST_EQUAL(std::distance(variables2[variablesIter].begin(), variables2[variablesIter].end()), 1);
         OPENGM_TEST_EQUAL(variables2[variablesIter].begin()->first, currentVariable);
         OPENGM_TEST_EQUAL(variables2[variablesIter].begin()->second, currentLabel);
         OPENGM_TEST_EQUAL(variables2[variablesIter].getLogicalOperatorType(), LPFunctionTransformationType::IndicatorVariableType::And);
         ++currentLabel;
         if(currentLabel == shape[currentVariable]) {
            ++currentVariable;
            currentLabel = 0;
         }
      }
      OPENGM_TEST_EQUAL(currentVariable, numVariables);
      OPENGM_TEST_EQUAL(currentLabel, 0);

      OPENGM_TEST_EQUAL(std::distance(variables1[variables1.size() - 1].begin(), variables1[variables1.size() - 1].end()), 1);
      OPENGM_TEST_EQUAL(variables1[variables1.size() - 1].begin()->first, numVariables);
      OPENGM_TEST_EQUAL(variables1[variables1.size() - 1].begin()->second, 0);
      OPENGM_TEST_EQUAL(variables1[variables1.size() - 1].getLogicalOperatorType(), LPFunctionTransformationType::IndicatorVariableType::And);
      OPENGM_TEST_EQUAL(std::distance(variables2[variables2.size() - 1].begin(), variables2[variables2.size() - 1].end()), 1);
      OPENGM_TEST_EQUAL(variables2[variables2.size() - 1].begin()->first, numVariables);
      OPENGM_TEST_EQUAL(variables2[variables2.size() - 1].begin()->second, 0);
      OPENGM_TEST_EQUAL(variables2[variables2.size() - 1].getLogicalOperatorType(), LPFunctionTransformationType::IndicatorVariableType::And);

      currentVariable = 0;
      currentLabel = 0;
      for(size_t variablesIter = 0; variablesIter < variables3.size() - 1; ++variablesIter) {
         OPENGM_TEST_EQUAL(std::distance(variables3[variablesIter].begin(), variables3[variablesIter].end()), 1);
         OPENGM_TEST_EQUAL(variables3[variablesIter].begin()->first, currentVariable);
         OPENGM_TEST_EQUAL(variables3[variablesIter].begin()->second, currentLabel);
         OPENGM_TEST_EQUAL(variables3[variablesIter].getLogicalOperatorType(), LPFunctionTransformationType::IndicatorVariableType::And);
         OPENGM_TEST_EQUAL(std::distance(variables4[variablesIter].begin(), variables4[variablesIter].end()), 1);
         OPENGM_TEST_EQUAL(variables4[variablesIter].begin()->first, currentVariable);
         OPENGM_TEST_EQUAL(variables4[variablesIter].begin()->second, currentLabel);
         OPENGM_TEST_EQUAL(variables4[variablesIter].getLogicalOperatorType(), LPFunctionTransformationType::IndicatorVariableType::And);
         ++currentLabel;
         if(currentLabel == shape[0]) {
            ++currentVariable;
            currentLabel = 0;
         }
      }
      OPENGM_TEST_EQUAL(currentVariable, numVariables);
      OPENGM_TEST_EQUAL(currentLabel, 0);

      OPENGM_TEST_EQUAL(std::distance(variables3[variables3.size() - 1].begin(), variables3[variables3.size() - 1].end()), 1);
      OPENGM_TEST_EQUAL(variables3[variables3.size() - 1].begin()->first, numVariables);
      OPENGM_TEST_EQUAL(variables3[variables3.size() - 1].begin()->second, 0);
      OPENGM_TEST_EQUAL(variables3[variables3.size() - 1].getLogicalOperatorType(), LPFunctionTransformationType::IndicatorVariableType::And);
      OPENGM_TEST_EQUAL(std::distance(variables4[variables4.size() - 1].begin(), variables4[variables4.size() - 1].end()), 1);
      OPENGM_TEST_EQUAL(variables4[variables4.size() - 1].begin()->first, numVariables);
      OPENGM_TEST_EQUAL(variables4[variables4.size() - 1].begin()->second, 0);
      OPENGM_TEST_EQUAL(variables4[variables4.size() - 1].getLogicalOperatorType(), LPFunctionTransformationType::IndicatorVariableType::And);
   } catch(opengm::RuntimeError& error) {
      catchGetIndicatorVariablesError = true;
   }
   OPENGM_TEST_EQUAL(catchGetIndicatorVariablesError, false);

   bool catchGetLinearConstraintsError = false;
   try {
      LPFunctionTransformationType::LinearConstraintsContainerType constraints1;
      LPFunctionTransformationType::LinearConstraintsContainerType constraints2;
      LPFunctionTransformationType::LinearConstraintsContainerType constraints3;
      LPFunctionTransformationType::LinearConstraintsContainerType constraints4;
      LPFunctionTransformationType::getLinearConstraints(SumConstraintFunctionDifferentNumLabels, constraints1);
      LPFunctionTransformationType::getLinearConstraints(SumConstraintFunctionDifferentNumLabelsSharedCoefficients, constraints2);
      LPFunctionTransformationType::getLinearConstraints(SumConstraintFunctionSameNumLabels, constraints3);
      LPFunctionTransformationType::getLinearConstraints(SumConstraintFunctionSameNumLabelsSharedCoefficients, constraints4);

      OPENGM_TEST_EQUAL(constraints1.size(), 2);
      OPENGM_TEST_EQUAL(constraints2.size(), 2);
      OPENGM_TEST_EQUAL(constraints3.size(), 2);
      OPENGM_TEST_EQUAL(constraints4.size(), 2);

      OPENGM_TEST_EQUAL(constraints1[0].getBound(), bound);
      OPENGM_TEST_EQUAL(constraints2[1].getBound(), -bound);
      OPENGM_TEST_EQUAL(constraints3[0].getBound(), bound);
      OPENGM_TEST_EQUAL(constraints4[1].getBound(), -bound);

      OPENGM_TEST_EQUAL(constraints1[0].getConstraintOperator(), LPFunctionTransformationType::LinearConstraintType::LinearConstraintOperatorType::LessEqual);
      OPENGM_TEST_EQUAL(constraints1[1].getConstraintOperator(), LPFunctionTransformationType::LinearConstraintType::LinearConstraintOperatorType::LessEqual);
      OPENGM_TEST_EQUAL(constraints2[0].getConstraintOperator(), LPFunctionTransformationType::LinearConstraintType::LinearConstraintOperatorType::LessEqual);
      OPENGM_TEST_EQUAL(constraints2[1].getConstraintOperator(), LPFunctionTransformationType::LinearConstraintType::LinearConstraintOperatorType::LessEqual);
      OPENGM_TEST_EQUAL(constraints3[0].getConstraintOperator(), LPFunctionTransformationType::LinearConstraintType::LinearConstraintOperatorType::LessEqual);
      OPENGM_TEST_EQUAL(constraints3[1].getConstraintOperator(), LPFunctionTransformationType::LinearConstraintType::LinearConstraintOperatorType::LessEqual);
      OPENGM_TEST_EQUAL(constraints4[0].getConstraintOperator(), LPFunctionTransformationType::LinearConstraintType::LinearConstraintOperatorType::LessEqual);
      OPENGM_TEST_EQUAL(constraints4[1].getConstraintOperator(), LPFunctionTransformationType::LinearConstraintType::LinearConstraintOperatorType::LessEqual);

      OPENGM_TEST_EQUAL(static_cast<size_t>(std::distance(constraints1[0].coefficientsBegin(), constraints1[0].coefficientsEnd())), numCoefficients + 1);
      OPENGM_TEST_EQUAL_SEQUENCE(coefficients.begin(), coefficients.begin() + numCoefficients, constraints1[0].coefficientsBegin());
      OPENGM_TEST_EQUAL(*(constraints1[0].coefficientsEnd() - 1), -1.0);

      OPENGM_TEST_EQUAL(static_cast<size_t>(std::distance(constraints2[0].coefficientsBegin(), constraints2[0].coefficientsEnd())), numCoefficients + 1);
      for(IndexType i = 0; i < numVariables; ++i) {
         OPENGM_TEST_EQUAL_SEQUENCE(coefficients.begin(), coefficients.begin() + shape[i], constraints2[0].coefficientsBegin() + std::accumulate(shape.begin(), shape.begin() + i, size_t(0)));
      }
      OPENGM_TEST_EQUAL(*(constraints2[0].coefficientsEnd() - 1), -1.0);

      OPENGM_TEST_EQUAL(static_cast<size_t>(std::distance(constraints3[0].coefficientsBegin(), constraints3[0].coefficientsEnd())), (numVariables * shape[0]) + 1);
      OPENGM_TEST_EQUAL_SEQUENCE(coefficients.begin(), coefficients.begin() + (numVariables * shape[0]), constraints3[0].coefficientsBegin());
      OPENGM_TEST_EQUAL(*(constraints3[0].coefficientsEnd() - 1), -1.0);

      OPENGM_TEST_EQUAL(static_cast<size_t>(std::distance(constraints4[0].coefficientsBegin(), constraints4[0].coefficientsEnd())), (numVariables * shape[0]) + 1);
      for(IndexType i = 0; i < numVariables; ++i) {
         OPENGM_TEST_EQUAL_SEQUENCE(coefficients.begin(), coefficients.begin() + shape[0], constraints4[0].coefficientsBegin() + (i * shape[0]));
      }
      OPENGM_TEST_EQUAL(*(constraints4[0].coefficientsEnd() - 1), -1.0);

      OPENGM_TEST_EQUAL(static_cast<size_t>(std::distance(constraints1[1].coefficientsBegin(), constraints1[1].coefficientsEnd())), numCoefficients + 1);
      OPENGM_TEST_EQUAL_SEQUENCE(coefficientsInverted.begin(), coefficientsInverted.begin() + numCoefficients, constraints1[1].coefficientsBegin());
      OPENGM_TEST_EQUAL(*(constraints1[1].coefficientsEnd() - 1), -1.0);

      OPENGM_TEST_EQUAL(static_cast<size_t>(std::distance(constraints2[1].coefficientsBegin(), constraints2[1].coefficientsEnd())), numCoefficients + 1);
      for(IndexType i = 0; i < numVariables; ++i) {
         OPENGM_TEST_EQUAL_SEQUENCE(coefficientsInverted.begin(), coefficientsInverted.begin() + shape[i], constraints2[1].coefficientsBegin() + std::accumulate(shape.begin(), shape.begin() + i, size_t(0)));
      }
      OPENGM_TEST_EQUAL(*(constraints2[1].coefficientsEnd() - 1), -1.0);

      OPENGM_TEST_EQUAL(static_cast<size_t>(std::distance(constraints3[1].coefficientsBegin(), constraints3[1].coefficientsEnd())), (numVariables * shape[0]) + 1);
      OPENGM_TEST_EQUAL_SEQUENCE(coefficientsInverted.begin(), coefficientsInverted.begin() + (numVariables * shape[0]), constraints3[1].coefficientsBegin());
      OPENGM_TEST_EQUAL(*(constraints3[1].coefficientsEnd() - 1), -1.0);

      OPENGM_TEST_EQUAL(static_cast<size_t>(std::distance(constraints4[1].coefficientsBegin(), constraints4[1].coefficientsEnd())), (numVariables * shape[0]) + 1);
      for(IndexType i = 0; i < numVariables; ++i) {
         OPENGM_TEST_EQUAL_SEQUENCE(coefficientsInverted.begin(), coefficientsInverted.begin() + shape[0], constraints4[1].coefficientsBegin() + (i * shape[0]));
      }
      OPENGM_TEST_EQUAL(*(constraints4[1].coefficientsEnd() - 1), -1.0);

      OPENGM_TEST_EQUAL(static_cast<size_t>(std::distance(constraints1[0].indicatorVariablesBegin(), constraints1[0].indicatorVariablesEnd())), numCoefficients + 1);
      OPENGM_TEST_EQUAL(static_cast<size_t>(std::distance(constraints1[1].indicatorVariablesBegin(), constraints1[1].indicatorVariablesEnd())), numCoefficients + 1);
      OPENGM_TEST_EQUAL(static_cast<size_t>(std::distance(constraints2[0].indicatorVariablesBegin(), constraints2[0].indicatorVariablesEnd())), numCoefficients + 1);
      OPENGM_TEST_EQUAL(static_cast<size_t>(std::distance(constraints2[1].indicatorVariablesBegin(), constraints2[1].indicatorVariablesEnd())), numCoefficients + 1);
      OPENGM_TEST_EQUAL(static_cast<size_t>(std::distance(constraints3[0].indicatorVariablesBegin(), constraints3[0].indicatorVariablesEnd())), (numVariables * shape[0]) + 1);
      OPENGM_TEST_EQUAL(static_cast<size_t>(std::distance(constraints3[1].indicatorVariablesBegin(), constraints3[1].indicatorVariablesEnd())), (numVariables * shape[0]) + 1);
      OPENGM_TEST_EQUAL(static_cast<size_t>(std::distance(constraints4[0].indicatorVariablesBegin(), constraints4[0].indicatorVariablesEnd())), (numVariables * shape[0]) + 1);
      OPENGM_TEST_EQUAL(static_cast<size_t>(std::distance(constraints4[1].indicatorVariablesBegin(), constraints4[1].indicatorVariablesEnd())), (numVariables * shape[0]) + 1);
      size_t currentIndicatorVariableID = 0;
      for(IndexType i = 0; i < numVariables; ++i) {
         for(LabelType j = 0; j < shape[i]; ++j) {
            OPENGM_TEST_EQUAL(std::distance((constraints1[0].indicatorVariablesBegin() + currentIndicatorVariableID)->begin(), (constraints1[0].indicatorVariablesBegin() + currentIndicatorVariableID)->end()), 1);
            OPENGM_TEST_EQUAL((constraints1[0].indicatorVariablesBegin() + currentIndicatorVariableID)->begin()->first, i);
            OPENGM_TEST_EQUAL((constraints1[0].indicatorVariablesBegin() + currentIndicatorVariableID)->begin()->second, j);
            OPENGM_TEST_EQUAL((constraints1[0].indicatorVariablesBegin() + currentIndicatorVariableID)->getLogicalOperatorType(), LPFunctionTransformationType::IndicatorVariableType::And);
            OPENGM_TEST_EQUAL(std::distance((constraints2[0].indicatorVariablesBegin() + currentIndicatorVariableID)->begin(), (constraints2[0].indicatorVariablesBegin() + currentIndicatorVariableID)->end()), 1);
            OPENGM_TEST_EQUAL((constraints2[0].indicatorVariablesBegin() + currentIndicatorVariableID)->begin()->first, i);
            OPENGM_TEST_EQUAL((constraints2[0].indicatorVariablesBegin() + currentIndicatorVariableID)->begin()->second, j);
            OPENGM_TEST_EQUAL((constraints2[0].indicatorVariablesBegin() + currentIndicatorVariableID)->getLogicalOperatorType(), LPFunctionTransformationType::IndicatorVariableType::And);

            OPENGM_TEST_EQUAL(std::distance((constraints1[1].indicatorVariablesBegin() + currentIndicatorVariableID)->begin(), (constraints1[1].indicatorVariablesBegin() + currentIndicatorVariableID)->end()), 1);
            OPENGM_TEST_EQUAL((constraints1[1].indicatorVariablesBegin() + currentIndicatorVariableID)->begin()->first, i);
            OPENGM_TEST_EQUAL((constraints1[1].indicatorVariablesBegin() + currentIndicatorVariableID)->begin()->second, j);
            OPENGM_TEST_EQUAL((constraints1[1].indicatorVariablesBegin() + currentIndicatorVariableID)->getLogicalOperatorType(), LPFunctionTransformationType::IndicatorVariableType::And);
            OPENGM_TEST_EQUAL(std::distance((constraints2[1].indicatorVariablesBegin() + currentIndicatorVariableID)->begin(), (constraints2[1].indicatorVariablesBegin() + currentIndicatorVariableID)->end()), 1);
            OPENGM_TEST_EQUAL((constraints2[1].indicatorVariablesBegin() + currentIndicatorVariableID)->begin()->first, i);
            OPENGM_TEST_EQUAL((constraints2[1].indicatorVariablesBegin() + currentIndicatorVariableID)->begin()->second, j);
            OPENGM_TEST_EQUAL((constraints2[1].indicatorVariablesBegin() + currentIndicatorVariableID)->getLogicalOperatorType(), LPFunctionTransformationType::IndicatorVariableType::And);

            ++currentIndicatorVariableID;
         }
      }
      OPENGM_TEST_EQUAL(currentIndicatorVariableID, numCoefficients);

      currentIndicatorVariableID = 0;
      for(IndexType i = 0; i < numVariables; ++i) {
         for(LabelType j = 0; j < shape[0]; ++j) {
            OPENGM_TEST_EQUAL(std::distance((constraints3[0].indicatorVariablesBegin() + currentIndicatorVariableID)->begin(), (constraints3[0].indicatorVariablesBegin() + currentIndicatorVariableID)->end()), 1);
            OPENGM_TEST_EQUAL((constraints3[0].indicatorVariablesBegin() + currentIndicatorVariableID)->begin()->first, i);
            OPENGM_TEST_EQUAL((constraints3[0].indicatorVariablesBegin() + currentIndicatorVariableID)->begin()->second, j);
            OPENGM_TEST_EQUAL((constraints3[0].indicatorVariablesBegin() + currentIndicatorVariableID)->getLogicalOperatorType(), LPFunctionTransformationType::IndicatorVariableType::And);
            OPENGM_TEST_EQUAL(std::distance((constraints4[0].indicatorVariablesBegin() + currentIndicatorVariableID)->begin(), (constraints4[0].indicatorVariablesBegin() + currentIndicatorVariableID)->end()), 1);
            OPENGM_TEST_EQUAL((constraints4[0].indicatorVariablesBegin() + currentIndicatorVariableID)->begin()->first, i);
            OPENGM_TEST_EQUAL((constraints4[0].indicatorVariablesBegin() + currentIndicatorVariableID)->begin()->second, j);
            OPENGM_TEST_EQUAL((constraints4[0].indicatorVariablesBegin() + currentIndicatorVariableID)->getLogicalOperatorType(), LPFunctionTransformationType::IndicatorVariableType::And);

            OPENGM_TEST_EQUAL(std::distance((constraints3[1].indicatorVariablesBegin() + currentIndicatorVariableID)->begin(), (constraints3[1].indicatorVariablesBegin() + currentIndicatorVariableID)->end()), 1);
            OPENGM_TEST_EQUAL((constraints3[1].indicatorVariablesBegin() + currentIndicatorVariableID)->begin()->first, i);
            OPENGM_TEST_EQUAL((constraints3[1].indicatorVariablesBegin() + currentIndicatorVariableID)->begin()->second, j);
            OPENGM_TEST_EQUAL((constraints3[1].indicatorVariablesBegin() + currentIndicatorVariableID)->getLogicalOperatorType(), LPFunctionTransformationType::IndicatorVariableType::And);
            OPENGM_TEST_EQUAL(std::distance((constraints4[1].indicatorVariablesBegin() + currentIndicatorVariableID)->begin(), (constraints4[1].indicatorVariablesBegin() + currentIndicatorVariableID)->end()), 1);
            OPENGM_TEST_EQUAL((constraints4[1].indicatorVariablesBegin() + currentIndicatorVariableID)->begin()->first, i);
            OPENGM_TEST_EQUAL((constraints4[1].indicatorVariablesBegin() + currentIndicatorVariableID)->begin()->second, j);
            OPENGM_TEST_EQUAL((constraints4[1].indicatorVariablesBegin() + currentIndicatorVariableID)->getLogicalOperatorType(), LPFunctionTransformationType::IndicatorVariableType::And);

            ++currentIndicatorVariableID;
         }
      }
      OPENGM_TEST_EQUAL(currentIndicatorVariableID, numVariables * shape[0]);

      OPENGM_TEST_EQUAL(std::distance((constraints1[0].indicatorVariablesBegin() + numCoefficients)->begin(), (constraints1[0].indicatorVariablesBegin() + numCoefficients)->end()), 1);
      OPENGM_TEST_EQUAL((constraints1[0].indicatorVariablesBegin() + numCoefficients)->begin()->first, numVariables);
      OPENGM_TEST_EQUAL((constraints1[0].indicatorVariablesBegin() + numCoefficients)->begin()->second, 0);
      OPENGM_TEST_EQUAL((constraints1[0].indicatorVariablesBegin() + numCoefficients)->getLogicalOperatorType(), LPFunctionTransformationType::IndicatorVariableType::And);
      OPENGM_TEST_EQUAL(std::distance((constraints2[0].indicatorVariablesBegin() + numCoefficients)->begin(), (constraints2[0].indicatorVariablesBegin() + numCoefficients)->end()), 1);
      OPENGM_TEST_EQUAL((constraints2[0].indicatorVariablesBegin() + numCoefficients)->begin()->first, numVariables);
      OPENGM_TEST_EQUAL((constraints2[0].indicatorVariablesBegin() + numCoefficients)->begin()->second, 0);
      OPENGM_TEST_EQUAL((constraints2[0].indicatorVariablesBegin() + numCoefficients)->getLogicalOperatorType(), LPFunctionTransformationType::IndicatorVariableType::And);
      OPENGM_TEST_EQUAL(std::distance((constraints3[0].indicatorVariablesBegin() + (numVariables * shape[0]))->begin(), (constraints3[0].indicatorVariablesBegin() + (numVariables * shape[0]))->end()), 1);
      OPENGM_TEST_EQUAL((constraints3[0].indicatorVariablesBegin() + (numVariables * shape[0]))->begin()->first, numVariables);
      OPENGM_TEST_EQUAL((constraints3[0].indicatorVariablesBegin() + (numVariables * shape[0]))->begin()->second, 0);
      OPENGM_TEST_EQUAL((constraints3[0].indicatorVariablesBegin() + (numVariables * shape[0]))->getLogicalOperatorType(), LPFunctionTransformationType::IndicatorVariableType::And);
      OPENGM_TEST_EQUAL(std::distance((constraints4[0].indicatorVariablesBegin() + (numVariables * shape[0]))->begin(), (constraints4[0].indicatorVariablesBegin() + (numVariables * shape[0]))->end()), 1);
      OPENGM_TEST_EQUAL((constraints4[0].indicatorVariablesBegin() + (numVariables * shape[0]))->begin()->first, numVariables);
      OPENGM_TEST_EQUAL((constraints4[0].indicatorVariablesBegin() + (numVariables * shape[0]))->begin()->second, 0);
      OPENGM_TEST_EQUAL((constraints4[0].indicatorVariablesBegin() + (numVariables * shape[0]))->getLogicalOperatorType(), LPFunctionTransformationType::IndicatorVariableType::And);

      OPENGM_TEST_EQUAL(std::distance((constraints1[1].indicatorVariablesBegin() + numCoefficients)->begin(), (constraints1[1].indicatorVariablesBegin() + numCoefficients)->end()), 1);
      OPENGM_TEST_EQUAL((constraints1[1].indicatorVariablesBegin() + numCoefficients)->begin()->first, numVariables);
      OPENGM_TEST_EQUAL((constraints1[1].indicatorVariablesBegin() + numCoefficients)->begin()->second, 0);
      OPENGM_TEST_EQUAL((constraints1[1].indicatorVariablesBegin() + numCoefficients)->getLogicalOperatorType(), LPFunctionTransformationType::IndicatorVariableType::And);
      OPENGM_TEST_EQUAL(std::distance((constraints2[1].indicatorVariablesBegin() + numCoefficients)->begin(), (constraints2[1].indicatorVariablesBegin() + numCoefficients)->end()), 1);
      OPENGM_TEST_EQUAL((constraints2[1].indicatorVariablesBegin() + numCoefficients)->begin()->first, numVariables);
      OPENGM_TEST_EQUAL((constraints2[1].indicatorVariablesBegin() + numCoefficients)->begin()->second, 0);
      OPENGM_TEST_EQUAL((constraints2[1].indicatorVariablesBegin() + numCoefficients)->getLogicalOperatorType(), LPFunctionTransformationType::IndicatorVariableType::And);
      OPENGM_TEST_EQUAL(std::distance((constraints3[1].indicatorVariablesBegin() + (numVariables * shape[0]))->begin(), (constraints3[1].indicatorVariablesBegin() + (numVariables * shape[0]))->end()), 1);
      OPENGM_TEST_EQUAL((constraints3[1].indicatorVariablesBegin() + (numVariables * shape[0]))->begin()->first, numVariables);
      OPENGM_TEST_EQUAL((constraints3[1].indicatorVariablesBegin() + (numVariables * shape[0]))->begin()->second, 0);
      OPENGM_TEST_EQUAL((constraints3[1].indicatorVariablesBegin() + (numVariables * shape[0]))->getLogicalOperatorType(), LPFunctionTransformationType::IndicatorVariableType::And);
      OPENGM_TEST_EQUAL(std::distance((constraints4[1].indicatorVariablesBegin() + (numVariables * shape[0]))->begin(), (constraints4[1].indicatorVariablesBegin() + (numVariables * shape[0]))->end()), 1);
      OPENGM_TEST_EQUAL((constraints4[1].indicatorVariablesBegin() + (numVariables * shape[0]))->begin()->first, numVariables);
      OPENGM_TEST_EQUAL((constraints4[1].indicatorVariablesBegin() + (numVariables * shape[0]))->begin()->second, 0);
      OPENGM_TEST_EQUAL((constraints4[1].indicatorVariablesBegin() + (numVariables * shape[0]))->getLogicalOperatorType(), LPFunctionTransformationType::IndicatorVariableType::And);
   } catch(opengm::RuntimeError& error) {
      catchGetLinearConstraintsError = true;
   }
   OPENGM_TEST_EQUAL(catchGetLinearConstraintsError, false);
}

void testLabelCostFunction() {
   typedef double ValueType;
   typedef size_t IndexType;
   typedef size_t LabelType;

   typedef opengm::LPFunctionTransfer<ValueType, IndexType, LabelType> LPFunctionTransformationType;
   typedef opengm::LabelCostFunction<ValueType, IndexType, LabelType>        LabelCostFunctionType;

   const IndexType numVariables = 15;
   const LabelType minNumLabels = 1;
   const LabelType maxNumLabels = 10;
   const ValueType minCostValue = 0.0;
   const ValueType maxCostValue = 1.0;

   typedef opengm::RandomUniformInteger<LabelType> RandomUniformLabelType;
   RandomUniformLabelType labelGenerator(minNumLabels, maxNumLabels + 1);

   typedef opengm::RandomUniformFloatingPoint<double> RandomUniformValueType;
   RandomUniformValueType costsGenerator(minCostValue, maxCostValue);

   // create functions
   LabelType currentMaxNumLabels = 0;
   std::vector<LabelType> shape(numVariables);
   for(IndexType i = 0; i < numVariables; ++i) {
      shape[i] = labelGenerator();
      if(shape[i] > currentMaxNumLabels) {
         currentMaxNumLabels = shape[i];
      }
   }

   LabelType numZeroCosts = 0;
   std::vector<ValueType> costs(currentMaxNumLabels);
   for(LabelType i = 0; i < currentMaxNumLabels; ++i) {
      costs[i] = costsGenerator();
      if(costs[i] == 0.0) {
         ++numZeroCosts;
      }
   }

   // ensure that at least one label(!= 0) cost is zero to check if zero cost is treated correctly
   if(numZeroCosts == 0) {
      LabelType zeroCostLabel = 0;
      while(zeroCostLabel == 0 || zeroCostLabel >= currentMaxNumLabels) {
         zeroCostLabel = labelGenerator() - 1;
      }
      ++numZeroCosts;
      costs[zeroCostLabel] = 0.0;
   }

   LabelCostFunctionType SameNumLabelsSingleLabelCostFunction(numVariables, currentMaxNumLabels, currentMaxNumLabels - 1, costs[0]);
   LabelCostFunctionType SameNumLabelsMultipleLabelCostFunction(numVariables, currentMaxNumLabels, costs.begin(), costs.end());
   LabelCostFunctionType DifferentNumLabelsSingleLabelCostFunction(shape.begin(), shape.end(), currentMaxNumLabels - 1, costs[0]);
   LabelCostFunctionType DifferentNumLabelsMultipleLabelCostFunction(shape.begin(), shape.end(), costs.begin(), costs.end());

   OPENGM_TEST_EQUAL(LPFunctionTransformationType::isTransferable<LabelCostFunctionType>(), true);

   bool catchNumSlackVariablesError = false;
   try {
      OPENGM_TEST_EQUAL(LPFunctionTransformationType::numSlackVariables(SameNumLabelsSingleLabelCostFunction), 1);
      OPENGM_TEST_EQUAL(LPFunctionTransformationType::numSlackVariables(SameNumLabelsMultipleLabelCostFunction), currentMaxNumLabels - numZeroCosts);
      OPENGM_TEST_EQUAL(LPFunctionTransformationType::numSlackVariables(DifferentNumLabelsSingleLabelCostFunction), 1);
      OPENGM_TEST_EQUAL(LPFunctionTransformationType::numSlackVariables(DifferentNumLabelsMultipleLabelCostFunction), currentMaxNumLabels - numZeroCosts);
   } catch(opengm::RuntimeError& error) {
      catchNumSlackVariablesError = true;
   }
   OPENGM_TEST_EQUAL(catchNumSlackVariablesError, false);

   bool catchGetSlackVariablesOrderError = false;
   try {
      LPFunctionTransformationType::IndicatorVariablesContainerType order;

      {
         LPFunctionTransformationType::getSlackVariablesOrder(SameNumLabelsSingleLabelCostFunction, order);
         OPENGM_TEST_EQUAL(order.size(), 1);
         OPENGM_TEST_EQUAL(std::distance(order[0].begin(), order[0].end()), 1);
         OPENGM_TEST_EQUAL(order[0].begin()->first, numVariables);
         OPENGM_TEST_EQUAL(order[0].begin()->second, 0);
         OPENGM_TEST_EQUAL(order[0].getLogicalOperatorType(), LPFunctionTransformationType::IndicatorVariableType::And);
      }

      {
         LPFunctionTransformationType::getSlackVariablesOrder(SameNumLabelsMultipleLabelCostFunction, order);
         OPENGM_TEST_EQUAL(order.size(), currentMaxNumLabels - numZeroCosts);
         for(size_t i = 0; i < order.size(); ++i) {
            OPENGM_TEST_EQUAL(std::distance(order[i].begin(), order[i].end()), 1);
            OPENGM_TEST_EQUAL(order[i].begin()->first, numVariables + i);
            OPENGM_TEST_EQUAL(order[i].begin()->second, 0);
            OPENGM_TEST_EQUAL(order[i].getLogicalOperatorType(), LPFunctionTransformationType::IndicatorVariableType::And);
         }
      }

      {
         LPFunctionTransformationType::getSlackVariablesOrder(DifferentNumLabelsSingleLabelCostFunction, order);
         OPENGM_TEST_EQUAL(order.size(), 1);
         OPENGM_TEST_EQUAL(std::distance(order[0].begin(), order[0].end()), 1);
         OPENGM_TEST_EQUAL(order[0].begin()->first, numVariables);
         OPENGM_TEST_EQUAL(order[0].begin()->second, 0);
         OPENGM_TEST_EQUAL(order[0].getLogicalOperatorType(), LPFunctionTransformationType::IndicatorVariableType::And);
      }

      {
         LPFunctionTransformationType::getSlackVariablesOrder(DifferentNumLabelsMultipleLabelCostFunction, order);
         OPENGM_TEST_EQUAL(order.size(), currentMaxNumLabels - numZeroCosts);
         for(size_t i = 0; i < order.size(); ++i) {
            OPENGM_TEST_EQUAL(std::distance(order[i].begin(), order[i].end()), 1);
            OPENGM_TEST_EQUAL(order[i].begin()->first, numVariables + i);
            OPENGM_TEST_EQUAL(order[i].begin()->second, 0);
            OPENGM_TEST_EQUAL(order[i].getLogicalOperatorType(), LPFunctionTransformationType::IndicatorVariableType::And);
         }
      }
   } catch(opengm::RuntimeError& error) {
      catchGetSlackVariablesOrderError = true;
   }
   OPENGM_TEST_EQUAL(catchGetSlackVariablesOrderError, false);

   bool catchGetSlackVariablesObjectiveCoefficientsError = false;
   try {
      LPFunctionTransformationType::SlackVariablesObjectiveCoefficientsContainerType slackVariablesObjectiveCoefficients;

      {
         LPFunctionTransformationType::getSlackVariablesObjectiveCoefficients(SameNumLabelsSingleLabelCostFunction, slackVariablesObjectiveCoefficients);
         OPENGM_TEST_EQUAL(slackVariablesObjectiveCoefficients.size(), 1);
         OPENGM_TEST_EQUAL(slackVariablesObjectiveCoefficients[0], costs[0]);
      }

      {
         LPFunctionTransformationType::getSlackVariablesObjectiveCoefficients(SameNumLabelsMultipleLabelCostFunction, slackVariablesObjectiveCoefficients);
         OPENGM_TEST_EQUAL(slackVariablesObjectiveCoefficients.size(), currentMaxNumLabels - numZeroCosts);
         LabelType currentSlackVariablesObjectiveCoefficientsPosition = 0;
         for(LabelType i = 0; i < currentMaxNumLabels; ++i) {
            if(costs[i] != 0.0) {
               OPENGM_TEST_EQUAL(slackVariablesObjectiveCoefficients[currentSlackVariablesObjectiveCoefficientsPosition], costs[i]);
               ++currentSlackVariablesObjectiveCoefficientsPosition;
            }
         }
         OPENGM_TEST_EQUAL(currentSlackVariablesObjectiveCoefficientsPosition, currentMaxNumLabels - numZeroCosts);
      }

      {
         LPFunctionTransformationType::getSlackVariablesObjectiveCoefficients(DifferentNumLabelsSingleLabelCostFunction, slackVariablesObjectiveCoefficients);
         OPENGM_TEST_EQUAL(slackVariablesObjectiveCoefficients.size(), 1);
         OPENGM_TEST_EQUAL(slackVariablesObjectiveCoefficients[0], costs[0]);
      }

      {
         LPFunctionTransformationType::getSlackVariablesObjectiveCoefficients(DifferentNumLabelsMultipleLabelCostFunction, slackVariablesObjectiveCoefficients);
         OPENGM_TEST_EQUAL(slackVariablesObjectiveCoefficients.size(), currentMaxNumLabels - numZeroCosts);
         LabelType currentSlackVariablesObjectiveCoefficientsPosition = 0;
         for(LabelType i = 0; i < currentMaxNumLabels; ++i) {
            if(costs[i] != 0.0) {
               OPENGM_TEST_EQUAL(slackVariablesObjectiveCoefficients[currentSlackVariablesObjectiveCoefficientsPosition], costs[i]);
               ++currentSlackVariablesObjectiveCoefficientsPosition;
            }
         }
         OPENGM_TEST_EQUAL(currentSlackVariablesObjectiveCoefficientsPosition, currentMaxNumLabels - numZeroCosts);
      }
   } catch(opengm::RuntimeError& error) {
      catchGetSlackVariablesObjectiveCoefficientsError = true;
   }
   OPENGM_TEST_EQUAL(catchGetSlackVariablesObjectiveCoefficientsError, false);

   bool catchGetIndicatorVariablesError = false;
   try {
      LPFunctionTransformationType::IndicatorVariablesContainerType variables;

      {
         LPFunctionTransformationType::getIndicatorVariables(SameNumLabelsSingleLabelCostFunction, variables);
         OPENGM_TEST_EQUAL(variables.size(), 2);
         OPENGM_TEST_EQUAL(std::distance(variables[0].begin(), variables[0].end()), numVariables);
         OPENGM_TEST_EQUAL(variables[0].getLogicalOperatorType(), LPFunctionTransformationType::IndicatorVariableType::Or);
         for(IndexType i = 0; i < numVariables; ++i) {
            OPENGM_TEST_EQUAL((variables[0].begin() + i)->first, i);
            OPENGM_TEST_EQUAL((variables[0].begin() + i)->second, currentMaxNumLabels - 1);
         }
         OPENGM_TEST_EQUAL(std::distance(variables[1].begin(), variables[1].end()), 1);
         OPENGM_TEST_EQUAL(variables[1].begin()->first, numVariables);
         OPENGM_TEST_EQUAL(variables[1].begin()->second, 0);
         OPENGM_TEST_EQUAL(variables[1].getLogicalOperatorType(), LPFunctionTransformationType::IndicatorVariableType::And);
      }

      {
         LPFunctionTransformationType::getIndicatorVariables(SameNumLabelsMultipleLabelCostFunction, variables);
         LabelType currentLabel = 0;
         OPENGM_TEST_EQUAL(variables.size(), (currentMaxNumLabels - numZeroCosts) * 2);
         for(size_t currentIndicatorVariable = 0; currentIndicatorVariable < variables.size(); ++currentIndicatorVariable) {
            if(currentIndicatorVariable % 2 == 0) {
               OPENGM_TEST_EQUAL(std::distance(variables[currentIndicatorVariable].begin(), variables[currentIndicatorVariable].end()), numVariables);
               OPENGM_TEST_EQUAL(variables[currentIndicatorVariable].getLogicalOperatorType(), LPFunctionTransformationType::IndicatorVariableType::Or);
               for(IndexType i = 0; i < numVariables; ++i) {
                  OPENGM_TEST_EQUAL((variables[currentIndicatorVariable].begin() + i)->first, i);
                  OPENGM_TEST_EQUAL((variables[currentIndicatorVariable].begin() + i)->second, currentLabel);
               }
               ++currentLabel;
               while((currentLabel < maxNumLabels) && (costs[currentLabel] == 0.0)) {
                  ++currentLabel;
               }
            } else {
               OPENGM_TEST_EQUAL(std::distance(variables[currentIndicatorVariable].begin(), variables[currentIndicatorVariable].end()), 1);
               OPENGM_TEST_EQUAL(variables[currentIndicatorVariable].begin()->first, numVariables + (currentIndicatorVariable / 2));
               OPENGM_TEST_EQUAL(variables[currentIndicatorVariable].begin()->second, 0);
               OPENGM_TEST_EQUAL(variables[currentIndicatorVariable].getLogicalOperatorType(), LPFunctionTransformationType::IndicatorVariableType::And);
            }
         }
      }

      {
         LPFunctionTransformationType::getIndicatorVariables(DifferentNumLabelsSingleLabelCostFunction, variables);
         OPENGM_TEST_EQUAL(variables.size(), 2);
         std::vector<IndexType> expectedVariables;
         for(IndexType i = 0; i < numVariables; ++i) {
            if(shape[i] > currentMaxNumLabels - 1) {
               expectedVariables.push_back(i);
            }
         }
         OPENGM_TEST_EQUAL(static_cast<size_t>(std::distance(variables[0].begin(), variables[0].end())), expectedVariables.size());
         OPENGM_TEST_EQUAL(variables[0].getLogicalOperatorType(), LPFunctionTransformationType::IndicatorVariableType::Or);
         for(IndexType i = 0; i < expectedVariables.size(); ++i) {
            OPENGM_TEST_EQUAL((variables[0].begin() + i)->first, expectedVariables[i]);
            OPENGM_TEST_EQUAL((variables[0].begin() + i)->second, currentMaxNumLabels - 1);
         }
         OPENGM_TEST_EQUAL(std::distance(variables[1].begin(), variables[1].end()), 1);
         OPENGM_TEST_EQUAL(variables[1].begin()->first, numVariables);
         OPENGM_TEST_EQUAL(variables[1].begin()->second, 0);
         OPENGM_TEST_EQUAL(variables[1].getLogicalOperatorType(), LPFunctionTransformationType::IndicatorVariableType::And);
      }

      {
         LPFunctionTransformationType::getIndicatorVariables(DifferentNumLabelsMultipleLabelCostFunction, variables);
         LabelType currentLabel = 0;
         OPENGM_TEST_EQUAL(variables.size(), (currentMaxNumLabels - numZeroCosts) * 2);
         for(size_t currentIndicatorVariable = 0; currentIndicatorVariable < variables.size(); ++currentIndicatorVariable) {
            if(currentIndicatorVariable % 2 == 0) {
               std::vector<IndexType> expectedVariables;
               for(IndexType i = 0; i < numVariables; ++i) {
                  if(shape[i] > currentLabel) {
                     expectedVariables.push_back(i);
                  }
               }
               OPENGM_TEST_EQUAL(static_cast<size_t>(std::distance(variables[currentIndicatorVariable].begin(), variables[currentIndicatorVariable].end())), expectedVariables.size());
               OPENGM_TEST_EQUAL(variables[currentIndicatorVariable].getLogicalOperatorType(), LPFunctionTransformationType::IndicatorVariableType::Or);
               for(IndexType i = 0; i < expectedVariables.size(); ++i) {
                  OPENGM_TEST_EQUAL((variables[currentIndicatorVariable].begin() + i)->first, expectedVariables[i]);
                  OPENGM_TEST_EQUAL((variables[currentIndicatorVariable].begin() + i)->second, currentLabel);
               }
               ++currentLabel;
               while((currentLabel < maxNumLabels) && (costs[currentLabel] == 0.0)) {
                  ++currentLabel;
               }
            } else {
               OPENGM_TEST_EQUAL(std::distance(variables[currentIndicatorVariable].begin(), variables[currentIndicatorVariable].end()), 1);
               OPENGM_TEST_EQUAL(variables[currentIndicatorVariable].begin()->first, numVariables + (currentIndicatorVariable / 2));
               OPENGM_TEST_EQUAL(variables[currentIndicatorVariable].begin()->second, 0);
               OPENGM_TEST_EQUAL(variables[currentIndicatorVariable].getLogicalOperatorType(), LPFunctionTransformationType::IndicatorVariableType::And);
            }
         }
      }
   } catch(opengm::RuntimeError& error) {
      catchGetIndicatorVariablesError = true;
   }
   OPENGM_TEST_EQUAL(catchGetIndicatorVariablesError, false);

   bool catchGetLinearConstraintsError = false;
   try {
      LPFunctionTransformationType::LinearConstraintsContainerType constraints;

      {
         LPFunctionTransformationType::getLinearConstraints(SameNumLabelsSingleLabelCostFunction, constraints);
         OPENGM_TEST_EQUAL(constraints.size(), 1);
         OPENGM_TEST_EQUAL(constraints[0].getBound(), 0.0);
         OPENGM_TEST_EQUAL(constraints[0].getConstraintOperator(), LPFunctionTransformationType::LinearConstraintType::LinearConstraintOperatorType::Equal);

         OPENGM_TEST_EQUAL(static_cast<size_t>(std::distance(constraints[0].coefficientsBegin(), constraints[0].coefficientsEnd())), 2);
         OPENGM_TEST_EQUAL(*(constraints[0].coefficientsBegin()), 1.0);
         OPENGM_TEST_EQUAL(*(constraints[0].coefficientsBegin() + 1), -1.0);

         OPENGM_TEST_EQUAL(static_cast<size_t>(std::distance(constraints[0].indicatorVariablesBegin(), constraints[0].indicatorVariablesEnd())), 2);

         const LPFunctionTransformationType::IndicatorVariableType& orVariable = *(constraints[0].indicatorVariablesBegin());
         const LPFunctionTransformationType::IndicatorVariableType& slackVariable = *(constraints[0].indicatorVariablesBegin() + 1);

         OPENGM_TEST_EQUAL(std::distance(orVariable.begin(), orVariable.end()), numVariables);
         OPENGM_TEST_EQUAL(orVariable.getLogicalOperatorType(), LPFunctionTransformationType::IndicatorVariableType::Or);
         for(IndexType i = 0; i < numVariables; ++i) {
            OPENGM_TEST_EQUAL((orVariable.begin() + i)->first, i);
            OPENGM_TEST_EQUAL((orVariable.begin() + i)->second, currentMaxNumLabels - 1);
         }
         OPENGM_TEST_EQUAL(std::distance(slackVariable.begin(), slackVariable.end()), 1);
         OPENGM_TEST_EQUAL(slackVariable.begin()->first, numVariables);
         OPENGM_TEST_EQUAL(slackVariable.begin()->second, 0);
         OPENGM_TEST_EQUAL(slackVariable.getLogicalOperatorType(), LPFunctionTransformationType::IndicatorVariableType::And);
      }

      {
         LPFunctionTransformationType::getLinearConstraints(SameNumLabelsMultipleLabelCostFunction, constraints);
         OPENGM_TEST_EQUAL(constraints.size(), currentMaxNumLabels - numZeroCosts);
         LabelType currentLabel = 0;
         for(size_t currentConstraint = 0; currentConstraint < constraints.size(); ++currentConstraint) {
            OPENGM_TEST_EQUAL(constraints[currentConstraint].getBound(), 0.0);
            OPENGM_TEST_EQUAL(constraints[currentConstraint].getConstraintOperator(), LPFunctionTransformationType::LinearConstraintType::LinearConstraintOperatorType::Equal);

            OPENGM_TEST_EQUAL(static_cast<size_t>(std::distance(constraints[currentConstraint].coefficientsBegin(), constraints[currentConstraint].coefficientsEnd())), 2);
            OPENGM_TEST_EQUAL(*(constraints[currentConstraint].coefficientsBegin()), 1.0);
            OPENGM_TEST_EQUAL(*(constraints[currentConstraint].coefficientsBegin() + 1), -1.0);

            OPENGM_TEST_EQUAL(static_cast<size_t>(std::distance(constraints[currentConstraint].indicatorVariablesBegin(), constraints[currentConstraint].indicatorVariablesEnd())), 2);
            const LPFunctionTransformationType::IndicatorVariableType& currentOrVariable = *(constraints[currentConstraint].indicatorVariablesBegin());
            const LPFunctionTransformationType::IndicatorVariableType& currentSlackVariable = *(constraints[currentConstraint].indicatorVariablesBegin() + 1);

            OPENGM_TEST_EQUAL(std::distance(currentOrVariable.begin(), currentOrVariable.end()), numVariables);
            OPENGM_TEST_EQUAL(currentOrVariable.getLogicalOperatorType(), LPFunctionTransformationType::IndicatorVariableType::Or);
            for(IndexType i = 0; i < numVariables; ++i) {
               OPENGM_TEST_EQUAL((currentOrVariable.begin() + i)->first, i);
               OPENGM_TEST_EQUAL((currentOrVariable.begin() + i)->second, currentLabel);
            }
            ++currentLabel;
            while((currentLabel < maxNumLabels) && (costs[currentLabel] == 0.0)) {
               ++currentLabel;
            }

            OPENGM_TEST_EQUAL(std::distance(currentSlackVariable.begin(), currentSlackVariable.end()), 1);
            OPENGM_TEST_EQUAL(currentSlackVariable.begin()->first, numVariables + currentConstraint);
            OPENGM_TEST_EQUAL(currentSlackVariable.begin()->second, 0);
            OPENGM_TEST_EQUAL(currentSlackVariable.getLogicalOperatorType(), LPFunctionTransformationType::IndicatorVariableType::And);
         }
      }

      {
         LPFunctionTransformationType::getLinearConstraints(DifferentNumLabelsSingleLabelCostFunction, constraints);
         OPENGM_TEST_EQUAL(constraints.size(), 1);
         OPENGM_TEST_EQUAL(constraints[0].getBound(), 0.0);
         OPENGM_TEST_EQUAL(constraints[0].getConstraintOperator(), LPFunctionTransformationType::LinearConstraintType::LinearConstraintOperatorType::Equal);

         OPENGM_TEST_EQUAL(static_cast<size_t>(std::distance(constraints[0].coefficientsBegin(), constraints[0].coefficientsEnd())), 2);
         OPENGM_TEST_EQUAL(*(constraints[0].coefficientsBegin()), 1.0);
         OPENGM_TEST_EQUAL(*(constraints[0].coefficientsBegin() + 1), -1.0);

         OPENGM_TEST_EQUAL(static_cast<size_t>(std::distance(constraints[0].indicatorVariablesBegin(), constraints[0].indicatorVariablesEnd())), 2);

         const LPFunctionTransformationType::IndicatorVariableType& orVariable = *(constraints[0].indicatorVariablesBegin());
         const LPFunctionTransformationType::IndicatorVariableType& slackVariable = *(constraints[0].indicatorVariablesBegin() + 1);

         std::vector<IndexType> expectedVariables;
         for(IndexType i = 0; i < numVariables; ++i) {
            if(shape[i] > currentMaxNumLabels - 1) {
               expectedVariables.push_back(i);
            }
         }
         OPENGM_TEST_EQUAL(static_cast<size_t>(std::distance(orVariable.begin(), orVariable.end())), expectedVariables.size());
         OPENGM_TEST_EQUAL(orVariable.getLogicalOperatorType(), LPFunctionTransformationType::IndicatorVariableType::Or);

         for(IndexType i = 0; i < expectedVariables.size(); ++i) {
            OPENGM_TEST_EQUAL((orVariable.begin() + i)->first, expectedVariables[i]);
            OPENGM_TEST_EQUAL((orVariable.begin() + i)->second, currentMaxNumLabels - 1);
         }
         OPENGM_TEST_EQUAL(std::distance(slackVariable.begin(), slackVariable.end()), 1);
         OPENGM_TEST_EQUAL(slackVariable.begin()->first, numVariables);
         OPENGM_TEST_EQUAL(slackVariable.begin()->second, 0);
         OPENGM_TEST_EQUAL(slackVariable.getLogicalOperatorType(), LPFunctionTransformationType::IndicatorVariableType::And);
      }

      {
         LPFunctionTransformationType::getLinearConstraints(DifferentNumLabelsMultipleLabelCostFunction, constraints);
         OPENGM_TEST_EQUAL(constraints.size(), currentMaxNumLabels - numZeroCosts);
         LabelType currentLabel = 0;
         for(size_t currentConstraint = 0; currentConstraint < constraints.size(); ++currentConstraint) {
            OPENGM_TEST_EQUAL(constraints[currentConstraint].getBound(), 0.0);
            OPENGM_TEST_EQUAL(constraints[currentConstraint].getConstraintOperator(), LPFunctionTransformationType::LinearConstraintType::LinearConstraintOperatorType::Equal);

            OPENGM_TEST_EQUAL(static_cast<size_t>(std::distance(constraints[currentConstraint].coefficientsBegin(), constraints[currentConstraint].coefficientsEnd())), 2);
            OPENGM_TEST_EQUAL(*(constraints[currentConstraint].coefficientsBegin()), 1.0);
            OPENGM_TEST_EQUAL(*(constraints[currentConstraint].coefficientsBegin() + 1), -1.0);

            OPENGM_TEST_EQUAL(static_cast<size_t>(std::distance(constraints[currentConstraint].indicatorVariablesBegin(), constraints[currentConstraint].indicatorVariablesEnd())), 2);
            const LPFunctionTransformationType::IndicatorVariableType& currentOrVariable = *(constraints[currentConstraint].indicatorVariablesBegin());
            const LPFunctionTransformationType::IndicatorVariableType& currentSlackVariable = *(constraints[currentConstraint].indicatorVariablesBegin() + 1);

            std::vector<IndexType> expectedVariables;
            for(IndexType i = 0; i < numVariables; ++i) {
               if(shape[i] > currentLabel) {
                  expectedVariables.push_back(i);
               }
            }
            OPENGM_TEST_EQUAL(static_cast<size_t>(std::distance(currentOrVariable.begin(), currentOrVariable.end())), expectedVariables.size());
            OPENGM_TEST_EQUAL(currentOrVariable.getLogicalOperatorType(), LPFunctionTransformationType::IndicatorVariableType::Or);
            for(IndexType i = 0; i < expectedVariables.size(); ++i) {
               OPENGM_TEST_EQUAL((currentOrVariable.begin() + i)->first, expectedVariables[i]);
               OPENGM_TEST_EQUAL((currentOrVariable.begin() + i)->second, currentLabel);
            }
            ++currentLabel;
            while((currentLabel < maxNumLabels) && (costs[currentLabel] == 0.0)) {
               ++currentLabel;
            }

            OPENGM_TEST_EQUAL(std::distance(currentSlackVariable.begin(), currentSlackVariable.end()), 1);
            OPENGM_TEST_EQUAL(currentSlackVariable.begin()->first, numVariables + currentConstraint);
            OPENGM_TEST_EQUAL(currentSlackVariable.begin()->second, 0);
            OPENGM_TEST_EQUAL(currentSlackVariable.getLogicalOperatorType(), LPFunctionTransformationType::IndicatorVariableType::And);
         }
      }
   } catch(opengm::RuntimeError& error) {
      catchGetLinearConstraintsError = true;
   }
   OPENGM_TEST_EQUAL(catchGetLinearConstraintsError, false);
}
