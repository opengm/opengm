#include <iostream>

#include <opengm/unittests/test.hxx>

#include <opengm/datastructures/linear_constraint.hxx>

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
void testLinearConstraint();

int main(int argc, char** argv){
   std::cout << "Linear Constraint test... " << std::endl;

   testLinearConstraint<double, size_t, size_t>();
   testLinearConstraint<double, size_t, int>();
   testLinearConstraint<double, size_t, char>();

   testLinearConstraint<double, int, size_t>();
   testLinearConstraint<double, int, int>();
   testLinearConstraint<double, int, char>();

   testLinearConstraint<double, unsigned char, size_t>();
   testLinearConstraint<double, unsigned char, int>();
   testLinearConstraint<double, unsigned char, char>();

   testLinearConstraint<int, size_t, size_t>();
   testLinearConstraint<int, size_t, int>();
   testLinearConstraint<int, size_t, char>();

   testLinearConstraint<int, int, size_t>();
   testLinearConstraint<int, int, int>();
   testLinearConstraint<int, int, char>();

   testLinearConstraint<int, unsigned char, size_t>();
   testLinearConstraint<int, unsigned char, int>();
   testLinearConstraint<int, unsigned char, char>();

   std::cout << "done..." << std::endl;
   return 0;
   
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
void testLinearConstraint() {
   typedef VALUE_TYPE ValueType;
   typedef INDEX_TYPE IndexType;
   typedef LABEL_TYPE LabelType;

   typedef opengm::LinearConstraint<ValueType, IndexType, LabelType> LinearConstraintType;

   typedef typename LinearConstraintType::IndicatorVariableType             IndicatorVariableType;
   typedef typename LinearConstraintType::IndicatorVariablesContainerType   IndicatorVariablesContainerType;
   typedef typename LinearConstraintType::CoefficientsContainerType         CoefficientsContainerType;
   typedef typename LinearConstraintType::BoundType                         BoundType;
   typedef typename LinearConstraintType::LinearConstraintOperatorType      LinearConstraintOperatorType;

   // construction
   const LinearConstraintType emptyLinearConstraint1;
   LinearConstraintType       emptyLinearConstraint2(emptyLinearConstraint1);
   IndicatorVariablesContainerType IndicatorVariablesAnd(3);
   IndicatorVariablesAnd[0].setLogicalOperatorType(IndicatorVariableType::And);
   IndicatorVariablesAnd[1].setLogicalOperatorType(IndicatorVariableType::And);
   IndicatorVariablesAnd[2].setLogicalOperatorType(IndicatorVariableType::And);
   IndicatorVariablesAnd[0].add(IndexType(0), LabelType(0));
   IndicatorVariablesAnd[1].add(IndexType(0), LabelType(0));
   IndicatorVariablesAnd[1].add(IndexType(1), LabelType(1));
   IndicatorVariablesAnd[2].add(IndexType(0), LabelType(0));
   IndicatorVariablesAnd[2].add(IndexType(1), LabelType(1));
   IndicatorVariablesAnd[2].add(IndexType(2), LabelType(2));

   IndicatorVariablesContainerType IndicatorVariablesOr(3);
   IndicatorVariablesOr[0].setLogicalOperatorType(IndicatorVariableType::Or);
   IndicatorVariablesOr[1].setLogicalOperatorType(IndicatorVariableType::Or);
   IndicatorVariablesOr[2].setLogicalOperatorType(IndicatorVariableType::Or);
   IndicatorVariablesOr[0].add(IndexType(0), LabelType(0));
   IndicatorVariablesOr[1].add(IndexType(0), LabelType(0));
   IndicatorVariablesOr[1].add(IndexType(1), LabelType(1));
   IndicatorVariablesOr[2].add(IndexType(0), LabelType(0));
   IndicatorVariablesOr[2].add(IndexType(1), LabelType(1));
   IndicatorVariablesOr[2].add(IndexType(2), LabelType(2));

   IndicatorVariablesContainerType IndicatorVariablesNot(3);
   IndicatorVariablesNot[0].setLogicalOperatorType(IndicatorVariableType::Not);
   IndicatorVariablesNot[1].setLogicalOperatorType(IndicatorVariableType::Not);
   IndicatorVariablesNot[2].setLogicalOperatorType(IndicatorVariableType::Not);
   IndicatorVariablesNot[0].add(IndexType(0), LabelType(0));
   IndicatorVariablesNot[1].add(IndexType(0), LabelType(0));
   IndicatorVariablesNot[1].add(IndexType(1), LabelType(1));
   IndicatorVariablesNot[2].add(IndexType(0), LabelType(0));
   IndicatorVariablesNot[2].add(IndexType(1), LabelType(1));
   IndicatorVariablesNot[2].add(IndexType(2), LabelType(2));

   CoefficientsContainerType coefficients;
   coefficients.push_back(ValueType(1.0));
   coefficients.push_back(ValueType(2.0));
   coefficients.push_back(ValueType(4.0));

   const BoundType bound(0.5);

   LinearConstraintType LinearConstraintAndLessEqual1(IndicatorVariablesAnd, coefficients, bound, LinearConstraintOperatorType::LessEqual);
   LinearConstraintType LinearConstraintAndLessEqual2(IndicatorVariablesAnd.begin(), IndicatorVariablesAnd.begin() + 1, coefficients.begin(), bound, LinearConstraintOperatorType::LessEqual);
   LinearConstraintType LinearConstraintAndEqual1(IndicatorVariablesAnd, coefficients, bound, LinearConstraintOperatorType::Equal);
   LinearConstraintType LinearConstraintAndEqual2(IndicatorVariablesAnd.begin(), IndicatorVariablesAnd.begin() + 1, coefficients.begin(), bound, LinearConstraintOperatorType::Equal);
   LinearConstraintType LinearConstraintAndGreaterEqual1(IndicatorVariablesAnd, coefficients, bound, LinearConstraintOperatorType::GreaterEqual);
   LinearConstraintType LinearConstraintAndGreaterEqual2(IndicatorVariablesAnd.begin(), IndicatorVariablesAnd.begin() + 1, coefficients.begin(), bound, LinearConstraintOperatorType::GreaterEqual);

   LinearConstraintType LinearConstraintOrLessEqual1(IndicatorVariablesOr, coefficients, bound, LinearConstraintOperatorType::LessEqual);
   LinearConstraintType LinearConstraintOrLessEqual2(IndicatorVariablesOr.begin(), IndicatorVariablesOr.begin() + 1, coefficients.begin(), bound, LinearConstraintOperatorType::LessEqual);
   LinearConstraintType LinearConstraintOrEqual1(IndicatorVariablesOr, coefficients, bound, LinearConstraintOperatorType::Equal);
   LinearConstraintType LinearConstraintOrEqual2(IndicatorVariablesOr.begin(), IndicatorVariablesOr.begin() + 1, coefficients.begin(), bound, LinearConstraintOperatorType::Equal);
   LinearConstraintType LinearConstraintOrGreaterEqual1(IndicatorVariablesOr, coefficients, bound, LinearConstraintOperatorType::GreaterEqual);
   LinearConstraintType LinearConstraintOrGreaterEqual2(IndicatorVariablesOr.begin(), IndicatorVariablesOr.begin() + 1, coefficients.begin(), bound, LinearConstraintOperatorType::GreaterEqual);

   LinearConstraintType LinearConstraintNotLessEqual1(IndicatorVariablesNot, coefficients, bound, LinearConstraintOperatorType::LessEqual);
   LinearConstraintType LinearConstraintNotLessEqual2(IndicatorVariablesNot.begin(), IndicatorVariablesNot.begin() + 1, coefficients.begin(), bound, LinearConstraintOperatorType::LessEqual);
   LinearConstraintType LinearConstraintNotEqual1(IndicatorVariablesNot, coefficients, bound, LinearConstraintOperatorType::Equal);
   LinearConstraintType LinearConstraintNotEqual2(IndicatorVariablesNot.begin(), IndicatorVariablesNot.begin() + 1, coefficients.begin(), bound, LinearConstraintOperatorType::Equal);
   LinearConstraintType LinearConstraintNotGreaterEqual1(IndicatorVariablesNot, coefficients, bound, LinearConstraintOperatorType::GreaterEqual);
   LinearConstraintType LinearConstraintNotGreaterEqual2(IndicatorVariablesNot.begin(), IndicatorVariablesNot.begin() + 1, coefficients.begin(), bound, LinearConstraintOperatorType::GreaterEqual);

   // reserve
   LinearConstraintAndLessEqual2.reserve(2);
   LinearConstraintAndEqual2.reserve(2);
   LinearConstraintAndGreaterEqual2.reserve(2);

   LinearConstraintOrLessEqual2.reserve(2);
   LinearConstraintOrEqual2.reserve(2);
   LinearConstraintOrGreaterEqual2.reserve(2);

   LinearConstraintNotLessEqual2.reserve(2);
   LinearConstraintNotEqual2.reserve(2);
   LinearConstraintNotGreaterEqual2.reserve(2);

   // add
   LinearConstraintAndLessEqual2.add(IndicatorVariablesAnd[1], coefficients[1]);
   LinearConstraintAndEqual2.add(IndicatorVariablesContainerType(1, IndicatorVariablesAnd[1]), CoefficientsContainerType(1, coefficients[1]));
   LinearConstraintAndGreaterEqual2.add(IndicatorVariablesAnd.begin() + 1, IndicatorVariablesAnd.begin() + 2, coefficients.begin() + 1);

   LinearConstraintOrLessEqual2.add(IndicatorVariablesOr[1], coefficients[1]);
   LinearConstraintOrEqual2.add(IndicatorVariablesContainerType(1, IndicatorVariablesOr[1]), CoefficientsContainerType(1, coefficients[1]));
   LinearConstraintOrGreaterEqual2.add(IndicatorVariablesOr.begin() + 1, IndicatorVariablesOr.begin() + 2, coefficients.begin() + 1);

   LinearConstraintNotLessEqual2.add(IndicatorVariablesNot[1], coefficients[1]);
   LinearConstraintNotEqual2.add(IndicatorVariablesContainerType(1, IndicatorVariablesNot[1]), CoefficientsContainerType(1, coefficients[1]));
   LinearConstraintNotGreaterEqual2.add(IndicatorVariablesNot.begin() + 1, IndicatorVariablesNot.begin() + 2, coefficients.begin() + 1);

   // set bound
   emptyLinearConstraint2.setBound(BoundType(-1.0));
   LinearConstraintAndEqual1.setBound(BoundType(coefficients[0]));
   LinearConstraintAndEqual2.setBound(BoundType(coefficients[0] + coefficients[1]));
   LinearConstraintOrEqual1.setBound(BoundType(coefficients[2]));
   LinearConstraintOrEqual2.setBound(BoundType(coefficients[0] + coefficients[1]));
   LinearConstraintNotEqual1.setBound(BoundType(coefficients[0]));
   LinearConstraintNotEqual2.setBound(BoundType(coefficients[0] + coefficients[1]));

   // set constraint operator
   emptyLinearConstraint2.setConstraintOperator(LinearConstraintOperatorType::GreaterEqual);

   // evaluate
   const LabelType labeling[] = {0, 1, 2,
                                 0, 1, 0,
                                 0, 0, 2,
                                 1, 1, 2,
                                 1, 0, 0,
                                 0, 0, 0,
                                 1, 0, 2};

   OPENGM_TEST_EQUAL(emptyLinearConstraint1(labeling), 0.0);
   OPENGM_TEST_EQUAL(emptyLinearConstraint2(labeling), 0.0);

   OPENGM_TEST_EQUAL(LinearConstraintAndLessEqual1(labeling + (4 * 3)), 0.0);
   OPENGM_TEST_EQUAL(LinearConstraintAndLessEqual1(labeling), coefficients[0] + coefficients[1] + coefficients[2] - bound);
   OPENGM_TEST_EQUAL(LinearConstraintAndLessEqual2(labeling + (4 * 3)), 0.0);
   OPENGM_TEST_EQUAL(LinearConstraintAndLessEqual2(labeling), coefficients[0] + coefficients[1] - bound);

   OPENGM_TEST_EQUAL(LinearConstraintAndEqual1(labeling + (5 * 3)), 0.0);
   OPENGM_TEST_EQUAL(LinearConstraintAndEqual1(labeling + (1 * 3)), coefficients[1]);
   OPENGM_TEST_EQUAL(LinearConstraintAndEqual2(labeling), 0.0);
   OPENGM_TEST_EQUAL(LinearConstraintAndEqual2(labeling + (3 * 3)), coefficients[0] + coefficients[1]);

   OPENGM_TEST_EQUAL(LinearConstraintAndGreaterEqual1(labeling + (4 * 3)), bound);
   OPENGM_TEST_EQUAL(LinearConstraintAndGreaterEqual1(labeling), 0.0);
   OPENGM_TEST_EQUAL(LinearConstraintAndGreaterEqual2(labeling + (4 * 3)), bound);
   OPENGM_TEST_EQUAL(LinearConstraintAndGreaterEqual2(labeling), 0.0);

   OPENGM_TEST_EQUAL(LinearConstraintOrLessEqual1(labeling + (4 * 3)), 0.0);
   OPENGM_TEST_EQUAL(LinearConstraintOrLessEqual1(labeling), coefficients[0] + coefficients[1] + coefficients[2] - bound);
   OPENGM_TEST_EQUAL(LinearConstraintOrLessEqual2(labeling + (4 * 3)), 0.0);
   OPENGM_TEST_EQUAL(LinearConstraintOrLessEqual2(labeling), coefficients[0] + coefficients[1] - bound);

   OPENGM_TEST_EQUAL(LinearConstraintOrEqual1(labeling + (6 * 3)), 0.0);
   OPENGM_TEST_EQUAL(LinearConstraintOrEqual1(labeling + (5 * 3)), coefficients[0] + coefficients[1]);
   OPENGM_TEST_EQUAL(LinearConstraintOrEqual2(labeling), 0.0);
   OPENGM_TEST_EQUAL(LinearConstraintOrEqual2(labeling + (4 * 3)), coefficients[0] + coefficients[1]);

   OPENGM_TEST_EQUAL(LinearConstraintOrGreaterEqual1(labeling + (4 * 3)), bound);
   OPENGM_TEST_EQUAL(LinearConstraintOrGreaterEqual1(labeling), 0.0);
   OPENGM_TEST_EQUAL(LinearConstraintOrGreaterEqual2(labeling + (4 * 3)), bound);
   OPENGM_TEST_EQUAL(LinearConstraintOrGreaterEqual2(labeling), 0.0);

   OPENGM_TEST_EQUAL(LinearConstraintNotLessEqual1(labeling), 0.0);
   OPENGM_TEST_EQUAL(LinearConstraintNotLessEqual1(labeling + (6 * 3)), coefficients[0] + coefficients[1] - bound);
   OPENGM_TEST_EQUAL(LinearConstraintNotLessEqual2(labeling), 0.0);
   OPENGM_TEST_EQUAL(LinearConstraintNotLessEqual2(labeling + (3 * 3)), coefficients[0] - bound);

   OPENGM_TEST_EQUAL(LinearConstraintNotEqual1(labeling + (3 * 3)), 0.0);
   OPENGM_TEST_EQUAL(LinearConstraintNotEqual1(labeling + (5 * 3)), coefficients[0]);
   OPENGM_TEST_EQUAL(LinearConstraintNotEqual2(labeling + (4 * 3)), 0.0);
   OPENGM_TEST_EQUAL(LinearConstraintNotEqual2(labeling), coefficients[0] + coefficients[1]);

   OPENGM_TEST_EQUAL(LinearConstraintNotGreaterEqual1(labeling), bound);
   OPENGM_TEST_EQUAL(LinearConstraintNotGreaterEqual1(labeling + (4 * 3)), 0.0);
   OPENGM_TEST_EQUAL(LinearConstraintNotGreaterEqual2(labeling + (1 * 3)), bound);
   OPENGM_TEST_EQUAL(LinearConstraintNotGreaterEqual2(labeling + (3 * 3)), 0.0);

   // indicator variables iterator
   OPENGM_TEST_EQUAL(std::distance(emptyLinearConstraint1.indicatorVariablesBegin(), emptyLinearConstraint1.indicatorVariablesEnd()), 0);

   OPENGM_TEST_EQUAL(std::distance(LinearConstraintAndLessEqual1.indicatorVariablesBegin(), LinearConstraintAndLessEqual1.indicatorVariablesEnd()), 3);
   OPENGM_TEST(LinearConstraintAndLessEqual1.indicatorVariablesBegin()[0] == IndicatorVariablesAnd[0]);
   OPENGM_TEST(LinearConstraintAndLessEqual1.indicatorVariablesBegin()[1] == IndicatorVariablesAnd[1]);
   OPENGM_TEST(LinearConstraintAndLessEqual1.indicatorVariablesBegin()[2] == IndicatorVariablesAnd[2]);

   OPENGM_TEST_EQUAL(std::distance(LinearConstraintAndLessEqual2.indicatorVariablesBegin(), LinearConstraintAndLessEqual2.indicatorVariablesEnd()), 2);
   OPENGM_TEST(LinearConstraintAndLessEqual2.indicatorVariablesBegin()[0] == IndicatorVariablesAnd[0]);
   OPENGM_TEST(LinearConstraintAndLessEqual2.indicatorVariablesBegin()[1] == IndicatorVariablesAnd[1]);

   OPENGM_TEST_EQUAL(std::distance(LinearConstraintAndEqual1.indicatorVariablesBegin(), LinearConstraintAndEqual1.indicatorVariablesEnd()), 3);
   OPENGM_TEST(LinearConstraintAndEqual1.indicatorVariablesBegin()[0] == IndicatorVariablesAnd[0]);
   OPENGM_TEST(LinearConstraintAndEqual1.indicatorVariablesBegin()[1] == IndicatorVariablesAnd[1]);
   OPENGM_TEST(LinearConstraintAndEqual1.indicatorVariablesBegin()[2] == IndicatorVariablesAnd[2]);

   OPENGM_TEST_EQUAL(std::distance(LinearConstraintAndEqual2.indicatorVariablesBegin(), LinearConstraintAndEqual2.indicatorVariablesEnd()), 2);
   OPENGM_TEST(LinearConstraintAndEqual2.indicatorVariablesBegin()[0] == IndicatorVariablesAnd[0]);
   OPENGM_TEST(LinearConstraintAndEqual2.indicatorVariablesBegin()[1] == IndicatorVariablesAnd[1]);

   OPENGM_TEST_EQUAL(std::distance(LinearConstraintAndGreaterEqual1.indicatorVariablesBegin(), LinearConstraintAndGreaterEqual1.indicatorVariablesEnd()), 3);
   OPENGM_TEST(LinearConstraintAndGreaterEqual1.indicatorVariablesBegin()[0] == IndicatorVariablesAnd[0]);
   OPENGM_TEST(LinearConstraintAndGreaterEqual1.indicatorVariablesBegin()[1] == IndicatorVariablesAnd[1]);
   OPENGM_TEST(LinearConstraintAndGreaterEqual1.indicatorVariablesBegin()[2] == IndicatorVariablesAnd[2]);

   OPENGM_TEST_EQUAL(std::distance(LinearConstraintAndGreaterEqual2.indicatorVariablesBegin(), LinearConstraintAndGreaterEqual2.indicatorVariablesEnd()), 2);
   OPENGM_TEST(LinearConstraintAndGreaterEqual2.indicatorVariablesBegin()[0] == IndicatorVariablesAnd[0]);
   OPENGM_TEST(LinearConstraintAndGreaterEqual2.indicatorVariablesBegin()[1] == IndicatorVariablesAnd[1]);

   OPENGM_TEST_EQUAL(std::distance(LinearConstraintOrLessEqual1.indicatorVariablesBegin(), LinearConstraintOrLessEqual1.indicatorVariablesEnd()), 3);
   OPENGM_TEST(LinearConstraintOrLessEqual1.indicatorVariablesBegin()[0] == IndicatorVariablesOr[0]);
   OPENGM_TEST(LinearConstraintOrLessEqual1.indicatorVariablesBegin()[1] == IndicatorVariablesOr[1]);
   OPENGM_TEST(LinearConstraintOrLessEqual1.indicatorVariablesBegin()[2] == IndicatorVariablesOr[2]);

   OPENGM_TEST_EQUAL(std::distance(LinearConstraintOrLessEqual2.indicatorVariablesBegin(), LinearConstraintOrLessEqual2.indicatorVariablesEnd()), 2);
   OPENGM_TEST(LinearConstraintOrLessEqual2.indicatorVariablesBegin()[0] == IndicatorVariablesOr[0]);
   OPENGM_TEST(LinearConstraintOrLessEqual2.indicatorVariablesBegin()[1] == IndicatorVariablesOr[1]);

   OPENGM_TEST_EQUAL(std::distance(LinearConstraintOrEqual1.indicatorVariablesBegin(), LinearConstraintOrEqual1.indicatorVariablesEnd()), 3);
   OPENGM_TEST(LinearConstraintOrEqual1.indicatorVariablesBegin()[0] == IndicatorVariablesOr[0]);
   OPENGM_TEST(LinearConstraintOrEqual1.indicatorVariablesBegin()[1] == IndicatorVariablesOr[1]);
   OPENGM_TEST(LinearConstraintOrEqual1.indicatorVariablesBegin()[2] == IndicatorVariablesOr[2]);

   OPENGM_TEST_EQUAL(std::distance(LinearConstraintOrEqual2.indicatorVariablesBegin(), LinearConstraintOrEqual2.indicatorVariablesEnd()), 2);
   OPENGM_TEST(LinearConstraintOrEqual2.indicatorVariablesBegin()[0] == IndicatorVariablesOr[0]);
   OPENGM_TEST(LinearConstraintOrEqual2.indicatorVariablesBegin()[1] == IndicatorVariablesOr[1]);

   OPENGM_TEST_EQUAL(std::distance(LinearConstraintOrGreaterEqual1.indicatorVariablesBegin(), LinearConstraintOrGreaterEqual1.indicatorVariablesEnd()), 3);
   OPENGM_TEST(LinearConstraintOrGreaterEqual1.indicatorVariablesBegin()[0] == IndicatorVariablesOr[0]);
   OPENGM_TEST(LinearConstraintOrGreaterEqual1.indicatorVariablesBegin()[1] == IndicatorVariablesOr[1]);
   OPENGM_TEST(LinearConstraintOrGreaterEqual1.indicatorVariablesBegin()[2] == IndicatorVariablesOr[2]);

   OPENGM_TEST_EQUAL(std::distance(LinearConstraintOrGreaterEqual2.indicatorVariablesBegin(), LinearConstraintOrGreaterEqual2.indicatorVariablesEnd()), 2);
   OPENGM_TEST(LinearConstraintOrGreaterEqual2.indicatorVariablesBegin()[0] == IndicatorVariablesOr[0]);
   OPENGM_TEST(LinearConstraintOrGreaterEqual2.indicatorVariablesBegin()[1] == IndicatorVariablesOr[1]);

   OPENGM_TEST_EQUAL(std::distance(LinearConstraintNotLessEqual1.indicatorVariablesBegin(), LinearConstraintNotLessEqual1.indicatorVariablesEnd()), 3);
   OPENGM_TEST(LinearConstraintNotLessEqual1.indicatorVariablesBegin()[0] == IndicatorVariablesNot[0]);
   OPENGM_TEST(LinearConstraintNotLessEqual1.indicatorVariablesBegin()[1] == IndicatorVariablesNot[1]);
   OPENGM_TEST(LinearConstraintNotLessEqual1.indicatorVariablesBegin()[2] == IndicatorVariablesNot[2]);

   OPENGM_TEST_EQUAL(std::distance(LinearConstraintNotLessEqual2.indicatorVariablesBegin(), LinearConstraintNotLessEqual2.indicatorVariablesEnd()), 2);
   OPENGM_TEST(LinearConstraintNotLessEqual2.indicatorVariablesBegin()[0] == IndicatorVariablesNot[0]);
   OPENGM_TEST(LinearConstraintNotLessEqual2.indicatorVariablesBegin()[1] == IndicatorVariablesNot[1]);

   OPENGM_TEST_EQUAL(std::distance(LinearConstraintNotEqual1.indicatorVariablesBegin(), LinearConstraintNotEqual1.indicatorVariablesEnd()), 3);
   OPENGM_TEST(LinearConstraintNotEqual1.indicatorVariablesBegin()[0] == IndicatorVariablesNot[0]);
   OPENGM_TEST(LinearConstraintNotEqual1.indicatorVariablesBegin()[1] == IndicatorVariablesNot[1]);
   OPENGM_TEST(LinearConstraintNotEqual1.indicatorVariablesBegin()[2] == IndicatorVariablesNot[2]);

   OPENGM_TEST_EQUAL(std::distance(LinearConstraintNotEqual2.indicatorVariablesBegin(), LinearConstraintNotEqual2.indicatorVariablesEnd()), 2);
   OPENGM_TEST(LinearConstraintNotEqual2.indicatorVariablesBegin()[0] == IndicatorVariablesNot[0]);
   OPENGM_TEST(LinearConstraintNotEqual2.indicatorVariablesBegin()[1] == IndicatorVariablesNot[1]);

   OPENGM_TEST_EQUAL(std::distance(LinearConstraintNotGreaterEqual1.indicatorVariablesBegin(), LinearConstraintNotGreaterEqual1.indicatorVariablesEnd()), 3);
   OPENGM_TEST(LinearConstraintNotGreaterEqual1.indicatorVariablesBegin()[0] == IndicatorVariablesNot[0]);
   OPENGM_TEST(LinearConstraintNotGreaterEqual1.indicatorVariablesBegin()[1] == IndicatorVariablesNot[1]);
   OPENGM_TEST(LinearConstraintNotGreaterEqual1.indicatorVariablesBegin()[2] == IndicatorVariablesNot[2]);

   OPENGM_TEST_EQUAL(std::distance(LinearConstraintNotGreaterEqual2.indicatorVariablesBegin(), LinearConstraintNotGreaterEqual2.indicatorVariablesEnd()), 2);
   OPENGM_TEST(LinearConstraintNotGreaterEqual2.indicatorVariablesBegin()[0] == IndicatorVariablesNot[0]);
   OPENGM_TEST(LinearConstraintNotGreaterEqual2.indicatorVariablesBegin()[1] == IndicatorVariablesNot[1]);

   // coefficients iterator
   OPENGM_TEST_EQUAL(std::distance(emptyLinearConstraint1.coefficientsBegin(), emptyLinearConstraint1.coefficientsEnd()), 0);

   OPENGM_TEST_EQUAL(std::distance(LinearConstraintAndLessEqual1.coefficientsBegin(), LinearConstraintAndLessEqual1.coefficientsEnd()), 3);
   OPENGM_TEST(LinearConstraintAndLessEqual1.coefficientsBegin()[0] == coefficients[0]);
   OPENGM_TEST(LinearConstraintAndLessEqual1.coefficientsBegin()[1] == coefficients[1]);
   OPENGM_TEST(LinearConstraintAndLessEqual1.coefficientsBegin()[2] == coefficients[2]);

   OPENGM_TEST_EQUAL(std::distance(LinearConstraintAndLessEqual2.coefficientsBegin(), LinearConstraintAndLessEqual2.coefficientsEnd()), 2);
   OPENGM_TEST(LinearConstraintAndLessEqual2.coefficientsBegin()[0] == coefficients[0]);
   OPENGM_TEST(LinearConstraintAndLessEqual2.coefficientsBegin()[1] == coefficients[1]);

   OPENGM_TEST_EQUAL(std::distance(LinearConstraintAndEqual1.coefficientsBegin(), LinearConstraintAndEqual1.coefficientsEnd()), 3);
   OPENGM_TEST(LinearConstraintAndEqual1.coefficientsBegin()[0] == coefficients[0]);
   OPENGM_TEST(LinearConstraintAndEqual1.coefficientsBegin()[1] == coefficients[1]);
   OPENGM_TEST(LinearConstraintAndEqual1.coefficientsBegin()[2] == coefficients[2]);

   OPENGM_TEST_EQUAL(std::distance(LinearConstraintAndEqual2.coefficientsBegin(), LinearConstraintAndEqual2.coefficientsEnd()), 2);
   OPENGM_TEST(LinearConstraintAndEqual2.coefficientsBegin()[0] == coefficients[0]);
   OPENGM_TEST(LinearConstraintAndEqual2.coefficientsBegin()[1] == coefficients[1]);

   OPENGM_TEST_EQUAL(std::distance(LinearConstraintAndGreaterEqual1.coefficientsBegin(), LinearConstraintAndGreaterEqual1.coefficientsEnd()), 3);
   OPENGM_TEST(LinearConstraintAndGreaterEqual1.coefficientsBegin()[0] == coefficients[0]);
   OPENGM_TEST(LinearConstraintAndGreaterEqual1.coefficientsBegin()[1] == coefficients[1]);
   OPENGM_TEST(LinearConstraintAndGreaterEqual1.coefficientsBegin()[2] == coefficients[2]);

   OPENGM_TEST_EQUAL(std::distance(LinearConstraintAndGreaterEqual2.coefficientsBegin(), LinearConstraintAndGreaterEqual2.coefficientsEnd()), 2);
   OPENGM_TEST(LinearConstraintAndGreaterEqual2.coefficientsBegin()[0] == coefficients[0]);
   OPENGM_TEST(LinearConstraintAndGreaterEqual2.coefficientsBegin()[1] == coefficients[1]);

   OPENGM_TEST_EQUAL(std::distance(LinearConstraintOrLessEqual1.coefficientsBegin(), LinearConstraintOrLessEqual1.coefficientsEnd()), 3);
   OPENGM_TEST(LinearConstraintOrLessEqual1.coefficientsBegin()[0] == coefficients[0]);
   OPENGM_TEST(LinearConstraintOrLessEqual1.coefficientsBegin()[1] == coefficients[1]);
   OPENGM_TEST(LinearConstraintOrLessEqual1.coefficientsBegin()[2] == coefficients[2]);

   OPENGM_TEST_EQUAL(std::distance(LinearConstraintOrLessEqual2.coefficientsBegin(), LinearConstraintOrLessEqual2.coefficientsEnd()), 2);
   OPENGM_TEST(LinearConstraintOrLessEqual2.coefficientsBegin()[0] == coefficients[0]);
   OPENGM_TEST(LinearConstraintOrLessEqual2.coefficientsBegin()[1] == coefficients[1]);

   OPENGM_TEST_EQUAL(std::distance(LinearConstraintOrEqual1.coefficientsBegin(), LinearConstraintOrEqual1.coefficientsEnd()), 3);
   OPENGM_TEST(LinearConstraintOrEqual1.coefficientsBegin()[0] == coefficients[0]);
   OPENGM_TEST(LinearConstraintOrEqual1.coefficientsBegin()[1] == coefficients[1]);
   OPENGM_TEST(LinearConstraintOrEqual1.coefficientsBegin()[2] == coefficients[2]);

   OPENGM_TEST_EQUAL(std::distance(LinearConstraintOrEqual2.coefficientsBegin(), LinearConstraintOrEqual2.coefficientsEnd()), 2);
   OPENGM_TEST(LinearConstraintOrEqual2.coefficientsBegin()[0] == coefficients[0]);
   OPENGM_TEST(LinearConstraintOrEqual2.coefficientsBegin()[1] == coefficients[1]);

   OPENGM_TEST_EQUAL(std::distance(LinearConstraintOrGreaterEqual1.coefficientsBegin(), LinearConstraintOrGreaterEqual1.coefficientsEnd()), 3);
   OPENGM_TEST(LinearConstraintOrGreaterEqual1.coefficientsBegin()[0] == coefficients[0]);
   OPENGM_TEST(LinearConstraintOrGreaterEqual1.coefficientsBegin()[1] == coefficients[1]);
   OPENGM_TEST(LinearConstraintOrGreaterEqual1.coefficientsBegin()[2] == coefficients[2]);

   OPENGM_TEST_EQUAL(std::distance(LinearConstraintOrGreaterEqual2.coefficientsBegin(), LinearConstraintOrGreaterEqual2.coefficientsEnd()), 2);
   OPENGM_TEST(LinearConstraintOrGreaterEqual2.coefficientsBegin()[0] == coefficients[0]);
   OPENGM_TEST(LinearConstraintOrGreaterEqual2.coefficientsBegin()[1] == coefficients[1]);

   OPENGM_TEST_EQUAL(std::distance(LinearConstraintNotLessEqual1.coefficientsBegin(), LinearConstraintNotLessEqual1.coefficientsEnd()), 3);
   OPENGM_TEST(LinearConstraintNotLessEqual1.coefficientsBegin()[0] == coefficients[0]);
   OPENGM_TEST(LinearConstraintNotLessEqual1.coefficientsBegin()[1] == coefficients[1]);
   OPENGM_TEST(LinearConstraintNotLessEqual1.coefficientsBegin()[2] == coefficients[2]);

   OPENGM_TEST_EQUAL(std::distance(LinearConstraintNotLessEqual2.coefficientsBegin(), LinearConstraintNotLessEqual2.coefficientsEnd()), 2);
   OPENGM_TEST(LinearConstraintNotLessEqual2.coefficientsBegin()[0] == coefficients[0]);
   OPENGM_TEST(LinearConstraintNotLessEqual2.coefficientsBegin()[1] == coefficients[1]);

   OPENGM_TEST_EQUAL(std::distance(LinearConstraintNotEqual1.coefficientsBegin(), LinearConstraintNotEqual1.coefficientsEnd()), 3);
   OPENGM_TEST(LinearConstraintNotEqual1.coefficientsBegin()[0] == coefficients[0]);
   OPENGM_TEST(LinearConstraintNotEqual1.coefficientsBegin()[1] == coefficients[1]);
   OPENGM_TEST(LinearConstraintNotEqual1.coefficientsBegin()[2] == coefficients[2]);

   OPENGM_TEST_EQUAL(std::distance(LinearConstraintNotEqual2.coefficientsBegin(), LinearConstraintNotEqual2.coefficientsEnd()), 2);
   OPENGM_TEST(LinearConstraintNotEqual2.coefficientsBegin()[0] == coefficients[0]);
   OPENGM_TEST(LinearConstraintNotEqual2.coefficientsBegin()[1] == coefficients[1]);

   OPENGM_TEST_EQUAL(std::distance(LinearConstraintNotGreaterEqual1.coefficientsBegin(), LinearConstraintNotGreaterEqual1.coefficientsEnd()), 3);
   OPENGM_TEST(LinearConstraintNotGreaterEqual1.coefficientsBegin()[0] == coefficients[0]);
   OPENGM_TEST(LinearConstraintNotGreaterEqual1.coefficientsBegin()[1] == coefficients[1]);
   OPENGM_TEST(LinearConstraintNotGreaterEqual1.coefficientsBegin()[2] == coefficients[2]);

   OPENGM_TEST_EQUAL(std::distance(LinearConstraintNotGreaterEqual2.coefficientsBegin(), LinearConstraintNotGreaterEqual2.coefficientsEnd()), 2);
   OPENGM_TEST(LinearConstraintNotGreaterEqual2.coefficientsBegin()[0] == coefficients[0]);
   OPENGM_TEST(LinearConstraintNotGreaterEqual2.coefficientsBegin()[1] == coefficients[1]);

   // get bound
   OPENGM_TEST_EQUAL(emptyLinearConstraint1.getBound(), BoundType(0.0));
   OPENGM_TEST_EQUAL(emptyLinearConstraint2.getBound(), BoundType(-1.0));

   OPENGM_TEST_EQUAL(LinearConstraintAndLessEqual1.getBound(), bound);
   OPENGM_TEST_EQUAL(LinearConstraintAndLessEqual2.getBound(), bound);
   OPENGM_TEST_EQUAL(LinearConstraintAndEqual1.getBound(), BoundType(coefficients[0]));
   OPENGM_TEST_EQUAL(LinearConstraintAndEqual2.getBound(), BoundType(coefficients[0] + coefficients[1]));
   OPENGM_TEST_EQUAL(LinearConstraintAndGreaterEqual1.getBound(), bound);
   OPENGM_TEST_EQUAL(LinearConstraintAndGreaterEqual2.getBound(), bound);

   OPENGM_TEST_EQUAL(LinearConstraintOrLessEqual1.getBound(), bound);
   OPENGM_TEST_EQUAL(LinearConstraintOrLessEqual2.getBound(), bound);
   OPENGM_TEST_EQUAL(LinearConstraintOrEqual1.getBound(), BoundType(coefficients[2]));
   OPENGM_TEST_EQUAL(LinearConstraintOrEqual2.getBound(), BoundType(coefficients[0] + coefficients[1]));
   OPENGM_TEST_EQUAL(LinearConstraintOrGreaterEqual1.getBound(), bound);
   OPENGM_TEST_EQUAL(LinearConstraintOrGreaterEqual2.getBound(), bound);

   OPENGM_TEST_EQUAL(LinearConstraintNotLessEqual1.getBound(), bound);
   OPENGM_TEST_EQUAL(LinearConstraintNotLessEqual2.getBound(), bound);
   OPENGM_TEST_EQUAL(LinearConstraintNotEqual1.getBound(), BoundType(coefficients[0]));
   OPENGM_TEST_EQUAL(LinearConstraintNotEqual2.getBound(), BoundType(coefficients[0] + coefficients[1]));
   OPENGM_TEST_EQUAL(LinearConstraintNotGreaterEqual1.getBound(), bound);
   OPENGM_TEST_EQUAL(LinearConstraintNotGreaterEqual2.getBound(), bound);

   // get constraint operator
   OPENGM_TEST_EQUAL(emptyLinearConstraint1.getConstraintOperator(), LinearConstraintOperatorType::LessEqual);
   OPENGM_TEST_EQUAL(emptyLinearConstraint2.getConstraintOperator(), LinearConstraintOperatorType::GreaterEqual);

   OPENGM_TEST_EQUAL(LinearConstraintAndLessEqual1.getConstraintOperator(), LinearConstraintOperatorType::LessEqual);
   OPENGM_TEST_EQUAL(LinearConstraintAndLessEqual2.getConstraintOperator(), LinearConstraintOperatorType::LessEqual);
   OPENGM_TEST_EQUAL(LinearConstraintAndEqual1.getConstraintOperator(), LinearConstraintOperatorType::Equal);
   OPENGM_TEST_EQUAL(LinearConstraintAndEqual2.getConstraintOperator(), LinearConstraintOperatorType::Equal);
   OPENGM_TEST_EQUAL(LinearConstraintAndGreaterEqual1.getConstraintOperator(), LinearConstraintOperatorType::GreaterEqual);
   OPENGM_TEST_EQUAL(LinearConstraintAndGreaterEqual2.getConstraintOperator(), LinearConstraintOperatorType::GreaterEqual);

   OPENGM_TEST_EQUAL(LinearConstraintOrLessEqual1.getConstraintOperator(), LinearConstraintOperatorType::LessEqual);
   OPENGM_TEST_EQUAL(LinearConstraintOrLessEqual2.getConstraintOperator(), LinearConstraintOperatorType::LessEqual);
   OPENGM_TEST_EQUAL(LinearConstraintOrEqual1.getConstraintOperator(), LinearConstraintOperatorType::Equal);
   OPENGM_TEST_EQUAL(LinearConstraintOrEqual2.getConstraintOperator(), LinearConstraintOperatorType::Equal);
   OPENGM_TEST_EQUAL(LinearConstraintOrGreaterEqual1.getConstraintOperator(), LinearConstraintOperatorType::GreaterEqual);
   OPENGM_TEST_EQUAL(LinearConstraintOrGreaterEqual2.getConstraintOperator(), LinearConstraintOperatorType::GreaterEqual);

   OPENGM_TEST_EQUAL(LinearConstraintNotLessEqual1.getConstraintOperator(), LinearConstraintOperatorType::LessEqual);
   OPENGM_TEST_EQUAL(LinearConstraintNotLessEqual2.getConstraintOperator(), LinearConstraintOperatorType::LessEqual);
   OPENGM_TEST_EQUAL(LinearConstraintNotEqual1.getConstraintOperator(), LinearConstraintOperatorType::Equal);
   OPENGM_TEST_EQUAL(LinearConstraintNotEqual2.getConstraintOperator(), LinearConstraintOperatorType::Equal);
   OPENGM_TEST_EQUAL(LinearConstraintNotGreaterEqual1.getConstraintOperator(), LinearConstraintOperatorType::GreaterEqual);
   OPENGM_TEST_EQUAL(LinearConstraintNotGreaterEqual2.getConstraintOperator(), LinearConstraintOperatorType::GreaterEqual);
}
