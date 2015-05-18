#include <iostream>

#include <opengm/unittests/test.hxx>

#include <opengm/datastructures/indicator_variable.hxx>

template<class INDEX_TYPE, class LABEL_TYPE>
void testIndicatorVariable();

int main(int argc, char** argv){
   std::cout << "Indicator Variable test... " << std::endl;

   testIndicatorVariable<size_t, size_t>();
   testIndicatorVariable<size_t, int>();
   testIndicatorVariable<size_t, char>();

   testIndicatorVariable<int, size_t>();
   testIndicatorVariable<int, int>();
   testIndicatorVariable<int, char>();

   testIndicatorVariable<unsigned char, size_t>();
   testIndicatorVariable<unsigned char, int>();
   testIndicatorVariable<unsigned char, char>();

   std::cout << "done..." << std::endl;
   return 0;
}

template<class INDEX_TYPE, class LABEL_TYPE>
void testIndicatorVariable() {
   typedef INDEX_TYPE IndexType;
   typedef LABEL_TYPE LabelType;

   typedef opengm::IndicatorVariable<IndexType, LabelType>   IndicatorVariableType;
   typedef typename IndicatorVariableType::VariableLabelPair VariableLabelPairType;
   typedef typename IndicatorVariableType::VariableLabelPairContainerType VariableLabelPairContainerType;

   // construction
   const IndicatorVariableType emptyIndicatorVariable1;
   IndicatorVariableType emptyIndicatorVariable2;
   const IndicatorVariableType singleAndIndicatorVariable(IndexType(0), LabelType(0));
   const IndicatorVariableType singleOrIndicatorVariable(VariableLabelPairType(IndexType(0), LabelType(0)), IndicatorVariableType::Or);
   const VariableLabelPairContainerType singleVariableLabelPair1(1, VariableLabelPairType(IndexType(0), LabelType(0)));
   const IndicatorVariableType singleNotIndicatorVariable1(singleVariableLabelPair1, IndicatorVariableType::Not);
   const IndicatorVariableType singleNotIndicatorVariable2(singleVariableLabelPair1.begin(), singleVariableLabelPair1.end(), IndicatorVariableType::Not);

   // copy
   IndicatorVariableType multipleAndIndicatorVariable = singleAndIndicatorVariable;
   IndicatorVariableType multipleOrIndicatorVariable  = singleOrIndicatorVariable;
   IndicatorVariableType multipleNotIndicatorVariable1 = singleNotIndicatorVariable1;
   IndicatorVariableType multipleNotIndicatorVariable2 = singleNotIndicatorVariable2;

   // reserve
   multipleAndIndicatorVariable.reserve(2);
   multipleOrIndicatorVariable.reserve(2);
   multipleNotIndicatorVariable1.reserve(2);

   // add
   multipleAndIndicatorVariable.add(IndexType(1), LabelType(1));
   multipleOrIndicatorVariable.add(VariableLabelPairType(IndexType(1), LabelType(1)));
   const VariableLabelPairContainerType singleVariableLabelPair2(1, VariableLabelPairType(IndexType(1), LabelType(1)));
   multipleNotIndicatorVariable1.add(singleVariableLabelPair2);
   multipleNotIndicatorVariable2.add(singleVariableLabelPair2.begin(), singleVariableLabelPair2.end());

   // set logical operator type
   emptyIndicatorVariable2.setLogicalOperatorType(IndicatorVariableType::Or);

   // evaluate
   const LabelType labeling[] = {0, 0, 0, 1, 1, 1, 1, 0};
   OPENGM_TEST_EQUAL(singleAndIndicatorVariable(labeling), true);
   OPENGM_TEST_EQUAL(singleAndIndicatorVariable(labeling + 3), false);
   OPENGM_TEST_EQUAL(singleOrIndicatorVariable(labeling), true);
   OPENGM_TEST_EQUAL(singleOrIndicatorVariable(labeling + 3), false);
   OPENGM_TEST_EQUAL(singleNotIndicatorVariable1(labeling), false);
   OPENGM_TEST_EQUAL(singleNotIndicatorVariable1(labeling + 3), true);

   OPENGM_TEST_EQUAL(multipleAndIndicatorVariable(labeling), false);
   OPENGM_TEST_EQUAL(multipleAndIndicatorVariable(labeling + 2), true);
   OPENGM_TEST_EQUAL(multipleAndIndicatorVariable(labeling + 4), false);
   OPENGM_TEST_EQUAL(multipleAndIndicatorVariable(labeling + 6), false);

   OPENGM_TEST_EQUAL(multipleOrIndicatorVariable(labeling), true);
   OPENGM_TEST_EQUAL(multipleOrIndicatorVariable(labeling + 2), true);
   OPENGM_TEST_EQUAL(multipleOrIndicatorVariable(labeling + 4), true);
   OPENGM_TEST_EQUAL(multipleOrIndicatorVariable(labeling + 6), false);

   OPENGM_TEST_EQUAL(multipleNotIndicatorVariable1(labeling), false);
   OPENGM_TEST_EQUAL(multipleNotIndicatorVariable1(labeling + 2), false);
   OPENGM_TEST_EQUAL(multipleNotIndicatorVariable1(labeling + 4), false);
   OPENGM_TEST_EQUAL(multipleNotIndicatorVariable1(labeling + 6), true);

   // iterator
   OPENGM_TEST_EQUAL(std::distance(emptyIndicatorVariable1.begin(), emptyIndicatorVariable1.end()), 0);
   OPENGM_TEST_EQUAL(std::distance(emptyIndicatorVariable2.begin(), emptyIndicatorVariable2.end()), 0);

   OPENGM_TEST_EQUAL(std::distance(singleAndIndicatorVariable.begin(), singleAndIndicatorVariable.end()),   1);
   OPENGM_TEST_EQUAL(std::distance(singleOrIndicatorVariable.begin(),  singleOrIndicatorVariable.end()),    1);
   OPENGM_TEST_EQUAL(std::distance(singleNotIndicatorVariable1.begin(), singleNotIndicatorVariable1.end()), 1);

   OPENGM_TEST_EQUAL(singleAndIndicatorVariable.begin()->first,   IndexType(0));
   OPENGM_TEST_EQUAL(singleAndIndicatorVariable.begin()->second,  LabelType(0));
   OPENGM_TEST_EQUAL(singleOrIndicatorVariable.begin()->first,    IndexType(0));
   OPENGM_TEST_EQUAL(singleOrIndicatorVariable.begin()->second,   LabelType(0));
   OPENGM_TEST_EQUAL(singleNotIndicatorVariable1.begin()->first,  IndexType(0));
   OPENGM_TEST_EQUAL(singleNotIndicatorVariable1.begin()->second, LabelType(0));

   OPENGM_TEST_EQUAL(std::distance(multipleAndIndicatorVariable.begin(), multipleAndIndicatorVariable.end()), 2);
   OPENGM_TEST_EQUAL(std::distance(multipleOrIndicatorVariable.begin(), multipleOrIndicatorVariable.end()), 2);
   OPENGM_TEST_EQUAL(std::distance(multipleNotIndicatorVariable1.begin(), multipleNotIndicatorVariable1.end()), 2);

   OPENGM_TEST_EQUAL(multipleAndIndicatorVariable.begin()->first,         IndexType(0));
   OPENGM_TEST_EQUAL(multipleAndIndicatorVariable.begin()->second,        LabelType(0));
   OPENGM_TEST_EQUAL((multipleAndIndicatorVariable.begin() + 1)->first,   IndexType(1));
   OPENGM_TEST_EQUAL((multipleAndIndicatorVariable.begin() + 1)->second,  LabelType(1));
   OPENGM_TEST_EQUAL(multipleOrIndicatorVariable.begin()->first,          IndexType(0));
   OPENGM_TEST_EQUAL(multipleOrIndicatorVariable.begin()->second,         LabelType(0));
   OPENGM_TEST_EQUAL((multipleOrIndicatorVariable.begin() + 1)->first,    IndexType(1));
   OPENGM_TEST_EQUAL((multipleOrIndicatorVariable.begin() + 1)->second,   LabelType(1));
   OPENGM_TEST_EQUAL(multipleNotIndicatorVariable1.begin()->first,        IndexType(0));
   OPENGM_TEST_EQUAL(multipleNotIndicatorVariable1.begin()->second,       LabelType(0));
   OPENGM_TEST_EQUAL((multipleNotIndicatorVariable1.begin() + 1)->first,  IndexType(1));
   OPENGM_TEST_EQUAL((multipleNotIndicatorVariable1.begin() + 1)->second, LabelType(1));

   // get logical operator type
   OPENGM_TEST_EQUAL(emptyIndicatorVariable1.getLogicalOperatorType(), IndicatorVariableType::And);
   OPENGM_TEST_EQUAL(emptyIndicatorVariable2.getLogicalOperatorType(), IndicatorVariableType::Or);

   OPENGM_TEST_EQUAL(singleAndIndicatorVariable.getLogicalOperatorType(),  IndicatorVariableType::And);
   OPENGM_TEST_EQUAL(singleOrIndicatorVariable.getLogicalOperatorType(),   IndicatorVariableType::Or);
   OPENGM_TEST_EQUAL(singleNotIndicatorVariable1.getLogicalOperatorType(), IndicatorVariableType::Not);

   OPENGM_TEST_EQUAL(multipleAndIndicatorVariable.getLogicalOperatorType(),  IndicatorVariableType::And);
   OPENGM_TEST_EQUAL(multipleOrIndicatorVariable.getLogicalOperatorType(),   IndicatorVariableType::Or);
   OPENGM_TEST_EQUAL(multipleNotIndicatorVariable1.getLogicalOperatorType(), IndicatorVariableType::Not);

   // comparison
   // ==
   OPENGM_TEST(emptyIndicatorVariable1 == emptyIndicatorVariable1);
   OPENGM_TEST(!(emptyIndicatorVariable1 == emptyIndicatorVariable2));
   OPENGM_TEST(!(emptyIndicatorVariable2 == emptyIndicatorVariable1));

   OPENGM_TEST(!(emptyIndicatorVariable1 == singleAndIndicatorVariable));
   OPENGM_TEST(!(singleAndIndicatorVariable == emptyIndicatorVariable1));
   OPENGM_TEST(!(emptyIndicatorVariable1 == singleOrIndicatorVariable));
   OPENGM_TEST(!(singleOrIndicatorVariable == emptyIndicatorVariable1));
   OPENGM_TEST(!(emptyIndicatorVariable1 == singleNotIndicatorVariable1));
   OPENGM_TEST(!(singleNotIndicatorVariable1 == emptyIndicatorVariable1));

   OPENGM_TEST(!(emptyIndicatorVariable1 == multipleAndIndicatorVariable));
   OPENGM_TEST(!(multipleAndIndicatorVariable == emptyIndicatorVariable1));
   OPENGM_TEST(!(emptyIndicatorVariable1 == multipleOrIndicatorVariable));
   OPENGM_TEST(!(multipleOrIndicatorVariable == emptyIndicatorVariable1));
   OPENGM_TEST(!(emptyIndicatorVariable1 == multipleNotIndicatorVariable1));
   OPENGM_TEST(!(multipleNotIndicatorVariable1 == emptyIndicatorVariable1));

   OPENGM_TEST(singleAndIndicatorVariable  == singleAndIndicatorVariable);
   OPENGM_TEST(singleOrIndicatorVariable   == singleOrIndicatorVariable);
   OPENGM_TEST(singleNotIndicatorVariable1 == singleNotIndicatorVariable1);

   OPENGM_TEST(multipleAndIndicatorVariable  == multipleAndIndicatorVariable);
   OPENGM_TEST(multipleOrIndicatorVariable   == multipleOrIndicatorVariable);
   OPENGM_TEST(multipleNotIndicatorVariable1 == multipleNotIndicatorVariable1);

   OPENGM_TEST(!(singleAndIndicatorVariable == singleOrIndicatorVariable));
   OPENGM_TEST(!(singleOrIndicatorVariable == singleAndIndicatorVariable));
   OPENGM_TEST(!(singleAndIndicatorVariable == singleNotIndicatorVariable1));
   OPENGM_TEST(!(singleNotIndicatorVariable1 == singleAndIndicatorVariable));

   OPENGM_TEST(!(singleAndIndicatorVariable == multipleAndIndicatorVariable));
   OPENGM_TEST(!(multipleAndIndicatorVariable == singleAndIndicatorVariable));
   OPENGM_TEST(!(singleAndIndicatorVariable == multipleOrIndicatorVariable));
   OPENGM_TEST(!(multipleOrIndicatorVariable == singleAndIndicatorVariable));
   OPENGM_TEST(!(singleAndIndicatorVariable == multipleNotIndicatorVariable1));
   OPENGM_TEST(!(multipleNotIndicatorVariable1 == singleAndIndicatorVariable));

   OPENGM_TEST(!(singleOrIndicatorVariable == singleNotIndicatorVariable1));
   OPENGM_TEST(!(singleNotIndicatorVariable1 == singleOrIndicatorVariable));

   OPENGM_TEST(!(singleOrIndicatorVariable == multipleAndIndicatorVariable));
   OPENGM_TEST(!(multipleAndIndicatorVariable == singleOrIndicatorVariable));
   OPENGM_TEST(!(singleOrIndicatorVariable == multipleOrIndicatorVariable));
   OPENGM_TEST(!(multipleOrIndicatorVariable == singleOrIndicatorVariable));
   OPENGM_TEST(!(singleOrIndicatorVariable == multipleNotIndicatorVariable1));
   OPENGM_TEST(!(multipleNotIndicatorVariable1 == singleOrIndicatorVariable));

   OPENGM_TEST(!(multipleAndIndicatorVariable == multipleOrIndicatorVariable));
   OPENGM_TEST(!(multipleOrIndicatorVariable == multipleAndIndicatorVariable));
   OPENGM_TEST(!(multipleAndIndicatorVariable == multipleNotIndicatorVariable1));
   OPENGM_TEST(!(multipleNotIndicatorVariable1 == multipleAndIndicatorVariable));

   OPENGM_TEST(!(multipleOrIndicatorVariable == multipleNotIndicatorVariable1));
   OPENGM_TEST(!(multipleNotIndicatorVariable1 == multipleOrIndicatorVariable));

   // !=
   OPENGM_TEST(!(emptyIndicatorVariable1 != emptyIndicatorVariable1));
   OPENGM_TEST(emptyIndicatorVariable1 != emptyIndicatorVariable2);
   OPENGM_TEST(emptyIndicatorVariable2 != emptyIndicatorVariable1);

   OPENGM_TEST(emptyIndicatorVariable1 != singleAndIndicatorVariable);
   OPENGM_TEST(singleAndIndicatorVariable != emptyIndicatorVariable1);
   OPENGM_TEST(emptyIndicatorVariable1 != singleOrIndicatorVariable);
   OPENGM_TEST(singleOrIndicatorVariable != emptyIndicatorVariable1);
   OPENGM_TEST(emptyIndicatorVariable1 != singleNotIndicatorVariable1);
   OPENGM_TEST(singleNotIndicatorVariable1 != emptyIndicatorVariable1);

   OPENGM_TEST(emptyIndicatorVariable1 != multipleAndIndicatorVariable);
   OPENGM_TEST(multipleAndIndicatorVariable != emptyIndicatorVariable1);
   OPENGM_TEST(emptyIndicatorVariable1 != multipleOrIndicatorVariable);
   OPENGM_TEST(multipleOrIndicatorVariable != emptyIndicatorVariable1);
   OPENGM_TEST(emptyIndicatorVariable1 != multipleNotIndicatorVariable1);
   OPENGM_TEST(multipleNotIndicatorVariable1 != emptyIndicatorVariable1);

   OPENGM_TEST(!(singleAndIndicatorVariable  != singleAndIndicatorVariable));
   OPENGM_TEST(!(singleOrIndicatorVariable   != singleOrIndicatorVariable));
   OPENGM_TEST(!(singleNotIndicatorVariable1 != singleNotIndicatorVariable1));

   OPENGM_TEST(!(multipleAndIndicatorVariable  != multipleAndIndicatorVariable));
   OPENGM_TEST(!(multipleOrIndicatorVariable   != multipleOrIndicatorVariable));
   OPENGM_TEST(!(multipleNotIndicatorVariable1 != multipleNotIndicatorVariable1));

   OPENGM_TEST(singleAndIndicatorVariable != singleOrIndicatorVariable);
   OPENGM_TEST(singleOrIndicatorVariable != singleAndIndicatorVariable);
   OPENGM_TEST(singleAndIndicatorVariable != singleNotIndicatorVariable1);
   OPENGM_TEST(singleNotIndicatorVariable1 != singleAndIndicatorVariable);

   OPENGM_TEST(singleAndIndicatorVariable != multipleAndIndicatorVariable);
   OPENGM_TEST(multipleAndIndicatorVariable != singleAndIndicatorVariable);
   OPENGM_TEST(singleAndIndicatorVariable != multipleOrIndicatorVariable);
   OPENGM_TEST(multipleOrIndicatorVariable != singleAndIndicatorVariable);
   OPENGM_TEST(singleAndIndicatorVariable != multipleNotIndicatorVariable1);
   OPENGM_TEST(multipleNotIndicatorVariable1 != singleAndIndicatorVariable);

   OPENGM_TEST(singleOrIndicatorVariable != singleNotIndicatorVariable1);
   OPENGM_TEST(singleNotIndicatorVariable1 != singleOrIndicatorVariable);

   OPENGM_TEST(singleOrIndicatorVariable != multipleAndIndicatorVariable);
   OPENGM_TEST(multipleAndIndicatorVariable != singleOrIndicatorVariable);
   OPENGM_TEST(singleOrIndicatorVariable != multipleOrIndicatorVariable);
   OPENGM_TEST(multipleOrIndicatorVariable != singleOrIndicatorVariable);
   OPENGM_TEST(singleOrIndicatorVariable != multipleNotIndicatorVariable1);
   OPENGM_TEST(multipleNotIndicatorVariable1 != singleOrIndicatorVariable);

   OPENGM_TEST(multipleAndIndicatorVariable != multipleOrIndicatorVariable);
   OPENGM_TEST(multipleOrIndicatorVariable != multipleAndIndicatorVariable);
   OPENGM_TEST(multipleAndIndicatorVariable != multipleNotIndicatorVariable1);
   OPENGM_TEST(multipleNotIndicatorVariable1 != multipleAndIndicatorVariable);

   OPENGM_TEST(multipleOrIndicatorVariable != multipleNotIndicatorVariable1);
   OPENGM_TEST(multipleNotIndicatorVariable1 != multipleOrIndicatorVariable);

   // <
   OPENGM_TEST(!(emptyIndicatorVariable1 < emptyIndicatorVariable1));
   OPENGM_TEST(emptyIndicatorVariable1 < emptyIndicatorVariable2);
   OPENGM_TEST(!(emptyIndicatorVariable2 < emptyIndicatorVariable1));

   OPENGM_TEST(emptyIndicatorVariable1 < singleAndIndicatorVariable);
   OPENGM_TEST(!(singleAndIndicatorVariable < emptyIndicatorVariable1));
   OPENGM_TEST(emptyIndicatorVariable1 < singleOrIndicatorVariable);
   OPENGM_TEST(!(singleOrIndicatorVariable < emptyIndicatorVariable1));
   OPENGM_TEST(emptyIndicatorVariable1 < singleNotIndicatorVariable1);
   OPENGM_TEST(!(singleNotIndicatorVariable1 < emptyIndicatorVariable1));

   OPENGM_TEST(emptyIndicatorVariable1 < multipleAndIndicatorVariable);
   OPENGM_TEST(!(multipleAndIndicatorVariable < emptyIndicatorVariable1));
   OPENGM_TEST(emptyIndicatorVariable1 < multipleOrIndicatorVariable);
   OPENGM_TEST(!(multipleOrIndicatorVariable < emptyIndicatorVariable1));
   OPENGM_TEST(emptyIndicatorVariable1 < multipleNotIndicatorVariable1);
   OPENGM_TEST(!(multipleNotIndicatorVariable1 < emptyIndicatorVariable1));

   OPENGM_TEST(!(singleAndIndicatorVariable  < singleAndIndicatorVariable));
   OPENGM_TEST(!(singleOrIndicatorVariable   < singleOrIndicatorVariable));
   OPENGM_TEST(!(singleNotIndicatorVariable1 < singleNotIndicatorVariable1));

   OPENGM_TEST(!(multipleAndIndicatorVariable  < multipleAndIndicatorVariable));
   OPENGM_TEST(!(multipleOrIndicatorVariable   < multipleOrIndicatorVariable));
   OPENGM_TEST(!(multipleNotIndicatorVariable1 < multipleNotIndicatorVariable1));

   OPENGM_TEST(singleAndIndicatorVariable < singleOrIndicatorVariable);
   OPENGM_TEST(!(singleOrIndicatorVariable < singleAndIndicatorVariable));
   OPENGM_TEST(singleAndIndicatorVariable < singleNotIndicatorVariable1);
   OPENGM_TEST(!(singleNotIndicatorVariable1 < singleAndIndicatorVariable));

   OPENGM_TEST(singleAndIndicatorVariable < multipleAndIndicatorVariable);
   OPENGM_TEST(!(multipleAndIndicatorVariable < singleAndIndicatorVariable));
   OPENGM_TEST(singleAndIndicatorVariable < multipleOrIndicatorVariable);
   OPENGM_TEST(!(multipleOrIndicatorVariable < singleAndIndicatorVariable));
   OPENGM_TEST(singleAndIndicatorVariable < multipleNotIndicatorVariable1);
   OPENGM_TEST(!(multipleNotIndicatorVariable1 < singleAndIndicatorVariable));

   OPENGM_TEST(singleOrIndicatorVariable < singleNotIndicatorVariable1);
   OPENGM_TEST(!(singleNotIndicatorVariable1 < singleOrIndicatorVariable));

   OPENGM_TEST(!(singleOrIndicatorVariable < multipleAndIndicatorVariable));
   OPENGM_TEST(multipleAndIndicatorVariable < singleOrIndicatorVariable);
   OPENGM_TEST(singleOrIndicatorVariable < multipleOrIndicatorVariable);
   OPENGM_TEST(!(multipleOrIndicatorVariable < singleOrIndicatorVariable));
   OPENGM_TEST(singleOrIndicatorVariable < multipleNotIndicatorVariable1);
   OPENGM_TEST(!(multipleNotIndicatorVariable1 < singleOrIndicatorVariable));

   OPENGM_TEST(multipleAndIndicatorVariable < multipleOrIndicatorVariable);
   OPENGM_TEST(!(multipleOrIndicatorVariable < multipleAndIndicatorVariable));
   OPENGM_TEST(multipleAndIndicatorVariable < multipleNotIndicatorVariable1);
   OPENGM_TEST(!(multipleNotIndicatorVariable1 < multipleAndIndicatorVariable));

   OPENGM_TEST(multipleOrIndicatorVariable < multipleNotIndicatorVariable1);
   OPENGM_TEST(!(multipleNotIndicatorVariable1 < multipleOrIndicatorVariable));

   // <=
   OPENGM_TEST(emptyIndicatorVariable1 <= emptyIndicatorVariable1);
   OPENGM_TEST(emptyIndicatorVariable1 <= emptyIndicatorVariable2);
   OPENGM_TEST(!(emptyIndicatorVariable2 <= emptyIndicatorVariable1));

   OPENGM_TEST(emptyIndicatorVariable1 <= singleAndIndicatorVariable);
   OPENGM_TEST(!(singleAndIndicatorVariable <= emptyIndicatorVariable1));
   OPENGM_TEST(emptyIndicatorVariable1 <= singleOrIndicatorVariable);
   OPENGM_TEST(!(singleOrIndicatorVariable <= emptyIndicatorVariable1));
   OPENGM_TEST(emptyIndicatorVariable1 <= singleNotIndicatorVariable1);
   OPENGM_TEST(!(singleNotIndicatorVariable1 <= emptyIndicatorVariable1));

   OPENGM_TEST(emptyIndicatorVariable1 <= multipleAndIndicatorVariable);
   OPENGM_TEST(!(multipleAndIndicatorVariable <= emptyIndicatorVariable1));
   OPENGM_TEST(emptyIndicatorVariable1 <= multipleOrIndicatorVariable);
   OPENGM_TEST(!(multipleOrIndicatorVariable <= emptyIndicatorVariable1));
   OPENGM_TEST(emptyIndicatorVariable1 <= multipleNotIndicatorVariable1);
   OPENGM_TEST(!(multipleNotIndicatorVariable1 <= emptyIndicatorVariable1));

   OPENGM_TEST(singleAndIndicatorVariable  <= singleAndIndicatorVariable);
   OPENGM_TEST(singleOrIndicatorVariable   <= singleOrIndicatorVariable);
   OPENGM_TEST(singleNotIndicatorVariable1 <= singleNotIndicatorVariable1);

   OPENGM_TEST(multipleAndIndicatorVariable  <= multipleAndIndicatorVariable);
   OPENGM_TEST(multipleOrIndicatorVariable   <= multipleOrIndicatorVariable);
   OPENGM_TEST(multipleNotIndicatorVariable1 <= multipleNotIndicatorVariable1);

   OPENGM_TEST(singleAndIndicatorVariable <= singleOrIndicatorVariable);
   OPENGM_TEST(!(singleOrIndicatorVariable <= singleAndIndicatorVariable));
   OPENGM_TEST(singleAndIndicatorVariable <= singleNotIndicatorVariable1);
   OPENGM_TEST(!(singleNotIndicatorVariable1 <= singleAndIndicatorVariable));

   OPENGM_TEST(singleAndIndicatorVariable <= multipleAndIndicatorVariable);
   OPENGM_TEST(!(multipleAndIndicatorVariable <= singleAndIndicatorVariable));
   OPENGM_TEST(singleAndIndicatorVariable <= multipleOrIndicatorVariable);
   OPENGM_TEST(!(multipleOrIndicatorVariable <= singleAndIndicatorVariable));
   OPENGM_TEST(singleAndIndicatorVariable <= multipleNotIndicatorVariable1);
   OPENGM_TEST(!(multipleNotIndicatorVariable1 <= singleAndIndicatorVariable));

   OPENGM_TEST(singleOrIndicatorVariable <= singleNotIndicatorVariable1);
   OPENGM_TEST(!(singleNotIndicatorVariable1 <= singleOrIndicatorVariable));

   OPENGM_TEST(!(singleOrIndicatorVariable <= multipleAndIndicatorVariable));
   OPENGM_TEST(multipleAndIndicatorVariable <= singleOrIndicatorVariable);
   OPENGM_TEST(singleOrIndicatorVariable <= multipleOrIndicatorVariable);
   OPENGM_TEST(!(multipleOrIndicatorVariable <= singleOrIndicatorVariable));
   OPENGM_TEST(singleOrIndicatorVariable <= multipleNotIndicatorVariable1);
   OPENGM_TEST(!(multipleNotIndicatorVariable1 <= singleOrIndicatorVariable));

   OPENGM_TEST(multipleAndIndicatorVariable <= multipleOrIndicatorVariable);
   OPENGM_TEST(!(multipleOrIndicatorVariable <= multipleAndIndicatorVariable));
   OPENGM_TEST(multipleAndIndicatorVariable <= multipleNotIndicatorVariable1);
   OPENGM_TEST(!(multipleNotIndicatorVariable1 <= multipleAndIndicatorVariable));

   OPENGM_TEST(multipleOrIndicatorVariable <= multipleNotIndicatorVariable1);
   OPENGM_TEST(!(multipleNotIndicatorVariable1 <= multipleOrIndicatorVariable));

   // >
   OPENGM_TEST(!(emptyIndicatorVariable1 > emptyIndicatorVariable1));
   OPENGM_TEST(!(emptyIndicatorVariable1 > emptyIndicatorVariable2));
   OPENGM_TEST(emptyIndicatorVariable2 > emptyIndicatorVariable1);

   OPENGM_TEST(!(emptyIndicatorVariable1 > singleAndIndicatorVariable));
   OPENGM_TEST(singleAndIndicatorVariable > emptyIndicatorVariable1);
   OPENGM_TEST(!(emptyIndicatorVariable1 > singleOrIndicatorVariable));
   OPENGM_TEST(singleOrIndicatorVariable > emptyIndicatorVariable1);
   OPENGM_TEST(!(emptyIndicatorVariable1 > singleNotIndicatorVariable1));
   OPENGM_TEST(singleNotIndicatorVariable1 > emptyIndicatorVariable1);

   OPENGM_TEST(!(emptyIndicatorVariable1 > multipleAndIndicatorVariable));
   OPENGM_TEST(multipleAndIndicatorVariable > emptyIndicatorVariable1);
   OPENGM_TEST(!(emptyIndicatorVariable1 > multipleOrIndicatorVariable));
   OPENGM_TEST(multipleOrIndicatorVariable > emptyIndicatorVariable1);
   OPENGM_TEST(!(emptyIndicatorVariable1 > multipleNotIndicatorVariable1));
   OPENGM_TEST(multipleNotIndicatorVariable1 > emptyIndicatorVariable1);

   OPENGM_TEST(!(singleAndIndicatorVariable  > singleAndIndicatorVariable));
   OPENGM_TEST(!(singleOrIndicatorVariable   > singleOrIndicatorVariable));
   OPENGM_TEST(!(singleNotIndicatorVariable1 > singleNotIndicatorVariable1));

   OPENGM_TEST(!(multipleAndIndicatorVariable  > multipleAndIndicatorVariable));
   OPENGM_TEST(!(multipleOrIndicatorVariable   > multipleOrIndicatorVariable));
   OPENGM_TEST(!(multipleNotIndicatorVariable1 > multipleNotIndicatorVariable1));

   OPENGM_TEST(!(singleAndIndicatorVariable > singleOrIndicatorVariable));
   OPENGM_TEST(singleOrIndicatorVariable > singleAndIndicatorVariable);
   OPENGM_TEST(!(singleAndIndicatorVariable > singleNotIndicatorVariable1));
   OPENGM_TEST(singleNotIndicatorVariable1 > singleAndIndicatorVariable);

   OPENGM_TEST(!(singleAndIndicatorVariable > multipleAndIndicatorVariable));
   OPENGM_TEST(multipleAndIndicatorVariable > singleAndIndicatorVariable);
   OPENGM_TEST(!(singleAndIndicatorVariable > multipleOrIndicatorVariable));
   OPENGM_TEST(multipleOrIndicatorVariable > singleAndIndicatorVariable);
   OPENGM_TEST(!(singleAndIndicatorVariable > multipleNotIndicatorVariable1));
   OPENGM_TEST(multipleNotIndicatorVariable1 > singleAndIndicatorVariable);

   OPENGM_TEST(!(singleOrIndicatorVariable > singleNotIndicatorVariable1));
   OPENGM_TEST(singleNotIndicatorVariable1 > singleOrIndicatorVariable);

   OPENGM_TEST(singleOrIndicatorVariable > multipleAndIndicatorVariable);
   OPENGM_TEST(!(multipleAndIndicatorVariable > singleOrIndicatorVariable));
   OPENGM_TEST(!(singleOrIndicatorVariable > multipleOrIndicatorVariable));
   OPENGM_TEST(multipleOrIndicatorVariable > singleOrIndicatorVariable);
   OPENGM_TEST(!(singleOrIndicatorVariable > multipleNotIndicatorVariable1));
   OPENGM_TEST(multipleNotIndicatorVariable1 > singleOrIndicatorVariable);

   OPENGM_TEST(!(multipleAndIndicatorVariable > multipleOrIndicatorVariable));
   OPENGM_TEST(multipleOrIndicatorVariable > multipleAndIndicatorVariable);
   OPENGM_TEST(!(multipleAndIndicatorVariable > multipleNotIndicatorVariable1));
   OPENGM_TEST(multipleNotIndicatorVariable1 > multipleAndIndicatorVariable);

   OPENGM_TEST(!(multipleOrIndicatorVariable > multipleNotIndicatorVariable1));
   OPENGM_TEST(multipleNotIndicatorVariable1 > multipleOrIndicatorVariable);

   // >=
   OPENGM_TEST(emptyIndicatorVariable1 >= emptyIndicatorVariable1);
   OPENGM_TEST(!(emptyIndicatorVariable1 >= emptyIndicatorVariable2));
   OPENGM_TEST(emptyIndicatorVariable2 >= emptyIndicatorVariable1);

   OPENGM_TEST(!(emptyIndicatorVariable1 >= singleAndIndicatorVariable));
   OPENGM_TEST(singleAndIndicatorVariable >= emptyIndicatorVariable1);
   OPENGM_TEST(!(emptyIndicatorVariable1 >= singleOrIndicatorVariable));
   OPENGM_TEST(singleOrIndicatorVariable >= emptyIndicatorVariable1);
   OPENGM_TEST(!(emptyIndicatorVariable1 >= singleNotIndicatorVariable1));
   OPENGM_TEST(singleNotIndicatorVariable1 >= emptyIndicatorVariable1);

   OPENGM_TEST(!(emptyIndicatorVariable1 >= multipleAndIndicatorVariable));
   OPENGM_TEST(multipleAndIndicatorVariable >= emptyIndicatorVariable1);
   OPENGM_TEST(!(emptyIndicatorVariable1 >= multipleOrIndicatorVariable));
   OPENGM_TEST(multipleOrIndicatorVariable >= emptyIndicatorVariable1);
   OPENGM_TEST(!(emptyIndicatorVariable1 >= multipleNotIndicatorVariable1));
   OPENGM_TEST(multipleNotIndicatorVariable1 >= emptyIndicatorVariable1);

   OPENGM_TEST(singleAndIndicatorVariable  >= singleAndIndicatorVariable);
   OPENGM_TEST(singleOrIndicatorVariable   >= singleOrIndicatorVariable);
   OPENGM_TEST(singleNotIndicatorVariable1 >= singleNotIndicatorVariable1);

   OPENGM_TEST(multipleAndIndicatorVariable  >= multipleAndIndicatorVariable);
   OPENGM_TEST(multipleOrIndicatorVariable   >= multipleOrIndicatorVariable);
   OPENGM_TEST(multipleNotIndicatorVariable1 >= multipleNotIndicatorVariable1);

   OPENGM_TEST(!(singleAndIndicatorVariable >= singleOrIndicatorVariable));
   OPENGM_TEST(singleOrIndicatorVariable >= singleAndIndicatorVariable);
   OPENGM_TEST(!(singleAndIndicatorVariable >= singleNotIndicatorVariable1));
   OPENGM_TEST(singleNotIndicatorVariable1 >= singleAndIndicatorVariable);

   OPENGM_TEST(!(singleAndIndicatorVariable >= multipleAndIndicatorVariable));
   OPENGM_TEST(multipleAndIndicatorVariable >= singleAndIndicatorVariable);
   OPENGM_TEST(!(singleAndIndicatorVariable >= multipleOrIndicatorVariable));
   OPENGM_TEST(multipleOrIndicatorVariable >= singleAndIndicatorVariable);
   OPENGM_TEST(!(singleAndIndicatorVariable >= multipleNotIndicatorVariable1));
   OPENGM_TEST(multipleNotIndicatorVariable1 >= singleAndIndicatorVariable);

   OPENGM_TEST(!(singleOrIndicatorVariable >= singleNotIndicatorVariable1));
   OPENGM_TEST(singleNotIndicatorVariable1 >= singleOrIndicatorVariable);

   OPENGM_TEST(singleOrIndicatorVariable >= multipleAndIndicatorVariable);
   OPENGM_TEST(!(multipleAndIndicatorVariable >= singleOrIndicatorVariable));
   OPENGM_TEST(!(singleOrIndicatorVariable >= multipleOrIndicatorVariable));
   OPENGM_TEST(multipleOrIndicatorVariable >= singleOrIndicatorVariable);
   OPENGM_TEST(!(singleOrIndicatorVariable >= multipleNotIndicatorVariable1));
   OPENGM_TEST(multipleNotIndicatorVariable1 >= singleOrIndicatorVariable);

   OPENGM_TEST(!(multipleAndIndicatorVariable >= multipleOrIndicatorVariable));
   OPENGM_TEST(multipleOrIndicatorVariable >= multipleAndIndicatorVariable);
   OPENGM_TEST(!(multipleAndIndicatorVariable >= multipleNotIndicatorVariable1));
   OPENGM_TEST(multipleNotIndicatorVariable1 >= multipleAndIndicatorVariable);

   OPENGM_TEST(!(multipleOrIndicatorVariable >= multipleNotIndicatorVariable1));
   OPENGM_TEST(multipleNotIndicatorVariable1 >= multipleOrIndicatorVariable);
}
