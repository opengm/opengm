#ifdef WITH_GUROBI
#include <opengm/operations/adder.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/operations/maximizer.hxx>

#include <opengm/unittests/blackboxtester.hxx>
#include <opengm/unittests/blackboxtests/blackboxtestgrid.hxx>
#include <opengm/unittests/blackboxtests/blackboxtestfull.hxx>
#include <opengm/unittests/blackboxtests/blackboxteststar.hxx>

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/unittests/test.hxx>
#include <opengm/inference/bruteforce.hxx>
#include <opengm/inference/lpgurobi2.hxx>

#include <opengm/functions/potts.hxx>
#include <opengm/functions/constraint_functions/linear_constraint_function.hxx>
#include <opengm/functions/constraint_functions/label_order_function.hxx>
#include <opengm/functions/soft_constraint_functions/sum_constraint_function.hxx>
#include <opengm/functions/soft_constraint_functions/label_cost_function.hxx>

#include <opengm/utilities/random.hxx>

#endif
#include <iostream>

#ifdef WITH_GUROBI
template <class VECTOR>
void showImage(const VECTOR& image) {
   for(size_t i = 0; i < image.size(); ++i) {
      if(image[i] == 1.0) {
         std::cout << "_";
      } else {
         std::cout << "*";
      }
      if((i + 1) % static_cast<size_t>(sqrt(image.size())) == 0) {
         std::cout << std::endl;
      }
   }
}
#endif

int main(){
#ifdef WITH_GUROBI
   {
      typedef opengm::GraphicalModel<double, opengm::Adder > SumGmType;
      typedef opengm::BlackBoxTestGrid<SumGmType> SumGridTest;
      typedef opengm::BlackBoxTestFull<SumGmType> SumFullTest;
      typedef opengm::BlackBoxTestStar<SumGmType> SumStarTest;

      opengm::InferenceBlackBoxTester<SumGmType> sumTester;
      sumTester.addTest(new SumGridTest(4, 4, 2, false, true, SumGridTest::RANDOM, opengm::PASS, 5));
      sumTester.addTest(new SumGridTest(4, 4, 2, false, false,SumGridTest::RANDOM, opengm::PASS, 5));
      sumTester.addTest(new SumStarTest(6,    4, false, true, SumStarTest::RANDOM, opengm::PASS, 20));
      sumTester.addTest(new SumFullTest(5,    2, false, 3,    SumFullTest::RANDOM, opengm::PASS, 5));

      opengm::InferenceBlackBoxTester<SumGmType> sumTesterOpt;
      sumTesterOpt.addTest(new SumGridTest(4, 4, 2, false, true, SumGridTest::RANDOM, opengm::OPTIMAL, 5));
      sumTesterOpt.addTest(new SumGridTest(4, 4, 2, false, false,SumGridTest::RANDOM, opengm::OPTIMAL, 5));
      sumTesterOpt.addTest(new SumStarTest(6,    4, false, true, SumStarTest::RANDOM, opengm::OPTIMAL, 20));
      sumTesterOpt.addTest(new SumFullTest(5,    2, false, 3,    SumFullTest::RANDOM, opengm::OPTIMAL, 5));

      std::cout << "LPGurobi2 Tests"<<std::endl;
      {
         std::cout << "  * Minimization/Adder LP ..."<<std::endl;
         typedef opengm::GraphicalModel<double,opengm::Adder > GmType;
         typedef opengm::LPGurobi2<GmType, opengm::Minimizer>    GUROBI;
         GUROBI::Parameter para;
         para.integerConstraintNodeVar_ = false;
         sumTester.test<GUROBI>(para);
         para.relaxation_ = GUROBI::Parameter::LoosePolytope;
         sumTester.test<GUROBI>(para);
         para.challengeHeuristic_ = GUROBI::Parameter::Weighted;
         para.maxNumConstraintsPerIter_ = 10;
         sumTester.test<GUROBI>(para);
         std::cout << " OK!"<<std::endl;
      }
      {
         std::cout << "  * Minimization/Adder ILP ..."<<std::endl;
         typedef opengm::GraphicalModel<double,opengm::Adder > GmType;
         typedef opengm::LPGurobi2<GmType, opengm::Minimizer>    GUROBI;
         GUROBI::Parameter para;
         para.integerConstraintNodeVar_ = true;
         sumTesterOpt.test<GUROBI>(para);
         para.relaxation_ = GUROBI::Parameter::LoosePolytope;
         sumTesterOpt.test<GUROBI>(para);
         para.challengeHeuristic_ = GUROBI::Parameter::Weighted;
         para.maxNumConstraintsPerIter_ = 10;
         sumTesterOpt.test<GUROBI>(para);

         std::cout << " OK!"<<std::endl;
      }

      {
         std::cout << "  * Maximization/Adder LP ..."<<std::endl;
         typedef opengm::GraphicalModel<double,opengm::Adder > GmType;
         typedef opengm::LPGurobi2<GmType, opengm::Maximizer>    GUROBI;
         GUROBI::Parameter para;
         para.integerConstraintNodeVar_ = false;
         sumTester.test<GUROBI>(para);
         para.relaxation_ = GUROBI::Parameter::LoosePolytope;
         sumTester.test<GUROBI>(para);
         para.challengeHeuristic_ = GUROBI::Parameter::Weighted;
         para.maxNumConstraintsPerIter_ = 10;
         sumTester.test<GUROBI>(para);
         std::cout << " OK!"<<std::endl;
      }
      {
         std::cout << "  * Maximization/Adder ILP ..."<<std::endl;
         typedef opengm::GraphicalModel<double,opengm::Adder > GmType;
         typedef opengm::LPGurobi2<GmType, opengm::Maximizer>    GUROBI;
         GUROBI::Parameter para;
         para.integerConstraintNodeVar_ = true;
         sumTesterOpt.test<GUROBI>(para);
         para.relaxation_ = GUROBI::Parameter::LoosePolytope;
         sumTesterOpt.test<GUROBI>(para);
         para.challengeHeuristic_ = GUROBI::Parameter::Weighted;
         para.maxNumConstraintsPerIter_ = 10;
         sumTesterOpt.test<GUROBI>(para);

         std::cout << " OK!"<<std::endl;
      }

      {
         std::cout << "Test LPGurobi2 with function transformation!"<<std::endl;
         typedef double ValueType;
         typedef size_t IndexType;
         typedef size_t LabelType;
         typedef opengm::Adder OperatorType;
         typedef opengm::Minimizer AccumulatorType;
         typedef opengm::DiscreteSpace<IndexType, LabelType> SpaceType;

         typedef opengm::SumConstraintFunction<ValueType, IndexType, LabelType> SumConstraintFunctionType;
         typedef opengm::LabelCostFunction<ValueType, IndexType, LabelType>     LabelCostFunctionType;

         typedef opengm::meta::TypeListGenerator<
               opengm::ExplicitFunction<ValueType, IndexType, LabelType>,
               opengm::PottsFunction<ValueType, IndexType, LabelType>,
               SumConstraintFunctionType,
               LabelCostFunctionType
               >::type FunctionTypeList;

         typedef opengm::GraphicalModel<
               ValueType,
               OperatorType,
               FunctionTypeList,
               SpaceType
               > GMType;
         typedef GMType::FunctionIdentifier   FunctionIdentifier;

         typedef opengm::LPGurobi2<GMType, AccumulatorType> LPGurobi;

         const IndexType gridSizeN = 4;
         const IndexType gridSizeM = 3;
         const LabelType numLabels = 2;
         const ValueType coefficientLabel0 = 1.1;
         const ValueType coefficientLabel1 = 2.7;
         const ValueType lambda = 42.0;

         // create test data
         std::vector<std::vector<ValueType> > data(gridSizeN, std::vector<ValueType>(gridSizeM, coefficientLabel0));
         typedef opengm::RandomUniformInteger<LabelType> RandomUniformIntegerType;
         RandomUniformIntegerType randomGenerator(0, numLabels);
         for(IndexType i = 0; i < gridSizeN; ++i) {
            for(IndexType j = 0; j < gridSizeM; ++j) {
               if(randomGenerator() == 1) {
                  data[i][j] = coefficientLabel1;
               }
            }
         }

         // build model
         GMType gm;

         // add variables
         for(IndexType i = 0; i < gridSizeN * gridSizeM; ++i) {
            gm.addVariable(numLabels);
         }

         // create shape and coefficients
         std::vector<LabelType> shape(gridSizeM, numLabels);
         std::vector<ValueType> coefficients;
         for(IndexType i = 0; i < gridSizeM; ++i) {
            coefficients.push_back(coefficientLabel0);
            coefficients.push_back(coefficientLabel1);

         }
         // add row constraints
         std::cout << "add row constraints" << std::endl;
         std::vector<IndexType> indices(gridSizeM);
         for(IndexType i = 0; i < gridSizeN; ++i) {
            ValueType rowSum = 0.0;
            for(IndexType j = 0; j < gridSizeM; ++j) {
               rowSum += data[i][j];
               indices[j] = i + (j * gridSizeN);
            }
            SumConstraintFunctionType f(shape.begin(), shape.end(), coefficients.begin(), coefficients.end(), false, lambda, rowSum);
            FunctionIdentifier fId = gm.addFunction(f);
            gm.addFactor(fId, indices.begin(), indices.end());
         }

         // add column constraints
         std::cout << "add column constraints" << std::endl;
         shape.resize(gridSizeN, numLabels);
         coefficients.clear();
         for(IndexType i = 0; i < gridSizeN; ++i) {
            coefficients.push_back(coefficientLabel0);
            coefficients.push_back(coefficientLabel1);

         }
         indices.resize(gridSizeN);
         for(IndexType i = 0; i < gridSizeM; ++i) {
            ValueType colSum = 0.0;
            for(IndexType j = 0; j < gridSizeN; ++j) {
               colSum += data[j][i];
               indices[j] = j + (i * gridSizeN);
            }
            SumConstraintFunctionType f(shape.begin(), shape.end(), coefficients.begin(), coefficients.end(), false, lambda, colSum);
            FunctionIdentifier fId = gm.addFunction(f);
            gm.addFactor(fId, indices.begin(), indices.end());
         }

         // add label cost function
         LabelCostFunctionType labelCostFunction(gridSizeN * gridSizeM, numLabels, 0, 3.14159);
         FunctionIdentifier labelCostFunctionId = gm.addFunction(labelCostFunction);
         std::vector<IndexType> labelCostFunctionIndices;
         for(IndexType i = 0; i < gridSizeM * gridSizeN; ++i) {
            labelCostFunctionIndices.push_back(i);
         }
         gm.addFactor(labelCostFunctionId, labelCostFunctionIndices.begin(), labelCostFunctionIndices.end());

         LPGurobi::Parameter para;
         //para.verbose_ = true;
         para.numberOfThreads_ = 1;
         para.integerConstraintNodeVar_ = false;
         para.integerConstraintFactorVar_ = false;
         para.useSoftConstraints_ = false;
         para.useFunctionTransfer_ = false;
         para.maxNumIterations_ = 0;

         std::cout << "build lpSolver!"<<std::endl;
         LPGurobi lpSolver(gm, para);

         std::cout << "build lpSolver with function transformation!"<<std::endl;
         para.useFunctionTransfer_ = true;

         LPGurobi lpSolverWithFunctionTransformation(gm, para);

         para.integerConstraintNodeVar_ = true;
         para.useSoftConstraints_ = false;
         para.useFunctionTransfer_ = false;
         para.relaxation_ = LPGurobi::Parameter::TightPolytope;

         std::cout << "build ilpSolver!"<<std::endl;
         LPGurobi ilpSolver(gm, para);

         std::cout << "build ilpSolver with function transformation!"<<std::endl;
         para.useFunctionTransfer_ = true;
         LPGurobi ilpSolverWithFunctionTransformation(gm, para);

         // infer
         std::vector<LabelType> result;

         std::cout << "infer lpSolver!"<<std::endl;
         lpSolver.infer();
         std::cout << "arg lpSolver!"<<std::endl;
         lpSolver.arg(result);
         std::cout << "Energy: " << gm.evaluate(result.begin()) << std::endl;

         std::cout << "infer lpSolver with function transformation!"<<std::endl;
         lpSolverWithFunctionTransformation.infer();
         std::cout << "arg lpSolver with function transformation!"<<std::endl;
         lpSolverWithFunctionTransformation.arg(result);
         std::cout << "Energy: " << gm.evaluate(result.begin()) << std::endl;

         std::cout << "infer ilpSolver!"<<std::endl;
         ilpSolver.infer();
         std::cout << "arg ilpSolver!"<<std::endl;
         ilpSolver.arg(result);
         const ValueType energyWithoutFunctionTransformation = gm.evaluate(result.begin());
         std::cout << "Energy: " << energyWithoutFunctionTransformation << std::endl;

         std::cout << "infer ilpSolver with function transformation!"<<std::endl;
         ilpSolverWithFunctionTransformation.infer();
         std::cout << "arg ilpSolver with function transformation!"<<std::endl;
         ilpSolverWithFunctionTransformation.arg(result);
         const ValueType energyWithFunctionTransformation = gm.evaluate(result.begin());
         std::cout << "Energy: " << energyWithFunctionTransformation << std::endl;

         OPENGM_TEST_EQUAL_TOLERANCE(energyWithFunctionTransformation, energyWithoutFunctionTransformation, OPENGM_FLOAT_TOL);
      }
      {
         std::cout << "Test LPGurobi with constraint functions!"<<std::endl;

         typedef double ValueType;
         typedef size_t IndexType;
         typedef size_t LabelType;
         typedef opengm::Adder OperatorType;
         typedef opengm::Minimizer AccumulatorType;
         typedef opengm::DiscreteSpace<IndexType, LabelType> SpaceType;

         typedef opengm::meta::TypeListGenerator<
            opengm::ExplicitFunction<ValueType, IndexType, LabelType>,
            opengm::PottsFunction<ValueType, IndexType, LabelType>,
            opengm::LabelOrderFunction<ValueType, IndexType, LabelType>,
            opengm::LinearConstraintFunction<ValueType, IndexType, LabelType>
            >::type FunctionTypeList;

         typedef opengm::GraphicalModel<
            ValueType,
            OperatorType,
            FunctionTypeList,
            SpaceType
         > GmWithConstraintFunctionsType;
         typedef GmWithConstraintFunctionsType::FunctionIdentifier   FunctionIdentifier;

         typedef opengm::LPGurobi2<GmWithConstraintFunctionsType, AccumulatorType> LPGurobi;

         // build model with Constraint functions
         const IndexType gridSize = 8;
         const LabelType numLabels = 2;

         // create binary image (upper half 1 lower half 0)
         std::vector<ValueType> binaryImage(gridSize * gridSize, 0.0);
         for(size_t i = 0; i < gridSize / 2; i++) {
            for(size_t j = 0; j < gridSize; ++j) {
               binaryImage[(i * gridSize) + j] = 1.0;
            }
         }

         // add 40% noise
         typedef opengm::RandomUniformInteger<LabelType> RandomUniformIntegerType;
         RandomUniformIntegerType randomInt(0, 101);
         for(size_t i = 0; i < gridSize * gridSize; i++) {
            if(randomInt() <= 40) {
               if(binaryImage[i] == 1.0) {
                  binaryImage[i] = 0.0;
               } else {
                  binaryImage[i] = 1.0;
               }
            }
         }

         // show image
         showImage(binaryImage);

         GmWithConstraintFunctionsType gmGrid;
         GmWithConstraintFunctionsType gmGrid2;

         for(size_t i = 0; i < gridSize * gridSize; i++) {
            gmGrid.addVariable(numLabels);
            gmGrid2.addVariable(numLabels);
         }

         // add unaries
         for(IndexType i = 0; i < gridSize * gridSize; i++) {
            size_t shape[] = {numLabels};
            size_t var[] = {i};

            opengm::ExplicitFunction<ValueType, IndexType, LabelType> function(shape,shape+1);
            for(LabelType j = 0; j < numLabels; j++) {
               function(j) = std::fabs(j - binaryImage[i]);
            }
            FunctionIdentifier funcId = gmGrid.addSharedFunction(function);
            gmGrid.addFactor(funcId, var, var+1);
            FunctionIdentifier funcId2 = gmGrid2.addSharedFunction(function);
            gmGrid2.addFactor(funcId2, var, var+1);
         }

         size_t var[2];
         // label order factors
         // ensure that each column has non decreasing labels from top to bottom
         opengm::LabelOrderFunction<ValueType, IndexType, LabelType>::LabelOrderType labelOrder(numLabels);
         for(size_t i = 0; i < numLabels; i++) {
            labelOrder[i] = static_cast<opengm::LabelOrderFunction<ValueType, IndexType, LabelType>::LabelOrderType::value_type>(numLabels - i);
         }
         opengm::LabelOrderFunction<ValueType, IndexType, LabelType> labelOrderFunction(numLabels, numLabels, labelOrder, 0.0, 1000.0);
         FunctionIdentifier labelOrderFuncId = gmGrid.addFunction(labelOrderFunction);
         FunctionIdentifier labelOrderFuncId2 = gmGrid2.addFunction(labelOrderFunction);
         for(size_t i = 0; i < gridSize; i++) {
            if(i + 1 < gridSize) {
               for(size_t j = 0; j < gridSize; j++) {
                  size_t v = (i * gridSize) + j;
                  var[0] = v;
                  var[1] = v + gridSize;
                  gmGrid.addFactor(labelOrderFuncId, var, var + 2);
                  gmGrid2.addFactor(labelOrderFuncId2, var, var + 2);
               }
            }
         }

         // linear constraint factors
         // ensure that each row has only one label for gmGrid
         opengm::LinearConstraintFunction<ValueType, IndexType, LabelType>::LinearConstraintsContainerType constraints(1);
         constraints[0].setBound(1.0);
         constraints[0].setConstraintOperator(opengm::LinearConstraintFunction<ValueType, IndexType, LabelType>::LinearConstraintType::LinearConstraintOperatorType::GreaterEqual);
         for(LabelType label = 0; label < numLabels; ++label) {
            opengm::LinearConstraintFunction<ValueType, IndexType, LabelType>::LinearConstraintType::IndicatorVariableType indicatorVar;
            for(IndexType i = 0; i < gridSize; ++i) {
               indicatorVar.add(i, label);
            }
            constraints[0].add(indicatorVar, 1.0);
         }

         std::vector<LabelType> shape(gridSize, numLabels);
         opengm::LinearConstraintFunction<ValueType, IndexType, LabelType> linearConstraintFunction(shape.begin(), shape.end(), constraints, 0.0, 100.0);

         FunctionIdentifier linearConstraintFuncId = gmGrid.addFunction(linearConstraintFunction);
         for(size_t i = 0; i < gridSize; ++i) {
            std::vector<IndexType> indices(gridSize);
            for(size_t j = 0; j < gridSize; ++j) {
               indices[j] = (i * gridSize) + j;
            }
            gmGrid.addFactor(linearConstraintFuncId, indices.begin(), indices.end());
         }

         // ensure that each row has each label assigned to at least one variable for gmGrid2
         opengm::LinearConstraintFunction<ValueType, IndexType, LabelType>::LinearConstraintsContainerType constraints2(numLabels);
         for(LabelType label = 0; label < numLabels; ++label) {
            constraints2[label].setBound(1.0);
            constraints2[label].setConstraintOperator(opengm::LinearConstraintFunction<ValueType, IndexType, LabelType>::LinearConstraintType::LinearConstraintOperatorType::Equal);
            opengm::LinearConstraintFunction<ValueType, IndexType, LabelType>::LinearConstraintType::IndicatorVariableType indicatorVar;
            indicatorVar.setLogicalOperatorType(opengm::LinearConstraintFunction<ValueType, IndexType, LabelType>::LinearConstraintType::IndicatorVariableType::Or);
            for(IndexType i = 0; i < gridSize; ++i) {
               indicatorVar.add(i, label);
            }
            constraints2[label].add(indicatorVar, 1.0);
         }

         std::vector<LabelType> shape2(gridSize, numLabels);
         opengm::LinearConstraintFunction<ValueType, IndexType, LabelType> linearConstraintFunction2(shape2.begin(), shape2.end(), constraints2, 0.0, 100.0);

         FunctionIdentifier linearConstraintFuncId2 = gmGrid2.addFunction(linearConstraintFunction2);
         for(size_t i = 0; i < gridSize; ++i) {
            std::vector<IndexType> indices(gridSize);
            for(size_t j = 0; j < gridSize; ++j) {
               indices[j] = (i * gridSize) + j;
            }
            gmGrid2.addFactor(linearConstraintFuncId2, indices.begin(), indices.end());
         }

         // ensure that variable ((gridSize - 3) * gridSize) and ((gridSize - 2) * gridSize) have different labels
         opengm::LinearConstraintFunction<ValueType, IndexType, LabelType>::LinearConstraintsContainerType constraints3(numLabels);
         for(LabelType label = 0; label < numLabels; ++label) {
            constraints3[label].setBound(0.0);
            constraints3[label].setConstraintOperator(opengm::LinearConstraintFunction<ValueType, IndexType, LabelType>::LinearConstraintType::LinearConstraintOperatorType::Equal);
            opengm::LinearConstraintFunction<ValueType, IndexType, LabelType>::LinearConstraintType::IndicatorVariableType indicatorVar;
            indicatorVar.setLogicalOperatorType(opengm::LinearConstraintFunction<ValueType, IndexType, LabelType>::LinearConstraintType::IndicatorVariableType::Not);
            for(IndexType i = 0; i < 2; ++i) {
               indicatorVar.add(i, label);
            }
            constraints3[label].add(indicatorVar, 1.0);
         }

         std::vector<LabelType> shape3(2, numLabels);
         opengm::LinearConstraintFunction<ValueType, IndexType, LabelType> linearConstraintFunction3(shape3.begin(), shape3.end(), constraints3, 0.0, 100.0);

         FunctionIdentifier linearConstraintFuncId3 = gmGrid2.addFunction(linearConstraintFunction3);
         std::vector<IndexType> indices3(2);
         indices3[0] = ((gridSize - 3) * gridSize);
         indices3[1] = ((gridSize - 2) * gridSize);
         gmGrid2.addFactor(linearConstraintFuncId3, indices3.begin(), indices3.end());

         // Disable label 1 for variable (3 * gridSize) + 4
         opengm::LinearConstraintFunction<ValueType, IndexType, LabelType>::LinearConstraintsContainerType constraints4(1);
         constraints4[0].setBound(1.0);
         constraints4[0].setConstraintOperator(opengm::LinearConstraintFunction<ValueType, IndexType, LabelType>::LinearConstraintType::LinearConstraintOperatorType::Equal);
         opengm::LinearConstraintFunction<ValueType, IndexType, LabelType>::LinearConstraintType::IndicatorVariableType indicatorVar;
         indicatorVar.setLogicalOperatorType(opengm::LinearConstraintFunction<ValueType, IndexType, LabelType>::LinearConstraintType::IndicatorVariableType::Not);
         indicatorVar.add(0, LabelType(1));
         constraints4[0].add(indicatorVar, 1.0);

         std::vector<LabelType> shape4(1, numLabels);
         opengm::LinearConstraintFunction<ValueType, IndexType, LabelType> linearConstraintFunction4(shape4.begin(), shape4.end(), constraints4, 0.0, 100000.0);

         FunctionIdentifier linearConstraintFuncId4 = gmGrid2.addFunction(linearConstraintFunction4);
         std::vector<IndexType> indices4(1);
         indices4[0] = (3 * gridSize) + 4;
         gmGrid2.addFactor(linearConstraintFuncId4, indices4.begin(), indices4.end());

         // solve problem
         LPGurobi::Parameter para;
         //para.verbose_ = true;
         para.integerConstraintNodeVar_ = false;
         para.integerConstraintFactorVar_ = false;
         para.useSoftConstraints_ = false;
         para.maxNumIterations_ = 0;
         std::cout << "build lpSolver!"<<std::endl;
         LPGurobi gmGridLpSolver(gmGrid, para);
         LPGurobi gmGrid2LpSolver(gmGrid2, para);
         std::cout << "build lpSolver2!"<<std::endl;
         para.relaxation_ = LPGurobi::Parameter::TightPolytope;
         LPGurobi gmGridLpSolver2(gmGrid, para);
         LPGurobi gmGrid2LpSolver2(gmGrid2, para);
         std::cout << "build lpSolver3!"<<std::endl;
         para.relaxation_ = LPGurobi::Parameter::LoosePolytope;
         LPGurobi gmGridLpSolver3(gmGrid, para);
         LPGurobi gmGrid2LpSolver3(gmGrid2, para);
         std::cout << "build lpSolver4!"<<std::endl;
         para.maxNumConstraintsPerIter_ = 10;
         LPGurobi gmGridLpSolver4(gmGrid, para);
         LPGurobi gmGrid2LpSolver4(gmGrid2, para);

         std::cout << "build lpSolver5!"<<std::endl;
         para.challengeHeuristic_ = LPGurobi::Parameter::Weighted;
         LPGurobi gmGridLpSolver5(gmGrid, para);
         LPGurobi gmGrid2LpSolver5(gmGrid2, para);

         para.integerConstraintNodeVar_ = true;
         para.integerConstraintFactorVar_ = true;
         para.useSoftConstraints_ = false;
         para.relaxation_ = LPGurobi::Parameter::LocalPolytope;
         para.challengeHeuristic_ = LPGurobi::Parameter::Random;
         para.maxNumConstraintsPerIter_ = 0;
         std::cout << "build ilpSolver!"<<std::endl;
         LPGurobi gmGridIlpSolver(gmGrid, para);
         LPGurobi gmGrid2IlpSolver(gmGrid2, para);

         std::cout << "build ilpSolver2!"<<std::endl;
         para.relaxation_ = LPGurobi::Parameter::TightPolytope;
         LPGurobi gmGridIlpSolver2(gmGrid, para);
         LPGurobi gmGrid2IlpSolver2(gmGrid2, para);
         std::cout << "build ilpSolver3!"<<std::endl;
         para.relaxation_ = LPGurobi::Parameter::LoosePolytope;
         LPGurobi gmGridIlpSolver3(gmGrid, para);
         LPGurobi gmGrid2IlpSolver3(gmGrid2, para);
         std::cout << "build ilpSolver4!"<<std::endl;
         para.maxNumConstraintsPerIter_ = 10;
         LPGurobi gmGridIlpSolver4(gmGrid, para);
         LPGurobi gmGrid2IlpSolver4(gmGrid2, para);
         std::cout << "build ilpSolver5!"<<std::endl;
         para.challengeHeuristic_ = LPGurobi::Parameter::Weighted;
         LPGurobi gmGridIlpSolver5(gmGrid, para);
         LPGurobi gmGrid2IlpSolver5(gmGrid2, para);

         para = LPGurobi::Parameter();
         para.useSoftConstraints_ = true;
         std::cout << "build lpSolverSoft!"<<std::endl;
         LPGurobi gmGridLpSolverSoft(gmGrid, para);
         LPGurobi gmGrid2LpSolverSoft(gmGrid2, para);

         para.integerConstraintNodeVar_ = true;
         std::cout << "build ilpSolverSoft!"<<std::endl;
         LPGurobi gmGridIlpSolverSoft(gmGrid, para);
         LPGurobi gmGrid2IlpSolverSoft(gmGrid2, para);

         std::vector<LabelType> gmGridResult;
         std::vector<LabelType> gmGrid2Result;

         std::cout << "infer lpSolver!"<<std::endl;
         gmGridLpSolver.infer();
         gmGrid2LpSolver.infer();
         std::cout << "arg lpSolver!"<<std::endl;
         gmGridLpSolver.arg(gmGridResult);
         gmGrid2LpSolver.arg(gmGrid2Result);
         // show image
         std::cout << std::endl;
         showImage(gmGridResult);
         std::cout << std::endl;
         showImage(gmGrid2Result);
         std::cout << std::endl;

         std::cout << "infer lpSolver2!"<<std::endl;
         gmGridLpSolver2.infer();
         gmGrid2LpSolver2.infer();
         std::cout << "arg lpSolver2!"<<std::endl;
         gmGridLpSolver2.arg(gmGridResult);
         gmGrid2LpSolver2.arg(gmGrid2Result);
         // show image
         std::cout << std::endl;
         showImage(gmGridResult);
         std::cout << std::endl;
         showImage(gmGrid2Result);
         std::cout << std::endl;

         std::cout << "infer lpSolver3!"<<std::endl;
         gmGridLpSolver3.infer();
         gmGrid2LpSolver3.infer();
         std::cout << "arg lpSolver3!"<<std::endl;
         gmGridLpSolver3.arg(gmGridResult);
         gmGrid2LpSolver3.arg(gmGrid2Result);
         // show image
         std::cout << std::endl;
         showImage(gmGridResult);
         std::cout << std::endl;
         showImage(gmGrid2Result);
         std::cout << std::endl;

         std::cout << "infer lpSolver4!"<<std::endl;
         gmGridLpSolver4.infer();
         gmGrid2LpSolver4.infer();
         std::cout << "arg lpSolver4!"<<std::endl;
         gmGridLpSolver4.arg(gmGridResult);
         gmGrid2LpSolver4.arg(gmGrid2Result);
         // show image
         std::cout << std::endl;
         showImage(gmGridResult);
         std::cout << std::endl;
         showImage(gmGrid2Result);
         std::cout << std::endl;

         std::cout << "infer lpSolver5!"<<std::endl;
         gmGridLpSolver5.infer();
         gmGrid2LpSolver5.infer();
         std::cout << "arg lpSolver5!"<<std::endl;
         gmGridLpSolver5.arg(gmGridResult);
         gmGrid2LpSolver5.arg(gmGrid2Result);
         // show image
         std::cout << std::endl;
         showImage(gmGridResult);
         std::cout << std::endl;
         showImage(gmGrid2Result);
         std::cout << std::endl;

         std::cout << "infer ilpSolver!"<<std::endl;
         gmGridIlpSolver.infer();
         gmGrid2IlpSolver.infer();
         std::cout << "arg ilpSolver!"<<std::endl;
         gmGridIlpSolver.arg(gmGridResult);
         gmGrid2IlpSolver.arg(gmGrid2Result);
         // show image
         std::cout << std::endl;
         showImage(gmGridResult);
         std::cout << std::endl;
         showImage(gmGrid2Result);
         std::cout << std::endl;

         std::cout << "infer ilpSolver2!"<<std::endl;
         gmGridIlpSolver2.infer();
         gmGrid2IlpSolver2.infer();
         std::cout << "arg ilpSolver2!"<<std::endl;
         gmGridIlpSolver2.arg(gmGridResult);
         gmGrid2IlpSolver2.arg(gmGrid2Result);
         // show image
         std::cout << std::endl;
         showImage(gmGridResult);
         std::cout << std::endl;
         showImage(gmGrid2Result);
         std::cout << std::endl;

         std::cout << "infer ilpSolver2!"<<std::endl;
         gmGridIlpSolver2.infer();
         gmGrid2IlpSolver2.infer();
         std::cout << "arg ilpSolver2!"<<std::endl;
         gmGridIlpSolver2.arg(gmGridResult);
         gmGrid2IlpSolver2.arg(gmGrid2Result);
         // show image
         std::cout << std::endl;
         showImage(gmGridResult);
         std::cout << std::endl;
         showImage(gmGrid2Result);
         std::cout << std::endl;

         std::cout << "infer ilpSolver3!"<<std::endl;
         gmGridIlpSolver3.infer();
         gmGrid2IlpSolver3.infer();
         std::cout << "arg ilpSolver3!"<<std::endl;
         gmGridIlpSolver3.arg(gmGridResult);
         gmGrid2IlpSolver3.arg(gmGrid2Result);
         // show image
         std::cout << std::endl;
         showImage(gmGridResult);
         std::cout << std::endl;
         showImage(gmGrid2Result);
         std::cout << std::endl;

         std::cout << "infer ilpSolver4!"<<std::endl;
         gmGridIlpSolver4.infer();
         gmGrid2IlpSolver4.infer();
         std::cout << "arg ilpSolver4!"<<std::endl;
         gmGridIlpSolver4.arg(gmGridResult);
         gmGrid2IlpSolver4.arg(gmGrid2Result);
         // show image
         std::cout << std::endl;
         showImage(gmGridResult);
         std::cout << std::endl;
         showImage(gmGrid2Result);
         std::cout << std::endl;

         std::cout << "infer ilpSolver5!"<<std::endl;
         gmGridIlpSolver5.infer();
         gmGrid2IlpSolver5.infer();
         std::cout << "arg ilpSolver5!"<<std::endl;
         gmGridIlpSolver5.arg(gmGridResult);
         gmGrid2IlpSolver5.arg(gmGrid2Result);
         // show image
         std::cout << std::endl;
         showImage(gmGridResult);
         std::cout << std::endl;
         showImage(gmGrid2Result);
         std::cout << std::endl;

         std::cout << "infer lpSolverSoft!"<<std::endl;
         gmGridLpSolverSoft.infer();
         gmGrid2LpSolverSoft.infer();
         std::cout << "arg lpSolverSoft!"<<std::endl;
         gmGridLpSolverSoft.arg(gmGridResult);
         gmGrid2LpSolverSoft.arg(gmGrid2Result);
         // show image
         std::cout << std::endl;
         showImage(gmGridResult);
         std::cout << std::endl;
         showImage(gmGrid2Result);
         std::cout << std::endl;

         std::cout << "infer ilpSolverSoft!"<<std::endl;
         gmGridIlpSolverSoft.infer();
         gmGrid2IlpSolverSoft.infer();
         std::cout << "arg ilpSolverSoft!"<<std::endl;
         gmGridIlpSolverSoft.arg(gmGridResult);
         gmGrid2IlpSolverSoft.arg(gmGrid2Result);
         // show image
         std::cout << std::endl;
         showImage(gmGridResult);
         std::cout << std::endl;
         showImage(gmGrid2Result);
         std::cout << std::endl;

         std::vector<LabelType> gmGridResult2;
         std::vector<LabelType> gmGrid2Result2;
         gmGridIlpSolver.arg(gmGridResult2);
         gmGrid2IlpSolver.arg(gmGrid2Result2);
         OPENGM_TEST_EQUAL(gmGrid.evaluate(gmGridResult.begin()), gmGrid.evaluate(gmGridResult2.begin()))
         OPENGM_TEST_EQUAL(gmGrid.evaluate(gmGrid2Result.begin()), gmGrid.evaluate(gmGrid2Result2.begin()))

         gmGridIlpSolver2.arg(gmGridResult2);
         gmGrid2IlpSolver2.arg(gmGrid2Result2);
         OPENGM_TEST_EQUAL(gmGrid.evaluate(gmGridResult.begin()), gmGrid.evaluate(gmGridResult2.begin()))
         OPENGM_TEST_EQUAL(gmGrid.evaluate(gmGrid2Result.begin()), gmGrid.evaluate(gmGrid2Result2.begin()))
      }
      std::cout << "done!"<<std::endl;
   }
#else
   std::cout << "LPGurobi2 test is disabled (compiled without Gurobi) "<< std::endl;
#endif
   return 0;
}
