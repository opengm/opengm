#include <iostream>

#if defined(WITH_CPLEX) || defined(WITH_GUROBI)
   #include <opengm/opengm.hxx>
   #include <opengm/graphicalmodel/graphicalmodel.hxx>
   #ifdef WITH_HDF5
      #include <cstdio>
      #include <opengm/graphicalmodel/graphicalmodel_hdf5.hxx>
   #endif
   #include <opengm/graphicalmodel/space/simplediscretespace.hxx>
   #include <opengm/operations/minimizer.hxx>
   #include <opengm/operations/adder.hxx>

   #include <opengm/functions/explicit_function.hxx>
   #include <opengm/functions/potts.hxx>
   #include <opengm/functions/constraint_functions/linear_constraint_function.hxx>

   #include "../../tutorials/c++/applications/utilities/pgmimage.hxx"
#endif
#ifdef WITH_CPLEX
   #include <opengm/inference/lpcplex2.hxx>
#endif
#ifdef WITH_GUROBI
   #include <opengm/inference/lpgurobi2.hxx>
#endif

#if defined(WITH_CPLEX) || defined(WITH_GUROBI)
   // forward declarations
   template <class VALUE_TYPE>
   void read_arguments(const int argc, char** const argv, std::string& inputImageFileName, std::string& outputImageFileName, VALUE_TYPE& foregroundExampleColor, VALUE_TYPE& backgroundExampleColor, VALUE_TYPE& minPercentForeground, VALUE_TYPE& maxPercentForeground, std::string& mode, std::string& relaxation, std::string& heuristic, VALUE_TYPE& lambda, std::string& solverName, double& timelimit);

   template <class PARAMETER_TYPE>
   void set_solver_parameter_settings(PARAMETER_TYPE& parameter, const std::string& mode, const std::string& relaxation, const std::string& heuristic, const double timelimit);

   #ifdef WITH_HDF5
      template <class VISITOR>
      void storeVisitor(const VISITOR& visitor, const std::string& filename);

      template <class VECTOR_TYPE>
      void storeVector(const std::string& filename, const std::string& dataset, VECTOR_TYPE& vec);

      bool fileExists(const std::string& filename);
   #endif
#endif

/*******************************************************************************
 * This is an example for foreground background image segmentation. It utilizes
 * the Constrained Graphical Models Framework to define a minimum and a maximum
 * count of pixels for the foreground. The segmentation is based on the tutorial
 * opengm/src/tutorials/imageSegSimple.cxx.
 *
 * Data-Term:    Truncated L1-distance of pixel-color to exemplary color
 * Regularizer:  Potts (penalize boundary length)
 * Constraints:  LinearConstraintFunction limiting the number of variables with
 *               foreground label.
 * Optimizer:    LPCplex2 or LPGurobi2
 * Dependencies: Support for Cplex and / or Gurobi.
 *
 * Note: If compiled with HDF5 support the timing visitor is used for logging
 *       and the logging from the visitor will be stored in a HDF5 file.
 *       Otherwise the verbose visitor is used and logging is only printed on
 *       console.
 *
 * Usage:        example-foreground-background-segmentation inputImageFileName outputImageFileName foregroundExampleColor backgroundExampleColor minPercentForeground maxPercentForeground [mode] [relaxation] [heuristic] [regularization] [solver] [timelimit]
 * Parameter:    inputImageFileName 8bit-gray images in PGM-format.
 *               outputImageFileName File name for the result image.
 *               foregroundExampleColor Example color value for foreground.
 *               backgroundExampleColor Example color value for background.
 *               minPercentForeground Minimum percent of variables which get
 *                                    assigned to foreground. Range: [0.0, 1.0]
 *               maxPercentForeground Maximum percent of variables which get
 *                                    assigned to foreground. Range: [0.0, 1.0]
 *               mode (Optional) Select optimization mode.
 *                               Possible Values:
 *                                  lp Use linear programming model.
 *                                  ilp Use integer linear programming model.
 *                               Default: ilp
 *              relaxation (Optional) Select relaxation mode.
 *                                    Possible Values:
 *                                       local Local Polytope constraints before
 *                                             first iteration. Linear
 *                                             Constraint Function constraints
 *                                              are added iteratively if
 *                                              violated.
 *                                       loose Local Polytope constraints and
 *                                             Linear Constraint Function
 *                                             constraints are added iteratively
 *                                             if violated.
 *                                       tight Local Polytope constraints and
 *                                             Linear Constraint Function
 *                                             constraints are added before
 *                                             first iteration.
 *                                    Default: local
 *              heuristic (Optional) Select heuristic mode.
 *                                   Possible Values:
 *                                      random In each iteration add all
 *                                             violated constraints.
 *                                      weighted In each iteration add only a
 *                                               limited number of violated
 *                                               constraints. Select them by
 *                                               amount of violation.
 *                                   Default: random
 *              regularization (Optional) Penalization value for boundary
 *                                        length. Only added if value is
 *                                        different from zero.
 *                                        Default: 0.0
 *              solver (Optional) Select the used lp / ilp solver.
 *                                Possible Values:
 *                                   cplex Select Cplex as solver. Only
 *                                         available if compiled with Cplex
 *                                         support.
 *                                   gurobi Select Gurobi as solver. Only
 *                                          available if compiled with Gurobi
 *                                          support.
 *                                Default: cplex if compiled with Cplex support,
 *                                         gurobi otherwise.
 *              timelimit (Optional) Maximum time the solver has. Infinite if
 *                                   set to zero.
 *                                   Default: 0.0
 * Example: example-foreground-background-segmentation opengm/src/tutorials/data/coins.pgm out.pgm 30 90 0.4 0.6 lp loose random 20.0 gurobi 3600.0
 ******************************************************************************/
int main(int argc, char** argv) {
   #if !(defined(WITH_CPLEX) || defined(WITH_GUROBI))
      std::cout << "Image Segmentation Example with Constrained Graphical Models requires Cplex or Gurobi" << std::endl;
   #else
      // typedefs
      typedef double ValueType;
      typedef size_t IndexType;
      typedef unsigned char LabelType;

      typedef opengm::ExplicitFunction<ValueType, IndexType, LabelType>         ExplicitFunctionType;
      typedef opengm::PottsFunction<ValueType, IndexType, LabelType>            PottsFunctionType;
      typedef opengm::LinearConstraintFunction<ValueType, IndexType, LabelType> LinearConstraintFunctionType;

      typedef opengm::meta::TypeListGenerator<ExplicitFunctionType, PottsFunctionType, LinearConstraintFunctionType>::type FunctionTypeList;
      typedef opengm::SimpleDiscreteSpace<IndexType, LabelType>                                                            SpaceType;
      typedef opengm::GraphicalModel<ValueType, opengm::Adder, FunctionTypeList, SpaceType>                                GraphicalModelType;
      typedef GraphicalModelType::FunctionIdentifier                                                                       FunctionIdentifierType;

      // read arguments
      std::string inputImageFileName;
      std::string outputImageFileName;

      ValueType foregroundExampleColor;
      ValueType backgroundExampleColor;
      ValueType minPercentForeground;
      ValueType maxPercentForeground;
      std::string mode("ilp");
      std::string relaxation("local");
      std::string heuristic("random");
      ValueType lambda = 0.0;
      #ifdef WITH_CPLEX
         std::string solverName("cplex");
      #else
         std::string solverName("gurobi");
      #endif
      double timelimit = 0.0;
      read_arguments(argc, argv, inputImageFileName, outputImageFileName, foregroundExampleColor, backgroundExampleColor, minPercentForeground, maxPercentForeground, mode, relaxation, heuristic, lambda, solverName, timelimit);

      // load image
      std::cout << "load image" << std::endl;
      opengm::PGMImage<unsigned char> inputImage;
      inputImage.readPGM(inputImageFileName);

      // get dimensions
      std::cout << "get dimensions" << std::endl;
      const IndexType gridSizeN = inputImage.height();
      const IndexType gridSizeM = inputImage.width();
      const IndexType numVariables = gridSizeN * gridSizeM;
      const LabelType numLabels = 2;

      std::cout << "image size: " << gridSizeN << "x" << gridSizeM << std::endl;

      // build model
      GraphicalModelType gm(SpaceType(numVariables, numLabels));
      const std::vector<LabelType> gmShape(numVariables, numLabels);
      std::vector<IndexType> gmVariables(numVariables);
      for(IndexType i = 0; i < numVariables; ++i) {
         gmVariables[i] = i;
      }

      // add functions
      std::cout << "add functions" << std::endl;
      std::vector<FunctionIdentifierType> unaryFunctionIDs(256); // 2^8 = 256 possible color values
      ExplicitFunctionType explicitFunction(&numLabels, &numLabels + 1);
      {
         unsigned char i = 0;
         const ValueType truncation = 100.0;
         do {
            const ValueType foregroundValue = static_cast<ValueType>(i) - foregroundExampleColor;
            const ValueType backgroundValue = static_cast<ValueType>(i) - backgroundExampleColor;
            explicitFunction(0) = std::min(truncation, std::fabs(foregroundValue));
            explicitFunction(1) = std::min(truncation, std::fabs(backgroundValue));
            unaryFunctionIDs[i] = gm.addFunction(explicitFunction);
         } while(++i); // iterate over all possible values of unsigned char
      }

      FunctionIdentifierType pottsFunctionID;
      if(lambda != 0.0) {
         PottsFunctionType pottsFunction(numLabels, numLabels, 0.0, lambda);
         pottsFunctionID = gm.addFunction(pottsFunction);
      }

      // add linear constraint functions
      FunctionIdentifierType minPercentForegroundLinearConstraintFunctionID;
      if(minPercentForeground > 0.0) {
         // build constraint
         LinearConstraintFunctionType::LinearConstraintType minPercentForegroundLinearConstraint;

         // left hand side
         minPercentForegroundLinearConstraint.reserve(numVariables);
         for(IndexType i = 0; i < numVariables; ++i) {
            const LinearConstraintFunctionType::LinearConstraintType::IndicatorVariableType indicatorVariable(i, LabelType(0)); // evaluates to one if variable i takes label 0
            minPercentForegroundLinearConstraint.add(indicatorVariable, 1.0);
         }

         // right hand side
         minPercentForegroundLinearConstraint.setBound(minPercentForeground * numVariables);

         // operator
         const LinearConstraintFunctionType::LinearConstraintType::LinearConstraintOperatorValueType greaterEqualOperator = LinearConstraintFunctionType::LinearConstraintType::LinearConstraintOperatorType::GreaterEqual;
         minPercentForegroundLinearConstraint.setConstraintOperator(greaterEqualOperator);

         // create constraint function (one function of maximum order)
         LinearConstraintFunctionType minPercentForegroundLinearConstraintFunction(gmShape.begin(), gmShape.end(), &minPercentForegroundLinearConstraint, &minPercentForegroundLinearConstraint + 1);

         // add constraint function
         minPercentForegroundLinearConstraintFunctionID = gm.addFunction(minPercentForegroundLinearConstraintFunction);
      }

      FunctionIdentifierType maxPercentForegroundLinearConstraintFunctionID;
      if(maxPercentForeground < 1.0) {
         // build constraint
         LinearConstraintFunctionType::LinearConstraintType maxPercentForegroundLinearConstraint;

         // left hand side
         maxPercentForegroundLinearConstraint.reserve(numVariables);
         for(IndexType i = 0; i < numVariables; ++i) {
            const LinearConstraintFunctionType::LinearConstraintType::IndicatorVariableType indicatorVariable(i, LabelType(0)); // evaluates to one if variable i takes label 0
            maxPercentForegroundLinearConstraint.add(indicatorVariable, 1.0);
         }

         // right hand side
         maxPercentForegroundLinearConstraint.setBound(maxPercentForeground * numVariables);

         // operator
         const LinearConstraintFunctionType::LinearConstraintType::LinearConstraintOperatorValueType lessEqualOperator = LinearConstraintFunctionType::LinearConstraintType::LinearConstraintOperatorType::LessEqual;
         maxPercentForegroundLinearConstraint.setConstraintOperator(lessEqualOperator);

         // create constraint function (one function of maximum order)
         LinearConstraintFunctionType maxPercentForegroundLinearConstraintFunction(gmShape.begin(), gmShape.end(), &maxPercentForegroundLinearConstraint, &maxPercentForegroundLinearConstraint + 1);

         // add constraint function
         maxPercentForegroundLinearConstraintFunctionID = gm.addFunction(maxPercentForegroundLinearConstraintFunction);
      }
      // NOTE: minPercentForegroundLinearConstraintFunction and maxPercentForegroundLinearConstraintFunction
      //       could have been joined into one single linear constraint function as linear constraint functions
      //       support multiple linear constraints. For clarity the linear constraints are separated into two
      //       different functions.

      // Add unary factors
      for(IndexType n = 0; n < gridSizeN; ++n){
         for(IndexType m = 0; m < gridSizeM; ++m){
            const IndexType variable = n + (m * gridSizeN);
            gm.addFactor(unaryFunctionIDs[inputImage(n, m)], &variable, &variable + 1);
         }
      }

      if(lambda != 0.0) {
         // add potts factors
         IndexType variables[]  = {0, 1};
         for(IndexType n = 0; n < gridSizeN; ++n){
            for(IndexType m = 0; m < gridSizeM; ++m){
               variables[0] = n + (m * gridSizeN);
               if(n + 1 < gridSizeN){
                  //add potts with lower neighbor
                  variables[1] = n + 1 + (m * gridSizeN);
                  gm.addFactor(pottsFunctionID, variables, variables + 2);
               }
               if(m + 1 < gridSizeM){
                  //add potts with right neighbor
                  variables[1] = n + ((m + 1) * gridSizeN);
                  gm.addFactor(pottsFunctionID, variables, variables + 2);
               }
            }
         }
      }

      if(minPercentForeground > 0.0) {
         // add min percent foreground constraint factor
         gm.addFactor(minPercentForegroundLinearConstraintFunctionID, gmVariables.begin(), gmVariables.end());
      }

      if(maxPercentForeground < 1.0) {
         // add max percent foreground constraint factor
         gm.addFactor(maxPercentForegroundLinearConstraintFunctionID, gmVariables.begin(), gmVariables.end());
      }

      // inference
      std::vector<LabelType> labeling(gm.numberOfVariables());
      #ifdef WITH_CPLEX
         typedef opengm::LPCplex2<GraphicalModelType, opengm::Minimizer> LPCplexType;
         #ifdef WITH_HDF5
            typedef LPCplexType::TimingVisitorType  LPCplexVisitorType;
         #else
            typedef LPCplexType::VerboseVisitorType LPCplexVisitorType;
         #endif

         LPCplexVisitorType cplexVisitor;
         if(solverName == "cplex") {
            std::cout << "cplex inference" << std::endl;
            LPCplexType::Parameter cplexParameter;
            set_solver_parameter_settings(cplexParameter, mode, relaxation, heuristic, timelimit);
            LPCplexType cplexSolver(gm, cplexParameter);
            cplexSolver.infer(cplexVisitor);
            cplexSolver.arg(labeling);
         }
      #endif
      #ifdef WITH_GUROBI
         typedef opengm::LPGurobi2<GraphicalModelType, opengm::Minimizer> LPGurobiType;
         #ifdef WITH_HDF5
            typedef LPGurobiType::TimingVisitorType  LPGurobiVisitorType;
         #else
            typedef LPGurobiType::VerboseVisitorType LPGurobiVisitorType;
         #endif

         LPGurobiVisitorType gurobiVisitor;
         if(solverName == "gurobi") {
            std::cout << "gurobi inference" << std::endl;
            LPGurobiType::Parameter gurobiParameter;
            set_solver_parameter_settings(gurobiParameter, mode, relaxation, heuristic, timelimit);
            LPGurobiType gurobiSolver(gm, gurobiParameter);
            gurobiSolver.infer(gurobiVisitor);
            gurobiSolver.arg(labeling);
         }
      #endif

      // read results
      std::cout << "read results" << std::endl;
      opengm::PGMImage<unsigned char> outputImage(gridSizeN, gridSizeM);
      for(IndexType i = 0; i < gridSizeN; ++i){
         for(IndexType j = 0; j < gridSizeM; ++j){
            const IndexType variable = i + (j * gridSizeN);
            if(labeling[variable] == 0) {
               // foreground
               outputImage(i, j) =  0;
            } else if(labeling[variable] == 1) {
               // background
               outputImage(i, j) =  255;
            } else {
               std::cerr << "Unexpected labeling!" << std::endl;
               return 1;
            }
         }
      }

      // store result image
      std::cout << "store result image" << std::endl;
      outputImage.writePGM(outputImageFileName);

      #ifdef WITH_HDF5
         // store visitor data
         const std::string outputVisitorFileName(outputImageFileName + std::string(".visitor.h5"));

         // clear file if it already exists
         if(fileExists(outputVisitorFileName)) {
            const std::string outputVisitorFileNameOld(outputVisitorFileName + std::string(".old"));
            std::cout << "Warning: Visitor file already exists moving it to " << outputVisitorFileNameOld << std::endl;
            if(std::rename(outputVisitorFileName.c_str(), outputVisitorFileNameOld.c_str())) {
               std::cerr << "Failed to move old visitor file. Please remove old files and try again." << std::endl;
               return 1;
            }
         }

         #ifdef WITH_CPLEX
            if(solverName == "cplex") {
               storeVisitor(cplexVisitor, outputVisitorFileName);
            }
         #endif
         #ifdef WITH_GUROBI
            if(solverName == "gurobi") {
               storeVisitor(gurobiVisitor, outputVisitorFileName);
            }
         #endif

         // store final states
         storeVector(outputVisitorFileName, "states", labeling);
      #endif

   #endif
   return 0;
}

#if defined(WITH_CPLEX) || defined(WITH_GUROBI)
   // forward declarations
   template <class VALUE_TYPE>
   void read_arguments(const int argc, char** const argv, std::string& inputImageFileName, std::string& outputImageFileName, VALUE_TYPE& foregroundExampleColor, VALUE_TYPE& backgroundExampleColor, VALUE_TYPE& minPercentForeground, VALUE_TYPE& maxPercentForeground, std::string& mode, std::string& relaxation, std::string& heuristic, VALUE_TYPE& lambda, std::string& solverName, double& timelimit) {
      if((argc < 7) || (argc > 13)) {
         std::cerr << "Usage: " << argv[0] << " inputImageFileName outputImageFileName foregroundExampleColor backgroundExampleColor minPercentForeground maxPercentForeground [mode] [relaxation] [heuristic] [regularization] [solver] [timelimit]" << std::endl;
         abort();
      }

      inputImageFileName  = std::string(argv[1]);
      outputImageFileName = std::string(argv[2]);

      foregroundExampleColor = atof(argv[3]);
      backgroundExampleColor = atof(argv[4]);

      minPercentForeground = atof(argv[5]);
      maxPercentForeground = atof(argv[6]);

      if(argc >= 8) {
         mode = std::string(argv[7]);
         if((mode != "lp") && (mode != "ilp")) {
            std::cerr << "Wrong mode! (lp or ilp)" << std::endl;
            abort();
         }
      }

      if(argc >= 9) {
         relaxation = std::string(argv[8]);
         if((relaxation != "local") && (relaxation != "loose") && (relaxation != "tight")) {
            std::cerr << "Wrong relaxation! (local, loose or tight)" << std::endl;
            abort();
         }
      }

      if(argc >= 10) {
         heuristic = std::string(argv[9]);
         if((heuristic != "random") && (heuristic != "weighted")) {
            std::cerr << "Wrong heuristic! (random or weighted)" << std::endl;
            abort();
         }
      }

      if(argc >= 11) {
         lambda = atof(argv[10]);
      }

      if(argc >= 12) {
         solverName = std::string(argv[11]);
         if(solverName == "cplex") {
            #ifdef WITH_CPLEX
            #else
               std::cerr << "Cplex not suported, recompile with Cplex enabled." << std::endl;
                  abort();
            #endif
         } else if(solverName == "gurobi") {
            #ifdef WITH_GUROBI
            #else
               std::cerr << "Gurobi not suported, recompile with Gurobi enabled." << std::endl;
            #endif
         } else {
            std::cerr << "Unknown solver: " << solverName << ". Use cplex or gurobi as solver." << std::endl;
         }
      }

      if(argc == 13) {
         timelimit = atof(argv[12]);
      }
   }

   template <class PARAMETER_TYPE>
   void set_solver_parameter_settings(PARAMETER_TYPE& parameter, const std::string& mode, const std::string& relaxation, const std::string& heuristic, const double timelimit) {
      if(mode == "lp") {
         parameter.integerConstraintNodeVar_ = false;
         parameter.integerConstraintFactorVar_ = false;
      } else if(mode == "ilp") {
         parameter.integerConstraintNodeVar_ = true;
         parameter.integerConstraintFactorVar_ = false;
      } else {
         std::cerr << "Unknown mode: "<< mode << std::endl;
         abort();
      }

      if(relaxation == "local") {
         parameter.relaxation_ = PARAMETER_TYPE::LocalPolytope;
      } else if(relaxation == "loose") {
         parameter.relaxation_ = PARAMETER_TYPE::LoosePolytope;
      } else if(relaxation == "tight") {
         parameter.relaxation_ = PARAMETER_TYPE::TightPolytope;
      } else {
         std::cerr << "Unknown relaxation: "<< relaxation << std::endl;
         abort();
      }

      if(heuristic == "random") {
         parameter.challengeHeuristic_ = PARAMETER_TYPE::Random;
      } else if(heuristic == "weighted") {
         parameter.challengeHeuristic_ = PARAMETER_TYPE::Weighted;
         // limit maximum number of violated constraints added as Weighted only makes sense in this case
         if(parameter.relaxation_ == PARAMETER_TYPE::LocalPolytope) {
            // We have only two linear constraints which will be added iteratively. Hence add at most one per iteration.
            parameter.maxNumConstraintsPerIter_ = 1;
         } else if(parameter.relaxation_ == PARAMETER_TYPE::LoosePolytope) {
            // We have multiple linear constraints which will be added iteratively. Hence add more than one per iteration.
            parameter.maxNumConstraintsPerIter_ = 100;
         }
      } else {
         std::cerr << "Unknown heuristic: "<< heuristic << std::endl;
         abort();
      }

      if(timelimit > 0.0) {
         parameter.timeLimit_ = timelimit;
      }

      // use only one thread
      parameter.numberOfThreads_ = 1;

      // enable verbose mode
      parameter.verbose_ = true;
   }

   #ifdef WITH_HDF5
      template <class VISITOR>
      void storeVisitor(const VISITOR& visitor, const std::string& filename) {
         typedef std::map<std::string, std::vector<double> > ProtocolMapType;
         const ProtocolMapType& protocolMap = visitor.protocolMap();
         for(typename ProtocolMapType::const_iterator iter = protocolMap.begin(); iter != protocolMap.end(); iter++) {
            storeVector(filename, iter->first, iter->second);
         }
      }

      template <class VECTOR_TYPE>
      void storeVector(const std::string& filename, const std::string& dataset, VECTOR_TYPE& vec) {

        hid_t handle;
        //check if file already exists and create file if it doesn't exist
        if(fileExists(filename)) {
           handle = marray::hdf5::openFile(filename, marray::hdf5::READ_WRITE);
        } else {
           handle = marray::hdf5::createFile(filename);
        }

        //check if dataset with same name already exists and if so, rename dataset
        try {
          marray::hdf5::save(handle, dataset, vec);
        } catch(...) {
          marray::hdf5::closeFile(handle);
          std::cout << "warning: was not able to store dataset" << dataset <<"! Maybe it alread exists or data is not valid." << std::endl;
          return;
        }

        marray::hdf5::closeFile(handle);
      }

      bool fileExists(const std::string& filename) {
         std::ifstream file(filename.c_str(), std::ifstream::in);
         if(file.is_open()) {
            file.close();
            return true;
         }
         return false;
      }
   #endif
#endif
