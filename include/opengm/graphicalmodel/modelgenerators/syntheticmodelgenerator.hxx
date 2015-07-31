#pragma once
#ifndef OPENGM_SYNTHETIC_MODEL_GENERATOR2_HXX
#define OPENGM_SYNTHETIC_MODEL_GENERATOR2_HXX

/// \cond HIDDEN_SYMBOLS

#include <vector>
#include <map>

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/functions/explicit_function.hxx>
#include <opengm/functions/constant.hxx>
#include <opengm/functions/potts.hxx>
#include <opengm/functions/absolute_difference.hxx>
#include <opengm/utilities/indexing.hxx>
#include <opengm/utilities/random.hxx>

namespace opengm {

   template<class GM>
   class SyntheticModelGenerator {
   public:
      // typedefs
      typedef GM GraphicalModelType;
      typedef typename GM::ValueType ValueType;
      typedef typename GM::IndexType IndexType;
      typedef typename GM::LabelType LabelType;

      typedef typename GM::FunctionIdentifier FunctionIdentifier;

      typedef opengm::ExplicitFunction<ValueType, IndexType, LabelType>           ExplicitFunctionType;
      typedef opengm::ConstantFunction<ValueType, IndexType, LabelType>           ConstantFunctionType;
      typedef opengm::PottsFunction<ValueType, IndexType, LabelType>              PottsFunctionType;
      typedef opengm::AbsoluteDifferenceFunction<ValueType, IndexType, LabelType> AbsoluteDifferenceFunctionType;

      typedef RandomUniform<double> URandomGenerator;
      typedef RandomUniform<int>    IRandomGenerator;

      struct Parameter {
         // typedefs
         enum FunctionTypes {CONSTF, URANDOM, IRANDOM, POTTS, L1};
         typedef std::pair<ValueType, ValueType> MinMax;

         // member
         LabelType                            maxNumberOfStates_;
         bool                                 randomNumberOfStates_;
         std::vector<FunctionTypes>           functionsType_;
         std::vector<MinMax>                  functionsMinMax_;
         std::vector<IndexType>               functionsOrder_;
         std::vector<bool>                    sharedFunctions_;
         std::vector<std::vector<ValueType> > functionsParameter_;

         // constructor
         Parameter(const LabelType maxNumberOfStatesIn = 2, const bool randomNumberOfStatesIn = false);

         // check parameter consistency
         bool sanityCheck() const;
      };

      // static public methods
      static GraphicalModelType buildGrid(const size_t id, const IndexType width, const IndexType height, const Parameter& parameter);
      static GraphicalModelType buildFull(const size_t id, const IndexType numberOfVariables, const Parameter& parameter);
      static GraphicalModelType buildStar(const size_t id, const IndexType numberOfVariables, const Parameter& parameter);
   protected:
      // static protected methods
      template<class IRANDOM_GENERATOR>
      static GraphicalModelType initGM(const IndexType numberOfVariables, IRANDOM_GENERATOR& integerRandomGenerator, const Parameter& parameter);
      template<class SHAPE_ITERATOR, class URANDOM_GENERATOR, class IRANDOM_GENERATOR>
      static FunctionIdentifier addFunction(GraphicalModelType& gm, const typename Parameter::FunctionTypes functionType, const std::vector<ValueType>& functionsParameter, SHAPE_ITERATOR shapeBegin, SHAPE_ITERATOR shapeEnd, URANDOM_GENERATOR& uniformRandomGenerator, IRANDOM_GENERATOR& integerRandomGenerator);
   };

   template <class GM>
   inline SyntheticModelGenerator<GM>::Parameter::Parameter(
         const LabelType maxNumberOfStatesIn, const bool randomNumberOfStatesIn)
         : maxNumberOfStates_(maxNumberOfStatesIn),
           randomNumberOfStates_(randomNumberOfStatesIn), functionsType_(),
           functionsMinMax_(), functionsOrder_(), sharedFunctions_(),
           functionsParameter_() {
   }

   template <class GM>
   inline bool SyntheticModelGenerator<GM>::Parameter::sanityCheck() const {
      return (functionsType_.size() == functionsMinMax_.size()) &&
            (functionsMinMax_.size() == functionsOrder_.size()) &&
            (functionsOrder_.size() == sharedFunctions_.size()) &&
            (sharedFunctions_.size() == functionsParameter_.size());
   }

   template <class GM>
   inline typename SyntheticModelGenerator<GM>::GraphicalModelType SyntheticModelGenerator<GM>::buildGrid(const size_t id, const IndexType width, const IndexType height, const Parameter& parameter) {
      OPENGM_ASSERT(parameter.sanityCheck());

      URandomGenerator uniformRandomGenerator(0.0, 1.0, id);
      IRandomGenerator integerRandomGenerator(0, 1, id);

      // init gm
      // TODO: width * height might overflow hence add overflow test (not trivial)
      GraphicalModelType gm = initGM(width * height, integerRandomGenerator, parameter);

      // build grid structure
      std::vector<LabelType> shape(1, parameter.maxNumberOfStates_);
      std::vector<IndexType> variables(1, 0);
      FunctionIdentifier functionId;
      std::map<std::vector<LabelType>, FunctionIdentifier> sharedFunctionMap;

      bool newFunction = true;

      for(size_t i = 0; i < parameter.functionsType_.size(); ++i) {
         //OPENGM_ASSERT(parameter.functionsOrder_[i] <= width);
         //OPENGM_ASSERT(parameter.functionsOrder_[i] <= height);

         // set new bounds for random generators
         uniformRandomGenerator.setLow(parameter.functionsMinMax_[i].first);
         uniformRandomGenerator.setHigh(parameter.functionsMinMax_[i].second);
         integerRandomGenerator.setLow(parameter.functionsMinMax_[i].first);
         integerRandomGenerator.setHigh(parameter.functionsMinMax_[i].second);

         // resize shape
         if(parameter.randomNumberOfStates_) {
            shape.resize(parameter.functionsOrder_[i]);
         } else {
            shape.resize(parameter.functionsOrder_[i], parameter.maxNumberOfStates_);
         }

         // resize variables
         variables.resize(parameter.functionsOrder_[i]);

         for(IndexType j = 0; j < width; ++j) {
            for(IndexType k = 0; k < height; ++k) {
               if((width >= parameter.functionsOrder_[i]) && (j < width - parameter.functionsOrder_[i] + 1)) {
                  // horizontal
                  IndexType variableOffset = j + (k * width);
                  for(IndexType l = 0; l < parameter.functionsOrder_[i]; ++l) {
                     if(parameter.randomNumberOfStates_) {
                        shape[l] = gm.numberOfLabels(variableOffset + l);
                     }
                     variables[l] = variableOffset + l;
                  }
                  if(parameter.sharedFunctions_[i]) {
                     const typename std::map<std::vector<LabelType>, FunctionIdentifier>::const_iterator position = sharedFunctionMap.find(shape);
                     if(position != sharedFunctionMap.end()) {
                        // use shared function
                        newFunction = false;
                        functionId = position->second;
                     } else {
                        // use new function
                        newFunction = true;
                     }
                  } else {
                     newFunction = true;
                  }

                  if(newFunction) {
                     // add new function
                     functionId = addFunction(gm, parameter.functionsType_[i], parameter.functionsParameter_[i], shape.begin(), shape.end(), uniformRandomGenerator, integerRandomGenerator);
                     if(parameter.sharedFunctions_[i]) {
                        sharedFunctionMap[shape] = functionId;
                     }
                  }

                  // add horizontal factor
                  gm.addFactor(functionId, variables.begin(), variables.end());
               }

               if((height >= parameter.functionsOrder_[i]) && (k < height - parameter.functionsOrder_[i] + 1) && (parameter.functionsOrder_[i] > 1)) {
                  // vertical (only if parameter.functionsOrder_[i] > 1 otherwise unary factors would be added twice)
                  IndexType variableOffset = j + (k * width);
                  for(IndexType l = 0; l < parameter.functionsOrder_[i]; ++l) {
                     if(parameter.randomNumberOfStates_) {
                        shape[l] = gm.numberOfLabels(variableOffset + (l * width));
                     }
                     variables[l] = variableOffset + (l * width);
                  }
                  if(parameter.sharedFunctions_[i]) {
                     const typename std::map<std::vector<LabelType>, FunctionIdentifier>::const_iterator position = sharedFunctionMap.find(shape);
                     if(position != sharedFunctionMap.end()) {
                        // use shared function
                        newFunction = false;
                        functionId = position->second;
                     } else {
                        // use new function
                        newFunction = true;
                     }
                  } else {
                     newFunction = true;
                  }

                  if(newFunction) {
                     // add new function
                     functionId = addFunction(gm, parameter.functionsType_[i], parameter.functionsParameter_[i], shape.begin(), shape.end(), uniformRandomGenerator, integerRandomGenerator);
                     if(parameter.sharedFunctions_[i]) {
                        sharedFunctionMap[shape] = functionId;
                     }
                  }

                  // add vertical factor
                  gm.addFactor(functionId, variables.begin(), variables.end());
               }
            }
         }

         // clear shared function map for next iteration
         if(parameter.sharedFunctions_[i]) {
            sharedFunctionMap.clear();
         }
      }

      return gm;
   }

   template <class GM>
   inline typename SyntheticModelGenerator<GM>::GraphicalModelType SyntheticModelGenerator<GM>::buildFull(const size_t id, const IndexType numberOfVariables, const Parameter& parameter) {

   }

   template <class GM>
   inline typename SyntheticModelGenerator<GM>::GraphicalModelType SyntheticModelGenerator<GM>::buildStar(const size_t id, const IndexType numberOfVariables, const Parameter& parameter) {

   }

   template <class GM>
   template<class IRANDOM_GENERATOR>
   inline typename SyntheticModelGenerator<GM>::GraphicalModelType SyntheticModelGenerator<GM>::initGM(const IndexType numberOfVariables, IRANDOM_GENERATOR& integerRandomGenerator, const Parameter& parameter) {
      if(parameter.randomNumberOfStates_) {
         std::vector<typename GM::LabelType> numberOfLabels(numberOfVariables);
         // generate random integer variables in the range [1, parameter.maxNumberOfStates_ + 1) = [1, parameter.maxNumberOfStates_]
         // TODO check if parameter.maxNumberOfStates_ + 1 fits in value type of IRANDOM_GENERATOR
         integerRandomGenerator.setLow(1);
         integerRandomGenerator.setHigh(parameter.maxNumberOfStates_ + 1);

         // generate number of labels
         for(size_t i = 0; i < numberOfVariables; i++) {
            numberOfLabels[i] = integerRandomGenerator();
         }

         return GM(opengm::DiscreteSpace<typename GM::IndexType,typename GM::LabelType>(numberOfLabels.begin(), numberOfLabels.end()));
      } else {
         std::vector<typename GM::LabelType> numberOfLabels(numberOfVariables, parameter.maxNumberOfStates_);
         return GM(opengm::DiscreteSpace<typename GM::IndexType,typename GM::LabelType>(numberOfLabels.begin(), numberOfLabels.end()));
      }
   }

   template <class GM>
   template<class SHAPE_ITERATOR, class URANDOM_GENERATOR, class IRANDOM_GENERATOR>
   inline typename SyntheticModelGenerator<GM>::FunctionIdentifier SyntheticModelGenerator<GM>::addFunction(GraphicalModelType& gm, const typename Parameter::FunctionTypes functionType, const std::vector<ValueType>& functionParameter, SHAPE_ITERATOR shapeBegin, SHAPE_ITERATOR shapeEnd, URANDOM_GENERATOR& uniformRandomGenerator, IRANDOM_GENERATOR& integerRandomGenerator) {
      switch (functionType) {
         case Parameter::CONSTF: {
            if(functionParameter.size() > 0) {
               OPENGM_ASSERT(functionParameter.size() == 1);
               ConstantFunctionType constantFunction(shapeBegin, shapeEnd, functionParameter[0]);
               return gm.addFunction(constantFunction);
            } else {
               ConstantFunctionType constantFunction(shapeBegin, shapeEnd, uniformRandomGenerator());
               return gm.addFunction(constantFunction);
            }
         }
         case Parameter::URANDOM: {
            ExplicitFunctionType function(shapeBegin, shapeEnd);
            ShapeWalker<SHAPE_ITERATOR> shapeWalker(shapeBegin, function.dimension());
            if(functionParameter.size() > 0) {
               OPENGM_ASSERT(functionParameter.size() == function.size());
               for(IndexType i = 0; i < function.size(); ++i) {
                  function(shapeWalker.coordinateTuple().begin()) = functionParameter[i];
                  ++shapeWalker;
               }
            } else {
               for(IndexType i = 0; i < function.size(); ++i) {
                  function(shapeWalker.coordinateTuple().begin()) = uniformRandomGenerator();
                  ++shapeWalker;
               }
            }
            return gm.addFunction(function);
         }
         case Parameter::IRANDOM: {
            ExplicitFunctionType function(shapeBegin, shapeEnd);
            ShapeWalker<SHAPE_ITERATOR> shapeWalker(shapeBegin, function.dimension());
            if(functionParameter.size() > 0) {
               OPENGM_ASSERT(functionParameter.size() == function.size());
               for(IndexType i = 0; i < function.size(); ++i) {
                  function(shapeWalker.coordinateTuple().begin()) = functionParameter[i];
                  ++shapeWalker;
               }
            } else {
               for(IndexType i = 0; i < function.size(); ++i) {
                  function(shapeWalker.coordinateTuple().begin()) = integerRandomGenerator();
                  ++shapeWalker;
               }
            }
            return gm.addFunction(function);
         }
         case Parameter::POTTS: {
            OPENGM_ASSERT(std::distance(shapeBegin, shapeEnd) == 2);
            if(functionParameter.size() > 0) {
               OPENGM_ASSERT(functionParameter.size() == 2);
               PottsFunctionType function(shapeBegin[0], shapeBegin[1], functionParameter[0], functionParameter[1]);
               return gm.addFunction(function);
            } else {
               PottsFunctionType function(shapeBegin[0], shapeBegin[1], uniformRandomGenerator(), uniformRandomGenerator());
               return gm.addFunction(function);
            }
         }
         case Parameter::L1: {
            OPENGM_ASSERT(std::distance(shapeBegin, shapeEnd) == 2);
            if(functionParameter.size() > 0) {
               OPENGM_ASSERT(functionParameter.size() == 1);
               AbsoluteDifferenceFunctionType function(shapeBegin[0], shapeBegin[1], functionParameter[0]);
               return gm.addFunction(function);
            } else {
               AbsoluteDifferenceFunctionType function(shapeBegin[0], shapeBegin[1], uniformRandomGenerator());
               return gm.addFunction(function);
            }
         }
         default:
            throw RuntimeError("Unknown function type.");
      }
   }


   template<class GM>
   class SyntheticModelGenerator2
   {
   public:
      typedef GM GraphicalModelType;
      typedef typename GM::ValueType ValueType;
      typedef typename GM::IndexType IndexType;
      typedef typename GM::LabelType LabelType;
      typedef opengm::ExplicitFunction<ValueType,IndexType,LabelType> ExplicitFunctionType;
      // typedef typename GM::SparseFunctionType SparseFunctionType;
      // typedef typename GM::ImplicitFunctionType ImplicitFunctionType;
      typedef typename GM::FunctionIdentifier FunctionIdentifier;
      typedef typename GM::OperatorType OperatorType;

      enum FunctionTypes
      {
         EMPTY, CONSTF, URANDOM, IRANDOM, GPOTTS, RGPOTTS, L1
      }; // GRANDOM

      class Parameter
      {
      public:
         std::vector<FunctionTypes> functionTypes_;
         std::vector<std::vector<ValueType> > functionParameters_;
         std::vector<bool> sharedFunctions_;
         bool randomNumberOfStates_;

         Parameter()
         {
            functionTypes_.resize(2, URANDOM);
            functionParameters_.resize(2);
            functionParameters_[0].resize(2);
            functionParameters_[0][0] = 0.1;
            functionParameters_[0][1] = 1;
            functionParameters_[1].resize(2);
            functionParameters_[1][0] = 0.1;
            functionParameters_[1][1] = 1;
            sharedFunctions_.resize(2, true);
            randomNumberOfStates_ = false;
         }

         bool isConsistent() const
         {
            return functionTypes_.size() == functionParameters_.size() &&
               functionParameters_.size() == sharedFunctions_.size();
         }
      };

      SyntheticModelGenerator2();
      GM buildGrid(const size_t, const size_t, const size_t, const size_t, const Parameter&) const;
      GM buildHigherOrderGrid(const size_t, const size_t, const size_t, const size_t, const size_t, const Parameter&) const;
      GM buildFull(const size_t, const size_t, const size_t, const Parameter&) const;
      GM buildStar(const size_t, const size_t, const size_t, const Parameter&) const;

   private:
      void addUnaries(GM&, const FunctionTypes, const std::vector<ValueType>&, const bool) const;
      template<class ITERATOR>
      FunctionIdentifier addFunction(GM&, const FunctionTypes, const std::vector<ValueType>&, ITERATOR, ITERATOR) const;
      GraphicalModelType getGM(size_t,size_t,bool) const;
   };

   template<class GM>
   SyntheticModelGenerator2<GM>::SyntheticModelGenerator2()
   {}

   template<class GM>
   GM SyntheticModelGenerator2<GM>::getGM(size_t numVar, size_t numStates, bool randomNumberOfStates) const
   {
      if(randomNumberOfStates) {
         std::vector<typename GM::LabelType> numberOfLabels(numVar);
         // generate random integer variables in the range [1, numStates + 1) = [1, numStates]
         opengm::RandomUniform<size_t> randomIntegerNumberGenerator(1,numStates + 1);
         for(size_t i = 0; i < numVar; i++) {
            numberOfLabels[i] = randomIntegerNumberGenerator();
         }
         return GM( opengm::DiscreteSpace<typename GM::IndexType,typename GM::LabelType>(numberOfLabels.begin(), numberOfLabels.end()));
      } else {
         std::vector<typename GM::LabelType> numberOfLabels(numVar,numStates);
         return GM( opengm::DiscreteSpace<typename GM::IndexType,typename GM::LabelType>(numberOfLabels.begin(), numberOfLabels.end()));
      }
   }

   template<class GM>
   void SyntheticModelGenerator2<GM>::addUnaries
   (
      GM& gm,
      const FunctionTypes functionType,
      const std::vector<ValueType>& functionParameter,
      const bool sharedFunction
   ) const
   {
      typename GM::LabelType shape[1];
      typename GM::IndexType var[] = {0};
      FunctionIdentifier funcId;
      switch (functionType) {
      case URANDOM:
         OPENGM_ASSERT(functionParameter.size() == 2);
         OPENGM_ASSERT(functionParameter[0] <= functionParameter[1]);
         for(size_t i = 0; i < gm.numberOfVariables(); ++i) {
            shape[0] = gm.numberOfLabels(i);
            var[0] = i;
            if(!sharedFunction|| i == 0) {
               ExplicitFunctionType function(shape, shape + 1);
               for(size_t ni = 0; ni < shape[0]; ++ni) {
                  function(ni) = functionParameter[0] + (functionParameter[1] - functionParameter[0]) * (rand() % 1000000) * 0.000001;
               }
               funcId = gm.addFunction(function);
            }
            gm.addFactor(funcId, var, var + 1);
         }
         break;
      case IRANDOM:
         OPENGM_ASSERT(functionParameter.size() == 2);
         OPENGM_ASSERT(functionParameter[0] <= functionParameter[1]);
         for(size_t i = 0; i < gm.numberOfVariables(); ++i) {
            shape[0] = gm.numberOfLabels(i);
            var[0] = i;
            if(!sharedFunction || i == 0) {
               ExplicitFunctionType function(shape, shape + 1);
               for(size_t ni = 0; ni < shape[0]; ++ni) {
                  function(ni) = functionParameter[0] + rand() % (size_t) (functionParameter[1] - functionParameter[0]);
               }
               funcId = gm.addFunction(function);
            }
            gm.addFactor(funcId, var, var + 1);
         }
         break;
      case EMPTY:
         // do nothing
         break;
      case CONSTF:
         OPENGM_ASSERT(functionParameter.size() == 1);
         for(typename GM::IndexType i = 0; i < gm.numberOfVariables(); ++i) {
            shape[0] = gm.numberOfLabels(i);
            var[0] = i;
            if(!sharedFunction || i == 0) {
               ExplicitFunctionType function(shape, shape + 1);
               for(typename GM::LabelType ni = 0; ni < shape[0]; ++ni) {
                  function(ni) = functionParameter[0];
               }
               funcId = gm.addFunction(function);
            }
            gm.addFactor(funcId, var, var + 1);
         }
         break;
      default:
         throw RuntimeError("Unknown function type for unary factors.");
      }
   }

   template<class GM>
   template<class ITERATOR>
   typename GM::FunctionIdentifier SyntheticModelGenerator2<GM>::addFunction
   (
      GM& gm,
      const FunctionTypes functionType,
      const std::vector<ValueType>& functionParameter,
      ITERATOR shapeBegin, ITERATOR shapeEnd
   ) const
   {
      switch (functionType) {
      case URANDOM:
         {
         OPENGM_ASSERT(functionParameter.size() == 2);
         OPENGM_ASSERT(functionParameter[0] <= functionParameter[1]);
         ExplicitFunctionType function(shapeBegin, shapeEnd);
         for(typename GM::LabelType ni = 0; ni < shapeBegin[0]; ++ni) {
            for(typename GM::LabelType nj = 0; nj < shapeBegin[1]; ++nj) {
               function(ni, nj) = functionParameter[0] + (functionParameter[1] - functionParameter[0]) * (rand() % 1000000) * 0.000001;
            }
         }
         return gm.addFunction(function);
      }
      case IRANDOM:
         {
         OPENGM_ASSERT(functionParameter.size() == 2);
         OPENGM_ASSERT(functionParameter[0] <= functionParameter[1]);
         ExplicitFunctionType function(shapeBegin, shapeEnd);
         for(typename GM::LabelType ni = 0; ni < shapeBegin[0]; ++ni) {
            for(typename GM::LabelType nj = 0; nj < shapeBegin[1]; ++nj) {
               function(ni, nj) = functionParameter[0] + rand() % (size_t) (functionParameter[0] - functionParameter[1]);
            }
         }
         return gm.addFunction(function);
      }
      case GPOTTS:
         {
         OPENGM_ASSERT(functionParameter.size() == 1);
         ExplicitFunctionType function(shapeBegin, shapeEnd, functionParameter[0]);
         for(typename GM::LabelType ni = 0; ni < shapeBegin[0]; ++ni) {
            for(typename GM::LabelType nj = 0; nj < shapeBegin[1]; ++nj) {
               if(ni == nj)
                  function(ni, nj) = 0;
            }
         }
         return gm.addFunction(function);
      } 
      case L1:
         {
         OPENGM_ASSERT(functionParameter.size() == 1);
         ExplicitFunctionType function(shapeBegin, shapeEnd, functionParameter[0]);
         for(typename GM::LabelType ni = 0; ni < shapeBegin[0]; ++ni) {
            for(typename GM::LabelType nj = 0; nj < shapeBegin[1]; ++nj) {
               function(ni, nj) = 0.1*fabs(1.0*ni-1.0*nj);
            }
         }
         return gm.addFunction(function);
      }
      case RGPOTTS:
         {
         OPENGM_ASSERT(functionParameter.size() == 2);
         OPENGM_ASSERT(functionParameter[0] <= functionParameter[1]);
         ValueType v = functionParameter[0] + (functionParameter[1] - functionParameter[0]) * (rand() % 1000000) * 0.000001;
         ExplicitFunctionType function(shapeBegin, shapeEnd, v);
         for(typename GM::LabelType ni = 0; ni < shapeBegin[0]; ++ni) {
            for(typename GM::LabelType nj = 0; nj < shapeBegin[1]; ++nj) {
               if(ni == nj)
                  function(ni, nj) = 0;
            }
         }
         return gm.addFunction(function);
      }
      case CONSTF:
         {
         OPENGM_ASSERT(functionParameter.size() == 1);
         ExplicitFunctionType function(shapeBegin, shapeEnd, functionParameter[0]);
         return gm.addFunction(function);
      }
      default:
         throw RuntimeError("Unknown function type for unary factors.");
      }
   }

   template<class GM>
   GM SyntheticModelGenerator2<GM>::buildGrid
   (
      const size_t id,
      const size_t height, const size_t width,
      const size_t numStates,
      const Parameter& parameter
   ) const
   {
      srand(id);
      OPENGM_ASSERT(parameter.isConsistent());
      OPENGM_ASSERT(parameter.functionTypes_.size() == 2);
      GraphicalModelType gm = getGM(height*width,numStates,parameter.randomNumberOfStates_);
      // UNARY
      addUnaries(gm, parameter.functionTypes_[0], parameter.functionParameters_[0], parameter.sharedFunctions_[0] && !parameter.randomNumberOfStates_);
      // PAIRWISE
      typename GM::LabelType shape[2];
      typename GM::IndexType var[2];
      bool newFunction = true;
      FunctionIdentifier funcId;
      for(size_t i = 0; i < height; ++i) {
         for(size_t j = 0; j < width; ++j) {
            size_t v = i + height * j;
            if(i + 1 < height) {
               var[0] = v;
               var[1] = i + 1 + height * j;
               shape[0] = gm.numberOfLabels(var[0]);
               shape[1] = gm.numberOfLabels(var[1]);
               if(newFunction) {
                  funcId = addFunction(gm, parameter.functionTypes_[1], parameter.functionParameters_[1], shape, shape + 2);
               }
               newFunction = !parameter.sharedFunctions_[1] || parameter.randomNumberOfStates_;
               gm.addFactor(funcId, var, var + 2);
            }
            if(j + 1 < width) {
               var[0] = v;
               var[1] = i + height * (j + 1);
               shape[0] = gm.numberOfLabels(var[0]);
               shape[1] = gm.numberOfLabels(var[1]);
               if(newFunction) {
                  funcId = addFunction(gm, parameter.functionTypes_[1], parameter.functionParameters_[1], shape, shape + 2);
               }
               newFunction = !parameter.sharedFunctions_[1] || parameter.randomNumberOfStates_;
               gm.addFactor(funcId, var, var + 2);
            }
         }
      }
      return gm;
   }

   template<class GM>
   GM SyntheticModelGenerator2<GM>::buildHigherOrderGrid(
         const size_t id,
         const size_t height, const size_t width,
         const size_t numStates, const size_t order,
         const Parameter& parameter) const {
      srand(id);
      OPENGM_ASSERT(parameter.isConsistent());
      OPENGM_ASSERT(parameter.functionTypes_.size() == 2);
      OPENGM_ASSERT(order >= 2);
      OPENGM_ASSERT(order <= height);
      OPENGM_ASSERT(order <= width);

      GraphicalModelType gm = getGM(height*width,numStates,parameter.randomNumberOfStates_);

      // add unaries
      addUnaries(gm, parameter.functionTypes_[0], parameter.functionParameters_[0], parameter.sharedFunctions_[0] && !parameter.randomNumberOfStates_);

      // add higher order factors
      typename GM::LabelType shape[order];
      typename GM::IndexType var[order];
      bool newFunction = true;
      FunctionIdentifier funcId;
      for(size_t i = 0; i < height - order + 1; ++i) {
         for(size_t j = 0; j < width - order + 1; ++j) {
            size_t v = i + height * j;
            if(i + 1 < height) {
               var[0] = v;
               var[1] = i + 1 + height * j;
               shape[0] = gm.numberOfLabels(var[0]);
               shape[1] = gm.numberOfLabels(var[1]);
               if(newFunction) {
                  funcId = addFunction(gm, parameter.functionTypes_[1], parameter.functionParameters_[1], shape, shape + 2);
               }
               newFunction = !parameter.sharedFunctions_[1] || parameter.randomNumberOfStates_;
               gm.addFactor(funcId, var, var + 2);
            }
            if(j + 1 < width) {
               var[0] = v;
               var[1] = i + height * (j + 1);
               shape[0] = gm.numberOfLabels(var[0]);
               shape[1] = gm.numberOfLabels(var[1]);
               if(newFunction) {
                  funcId = addFunction(gm, parameter.functionTypes_[1], parameter.functionParameters_[1], shape, shape + 2);
               }
               newFunction = !parameter.sharedFunctions_[1] || parameter.randomNumberOfStates_;
               gm.addFactor(funcId, var, var + 2);
            }
         }
      }
      return gm;

   }

   template<class GM>
   GM SyntheticModelGenerator2<GM>::buildFull
   (
      const size_t id,
      const size_t numVars,
      const size_t numStates,
      const Parameter& parameter
   ) const
   {
      srand(id);
      OPENGM_ASSERT(parameter.isConsistent());
      OPENGM_ASSERT(parameter.functionTypes_.size() == 2);
      GraphicalModelType gm = getGM(numVars,numStates,parameter.randomNumberOfStates_);
      //UNARY
      addUnaries(gm, parameter.functionTypes_[0], parameter.functionParameters_[0], parameter.sharedFunctions_[0]&& !parameter.randomNumberOfStates_);
      //PAIRWISE
      typename GM::LabelType shape[2];
      typename GM::IndexType var[2];
      bool newFunction = true;
      FunctionIdentifier funcId;
      for(typename GM::IndexType i = 0; i < numVars; ++i) {
         for(typename GM::IndexType j = i + 1; j < numVars; ++j) {
            var[0] = i;
            var[1] = j;
            shape[0] = gm.numberOfLabels(var[0]);
            shape[1] = gm.numberOfLabels(var[1]);
            if(newFunction) {
               funcId = addFunction(gm, parameter.functionTypes_[1], parameter.functionParameters_[1], shape, shape + 2);
            }
            newFunction = !parameter.sharedFunctions_[1] || parameter.randomNumberOfStates_;
            gm.addFactor(funcId, var, var + 2);
         }
      }
      return gm;
   }

   template<class GM>
   GM SyntheticModelGenerator2<GM>::buildStar
   (
      const size_t id,
      const size_t numVars,
      const size_t numStates,
      const Parameter& parameter
   ) const
   {
      srand(id);
      OPENGM_ASSERT(parameter.isConsistent());
      OPENGM_ASSERT(parameter.functionTypes_.size() == 2);
      GraphicalModelType gm = getGM(numVars,numStates,parameter.randomNumberOfStates_);
      // UNARY
      addUnaries(gm, parameter.functionTypes_[0], parameter.functionParameters_[0], parameter.sharedFunctions_[0]&& !parameter.randomNumberOfStates_);
      // PAIRWISE
      typename GM::LabelType shape[2];
      typename GM::IndexType var[2];
      bool newFunction = true;
      FunctionIdentifier funcId;
      const size_t root = (rand() % gm.numberOfVariables());
      for(size_t i = 0; i < root; ++i) {
         var[0] = i;
         var[1] = root;
         shape[0] = gm.numberOfLabels(var[0]);
         shape[1] = gm.numberOfLabels(var[1]);
         if(newFunction) {
            funcId = addFunction(gm, parameter.functionTypes_[1], parameter.functionParameters_[1], shape, shape + 2);
         }
         newFunction = !parameter.sharedFunctions_[1] || parameter.randomNumberOfStates_;
         gm.addFactor(funcId, var, var + 2);
      }
      for(size_t i = root + 1; i < gm.numberOfVariables(); ++i) {
         var[0] = root;
         var[1] = i;
         shape[0] = gm.numberOfLabels(var[0]);
         shape[1] = gm.numberOfLabels(var[1]);
         if(newFunction) {
            funcId = addFunction(gm, parameter.functionTypes_[1], parameter.functionParameters_[1], shape, shape + 2);
         }
         newFunction = !parameter.sharedFunctions_[1] || parameter.randomNumberOfStates_;
         gm.addFactor(funcId, var, var + 2);
      }
      return gm;
   }

} // namespace opengm

/// \endcond

#endif // #ifndef OPENGM_SYNTHETIC_MODEL_GENERATOR2_HXX
