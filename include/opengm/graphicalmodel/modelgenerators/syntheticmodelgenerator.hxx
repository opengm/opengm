#pragma once
#ifndef OPENGM_SYNTHETIC_MODEL_GENERATOR2_HXX
#define OPENGM_SYNTHETIC_MODEL_GENERATOR2_HXX

/// \cond HIDDEN_SYMBOLS

#include <cstdlib>
#include <vector>
#include <cstdlib>
#include <set>
#include <functional>

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/utilities/random.hxx>

namespace opengm {

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

