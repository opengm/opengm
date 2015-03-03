#ifndef OPENGM_BLACKBOXTESTHIGHERORDERGRID_HXX_
#define OPENGM_BLACKBOXTESTHIGHERORDERGRID_HXX_

#include <opengm/opengm.hxx>
#include <opengm/unittests/test.hxx>
#include <opengm/graphicalmodel/modelgenerators/syntheticmodelgenerator.hxx>
#include <opengm/unittests/blackboxtests/blackboxtestbase.hxx>

/// \cond HIDDEN_SYMBOLS

namespace opengm {

   template<class GM> class BlackBoxTestBase;

   template<class GM>
   class BlackBoxTestHigherOrderGrid : public BlackBoxTestBase<GM>
   {
   public:
      enum BlackBoxFunction
      {
         RANDOM, POTTS, IPOTTS, L1
      };
      const size_t height_;
      const size_t width_;
      const size_t numVar_;
      const size_t numStates_;
      const size_t maxOrder_;
      const bool varStates_;
      const bool withUnary_;
      const bool multipleOrders_;
      const BlackBoxFunction function_;
      const BlackBoxBehaviour behaviour_;
      const size_t numTests_;
      const size_t modelIdOffset_;

      BlackBoxTestHigherOrderGrid(const size_t height, const size_t width, const size_t numStates, const size_t maxOrder, const bool varStates, const bool withUnary, const bool multipleOrders, const BlackBoxFunction f, const BlackBoxBehaviour b, const size_t numTests, const size_t modelIdOffset=0);
      virtual ~BlackBoxTestHigherOrderGrid();

      virtual std::string infoText();
      virtual GM getModel(size_t id);
      virtual size_t numberOfTests();
      virtual BlackBoxBehaviour behaviour();

   private:
      std::string functionString(const BlackBoxFunction f) const;
   };

   template<class GM>
   inline BlackBoxTestHigherOrderGrid<GM>::BlackBoxTestHigherOrderGrid(const size_t height, const size_t width, const size_t numStates, const size_t maxOrder, const bool varStates, const bool withUnary, const bool multipleOrders, const BlackBoxFunction f, const BlackBoxBehaviour b, const size_t numTests, const size_t modelIdOffset)
      : height_(height), width_(width), numVar_(height * width), numStates_(numStates), maxOrder_(maxOrder), varStates_(varStates), withUnary_(withUnary), multipleOrders_(multipleOrders), function_(f), behaviour_(b), numTests_(numTests), modelIdOffset_(modelIdOffset) {
      OPENGM_ASSERT(maxOrder_ >= 2);
      //OPENGM_ASSERT(maxOrder_ <= height_);
      //OPENGM_ASSERT(maxOrder_ <= width_);
   }

   template<class GM>
   inline BlackBoxTestHigherOrderGrid<GM>::~BlackBoxTestHigherOrderGrid() {

   }

   template<class GM>
   inline std::string BlackBoxTestHigherOrderGrid<GM>::infoText() {
      std::string s = varStates_ ? "<" : "";
      std::string u = withUnary_ ? " with unary " : " without unary ";
      std::string o = multipleOrders_ ? "<=" : "";
      std::stringstream oss;
      oss << "    - higher order grid model (" << o << maxOrder_ << ") with " << numVar_ << " variables and " << s << numStates_ << " states ";
      oss << "(functionType = " << functionString(function_) << u << ")";
      return oss.str();
   }

   template<class GM>
   inline GM BlackBoxTestHigherOrderGrid<GM>::getModel(size_t id) {
      typedef opengm::SyntheticModelGenerator<GM> ModelGeneratorType;

      ModelGeneratorType modelGenerator;
      typename ModelGeneratorType::Parameter modelGeneratorPara(numStates_, varStates_);

      if(withUnary_) {
         modelGeneratorPara.functionsType_.push_back(ModelGeneratorType::Parameter::URANDOM);
         modelGeneratorPara.functionsOrder_.push_back(1);
         modelGeneratorPara.functionsMinMax_.push_back(typename ModelGeneratorType::Parameter::MinMax(1.0, 2.0));
         modelGeneratorPara.sharedFunctions_.push_back(false);
         modelGeneratorPara.functionsParameter_.push_back(std::vector<typename GM::ValueType>());
      }

      for(size_t i = multipleOrders_ ? 2 : maxOrder_; i <= maxOrder_; ++i) {
         switch (function_) {
         case RANDOM: {
            modelGeneratorPara.functionsType_.push_back(ModelGeneratorType::Parameter::URANDOM);
            modelGeneratorPara.functionsOrder_.push_back(i);
            modelGeneratorPara.functionsMinMax_.push_back(typename ModelGeneratorType::Parameter::MinMax(1.0, 2.0));
            modelGeneratorPara.sharedFunctions_.push_back(false);
            modelGeneratorPara.functionsParameter_.push_back(std::vector<typename GM::ValueType>());
            break;
         }
         case POTTS: {
            modelGeneratorPara.functionsType_.push_back(ModelGeneratorType::Parameter::POTTS);
            modelGeneratorPara.functionsOrder_.push_back(i);
            modelGeneratorPara.functionsMinMax_.push_back(typename ModelGeneratorType::Parameter::MinMax(1.0, 2.0));
            modelGeneratorPara.sharedFunctions_.push_back(true);

            std::vector<typename GM::ValueType> functionParameter;
            functionParameter.push_back(0.0);
            functionParameter.push_back(3.0);
            modelGeneratorPara.functionsParameter_.push_back(functionParameter);
            break;
         }
         case IPOTTS: {
            modelGeneratorPara.functionsType_.push_back(ModelGeneratorType::Parameter::POTTS);
            modelGeneratorPara.functionsOrder_.push_back(i);
            modelGeneratorPara.functionsMinMax_.push_back(typename ModelGeneratorType::Parameter::MinMax(1.0, 2.0));
            modelGeneratorPara.sharedFunctions_.push_back(true);

            std::vector<typename GM::ValueType> functionParameter;
            functionParameter.push_back(0.0);
            functionParameter.push_back(-3.0);
            modelGeneratorPara.functionsParameter_.push_back(functionParameter);
            break;
         }
         case L1: {
            modelGeneratorPara.functionsType_.push_back(ModelGeneratorType::Parameter::L1);
            modelGeneratorPara.functionsOrder_.push_back(i);
            modelGeneratorPara.functionsMinMax_.push_back(typename ModelGeneratorType::Parameter::MinMax(1.0, 2.0));
            modelGeneratorPara.sharedFunctions_.push_back(true);
            modelGeneratorPara.functionsParameter_.push_back(std::vector<typename GM::ValueType>(1, 1.0));
            break;
         }
         }
      }

      return modelGenerator.buildGrid(id + modelIdOffset_, width_, height_, modelGeneratorPara);
   }

   template<class GM>
   inline size_t BlackBoxTestHigherOrderGrid<GM>::numberOfTests() {
      return numTests_;
   }

   template<class GM>
   inline BlackBoxBehaviour BlackBoxTestHigherOrderGrid<GM>::behaviour() {
      return behaviour_;
   }

   template<class GM>
   inline std::string BlackBoxTestHigherOrderGrid<GM>::functionString(const BlackBoxFunction f) const {
      switch (f) {
      case RANDOM:
         return "random";
      case POTTS:
         return "potts";
      case IPOTTS:
         return "negative potts";
      case L1:
         return "L1";
      default:
         return "";
      };
      return "";
   }

} // namespace opengm

#endif /* OPENGM_BLACKBOXTESTHIGHERORDERGRID_HXX_ */
