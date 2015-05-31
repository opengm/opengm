#ifndef OPENGM_LABEL_COST_FUNCTION_HXX_
#define OPENGM_LABEL_COST_FUNCTION_HXX_

#include <cmath>

#include <opengm/opengm.hxx>
#include <opengm/functions/function_registration.hxx>
#include <opengm/functions/function_properties_base.hxx>
#include <opengm/utilities/unsigned_integer_pow.hxx>

namespace opengm {

/*********************
 * class definition *
 *********************/
template<class FUNCTION_TYPE, class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
class LPFunctionTransfer_impl;

template<class VALUE_TYPE, class INDEX_TYPE = size_t, class LABEL_TYPE = size_t>
class LabelCostFunction : public FunctionBase<LabelCostFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>, VALUE_TYPE, INDEX_TYPE, LABEL_TYPE> {
public:
   // typedefs
   typedef VALUE_TYPE   ValueType;
   typedef INDEX_TYPE   IndexType;
   typedef LABEL_TYPE   LabelType;

   // constructors
   LabelCostFunction();
   template <class SHAPE_ITERATOR_TYPE, class COST_ITERATOR_TYPE>
   LabelCostFunction(SHAPE_ITERATOR_TYPE shapeBegin, SHAPE_ITERATOR_TYPE shapeEnd, COST_ITERATOR_TYPE costsBegin, COST_ITERATOR_TYPE costsEnd);
   template <class SHAPE_ITERATOR_TYPE>
   LabelCostFunction(SHAPE_ITERATOR_TYPE shapeBegin, SHAPE_ITERATOR_TYPE shapeEnd, const LabelType label, const ValueType cost);
   template <class COST_ITERATOR_TYPE>
   LabelCostFunction(const IndexType numVariables, const LabelType numLabels, COST_ITERATOR_TYPE costsBegin, COST_ITERATOR_TYPE costsEnd);
   LabelCostFunction(const IndexType numVariables, const LabelType numLabels, const LabelType label, const ValueType cost);
   ~LabelCostFunction();

   // function access
   template<class Iterator>
   ValueType   operator()(Iterator statesBegin) const;   // function evaluation
   size_t      shape(const size_t i) const;              // number of labels of the indicated input variable
   size_t      dimension() const;                        // number of input variables
   size_t      size() const;                             // number of parameters

protected:
   // storage
   std::vector<LabelType> shape_;
   IndexType              numVariables_;
   LabelType              maxNumLabels_;
   bool                   useSameNumLabels_;
   size_t                 size_;
   std::vector<ValueType> costs_;
   bool                   useSingleCost_;
   LabelType              singleLabel_;
   ValueType              singleCost_;

   // friends
   friend class FunctionSerialization<LabelCostFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE> >;
   friend class LPFunctionTransfer_impl<LabelCostFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>, VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>;
};

/// \cond HIDDEN_SYMBOLS
/// FunctionRegistration
template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
struct FunctionRegistration<LabelCostFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE> > {
   enum ID {
      // TODO set final Id
      Id = opengm::FUNCTION_TYPE_ID_OFFSET - 5
   };
};

/// FunctionSerialization
template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
class FunctionSerialization<LabelCostFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE> > {
public:
   typedef typename LabelCostFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::ValueType ValueType;

   static size_t indexSequenceSize(const LabelCostFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>&);
   static size_t valueSequenceSize(const LabelCostFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>&);
   template<class INDEX_OUTPUT_ITERATOR, class VALUE_OUTPUT_ITERATOR>
   static void serialize(const LabelCostFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>&, INDEX_OUTPUT_ITERATOR, VALUE_OUTPUT_ITERATOR);
   template<class INDEX_INPUT_ITERATOR, class VALUE_INPUT_ITERATOR>
   static void deserialize( INDEX_INPUT_ITERATOR, VALUE_INPUT_ITERATOR, LabelCostFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>&);
};
/// \endcond

/***********************
 * class documentation *
 ***********************/
/*! \file label_cost_function.hxx
 *  \brief Provides implementation of a label cost function.
 */

/*! \class LabelCostFunction
 *  \brief A label cost function class penalizing the usage of labels.
 *
 *  This class implements a label cost function which penalizes the usage of
 *  labels.
 *  \f[
 *     f(x) = \sum_{l = 0}^{ l < n} c_{l}(x),
 *  \f]
 *  where \f$c_{l}(x)\f$ is the cost of label \f$l\f$ if \f$l\f$ is used in
 *  \f$x\f$ and zero otherwise. The function supports to assign each label a
 *  different cost or to set a cost for one label while all other costs are
 *  treated as zero.
 *
 *  \tparam VALUE_TYPE Value type.
 *  \tparam INDEX_TYPE Index type.
 *  \tparam LABEL_TYPE Label type.
 *
 *  \ingroup functions
 */

/*! \typedef LabelCostFunction::ValueType
 *  \brief Typedef of the VALUE_TYPE template parameter type from the class
 *         LabelCostFunction.
 */

/*! \typedef LabelCostFunction::IndexType
 *  \brief Typedef of the INDEX_TYPE template parameter type from the class
 *         LabelCostFunction.
 */

/*! \typedef LabelCostFunction::LabelType
 *  \brief Typedef of the LABEL_TYPE template parameter type from the class
 *         LabelCostFunction.
 */

/*! \fn LabelCostFunction::LabelCostFunction()
 *  \brief LabelCostFunction constructor.
 *
 *  This constructor will create an empty LabelCostFunction.
 */

/*! \fn LabelCostFunction::LabelCostFunction(SHAPE_ITERATOR_TYPE shapeBegin, SHAPE_ITERATOR_TYPE shapeEnd, COST_ITERATOR_TYPE costsBegin, COST_ITERATOR_TYPE costsEnd)
 *  \brief LabelCostFunction constructor.
 *
 *  This constructor will create a LabelCostFunction where each variable can
 *  have a different number of labels and where a cost value is set for each
 *  label.
 *
 *  \tparam SHAPE_ITERATOR_TYPE Iterator type used to iterate over the shape of
 *                              the function.
 *  \tparam COST_ITERATOR_TYPE Iterator type used to iterate over the costs for
 *                             each label.
 *
 *  \param[in] shapeBegin Iterator pointing to the begin of the sequence which
 *                        defines the shape of the function.
 *  \param[in] shapeEnd Iterator pointing to the end of the sequence which
 *                      defines the shape of the function.
 *  \param[in] costsBegin Iterator pointing to the begin of the sequence of the
 *                        costs for each label.
 *  \param[in] costsEnd Iterator pointing to the end of the sequence of the
 *                      costs for each label.
 */

/*! \fn LabelCostFunction::LabelCostFunction(SHAPE_ITERATOR_TYPE shapeBegin, SHAPE_ITERATOR_TYPE shapeEnd, const LabelType label, const ValueType cost)
 *  \brief LabelCostFunction constructor.
 *
 *  This constructor will create a LabelCostFunction where each variable can
 *  have a different number of labels and where a cost value is set for exactly
 *  one label while the costs for each other label will be treated as zero.
 *
 *  \tparam SHAPE_ITERATOR_TYPE Iterator type used to iterate over the shape of
 *                              the function.
 *
 *  \param[in] shapeBegin Iterator pointing to the begin of the sequence which
 *                        defines the shape of the function.
 *  \param[in] shapeEnd Iterator pointing to the end of the sequence which
 *                      defines the shape of the function.
 *  \param[in] label The label for which a cost is set.
 *  \param[in] cost The cost which is used for the selected label.
 */

/*! \fn LabelCostFunction::LabelCostFunction(const IndexType numVariables, const LabelType numLabels, COST_ITERATOR_TYPE costsBegin, COST_ITERATOR_TYPE costsEnd)
 *  \brief LabelCostFunction constructor.
 *
 *  This constructor will create a LabelCostFunction where each variable has the
 *  same number of labels and where a cost value is set for each label.
 *
 *  \tparam COST_ITERATOR_TYPE Iterator type used to iterate over the costs for
 *                             each label.
 *
 *  \param[in] numVariables The number of variables of the function.
 *  \param[in] numLabels The number of labels for each variable of the function.
 *  \param[in] costsBegin Iterator pointing to the begin of the sequence of the
 *                        costs for each label.
 *  \param[in] costsEnd Iterator pointing to the end of the sequence of the
 *                      costs for each label.
 */

/*! \fn LabelCostFunction::LabelCostFunction(const IndexType numVariables, const LabelType numLabels, const LabelType label, const ValueType cost)
 *  \brief LabelCostFunction constructor.
 *
 *  This constructor will create a LabelCostFunction where each variable has the
 *  same number of labels and where a cost value is set for exactly one label
 *  while the costs for each other label will be treated as zero.
 *
 *  \param[in] numVariables The number of variables of the function.
 *  \param[in] numLabels The number of labels for each variable of the function.
 *  \param[in] label The label for which a cost is set.
 *  \param[in] cost The cost which is used for the selected label.
 */

/*! \fn LabelCostFunction::~LabelCostFunction()
 *  \brief LabelCostFunction destructor.
 */

/*! \fn LabelCostFunction::ValueType LabelCostFunction::operator()(Iterator statesBegin) const
 *  \brief Function evaluation.
 *
 *  \tparam Iterator Iterator type
 *
 *  \param[in] statesBegin Iterator pointing to the begin of a sequence of
 *                         labels for the variables of the function.
 *
 *  \return The sum of the costs of the labels which are used in the states
 *          sequence.
 */

/*! \fn size_t LabelCostFunction::shape(const size_t i) const
 *  \brief Number of labels of the indicated input variable.
 *
 *  \param[in] i Index of the variable.
 *
 *  \return Returns the number of labels of the i-th variable.
 */

/*! \fn size_t LabelCostFunction::dimension() const
 *  \brief Number of input variables.
 *
 *  \return Returns the number of variables.
 */

/*! \fn size_t LabelCostFunction::size() const
 *  \brief Number of parameters.
 *
 *  \return Returns the number of parameters.
 */

/*! \var LabelCostFunction::shape_
 *  \brief Storage for the shape of the function which is used if all variables
 *         can have different numbers of labels.
 */

/*! \var LabelCostFunction::numVariables_
 *  \brief Storage for the number of variables of the function.
 */

/*! \var LabelCostFunction::maxNumLabels_
 *  \brief Storage for the maximum number of labels of the function.
 */

/*! \var LabelCostFunction::useSameNumLabels_
 *  \brief Indicator to tell if all variables have the same number of labels.
 */

/*! \var LabelCostFunction::size_
 *  \brief Stores the size of the label cost function.
 */

/*! \var LabelCostFunction::costs_
 *  \brief Storage for the label costs which is used if all labels can have
 *         label costs.
 */

/*! \var LabelCostFunction::useSingleCost_
 *  \brief Indicator to tell if only a single label has label costs.
 */

/*! \var LabelCostFunction::singleLabel_
 *  \brief Storage for the single label which has label costs. Is only used when
 *         useSingleCost_ is set to true.
 */

/*! \var LabelCostFunction::singleCost_
 *  \brief Storage for cost of the single label which has label costs. Is only
 *         used when useSingleCost_ is set to true.
 */

/******************
 * implementation *
 ******************/
template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline LabelCostFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::LabelCostFunction() : shape_(),
   numVariables_(), maxNumLabels_(), useSameNumLabels_(), size_(), costs_(),
   useSingleCost_(), singleLabel_(), singleCost_() {

}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
template <class SHAPE_ITERATOR_TYPE, class COST_ITERATOR_TYPE>
inline LabelCostFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::LabelCostFunction(SHAPE_ITERATOR_TYPE shapeBegin, SHAPE_ITERATOR_TYPE shapeEnd, COST_ITERATOR_TYPE costsBegin, COST_ITERATOR_TYPE costsEnd)
   : shape_(shapeBegin, shapeEnd), numVariables_(shape_.size()),
     maxNumLabels_(numVariables_ > 0 ? *std::max_element(shape_.begin(), shape_.end()) : 0),
     useSameNumLabels_(numVariables_ > 0 ? std::equal(shape_.begin() + 1, shape_.end(), shape_.begin()) : true),
     size_(std::accumulate(shapeBegin, shapeEnd, 1, std::multiplies<typename std::iterator_traits<SHAPE_ITERATOR_TYPE>::value_type>())),
     costs_(costsBegin, costsEnd), useSingleCost_(false), singleLabel_(),
     singleCost_() {
   OPENGM_ASSERT(costs_.size() >= maxNumLabels_);
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
template <class SHAPE_ITERATOR_TYPE>
inline LabelCostFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::LabelCostFunction(SHAPE_ITERATOR_TYPE shapeBegin, SHAPE_ITERATOR_TYPE shapeEnd, const LabelType label, const ValueType cost)
   : shape_(shapeBegin, shapeEnd), numVariables_(shape_.size()),
     maxNumLabels_(numVariables_ > 0 ? *std::max_element(shape_.begin(), shape_.end()) : 0),
     useSameNumLabels_(numVariables_ > 0 ? std::equal(shape_.begin() + 1, shape_.end(), shape_.begin()) : true),
     size_(std::accumulate(shapeBegin, shapeEnd, 1, std::multiplies<typename std::iterator_traits<SHAPE_ITERATOR_TYPE>::value_type>())),
     costs_(), useSingleCost_(true), singleLabel_(label), singleCost_(cost) {
   OPENGM_ASSERT(singleLabel_ < maxNumLabels_);
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
template <class COST_ITERATOR_TYPE>
inline LabelCostFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::LabelCostFunction(const IndexType numVariables, const LabelType numLabels, COST_ITERATOR_TYPE costsBegin, COST_ITERATOR_TYPE costsEnd)
   : shape_(), numVariables_(numVariables), maxNumLabels_(numLabels),
     useSameNumLabels_(true),
     size_(unsignedIntegerPow(maxNumLabels_, numVariables_)),
     costs_(costsBegin, costsEnd), useSingleCost_(false), singleLabel_(),
     singleCost_() {
   OPENGM_ASSERT(costs_.size() >= maxNumLabels_);
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline LabelCostFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::LabelCostFunction(const IndexType numVariables, const LabelType numLabels, const LabelType label, const ValueType cost)
   : shape_(), numVariables_(numVariables), maxNumLabels_(numLabels),
     useSameNumLabels_(true),
     size_(unsignedIntegerPow(maxNumLabels_, numVariables_)), costs_(),
     useSingleCost_(true), singleLabel_(label), singleCost_(cost) {
   OPENGM_ASSERT(singleLabel_ < maxNumLabels_);
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline LabelCostFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::~LabelCostFunction() {

}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
template<class Iterator>
inline typename LabelCostFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::ValueType LabelCostFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::operator()(Iterator statesBegin) const {
   if(useSingleCost_) {
      if(std::find(statesBegin, statesBegin + numVariables_, singleLabel_) == statesBegin + numVariables_) {
         return 0.0;
      } else {
         return singleCost_;
      }
   } else {
      std::vector<bool> labelIsUsed(maxNumLabels_, false);
      ValueType result = 0.0;
      LabelType numLabelsFound = 0;
      const Iterator statesEnd = statesBegin + numVariables_;
      while(statesBegin != statesEnd) {
         const LabelType currentLabel = *statesBegin;
         OPENGM_ASSERT(currentLabel < maxNumLabels_);
         if(!labelIsUsed[currentLabel]) {
            labelIsUsed[currentLabel] = true;
            ++numLabelsFound;
            result += costs_[currentLabel];
         }
         if(numLabelsFound == maxNumLabels_) {
            break;
         }
         ++statesBegin;
      }
      return result;
   }
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline size_t LabelCostFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::shape(const size_t i) const {
   OPENGM_ASSERT(i < numVariables_);
   if(useSameNumLabels_) {
      return maxNumLabels_;
   } else {
      return shape_[i];
   }
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline size_t LabelCostFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::dimension() const {
   return numVariables_;
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline size_t LabelCostFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::size() const {
   return size_;
}

/// \cond HIDDEN_SYMBOLS
template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline size_t FunctionSerialization<LabelCostFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE> >::indexSequenceSize(const LabelCostFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>& src) {
   const size_t dimensionSize = 1;
   const size_t useSameNumLabelsSize = 1;
   const size_t shapeSize = src.useSameNumLabels_ ? 1 : src.dimension();
   const size_t useSingleCostSize = 1;
   const size_t costsSize = 1;

   const size_t totalIndexSize = dimensionSize + useSameNumLabelsSize +
         shapeSize + useSingleCostSize + costsSize;
   return totalIndexSize;
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline size_t FunctionSerialization<LabelCostFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE> >::valueSequenceSize(const LabelCostFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>& src) {
   const size_t costsSize = src.useSingleCost_ ? 1 : src.costs_.size();

   const size_t totalValueSize = costsSize;
   return totalValueSize;
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
template<class INDEX_OUTPUT_ITERATOR, class VALUE_OUTPUT_ITERATOR>
inline void FunctionSerialization<LabelCostFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE> >::serialize(const LabelCostFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>& src, INDEX_OUTPUT_ITERATOR indexOutIterator, VALUE_OUTPUT_ITERATOR valueOutIterator) {
   // index output
   // dimension
   *indexOutIterator = src.dimension();
   ++indexOutIterator;

   // use same number of labels
   *indexOutIterator = src.useSameNumLabels_ ? 1 : 0;
   ++indexOutIterator;

   // shape
   if(src.useSameNumLabels_) {
      *indexOutIterator = src.maxNumLabels_;
      ++indexOutIterator;
   } else {
      for(size_t i = 0; i < src.dimension(); ++i) {
         *indexOutIterator = src.shape_[i];
         ++indexOutIterator;
      }
   }

   // use single cost
   *indexOutIterator = src.useSingleCost_ ? 1 : 0;
   ++indexOutIterator;
   if(src.useSingleCost_) {
      *indexOutIterator = src.singleLabel_;
   } else {
      *indexOutIterator = src.costs_.size();
   }

   // value output
   // costs
   if(src.useSingleCost_) {
      *valueOutIterator = src.singleCost_;
   } else {
      for(size_t i = 0; i < src.costs_.size(); ++i) {
         *valueOutIterator = src.costs_[i];
         ++valueOutIterator;
      }
   }
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
template<class INDEX_INPUT_ITERATOR, class VALUE_INPUT_ITERATOR>
inline void FunctionSerialization<LabelCostFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE> >::deserialize(INDEX_INPUT_ITERATOR indexInIterator, VALUE_INPUT_ITERATOR valueInIterator, LabelCostFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>& dst) {
   // index input
   // dimension
   const size_t dimension = *indexInIterator;
   ++indexInIterator;

   // use same number of Labels
   const bool useSameNumLabels = *indexInIterator == 1 ? true : false;
   ++indexInIterator;

   // shape
   INDEX_INPUT_ITERATOR shapeBegin = indexInIterator;
   INDEX_INPUT_ITERATOR shapeEnd = indexInIterator + (useSameNumLabels ? 1 : dimension);
   indexInIterator += (useSameNumLabels ? 1 : dimension);

   // use single cost
   const bool useSingleCost = *indexInIterator == 1 ? true : false;
   ++indexInIterator;
   const size_t costsSize = *indexInIterator;

   // value input
   // coefficients
   VALUE_INPUT_ITERATOR costsBegin = valueInIterator;
   VALUE_INPUT_ITERATOR costsEnd = valueInIterator + costsSize;

   if(useSameNumLabels) {
      if(useSingleCost) {
         dst = LabelCostFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>(dimension, *shapeBegin, costsSize, *costsBegin);
      } else {
         dst = LabelCostFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>(dimension, *shapeBegin, costsBegin, costsEnd);
      }
   } else {
      if(useSingleCost) {
         dst = LabelCostFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>(shapeBegin, shapeEnd, costsSize, *costsBegin);
      } else {
         dst = LabelCostFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>(shapeBegin, shapeEnd, costsBegin, costsEnd);
      }
   }
}
/// \endcond

} // namespace opengm



#endif /* OPENGM_LABEL_COST_FUNCTION_HXX_ */
