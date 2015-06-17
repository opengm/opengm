#ifndef OPENGM_SUM_CONSTRAINT_FUNCTION_HXX_
#define OPENGM_SUM_CONSTRAINT_FUNCTION_HXX_

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
class SumConstraintFunction : public FunctionBase<SumConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>, VALUE_TYPE, INDEX_TYPE, LABEL_TYPE> {
public:
   // typedefs
   typedef VALUE_TYPE   ValueType;
   typedef INDEX_TYPE   IndexType;
   typedef LABEL_TYPE   LabelType;

   // constructors
   SumConstraintFunction();
   template <class SHAPE_ITERATOR_TYPE, class COEFFICIENTS_ITERATOR_TYPE>
   SumConstraintFunction(SHAPE_ITERATOR_TYPE shapeBegin, SHAPE_ITERATOR_TYPE shapeEnd, COEFFICIENTS_ITERATOR_TYPE coefficientsBegin, COEFFICIENTS_ITERATOR_TYPE coefficientsEnd, const bool shareCoefficients, const ValueType lambda = 1.0, const ValueType bound = 0.0);
   template <class COEFFICIENTS_ITERATOR_TYPE>
   SumConstraintFunction(const IndexType numVariables, const LabelType numLabels, COEFFICIENTS_ITERATOR_TYPE coefficientsBegin, COEFFICIENTS_ITERATOR_TYPE coefficientsEnd, const bool shareCoefficients, const ValueType lambda = 1.0, const ValueType bound = 0.0);
   ~SumConstraintFunction();

   // function access
   template<class Iterator>
   ValueType   operator()(Iterator statesBegin) const;   // function evaluation
   size_t      shape(const size_t i) const;              // number of labels of the indicated input variable
   size_t      dimension() const;                        // number of input variables
   size_t      size() const;                             // number of parameters

protected:
   // storage
   std::vector<LabelType> shape_;
   size_t                 numVariables_;
   bool                   useSameNumLabels_;
   LabelType              maxNumLabels_;
   size_t                 size_;
   std::vector<ValueType> coefficients_;
   bool                   shareCoefficients_;
   std::vector<size_t>    coefficientsOffsets_;
   ValueType              lambda_;
   ValueType              bound_;

   // friends
   friend class FunctionSerialization<SumConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE> >;
   friend class LPFunctionTransfer_impl<SumConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>, VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>;
};

/// \cond HIDDEN_SYMBOLS
/// FunctionRegistration
template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
struct FunctionRegistration<SumConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE> > {
   enum ID {
      // TODO set final Id
      Id = opengm::FUNCTION_TYPE_ID_OFFSET - 4
   };
};

/// FunctionSerialization
template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
class FunctionSerialization<SumConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE> > {
public:
   typedef typename SumConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::ValueType ValueType;

   static size_t indexSequenceSize(const SumConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>&);
   static size_t valueSequenceSize(const SumConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>&);
   template<class INDEX_OUTPUT_ITERATOR, class VALUE_OUTPUT_ITERATOR>
   static void serialize(const SumConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>&, INDEX_OUTPUT_ITERATOR, VALUE_OUTPUT_ITERATOR);
   template<class INDEX_INPUT_ITERATOR, class VALUE_INPUT_ITERATOR>
   static void deserialize( INDEX_INPUT_ITERATOR, VALUE_INPUT_ITERATOR, SumConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>&);
};
/// \endcond

/***********************
 * class documentation *
 ***********************/
/*! \file sum_constraint_function.hxx
 *  \brief Provides implementation of a sum constraint function.
 */

/*! \class SumConstraintFunction
 *  \brief A sum constraint function class penalizing the violation of a given
 *         linear equality constraint.
 *
 *  This class implements a sum constraint function which penalizes the
 *  violation of a linear equality constraint. The function evaluates to
 *  \f[
 *     f(x) = |\sum c_i(x_i) - b| * \lambda
 *  \f]
 *  where \f$c_i(x_i)\f$ is the coefficient value of variable \f$i\f$ if the
 *  variable takes the label \f$x_i\f$.
 *
 *  \tparam VALUE_TYPE Value type.
 *  \tparam INDEX_TYPE Index type.
 *  \tparam LABEL_TYPE Label type.
 *
 *  \ingroup functions
 */

/*! \typedef SumConstraintFunction::ValueType
 *  \brief Typedef of the VALUE_TYPE template parameter type from the class
 *         SumConstraintFunction.
 */

/*! \typedef SumConstraintFunction::IndexType
 *  \brief Typedef of the INDEX_TYPE template parameter type from the class
 *         SumConstraintFunction.
 */

/*! \typedef SumConstraintFunction::LabelType
 *  \brief Typedef of the LABEL_TYPE template parameter type from the class
 *         SumConstraintFunction.
 */

/*! \fn SumConstraintFunction::SumConstraintFunction()
 * \brief SumConstraintFunction constructor.
 *
 *  This constructor will create an empty SumConstraintFunction.
 */

/*! \fn SumConstraintFunction::SumConstraintFunction(SHAPE_ITERATOR_TYPE shapeBegin, SHAPE_ITERATOR_TYPE shapeEnd, COEFFICIENTS_ITERATOR_TYPE coefficientsBegin, COEFFICIENTS_ITERATOR_TYPE coefficientsEnd, const bool shareCoefficients, const ValueType lambda = 1.0, const ValueType bound = 0.0)
 *  \brief SumConstraintFunction constructor.
 *
 *  This constructor will create a SumConstraintFunction where each variable can
 *  have a different number of labels.
 *
 *  \tparam SHAPE_ITERATOR_TYPE Iterator type used to iterate over the shape of
 *                              the function.
 *  \tparam COEFFICIENTS_ITERATOR_TYPE Iterator type used to iterate over the
 *                                              coefficients of the function.
 *
 *  \param[in] shapeBegin Iterator pointing to the begin of the sequence which
 *                        defines the shape of the function.
 *  \param[in] shapeEnd Iterator pointing to the end of the sequence which
 *                      defines the shape of the function.
 *  \param[in] coefficientsBegin Iterator pointing to the begin of the sequence
 *                               which defines the coefficients of the
 *                               linear constraint. There are two possibilities
 *                               for the coefficients.
 *                               -# For each label of each variable there has to
 *                                  be one coefficient. The coefficients are
 *                                  expected to be in the following order:
 *                                  \f$c_{0,0},...,c_{0,l_{0}-1},c_{1,0},...,
 *                                  c_{n-1,l_{n-1}-1}\f$, where \f$c_{i,j}\f$ is
 *                                  the coefficient for variable \f$i\f$
 *                                  assigned to label \f$j\f$, \f$l_{i}\f$ is
 *                                  the number of labels of variable \f$i\f$ and
 *                                  \f$n\f$ is the number of variables of the
 *                                  function.
 *                               -# The coefficients for the labels are the same
 *                                  for each variable. Hence there has to be one
 *                                  coefficient for every possible label and
 *                                  \f$c_l\f$ is the coefficient for label
 *                                  \f$l\f$.
 *  \param[in] coefficientsEnd Iterator pointing to the end of the sequence
 *                             which defines the coefficients of the linear
 *                             constraint.
 *  \param[in] shareCoefficients Tell if the variables share the coefficients
 *                               for the labels.
 *  \param[in] lambda The factor by which the violation of the equality
 *                    constraint will be penalized.
 *  \param[in] bound The bound of the equality constraint.
 */

/*! \fn SumConstraintFunction::SumConstraintFunction(const IndexType numVariables, const LabelType numLabels, COEFFICIENTS_ITERATOR_TYPE coefficientsBegin, COEFFICIENTS_ITERATOR_TYPE coefficientsEnd, const bool shareCoefficients, const ValueType lambda = 1.0, const ValueType bound = 0.0);
 *  \brief SumConstraintFunction constructor.
 *
 *  This constructor will create a SumConstraintFunction where each variable has
 *  the same number of labels.
 *
 *  \tparam COEFFICIENTS_ITERATOR_TYPE Iterator type used to iterate over the
 *                                              coefficients of the function.
 *
 *  \param[in] numVariables The number of variables of the
 *                          SumConstraintFunction.
 *  \param[in] numLabels The number of labels of each variable.
 *  \param[in] coefficientsBegin Iterator pointing to the begin of the sequence
 *                               which defines the coefficients of the
 *                               linear constraint. There are two possibilities
 *                               for the coefficients.
 *                               -# For each label of each variable there has to
 *                                  be one coefficient. The coefficients are
 *                                  expected to be in the following order:
 *                                  \f$c_{0,0},...,c_{0,l_{0}-1},c_{1,0},...,
 *                                  c_{n-1,l_{n-1}-1}\f$, where \f$c_{i,j}\f$ is
 *                                  the coefficient for variable \f$i\f$
 *                                  assigned to label \f$j\f$, \f$l_{i}\f$ is
 *                                  the number of labels of variable \f$i\f$ and
 *                                  \f$n\f$ is the number of variables of the
 *                                  function.
 *                               -# The coefficients for the labels are the same
 *                                  for each variable. Hence there has to be one
 *                                  coefficient for every possible label and
 *                                  \f$c_l\f$ is the coefficient for label
 *                                  \f$l\f$.
 *  \param[in] coefficientsEnd Iterator pointing to the end of the sequence
 *                             which defines the coefficients of the linear
 *                             constraint.
 *  \param[in] shareCoefficients Tell if the variables share the coefficients
 *                               for the labels.
 *  \param[in] lambda The factor by which the violation of the equality
 *                    constraint will be penalized.
 *  \param[in] bound The bound of the equality constraint.
 */

/*! \fn SumConstraintFunction::~SumConstraintFunction()
 *  \brief SumConstraintFunction destructor.
 */

/*! \fn SumConstraintFunction::ValueType SumConstraintFunction::operator()(Iterator statesBegin) const
 *  \brief Function evaluation.
 *
 *  \tparam Iterator Iterator type
 *
 *  \param[in] statesBegin Iterator pointing to the begin of a sequence of
 *                         labels for the variables of the function.
 *
 *  \return The absolute value by which the equality constraint is violated
 *          scaled with the factor lambda (\f$f(x) = |\sum c_i(x_i) - b| *
 *          \lambda\f$).
 */

/*! \fn size_t SumConstraintFunction::shape(const size_t i) const
 *  \brief Number of labels of the indicated input variable.
 *
 *  \param[in] i Index of the variable.
 *
 *  \return Returns the number of labels of the i-th variable.
 */

/*! \fn size_t SumConstraintFunction::dimension() const
 *  \brief Number of input variables.
 *
 *  \return Returns the number of variables.
 */

/*! \fn size_t SumConstraintFunction::size() const
 *  \brief Number of parameters.
 *
 *  \return Returns the number of parameters.
 */

/*! \var SumConstraintFunction::shape_
 *  \brief The shape of the function. Only valid if
 *         SumConstraintFunction::useSameNumLabels_ is set to false.
 */

/*! \var SumConstraintFunction::numVariables_
 *  \brief The number of variables of the function.
 */

/*! \var SumConstraintFunction::useSameNumLabels_
 *  \brief Tell if each variable of the function has the same number of labels.
 */

/*! \var SumConstraintFunction::maxNumLabels_
 *  \brief The maximum number of labels of the variables.
 */

/*! \var SumConstraintFunction::size_
 *  \brief Stores the size of the function.
 */

/*! \var SumConstraintFunction::coefficients_
 *  \brief The coefficients of the equality constraint.
 */

/*! \var SumConstraintFunction::shareCoefficients_
 *  \brief Tell if the labels of the variables share the same coefficients.
 */

/*! \var SumConstraintFunction::coefficientsOffsets_
 *  \brief The Offsets of the SumConstraintFunction::coefficients_ indices for
 *         each variable. Only valid if
 *         SumConstraintFunction::shareCoefficients_ is set to false.
 */

/*! \var SumConstraintFunction::lambda_
 *  \brief The scaling factor for the violation of the equality constraint.
 */

/*! \var SumConstraintFunction::bound_
 *  \brief The bound for the equality constraint.
 */

/******************
 * implementation *
 ******************/
template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline SumConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::SumConstraintFunction()
   : shape_(), numVariables_(), useSameNumLabels_(), maxNumLabels_(), size_(),
     coefficients_(), shareCoefficients_(),
     coefficientsOffsets_(), lambda_(), bound_() {

}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
template <class SHAPE_ITERATOR_TYPE, class COEFFICIENTS_ITERATOR_TYPE>
inline SumConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::SumConstraintFunction(SHAPE_ITERATOR_TYPE shapeBegin, SHAPE_ITERATOR_TYPE shapeEnd, COEFFICIENTS_ITERATOR_TYPE coefficientsBegin, COEFFICIENTS_ITERATOR_TYPE coefficientsEnd, const bool shareCoefficients, const ValueType lambda, const ValueType bound)
   : shape_(shapeBegin, shapeEnd), numVariables_(shape_.size()),
     useSameNumLabels_(numVariables_ > 0 ? std::equal(shape_.begin() + 1, shape_.end(), shape_.begin()) : true),
     maxNumLabels_(numVariables_ > 0 ? *std::max_element(shape_.begin(), shape_.end()) : 0),
     size_(std::accumulate(shapeBegin, shapeEnd, 1, std::multiplies<typename std::iterator_traits<SHAPE_ITERATOR_TYPE>::value_type>())),
     coefficients_(coefficientsBegin, coefficientsEnd),
     shareCoefficients_(shareCoefficients), coefficientsOffsets_(),
     lambda_(lambda), bound_(bound) {
   if(shareCoefficients_) {
      OPENGM_ASSERT(maxNumLabels_ == coefficients_.size());
   } else {
      OPENGM_ASSERT(std::accumulate(shape_.begin(), shape_.end(), size_t(0), std::plus<size_t>()) == coefficients_.size());
      coefficientsOffsets_.resize(numVariables_);
      // compute coefficients offsets
      size_t currentOffset = 0;
      for(size_t i = 0; i < numVariables_; ++i) {
         coefficientsOffsets_[i] = currentOffset;
         currentOffset += shape_[i];
      }
   }
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
template <class COEFFICIENTS_ITERATOR_TYPE>
inline SumConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::SumConstraintFunction(const IndexType numVariables, const LabelType numLabels, COEFFICIENTS_ITERATOR_TYPE coefficientsBegin, COEFFICIENTS_ITERATOR_TYPE coefficientsEnd, const bool shareCoefficients, const ValueType lambda, const ValueType bound)
   : shape_(), numVariables_(numVariables), useSameNumLabels_(true),
     maxNumLabels_(numLabels),
     size_(unsignedIntegerPow(maxNumLabels_, numVariables_)),
     coefficients_(coefficientsBegin, coefficientsEnd),
     shareCoefficients_(shareCoefficients), coefficientsOffsets_(),
     lambda_(lambda), bound_(bound) {
   if(shareCoefficients_) {
      OPENGM_ASSERT(maxNumLabels_ == coefficients_.size());
   } else {
      OPENGM_ASSERT(numVariables_ * maxNumLabels_ == coefficients_.size());
      coefficientsOffsets_.resize(numVariables_);
      // compute coefficients offsets
      size_t currentOffset = 0;
      for(size_t i = 0; i < numVariables_; ++i) {
         coefficientsOffsets_[i] = currentOffset;
         currentOffset += maxNumLabels_;
      }
   }
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline SumConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::~SumConstraintFunction() {

}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
template<class Iterator>
inline typename SumConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::ValueType SumConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::operator()(Iterator statesBegin) const {
   ValueType sumConstraintViolation = -bound_;
   if(shareCoefficients_) {
      for(size_t i = 0; i < numVariables_; ++i) {
         sumConstraintViolation += coefficients_[statesBegin[i]];
      }
   } else {
      for(size_t i = 0; i < numVariables_; ++i) {
         sumConstraintViolation += coefficients_[coefficientsOffsets_[i] + statesBegin[i]];
      }
   }
   return std::abs(sumConstraintViolation) * lambda_;
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline size_t SumConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::shape(const size_t i) const {
   OPENGM_ASSERT(i < numVariables_);
   if(useSameNumLabels_) {
      return maxNumLabels_;
   } else {
      return shape_[i];
   }
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline size_t SumConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::dimension() const {
   return numVariables_;
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline size_t SumConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::size() const {
   return size_;
}

/// \cond HIDDEN_SYMBOLS
template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline size_t FunctionSerialization<SumConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE> >::indexSequenceSize(const SumConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>& src) {
   const size_t sameNumLabelsSize    = 1;
   const size_t numVariablesSize     = 1;
   const size_t shapeSize            = src.useSameNumLabels_ ? 1 : src.shape_.size();
   const size_t shareCoefficientsSize = 1;
   const size_t coefficientsSize = 1;

   const size_t totalIndexSize = sameNumLabelsSize + numVariablesSize
         + shapeSize + shareCoefficientsSize + coefficientsSize;
   return totalIndexSize;
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline size_t FunctionSerialization<SumConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE> >::valueSequenceSize(const SumConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>& src) {
   const size_t coefficientsSize = src.coefficients_.size();
   const size_t lambdaSize = 1;
   const size_t boundSize = 1;

   const size_t totalValueSize = coefficientsSize + lambdaSize + boundSize;
   return totalValueSize;
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
template<class INDEX_OUTPUT_ITERATOR, class VALUE_OUTPUT_ITERATOR>
inline void FunctionSerialization<SumConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE> >::serialize(const SumConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>& src, INDEX_OUTPUT_ITERATOR indexOutIterator, VALUE_OUTPUT_ITERATOR valueOutIterator) {
   // index output
   // shape
   *indexOutIterator = static_cast<typename INDEX_OUTPUT_ITERATOR::value_type>(src.useSameNumLabels_);
   ++indexOutIterator;
   *indexOutIterator = src.numVariables_;
   ++indexOutIterator;
   if(src.useSameNumLabels_) {
      *indexOutIterator = src.maxNumLabels_;
      ++indexOutIterator;
   } else {
      for(size_t i = 0; i < src.shape_.size(); ++i) {
         *indexOutIterator = src.shape_[i];
         ++indexOutIterator;
      }
   }

   // share coefficients
   *indexOutIterator = src.shareCoefficients_;
   ++indexOutIterator;

   // coefficients size
   *indexOutIterator = src.coefficients_.size();

   // value output
   // coefficients
   for(size_t i = 0; i <src.coefficients_.size(); ++i) {
      *valueOutIterator = src.coefficients_[i];
      ++valueOutIterator;
   }

   // lambda
   *valueOutIterator = src.lambda_;
   ++valueOutIterator;

   // bound
   *valueOutIterator = src.bound_;
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
template<class INDEX_INPUT_ITERATOR, class VALUE_INPUT_ITERATOR>
inline void FunctionSerialization<SumConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE> >::deserialize(INDEX_INPUT_ITERATOR indexInIterator, VALUE_INPUT_ITERATOR valueInIterator, SumConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>& dst) {
   typedef VALUE_TYPE ValueType;
   typedef INDEX_TYPE IndexType;

   // index input
   // shape
   const bool useSameNumLabels = *indexInIterator;
   ++indexInIterator;
   const IndexType numVariables = *indexInIterator;
   ++indexInIterator;
   INDEX_INPUT_ITERATOR shapeBegin = indexInIterator;
   indexInIterator += (useSameNumLabels ? 1 : numVariables);
   INDEX_INPUT_ITERATOR shapeEnd = indexInIterator;
   // share coefficients
   const bool shareCoefficients = *indexInIterator;
   ++indexInIterator;
   // coefficients size
   const size_t numCoeffiecients = *indexInIterator;

   // value input
   // coefficients
   VALUE_INPUT_ITERATOR coefficientsBegin = valueInIterator;
   VALUE_INPUT_ITERATOR coefficientsEnd = valueInIterator + numCoeffiecients;
   valueInIterator += numCoeffiecients;

   // lambda
   const ValueType lambda = *valueInIterator;
   ++valueInIterator;

   // bound
   const ValueType bound = *valueInIterator;

   if(useSameNumLabels) {
      dst = SumConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>(numVariables, *shapeBegin, coefficientsBegin, coefficientsEnd, shareCoefficients, lambda, bound);
   } else {
      dst = SumConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>(shapeBegin, shapeEnd, coefficientsBegin, coefficientsEnd, shareCoefficients, lambda, bound);
   }
}
/// \endcond

} // namespace opengm

#endif /* OPENGM_SUM_CONSTRAINT_FUNCTION_HXX_ */
