#ifndef FLOAT_ARGUMENT_HXX_
#define FLOAT_ARGUMENT_HXX_

#include "argument_base.hxx"

namespace opengm {

namespace interface {

/*********************
 * class definitions *
 *********************/

template <class CONTAINER = std::vector<float> >
class FloatArgument : public ArgumentBase<float, CONTAINER> {
public:
   FloatArgument(float& storageIn, const std::string& shortNameIn,
         const std::string& longNameIn, const std::string& descriptionIn,
         const bool requiredIn = false);
   FloatArgument(float& storageIn, const std::string& shortNameIn,
         const std::string& longNameIn, const std::string& descriptionIn,
         const float& defaultValueIn);
   FloatArgument(float& storageIn, const std::string& shortNameIn,
         const std::string& longNameIn, const std::string& descriptionIn,
         const bool requiredIn, const CONTAINER& permittedValuesIn);
   FloatArgument(float& storageIn, const std::string& shortNameIn,
         const std::string& longNameIn, const std::string& descriptionIn,
         const float& defaultValueIn, const CONTAINER& permittedValuesIn);
};

/***********************
 * class documentation *
 ***********************/
//TODO add documentation

/******************
 * implementation *
 ******************/

template <class CONTAINER>
inline FloatArgument<CONTAINER>::FloatArgument(float& storageIn, const std::string& shortNameIn,
      const std::string& longNameIn, const std::string& descriptionIn,
      const bool requiredIn)
      : ArgumentBase<float, CONTAINER>(storageIn, shortNameIn, longNameIn, descriptionIn, requiredIn) {

}

template <class CONTAINER>
inline FloatArgument<CONTAINER>::FloatArgument(float& storageIn, const std::string& shortNameIn,
      const std::string& longNameIn, const std::string& descriptionIn,
      const float& defaultValueIn)
      : ArgumentBase<float, CONTAINER>(storageIn, shortNameIn, longNameIn, descriptionIn, defaultValueIn) {

}

template <class CONTAINER>
inline FloatArgument<CONTAINER>::FloatArgument(float& storageIn, const std::string& shortNameIn,
      const std::string& longNameIn, const std::string& descriptionIn,
      const bool requiredIn, const CONTAINER& permittedValuesIn)
      : ArgumentBase<float, CONTAINER>(storageIn, shortNameIn, longNameIn, descriptionIn, requiredIn, permittedValuesIn) {

}

template <class CONTAINER>
inline FloatArgument<CONTAINER>::FloatArgument(float& storageIn, const std::string& shortNameIn,
      const std::string& longNameIn, const std::string& descriptionIn,
      const float& defaultValueIn, const CONTAINER& permittedValuesIn)
      : ArgumentBase<float, CONTAINER>(storageIn, shortNameIn, longNameIn, descriptionIn, defaultValueIn, permittedValuesIn) {

}

} // namespace interface

} // namespace opengm

#endif /* FLOAT_ARGUMENT_HXX_ */
