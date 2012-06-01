#ifndef SIZE_T_ARGUMENT_HXX_
#define SIZE_T_ARGUMENT_HXX_

#include "argument_base.hxx"

namespace opengm {

namespace interface {

/*********************
 * class definitions *
 *********************/

template <class CONTAINER = std::vector<size_t> >
class Size_TArgument : public ArgumentBase<size_t, CONTAINER> {
public:
   Size_TArgument(size_t& storageIn, const std::string& shortNameIn,
         const std::string& longNameIn, const std::string& descriptionIn,
         const bool requiredIn = false);
   Size_TArgument(size_t& storageIn, const std::string& shortNameIn,
         const std::string& longNameIn, const std::string& descriptionIn,
         const size_t& defaultValueIn);
   Size_TArgument(size_t& storageIn, const std::string& shortNameIn,
         const std::string& longNameIn, const std::string& descriptionIn,
         const bool requiredIn, const CONTAINER& permittedValuesIn);
   Size_TArgument(size_t& storageIn, const std::string& shortNameIn,
         const std::string& longNameIn, const std::string& descriptionIn,
         const size_t& defaultValueIn, const CONTAINER& permittedValuesIn);
};

/***********************
 * class documentation *
 ***********************/
//TODO add documentation

/******************
 * implementation *
 ******************/

template <class CONTAINER>
inline Size_TArgument<CONTAINER>::Size_TArgument(size_t& storageIn, const std::string& shortNameIn,
      const std::string& longNameIn, const std::string& descriptionIn,
      const bool requiredIn)
      : ArgumentBase<size_t, CONTAINER>(storageIn, shortNameIn, longNameIn, descriptionIn, requiredIn) {

}

template <class CONTAINER>
inline Size_TArgument<CONTAINER>::Size_TArgument(size_t& storageIn, const std::string& shortNameIn,
      const std::string& longNameIn, const std::string& descriptionIn,
      const size_t& defaultValueIn)
      : ArgumentBase<size_t, CONTAINER>(storageIn, shortNameIn, longNameIn, descriptionIn, defaultValueIn) {

}

template <class CONTAINER>
inline Size_TArgument<CONTAINER>::Size_TArgument(size_t& storageIn, const std::string& shortNameIn,
      const std::string& longNameIn, const std::string& descriptionIn,
      const bool requiredIn, const CONTAINER& permittedValuesIn)
      : ArgumentBase<size_t, CONTAINER>(storageIn, shortNameIn, longNameIn, descriptionIn, requiredIn, permittedValuesIn) {

}

template <class CONTAINER>
inline Size_TArgument<CONTAINER>::Size_TArgument(size_t& storageIn, const std::string& shortNameIn,
      const std::string& longNameIn, const std::string& descriptionIn,
      const size_t& defaultValueIn, const CONTAINER& permittedValuesIn)
      : ArgumentBase<size_t, CONTAINER>(storageIn, shortNameIn, longNameIn, descriptionIn, defaultValueIn, permittedValuesIn) {

}

} // namespace interface

} // namespace opengm

#endif /* SIZE_T_ARGUMENT_HXX_ */
