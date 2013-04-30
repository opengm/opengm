#ifndef STRING_ARGUMENT_HXX_
#define STRING_ARGUMENT_HXX_

#include "argument_base.hxx"

namespace opengm {

namespace interface {

/*********************
 * class definitions *
 *********************/

template <class CONTAINER = std::vector<std::string> >
class StringArgument : public ArgumentBase<std::string, CONTAINER> {
public:
   StringArgument(std::string& storageIn, const std::string& shortNameIn,
         const std::string& longNameIn, const std::string& descriptionIn,
         const bool requiredIn = false);
   StringArgument(std::string& storageIn, const std::string& shortNameIn,
         const std::string& longNameIn, const std::string& descriptionIn,
         const std::string& defaultValueIn);
   StringArgument(std::string& storageIn, const std::string& shortNameIn,
         const std::string& longNameIn, const std::string& descriptionIn,
         const bool requiredIn, const CONTAINER& permittedValuesIn);
   StringArgument(std::string& storageIn, const std::string& shortNameIn,
         const std::string& longNameIn, const std::string& descriptionIn,
         const std::string& defaultValueIn, const CONTAINER& permittedValuesIn);
};

/***********************
 * class documentation *
 ***********************/
//TODO add documentation

/******************
 * implementation *
 ******************/

template <class CONTAINER>
inline StringArgument<CONTAINER>::StringArgument(std::string& storageIn, const std::string& shortNameIn,
    const std::string& longNameIn, const std::string& descriptionIn,
    const bool requiredIn)
    : ArgumentBase<std::string, CONTAINER>(storageIn, shortNameIn, longNameIn, descriptionIn, requiredIn) {
   ;
}

template <class CONTAINER>
inline StringArgument<CONTAINER>::StringArgument(std::string& storageIn, const std::string& shortNameIn,
    const std::string& longNameIn, const std::string& descriptionIn,
    const std::string& defaultValueIn)
    : ArgumentBase<std::string, CONTAINER>(storageIn, shortNameIn, longNameIn, descriptionIn, defaultValueIn) {
   ;
}

template <class CONTAINER>
inline StringArgument<CONTAINER>::StringArgument(std::string& storageIn, const std::string& shortNameIn,
      const std::string& longNameIn, const std::string& descriptionIn,
      const bool requiredIn, const CONTAINER& permittedValuesIn)
      : ArgumentBase<std::string, CONTAINER>(storageIn, shortNameIn, longNameIn, descriptionIn, requiredIn, permittedValuesIn) {
   ;
}

template <class CONTAINER>
inline StringArgument<CONTAINER>::StringArgument(std::string& storageIn, const std::string& shortNameIn,
      const std::string& longNameIn, const std::string& descriptionIn,
      const std::string& defaultValueIn, const CONTAINER& permittedValuesIn)
      : ArgumentBase<std::string, CONTAINER>(storageIn, shortNameIn, longNameIn, descriptionIn, defaultValueIn, permittedValuesIn) {
   ;
}

} // namespace interface

} // namespace opengm

#endif /* STRING_ARGUMENT_HXX_ */
