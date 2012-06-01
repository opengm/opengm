#ifndef INT_ARGUMENT_HXX_
#define INT_ARGUMENT_HXX_

#include "argument_base.hxx"

namespace opengm {

namespace interface {

/*********************
 * class definitions *
 *********************/

template <class CONTAINER = std::vector<int> >
class IntArgument : public ArgumentBase<int, CONTAINER> {
public:
   IntArgument(int& storageIn, const std::string& shortNameIn,
         const std::string& longNameIn, const std::string& descriptionIn,
         const bool requiredIn = false);
   IntArgument(int& storageIn, const std::string& shortNameIn,
         const std::string& longNameIn, const std::string& descriptionIn,
         const int& defaultValueIn);
   IntArgument(int& storageIn, const std::string& shortNameIn,
         const std::string& longNameIn, const std::string& descriptionIn,
         const bool requiredIn, const CONTAINER& permittedValuesIn);
   IntArgument(int& storageIn, const std::string& shortNameIn,
         const std::string& longNameIn, const std::string& descriptionIn,
         const int& defaultValueIn, const CONTAINER& permittedValuesIn);
};

/***********************
 * class documentation *
 ***********************/
//TODO add documentation

/******************
 * implementation *
 ******************/

template <class CONTAINER>
inline IntArgument<CONTAINER>::IntArgument(int& storageIn, const std::string& shortNameIn,
      const std::string& longNameIn, const std::string& descriptionIn,
      const bool requiredIn)
      : ArgumentBase<int, CONTAINER>(storageIn, shortNameIn, longNameIn, descriptionIn, requiredIn) {

}

template <class CONTAINER>
inline IntArgument<CONTAINER>::IntArgument(int& storageIn, const std::string& shortNameIn,
      const std::string& longNameIn, const std::string& descriptionIn,
      const int& defaultValueIn)
      : ArgumentBase<int, CONTAINER>(storageIn, shortNameIn, longNameIn, descriptionIn, defaultValueIn) {

}

template <class CONTAINER>
inline IntArgument<CONTAINER>::IntArgument(int& storageIn, const std::string& shortNameIn,
      const std::string& longNameIn, const std::string& descriptionIn,
      const bool requiredIn, const CONTAINER& permittedValuesIn)
      : ArgumentBase<int, CONTAINER>(storageIn, shortNameIn, longNameIn, descriptionIn, requiredIn, permittedValuesIn) {

}

template <class CONTAINER>
inline IntArgument<CONTAINER>::IntArgument(int& storageIn, const std::string& shortNameIn,
      const std::string& longNameIn, const std::string& descriptionIn,
      const int& defaultValueIn, const CONTAINER& permittedValuesIn)
      : ArgumentBase<int, CONTAINER>(storageIn, shortNameIn, longNameIn, descriptionIn, defaultValueIn, permittedValuesIn) {

}

} // namespace interface

} // namespace opengm

#endif /* INT_ARGUMENT_HXX_ */
