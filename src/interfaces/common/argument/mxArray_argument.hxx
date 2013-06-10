#ifndef MXARRAY_ARGUMENT_HXX_
#define MXARRAY_ARGUMENT_HXX_

#include <typeinfo>

#include <mex.h>

#include "argument_base.hxx"

namespace opengm {

namespace interface {

/*********************
 * class definitions *
 *********************/

template <class CONTAINER = std::vector<mxArray*> >
class mxArrayArgument : public ArgumentBase<mxArray*, CONTAINER> {
public:
   mxArrayArgument(mxArray*& storageIn, const std::string& shortNameIn,
         const std::string& longNameIn, const std::string& descriptionIn,
         const bool requiredIn = false);
   mxArrayArgument(mxArray*& storageIn, const std::string& shortNameIn,
         const std::string& longNameIn, const std::string& descriptionIn,
         const mxArray*& defaultValueIn);
   mxArrayArgument(mxArray*& storageIn, const std::string& shortNameIn,
         const std::string& longNameIn, const std::string& descriptionIn,
         const bool requiredIn, const CONTAINER& permittedValuesIn);
   mxArrayArgument(mxArray*& storageIn, const std::string& shortNameIn,
         const std::string& longNameIn, const std::string& descriptionIn,
         const mxArray*& defaultValueIn, const CONTAINER& permittedValuesIn);
   void printDefaultValue(std::ostream& stream) const;
};

template <class CONTAINER = std::vector<const mxArray*> >
class mxArrayConstArgument : public ArgumentBase<const mxArray*, CONTAINER> {
public:
   mxArrayConstArgument(const mxArray*& storageIn, const std::string& shortNameIn,
         const std::string& longNameIn, const std::string& descriptionIn,
         const bool requiredIn = false);
   mxArrayConstArgument(const mxArray*& storageIn, const std::string& shortNameIn,
         const std::string& longNameIn, const std::string& descriptionIn,
         const mxArray*& defaultValueIn);
   mxArrayConstArgument(const mxArray*& storageIn, const std::string& shortNameIn,
         const std::string& longNameIn, const std::string& descriptionIn,
         const bool requiredIn, const CONTAINER& permittedValuesIn);
   mxArrayConstArgument(const mxArray*& storageIn, const std::string& shortNameIn,
         const std::string& longNameIn, const std::string& descriptionIn,
         const mxArray*& defaultValueIn, const CONTAINER& permittedValuesIn);

   void printDefaultValue(std::ostream& stream) const;
};
/***********************
 * class documentation *
 ***********************/
//TODO add documentation

/******************
 * implementation *
 ******************/

template <class CONTAINER>
inline mxArrayArgument<CONTAINER>::mxArrayArgument(mxArray*& storageIn, const std::string& shortNameIn,
      const std::string& longNameIn, const std::string& descriptionIn,
      const bool requiredIn)
      : ArgumentBase<mxArray*, CONTAINER>(storageIn, shortNameIn, longNameIn, descriptionIn, requiredIn) {

}

template <class CONTAINER>
inline mxArrayArgument<CONTAINER>::mxArrayArgument(mxArray*& storageIn, const std::string& shortNameIn,
      const std::string& longNameIn, const std::string& descriptionIn,
      const mxArray*& defaultValueIn)
      : ArgumentBase<mxArray*, CONTAINER>(storageIn, shortNameIn, longNameIn, descriptionIn, defaultValueIn) {

}

template <class CONTAINER>
inline mxArrayArgument<CONTAINER>::mxArrayArgument(mxArray*& storageIn, const std::string& shortNameIn,
      const std::string& longNameIn, const std::string& descriptionIn,
      const bool requiredIn, const CONTAINER& permittedValuesIn)
      : ArgumentBase<mxArray*, CONTAINER>(storageIn, shortNameIn, longNameIn, descriptionIn, requiredIn, permittedValuesIn) {

}

template <class CONTAINER>
inline mxArrayArgument<CONTAINER>::mxArrayArgument(mxArray*& storageIn, const std::string& shortNameIn,
      const std::string& longNameIn, const std::string& descriptionIn,
      const mxArray*& defaultValueIn, const CONTAINER& permittedValuesIn)
      : ArgumentBase<mxArray*, CONTAINER>(storageIn, shortNameIn, longNameIn, descriptionIn, defaultValueIn, permittedValuesIn) {

}

template <class CONTAINER>
inline void mxArrayArgument<CONTAINER>::printDefaultValue(std::ostream& stream) const {
   if(mxIsChar(this->defaultValue_)) {
      stream << mxArrayToString(this->defaultValue_);
   }
}

template <class CONTAINER>
inline mxArrayConstArgument<CONTAINER>::mxArrayConstArgument(const mxArray*& storageIn, const std::string& shortNameIn,
      const std::string& longNameIn, const std::string& descriptionIn,
      bool requiredIn)
      : ArgumentBase<const mxArray*, CONTAINER>(storageIn, shortNameIn, longNameIn, descriptionIn, requiredIn) {

}

template <class CONTAINER>
inline mxArrayConstArgument<CONTAINER>::mxArrayConstArgument(const mxArray*& storageIn, const std::string& shortNameIn,
      const std::string& longNameIn, const std::string& descriptionIn,
      const mxArray*& defaultValueIn)
      : ArgumentBase<const mxArray*, CONTAINER>(storageIn, shortNameIn, longNameIn, descriptionIn, defaultValueIn) {

}

template <class CONTAINER>
inline mxArrayConstArgument<CONTAINER>::mxArrayConstArgument(const mxArray*& storageIn, const std::string& shortNameIn,
      const std::string& longNameIn, const std::string& descriptionIn,
      const bool requiredIn, const CONTAINER& permittedValuesIn)
      : ArgumentBase<const mxArray*, CONTAINER>(storageIn, shortNameIn, longNameIn, descriptionIn, requiredIn, permittedValuesIn) {

}

template <class CONTAINER>
inline mxArrayConstArgument<CONTAINER>::mxArrayConstArgument(const mxArray*& storageIn, const std::string& shortNameIn,
      const std::string& longNameIn, const std::string& descriptionIn,
      const mxArray*& defaultValueIn, const CONTAINER& permittedValuesIn)
      : ArgumentBase<const mxArray*, CONTAINER>(storageIn, shortNameIn, longNameIn, descriptionIn, defaultValueIn, permittedValuesIn) {

}

template <class CONTAINER>
inline void mxArrayConstArgument<CONTAINER>::printDefaultValue(std::ostream& stream) const {
   if(mxIsChar(this->defaultValue_)) {
      stream << mxArrayToString(this->defaultValue_);
   }
}

} // namespace interface

} // namespace opengm

#endif /* MXARRAY_ARGUMENT_HXX_ */
