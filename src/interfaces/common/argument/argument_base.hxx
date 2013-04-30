#ifndef ARGUMENT_BASE_HXX_
#define ARGUMENT_BASE_HXX_

#include <string>
#include <iostream>
#include <iomanip>
#include <vector>

#include <opengm/opengm.hxx>
#include <opengm/utilities/metaprogramming.hxx>

#include "argument_delimiter.hxx"

namespace opengm {

namespace interface {

/*********************
 * class definitions *
 *********************/

template <class T, class CONTAINER = std::vector<T> >
class ArgumentBase {
protected:
   static const size_t shortNameSize_ = 11;
   static const size_t longNameSize_ = 27;
   static const size_t requiredSize_ = 8;
   static const size_t descriptionSize_ = 50;
   typedef typename opengm::meta::Compare<typename CONTAINER::value_type, T>::type compiletimeTypecheck;
   T* storage_;
   std::string shortName_;
   std::string longName_;
   std::string description_;
   static const std::string delimiter_;
   bool required_;
   bool hasDefaultValue_;
   const T defaultValue_;
   CONTAINER permittedValues_;
   mutable bool isSet_;
   void printHelpBase(std::ostream& stream, bool verbose) const;
public:
   ArgumentBase(T& storageIn, const std::string& shortNameIn,
         const std::string& longNameIn, const std::string& descriptionIn,
         const bool requiredIn = false);
   ArgumentBase(T& storageIn, const std::string& shortNameIn,
         const std::string& longNameIn, const std::string& descriptionIn,
         const T& defaultValueIn);
   ArgumentBase(T& storageIn, const std::string& shortNameIn,
         const std::string& longNameIn, const std::string& descriptionIn,
         const bool requiredIn, const CONTAINER& permittedValuesIn);
   ArgumentBase(T& storageIn, const std::string& shortNameIn,
         const std::string& longNameIn, const std::string& descriptionIn,
         const T& defaultValueIn, const CONTAINER& permittedValuesIn);

   const std::string& getShortName() const;
   const std::string& getLongName() const;
   const std::string& getDescription() const;
   bool isRequired() const;
   bool hasDefaultValue() const;
   const T& getDefaultValue() const;
   const CONTAINER& GetPermittedValues() const;
   bool valueIsValid(const T& value) const;
   void printValidValues(std::ostream& stream) const;
   const ArgumentBase<T, CONTAINER>& operator()(const T& value, const bool isSet) const;
   const T& getValue() const;
   T& getReference() const;
   const bool& isSet() const;
   void printDefaultValue(std::ostream& stream) const;
   void printHelp(std::ostream& stream, bool verbose) const;
};

template <class T, class CONTAINER>
const std::string ArgumentBase<T, CONTAINER>::delimiter_ = ArgumentBaseDelimiter::delimiter_;

/***********************
 * class documentation *
 ***********************/
//TODO add documentation

/******************
 * implementation *
 ******************/
template <class T, class CONTAINER>
inline ArgumentBase<T, CONTAINER>::ArgumentBase(T& storageIn, const std::string& shortNameIn,
    const std::string& longNameIn, const std::string& descriptionIn,
    const bool requiredIn) : storage_(&storageIn),
    shortName_(shortNameIn), longName_(longNameIn), description_(descriptionIn),
    required_(requiredIn), hasDefaultValue_(false), defaultValue_(), isSet_(false)
{
   OPENGM_META_ASSERT(compiletimeTypecheck::value , CONTAINER_HAS_WRONG_VALUE_TYPE);
}

template <class T, class CONTAINER>
inline ArgumentBase<T, CONTAINER>::ArgumentBase(T& storageIn, const std::string& shortNameIn,
    const std::string& longNameIn, const std::string& descriptionIn,
    const T& defaultValueIn) : storage_(&storageIn),
    shortName_(shortNameIn), longName_(longNameIn), description_(descriptionIn),
    required_(false), hasDefaultValue_(true),
    defaultValue_(defaultValueIn), isSet_(false)
{
   OPENGM_META_ASSERT(compiletimeTypecheck::value , CONTAINER_HAS_WRONG_VALUE_TYPE);
}

template <class T, class CONTAINER>
inline ArgumentBase<T, CONTAINER>::ArgumentBase(T& storageIn, const std::string& shortNameIn,
    const std::string& longNameIn, const std::string& descriptionIn,
    const bool requiredIn, const CONTAINER& permittedValuesIn) :
    storage_(&storageIn), shortName_(shortNameIn), longName_(longNameIn),
    description_(descriptionIn), required_(requiredIn),
    hasDefaultValue_(false), permittedValues_(permittedValuesIn), isSet_(false)
{
   OPENGM_META_ASSERT(compiletimeTypecheck::value , CONTAINER_HAS_WRONG_VALUE_TYPE);
}

template <class T, class CONTAINER>
inline ArgumentBase<T, CONTAINER>::ArgumentBase(T& storageIn, const std::string& shortNameIn,
    const std::string& longNameIn, const std::string& descriptionIn,
    const T& defaultValueIn, const CONTAINER& permittedValuesIn)
    : storage_(&storageIn), shortName_(shortNameIn), longName_(longNameIn),
      description_(descriptionIn), required_(false), hasDefaultValue_(true),
    defaultValue_(defaultValueIn), permittedValues_(permittedValuesIn), isSet_(false)
{
   OPENGM_META_ASSERT(compiletimeTypecheck::value , TEST);
}

template <class T, class CONTAINER>
inline const std::string& ArgumentBase<T, CONTAINER>::getShortName() const {
   return this->shortName_;
}

template <class T, class CONTAINER>
inline const std::string& ArgumentBase<T, CONTAINER>::getLongName() const {
   return this->longName_;
}

template <class T, class CONTAINER>
inline const std::string& ArgumentBase<T, CONTAINER>::getDescription() const {
   return this->description_;
}

template <class T, class CONTAINER>
inline bool ArgumentBase<T, CONTAINER>::isRequired() const {
   return this->required_;
}

template <class T, class CONTAINER>
inline bool ArgumentBase<T, CONTAINER>::hasDefaultValue() const {
   return this->hasDefaultValue_;
}

template <class T, class CONTAINER>
inline const T& ArgumentBase<T, CONTAINER>::getDefaultValue() const {
   return this->defaultValue_;
}

template <class T, class CONTAINER>
inline const CONTAINER& ArgumentBase<T, CONTAINER>::GetPermittedValues() const {
   return this->permittedValues_;
}

template <class T, class CONTAINER>
inline bool ArgumentBase<T, CONTAINER>::valueIsValid(const T& value) const {
   //all values are allowed?
   if(this->permittedValues_.size() == 0) {
      return true;
   } else {
      for(typename CONTAINER::const_iterator iter = this->permittedValues_.begin(); iter != this->permittedValues_.end(); iter++) {
         if(*iter == value) {
            return true;
         }
      }
   }
   return false;
}

template <class T, class CONTAINER>
inline void ArgumentBase<T, CONTAINER>::printValidValues(std::ostream& stream) const {
   for(typename CONTAINER::const_iterator iter = this->permittedValues_.begin(); iter != this->permittedValues_.end(); iter++) {
      stream << *iter << "; ";
   }
}

template <class T, class CONTAINER>
inline const ArgumentBase<T, CONTAINER>& ArgumentBase<T, CONTAINER>::operator()(const T& value, const bool isSet) const {
   if(valueIsValid(value)) {
      *(this->storage_) = value;
      this->isSet_ = isSet;
      return *this;
   } else {
      std::stringstream error;
      error << value << " is not a valid value for argument: \"" << this->longName_ << "\". Possible are: " << std::endl;
      this->printValidValues(error);
      throw RuntimeError(error.str());
   }
}

template <class T, class CONTAINER>
inline const T& ArgumentBase<T, CONTAINER>::getValue() const {
   return *(this->storage_);
}

template <class T, class CONTAINER>
inline T& ArgumentBase<T, CONTAINER>::getReference() const {
   return *(this->storage_);
}

template <class T, class CONTAINER>
inline const bool& ArgumentBase<T, CONTAINER>::isSet() const {
   return this->isSet_;
}

template <class T, class CONTAINER>
inline void ArgumentBase<T, CONTAINER>::printHelpBase(std::ostream& stream, bool verbose) const {
   if(shortName_.size() != 0) {
      stream << "  " + delimiter_ << std::setw(shortNameSize_ - delimiter_.size()) << std::left << shortName_;
   } else {
      stream << std::setw(shortNameSize_ + 2) << std::left << "";
   }

   stream << " " + delimiter_ + delimiter_ << std::setw(longNameSize_ - (2 * delimiter_.size())) << std::left << longName_;
   if(required_) {
      stream << std::setw(requiredSize_) << std::left << "yes";
   } else {
      stream << std::setw(requiredSize_) << std::left << "no";
   }

   stream << description_ << std::endl;
}

template <class T, class CONTAINER>
inline void ArgumentBase<T, CONTAINER>::printDefaultValue(std::ostream& stream) const {
   stream << defaultValue_;
}

template <class T, class CONTAINER>
inline void ArgumentBase<T, CONTAINER>::printHelp(std::ostream& stream, bool verbose) const {
   printHelpBase(stream, verbose);

   if(verbose) {
      if(permittedValues_.size() != 0) {
         stream << std::setw(49) << "" << "permitted values: ";
         printValidValues(stream);
         stream << std::endl;
      }
      if(hasDefaultValue_) {
         stream << std::setw(49) << "" << "default value: ";
         printDefaultValue(stream);
         stream << std::endl;
      }
   }
}

} // namespace interface

} // namespace opengm

#endif /* ARGUMENT_BASE_HXX_ */
