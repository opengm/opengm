#ifndef BOOL_ARGUMENT_HXX_
#define BOOL_ARGUMENT_HXX_

#include "argument_base.hxx"

namespace opengm {

namespace interface {

/*********************
 * class definitions *
 *********************/

class BoolArgument {
protected:
   typedef std::vector<bool> CONTAINER;
   static const size_t shortNameSize_ = 11;
   static const size_t longNameSize_ = 27;
   static const size_t requiredSize_ = 8;
   static const size_t descriptionSize_ = 50;
   typedef opengm::meta::Compare<CONTAINER::value_type, bool>::type compiletimeTypecheck;
   bool* storage_;
   std::string shortName_;
   std::string longName_;
   std::string description_;
   static const std::string delimiter_;
   CONTAINER permittedValues_;
   mutable bool isSet_;
   void printHelpBase(std::ostream& stream, bool verbose) const;
public:
   BoolArgument(bool& storageIn, const std::string& shortNameIn,
       const std::string& longNameIn, const std::string& descriptionIn);

   const std::string& getShortName() const;
   const std::string& getLongName() const;
   const std::string& getDescription() const;
   bool isRequired() const;
   bool hasDefaultValue() const;
   const bool& getDefaultValue() const;
   const CONTAINER& GetPermittedValues() const;
   bool valueIsValid(const bool& value) const;
   void printValidValues(std::ostream& stream) const;
   const BoolArgument& operator()(const bool& value, const bool isSet) const;
   const bool& getValue() const;
   bool& getReference() const;
   const bool& isSet() const;
   void printDefaultValue(std::ostream& stream) const;
   void printHelp(std::ostream& stream, bool verbose) const;
};

const std::string BoolArgument::delimiter_ = ArgumentBaseDelimiter::delimiter_;

/***********************
 * class documentation *
 ***********************/
//TODO add documentation

/******************
 * implementation *
 ******************/

BoolArgument::BoolArgument(bool& storageIn, const std::string& shortNameIn, const std::string& longNameIn, const std::string& descriptionIn)
   : storage_(&storageIn), shortName_(shortNameIn), longName_(longNameIn), description_(descriptionIn), isSet_(false) {
   ;
}

inline const std::string& BoolArgument::getShortName() const {
   return this->shortName_;
}

inline const std::string& BoolArgument::getLongName() const {
   return this->longName_;
}

inline const std::string& BoolArgument::getDescription() const {
   return this->description_;
}

inline bool BoolArgument::isRequired() const {
   return false;
}

inline bool BoolArgument::hasDefaultValue() const {
   return false;
}

/*inline const T& BoolArgument::getDefaultValue() const {
   return this->defaultValue_;
}*/

inline const BoolArgument::CONTAINER& BoolArgument::GetPermittedValues() const {
   return this->permittedValues_;
}

inline bool BoolArgument::valueIsValid(const bool& value) const {
   //all values are allowed?
   if(this->permittedValues_.size() == 0) {
      return true;
   } else {
      for(CONTAINER::const_iterator iter = this->permittedValues_.begin(); iter != this->permittedValues_.end(); iter++) {
         if(*iter == value) {
            return true;
         }
      }
   }
   return false;
}

inline void BoolArgument::printValidValues(std::ostream& stream) const {
   for(CONTAINER::const_iterator iter = this->permittedValues_.begin(); iter != this->permittedValues_.end(); iter++) {
      stream << *iter << "; ";
   }
}

inline const BoolArgument& BoolArgument::operator()(const bool& value, const bool isSet) const {
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

inline const bool& BoolArgument::getValue() const {
   return *(this->storage_);
}

inline bool& BoolArgument::getReference() const {
   return *(this->storage_);
}

inline const bool& BoolArgument::isSet() const {
   return this->isSet_;
}

inline void BoolArgument::printHelpBase(std::ostream& stream, bool verbose) const {
   if(shortName_.size() != 0) {
      stream << "  " + delimiter_ << std::setw(shortNameSize_ - delimiter_.size()) << std::left << shortName_;
   } else {
      stream << std::setw(shortNameSize_ + 2) << std::left << "";
   }

   stream << " " + delimiter_ + delimiter_ << std::setw(longNameSize_ - (2 * delimiter_.size())) << std::left << longName_;
   if(false) {
      stream << std::setw(requiredSize_) << std::left << "yes";
   } else {
      stream << std::setw(requiredSize_) << std::left << "no";
   }

   stream << description_ << std::endl;
}

inline void BoolArgument::printDefaultValue(std::ostream& stream) const {
   stream << false;
}

inline void BoolArgument::printHelp(std::ostream& stream, bool verbose) const {
   printHelpBase(stream, verbose);

   if(verbose) {
      if(permittedValues_.size() != 0) {
         stream << std::setw(49) << "" << "permitted values: ";
         printValidValues(stream);
         stream << std::endl;
      }
      if(false) {
         stream << std::setw(49) << "" << "default value: ";
         printDefaultValue(stream);
         stream << std::endl;
      }
   }
}

} // namespace interface

} // namespace opengm

#endif /* BOOL_ARGUMENT_HXX_ */
