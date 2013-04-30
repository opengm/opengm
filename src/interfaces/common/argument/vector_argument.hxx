#ifndef VECTOR_ARGUMENT_HXX_
#define VECTOR_ARGUMENT_HXX_

#include "argument_base.hxx"

namespace opengm {

namespace interface {

/*********************
 * class definitions *
 *********************/

template <class VECTOR, class CONTAINER = std::vector<VECTOR> >
class VectorArgument : public ArgumentBase<VECTOR, CONTAINER> {
public:
   VectorArgument(VECTOR& storageIn, const std::string& shortNameIn,
         const std::string& longNameIn, const std::string& descriptionIn,
         const bool requiredIn = false);
   VectorArgument(VECTOR& storageIn, const std::string& shortNameIn,
         const std::string& longNameIn, const std::string& descriptionIn,
         const VECTOR& defaultValueIn);
   VectorArgument(VECTOR& storageIn, const std::string& shortNameIn,
         const std::string& longNameIn, const std::string& descriptionIn,
         const bool requiredIn, const CONTAINER& permittedValuesIn);
   VectorArgument(VECTOR& storageIn, const std::string& shortNameIn,
         const std::string& longNameIn, const std::string& descriptionIn,
         const VECTOR& defaultValueIn, const CONTAINER& permittedValuesIn);

   void printValidValues(std::ostream& stream) const;
   void printDefaultValue(std::ostream& stream) const;
   void printHelp(std::ostream& stream, bool verbose) const;
   void markAsSet() const;
};

/***********************
 * class documentation *
 ***********************/
//TODO add documentation

/******************
 * implementation *
 ******************/

template <class VECTOR, class CONTAINER>
inline VectorArgument<VECTOR, CONTAINER>::VectorArgument(VECTOR& storageIn, const std::string& shortNameIn,
      const std::string& longNameIn, const std::string& descriptionIn,
      const bool requiredIn)
      : ArgumentBase<VECTOR, CONTAINER>(storageIn, shortNameIn, longNameIn, descriptionIn, requiredIn) {

}

template <class VECTOR, class CONTAINER>
inline VectorArgument<VECTOR, CONTAINER>::VectorArgument(VECTOR& storageIn, const std::string& shortNameIn,
      const std::string& longNameIn, const std::string& descriptionIn,
      const VECTOR& defaultValueIn)
      : ArgumentBase<VECTOR, CONTAINER>(storageIn, shortNameIn, longNameIn, descriptionIn, defaultValueIn) {

}

template <class VECTOR, class CONTAINER>
inline VectorArgument<VECTOR, CONTAINER>::VectorArgument(VECTOR& storageIn, const std::string& shortNameIn,
      const std::string& longNameIn, const std::string& descriptionIn,
      const bool requiredIn, const CONTAINER& permittedValuesIn)
      : ArgumentBase<VECTOR, CONTAINER>(storageIn, shortNameIn, longNameIn, descriptionIn, requiredIn, permittedValuesIn) {

}

template <class VECTOR, class CONTAINER>
inline VectorArgument<VECTOR, CONTAINER>::VectorArgument(VECTOR& storageIn, const std::string& shortNameIn,
      const std::string& longNameIn, const std::string& descriptionIn,
      const VECTOR& defaultValueIn, const CONTAINER& permittedValuesIn)
      : ArgumentBase<VECTOR, CONTAINER>(storageIn, shortNameIn, longNameIn, descriptionIn, defaultValueIn, permittedValuesIn) {

}

template <class VECTOR, class CONTAINER>
inline void VectorArgument<VECTOR, CONTAINER>::printValidValues(std::ostream& stream) const {
   //TODO add vector print
   throw RuntimeError("Print vector Argument Values not yet implemented");
}

template <class VECTOR, class CONTAINER>
inline void VectorArgument<VECTOR, CONTAINER>::printDefaultValue(std::ostream& stream) const {
   stream << "(";
   for(typename VECTOR::const_iterator iter = ArgumentBase<VECTOR, CONTAINER>::defaultValue_.begin(); iter != ArgumentBase<VECTOR, CONTAINER>::defaultValue_.end(); iter++) {
      if(iter + 1 != ArgumentBase<VECTOR, CONTAINER>::defaultValue_.end()) {
         stream << *iter <<"; ";
      }
   }
   if(ArgumentBase<VECTOR, CONTAINER>::defaultValue_.begin() != ArgumentBase<VECTOR, CONTAINER>::defaultValue_.end()) {
      stream << ArgumentBase<VECTOR, CONTAINER>::defaultValue_.back();
   }
   stream << ")";
}


template <class VECTOR, class CONTAINER>
inline void VectorArgument<VECTOR, CONTAINER>::printHelp(std::ostream& stream, bool verbose) const {
   ArgumentBase<VECTOR, CONTAINER>::printHelpBase(stream, verbose);

   if(verbose) {
      if(ArgumentBase<VECTOR, CONTAINER>::permittedValues_.size() != 0) {
         stream << std::setw(49) << "" << "permitted values: ";
         printValidValues(stream);
         stream << std::endl;
      }
      if(ArgumentBase<VECTOR, CONTAINER>::hasDefaultValue_) {
         stream << std::setw(49) << "" << "default value: ";
         printDefaultValue(stream);
         stream << std::endl;
      }
   }
}

template <class VECTOR, class CONTAINER>
inline void VectorArgument<VECTOR, CONTAINER>::markAsSet() const {
   ArgumentBase<VECTOR, CONTAINER>::isSet_ = true;
}

} // namespace interface

} // namespace opengm

#endif /* VECTOR_ARGUMENT_HXX_ */
