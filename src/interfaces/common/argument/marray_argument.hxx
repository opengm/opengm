#ifndef MARRAY_ARGUMENT_HXX_
#define MARRAY_ARGUMENT_HXX_

#include "argument_base.hxx"

namespace opengm {

namespace interface {

/*********************
 * class definitions *
 *********************/

template <class MARRAY, class CONTAINER = std::vector<MARRAY> >
class MArrayArgument : public ArgumentBase<MARRAY, CONTAINER> {
public:
   MArrayArgument(MARRAY& storageIn, const std::string& shortNameIn,
         const std::string& longNameIn, const std::string& descriptionIn,
         const bool requiredIn = false);
   MArrayArgument(MARRAY& storageIn, const std::string& shortNameIn,
         const std::string& longNameIn, const std::string& descriptionIn,
         const MARRAY& defaultValueIn);
   MArrayArgument(MARRAY& storageIn, const std::string& shortNameIn,
         const std::string& longNameIn, const std::string& descriptionIn,
         const bool requiredIn, const CONTAINER& permittedValuesIn);
   MArrayArgument(MARRAY& storageIn, const std::string& shortNameIn,
         const std::string& longNameIn, const std::string& descriptionIn,
         const MARRAY& defaultValueIn, const CONTAINER& permittedValuesIn);

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

template <class MARRAY, class CONTAINER>
inline MArrayArgument<MARRAY, CONTAINER>::MArrayArgument(MARRAY& storageIn, const std::string& shortNameIn,
      const std::string& longNameIn, const std::string& descriptionIn,
      const bool requiredIn)
      : ArgumentBase<MARRAY, CONTAINER>(storageIn, shortNameIn, longNameIn, descriptionIn, requiredIn) {

}

template <class MARRAY, class CONTAINER>
inline MArrayArgument<MARRAY, CONTAINER>::MArrayArgument(MARRAY& storageIn, const std::string& shortNameIn,
      const std::string& longNameIn, const std::string& descriptionIn,
      const MARRAY& defaultValueIn)
      : ArgumentBase<MARRAY, CONTAINER>(storageIn, shortNameIn, longNameIn, descriptionIn, defaultValueIn) {

}

template <class MARRAY, class CONTAINER>
inline MArrayArgument<MARRAY, CONTAINER>::MArrayArgument(MARRAY& storageIn, const std::string& shortNameIn,
      const std::string& longNameIn, const std::string& descriptionIn,
      const bool requiredIn, const CONTAINER& permittedValuesIn)
      : ArgumentBase<MARRAY, CONTAINER>(storageIn, shortNameIn, longNameIn, descriptionIn, requiredIn, permittedValuesIn) {

}

template <class MARRAY, class CONTAINER>
inline MArrayArgument<MARRAY, CONTAINER>::MArrayArgument(MARRAY& storageIn, const std::string& shortNameIn,
      const std::string& longNameIn, const std::string& descriptionIn,
      const MARRAY& defaultValueIn, const CONTAINER& permittedValuesIn)
      : ArgumentBase<MARRAY, CONTAINER>(storageIn, shortNameIn, longNameIn, descriptionIn, defaultValueIn, permittedValuesIn) {

}

template <class MARRAY, class CONTAINER>
inline void MArrayArgument<MARRAY, CONTAINER>::printValidValues(std::ostream& stream) const {
   //TODO add marray print
   throw RuntimeError("Print marray Argument Values not yet implemented");
}

template <class MARRAY, class CONTAINER>
inline void MArrayArgument<MARRAY, CONTAINER>::printDefaultValue(std::ostream& stream) const {
   stream << "(";
   for(typename MARRAY::const_iterator iter = ArgumentBase<MARRAY, CONTAINER>::defaultValue_.begin(); iter != ArgumentBase<MARRAY, CONTAINER>::defaultValue_.end(); iter++) {
      if(iter + 1 != ArgumentBase<MARRAY, CONTAINER>::defaultValue_.end()) {
         stream << *iter <<"; ";
      }
   }
   if(ArgumentBase<MARRAY, CONTAINER>::defaultValue_.begin() != ArgumentBase<MARRAY, CONTAINER>::defaultValue_.end()) {
      stream << ArgumentBase<MARRAY, CONTAINER>::defaultValue_.back();
   }
   stream << ")";
}


template <class MARRAY, class CONTAINER>
inline void MArrayArgument<MARRAY, CONTAINER>::printHelp(std::ostream& stream, bool verbose) const {
   ArgumentBase<MARRAY, CONTAINER>::printHelpBase(stream, verbose);

   if(verbose) {
      if(ArgumentBase<MARRAY, CONTAINER>::permittedValues_.size() != 0) {
         stream << std::setw(49) << "" << "permitted values: ";
         printValidValues(stream);
         stream << std::endl;
      }
      if(ArgumentBase<MARRAY, CONTAINER>::hasDefaultValue_) {
         stream << std::setw(49) << "" << "default value: ";
         printDefaultValue(stream);
         stream << std::endl;
      }
   }
}

template <class MARRAY, class CONTAINER>
inline void MArrayArgument<MARRAY, CONTAINER>::markAsSet() const {
   ArgumentBase<MARRAY, CONTAINER>::isSet_ = true;
}

} // namespace interface

} // namespace opengm


#endif /* MARRAY_ARGUMENT_HXX_ */
