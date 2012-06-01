#ifndef ARGUMENT_HXX_
#define ARGUMENT_HXX_

#include <string>
#include <vector>
#include <cstdlib>
#include <map>

#include "helper/interfacesEnums.hxx"

namespace opengm
{

namespace interface
{

template <class T>
class Argument //COM-J: Rename 2 OpengmInterfaceArgument
{
protected:
  T* storage_;
  std::string shortName_;
  std::string longName_;
  std::string description_;
  bool required_;
  bool hasDefaultValue_;
  T defaultValue_;
  std::vector<T> permittedValues_;
  mutable bool isSet_;

public:
  Argument(T& storageIn, const std::string& shortNameIn,
      const std::string& longNameIn, const std::string& descriptionIn,
      const bool requiredIn);
  Argument(T& storageIn, const std::string& shortNameIn,
      const std::string& longNameIn, const std::string& descriptionIn,
      const bool requiredIn, const T& defaultValueIn);
  Argument(T& storageIn, const std::string& shortNameIn,
        const std::string& longNameIn, const std::string& descriptionIn,
        const bool requiredIn, const std::vector<T>& permittedValuesIn);
  Argument(T& storageIn, const std::string& shortNameIn,
        const std::string& longNameIn, const std::string& descriptionIn,
        const bool requiredIn, const T& defaultValueIn,
        const std::vector<T>& permittedValuesIn);

  const std::string& getShortName() const;
  const std::string& getLongName() const;
  const std::string& getDescription() const;
  bool isRequired() const;
  bool hasDefaultValue() const;
  const T& getDefaultValue() const;
  const std::vector<T>& GetPermittedValues() const;
  bool valueIsValid(const T& value) const;
  void printValidValues() const;
  const Argument<T>& operator()(const T& value) const;
  const T& getValue() const;
  const bool& isSet() const;
};

template <>
class Argument<void>
{
protected:
  bool* storage_;
  std::string shortName_;
  std::string longName_;
  std::string description_;
  bool required_;

public:
  Argument(bool& flagSet, const std::string& shortNameIn,
      const std::string& longNameIn, const std::string& descriptionIn,
      const bool requiredIn);
  const std::string& getShortName() const;
  const std::string& getLongName() const;
  const std::string& getDescription() const;
  bool isRequired() const;
  bool hasDefaultValue() const;
  const Argument<void>& operator()(const bool& value) const;
  const bool& getValue() const;
  const bool& isSet() const;
};

template <>
class Argument<interfaceType> : public Argument<std::string> {
protected:
  std::map<std::string, interfaceType>& possibleAlgorithms_;
public:
  Argument(std::string& storageIn, const std::string& shortNameIn,
      const std::string& longNameIn, const std::string& descriptionIn,
      const bool requiredIn,
      std::map<std::string, interfaceType>& possibleAlgorithmsIn);
  Argument(std::string& storageIn, const std::string& shortNameIn,
      const std::string& longNameIn, const std::string& descriptionIn,
      const bool requiredIn,
      std::map<std::string, interfaceType>& possibleAlgorithmsIn,
      std::string defaultValueIn);
  //assignment operator needed to use Argument<interfaceType> with boost::variant
  Argument<interfaceType>& operator=(const Argument<interfaceType>& argIn);
  interfaceType getInterfaceType() const;
};

template <typename T>
class Argument<std::vector<T> > : public Argument<std::string> {
protected:
   std::string userInput_;
   std::vector<T>* storage_;
public:
   Argument(std::vector<T>& storageIn, const std::string& shortNameIn,
         const std::string& longNameIn, const std::string& descriptionIn,
         const bool requiredIn);
   const std::string& getVecFileLocation() const;
   std::vector<T>& getValue() const;
};

template <class T>
Argument<T>::Argument(T& storageIn, const std::string& shortNameIn,
    const std::string& longNameIn, const std::string& descriptionIn,
    const bool requiredIn) : storage_(&storageIn),
    shortName_(shortNameIn), longName_(longNameIn), description_(descriptionIn),
    required_(requiredIn), hasDefaultValue_(false), isSet_(false)
{

}

template <class T>
Argument<T>::Argument(T& storageIn, const std::string& shortNameIn,
    const std::string& longNameIn, const std::string& descriptionIn,
    const bool requiredIn, const T& defaultValueIn) : storage_(&storageIn),
    shortName_(shortNameIn), longName_(longNameIn), description_(descriptionIn),
    required_(requiredIn), /*hasValue_(true),*/ hasDefaultValue_(true),
    defaultValue_(defaultValueIn), isSet_(false)
{

}

template <class T>
Argument<T>::Argument(T& storageIn, const std::string& shortNameIn,
    const std::string& longNameIn, const std::string& descriptionIn,
    const bool requiredIn, const std::vector<T>& permittedValuesIn) :
    storage_(&storageIn), shortName_(shortNameIn), longName_(longNameIn),
    description_(descriptionIn), required_(requiredIn), /*hasValue_(true),*/
    hasDefaultValue_(false), permittedValues_(permittedValuesIn), isSet_(false)
{

}

template <class T>
Argument<T>::Argument(T& storageIn, const std::string& shortNameIn,
    const std::string& longNameIn, const std::string& descriptionIn,
    const bool requiredIn, const T& defaultValueIn,
    const std::vector<T>& permittedValuesIn) : storage_(&storageIn),
    shortName_(shortNameIn), longName_(longNameIn), description_(descriptionIn),
    required_(requiredIn), /*hasValue_(true),*/ hasDefaultValue_(true),
    defaultValue_(defaultValueIn), permittedValues_(permittedValuesIn), isSet_(false)
{

}

template <class T>
const std::string& Argument<T>::getShortName() const
{
  return this->shortName_;
}

template <class T>
const std::string& Argument<T>::getLongName() const
{
  return this->longName_;
}

template <class T>
const std::string& Argument<T>::getDescription() const
{
  return this->description_;
}

template <class T>
bool Argument<T>::isRequired() const
{
  return this->required_;
}

template <class T>
bool Argument<T>::hasDefaultValue() const
{
  return this->hasDefaultValue_;
}

template <class T>
const T& Argument<T>::getDefaultValue() const
{
  return this->defaultValue_;
}

template <class T>
const std::vector<T>& Argument<T>::GetPermittedValues() const
{
  return this->permittedValues_;
}

template <class T>
bool Argument<T>::valueIsValid(const T& value) const
{
  //all values are allowed
  if(this->permittedValues_.size() == 0)
  {
    return true;
  }
  else
  {
    for(typename std::vector<T>::const_iterator iter =
        this->permittedValues_.begin(); iter != this->permittedValues_.end();
        iter++)
    {
      if(*iter == value)
      {
        return true;
      }
    }
  }
  return false;
}

template <class T>
void Argument<T>::printValidValues() const
{
  for(typename std::vector<T>::const_iterator iter =
      this->permittedValues_.begin(); iter != this->permittedValues_.end();
      iter++)
  {
    std::cout << *iter << "; ";
  }
}

template <class T>
const Argument<T>& Argument<T>::operator()(const T& value) const
{
  if(this->permittedValues_.size() != 0)
  {
    for(typename std::vector<T>::const_iterator iter =
        this->permittedValues_.begin(); iter != this->permittedValues_.end();
        iter++)
    {
      if(*iter == value)
      {
        *(this->storage_) = value;
        this->isSet_ = true;
        return *this;
      }
      else
      {
        continue;
      }
    }
    std::cerr << value << " is not a valid value for argument: \"" << this->longName_ << "\". Possible are: " << std::endl;
    this->printValidValues();
    std::cout << std::endl;
    std::abort();
  }
  else
  {
    *(this->storage_) = value;
    this->isSet_ = true;
    return *this;
  }
}

template <class T>
const T& Argument<T>::getValue() const {
  return *(this->storage_);
}

template <class T>
const bool& Argument<T>::isSet() const {
  return this->isSet_;
}

Argument<void>::Argument(bool& flagSet, const std::string& shortNameIn,
    const std::string& longNameIn, const std::string& descriptionIn,
    const bool requiredIn) : storage_(&flagSet), shortName_(shortNameIn),
    longName_(longNameIn), description_(descriptionIn), required_(requiredIn)
{

}

const std::string& Argument<void>::getShortName() const
{
  return this->shortName_;
}

const std::string& Argument<void>::getLongName() const
{
  return this->longName_;
}

const std::string& Argument<void>::getDescription() const
{
  return this->description_;
}

bool Argument<void>::isRequired() const
{
  return this->required_;
}

bool Argument<void>::hasDefaultValue() const
{
  return false;
}

const Argument<void>& Argument<void>::operator()(const bool& value) const {
  *storage_ = value;
}

const bool& Argument<void>::getValue() const {
  return *(this->storage_);
}

const bool& Argument<void>::isSet() const {
  return *(this->storage_);
}

Argument<interfaceType>::Argument(std::string& storageIn,
    const std::string& shortNameIn, const std::string& longNameIn,
    const std::string& descriptionIn, const bool requiredIn,
    std::map<std::string, interfaceType>& possibleAlgorithmsIn) :
    Argument<std::string>::Argument(storageIn, shortNameIn, longNameIn,
        descriptionIn, requiredIn), possibleAlgorithms_(possibleAlgorithmsIn) {
  std::map<std::string, interfaceType>::const_iterator iter;
  for(iter = this->possibleAlgorithms_.begin(); iter != this->possibleAlgorithms_.end(); iter++) {
   this->permittedValues_.push_back(iter->first);
  }
}

Argument<interfaceType>::Argument(std::string& storageIn,
    const std::string& shortNameIn, const std::string& longNameIn,
    const std::string& descriptionIn, const bool requiredIn,
    std::map<std::string, interfaceType>& possibleAlgorithmsIn,
    std::string defaultValueIn) :
    Argument<std::string>::Argument(storageIn, shortNameIn, longNameIn,
        descriptionIn, requiredIn, defaultValueIn),
        possibleAlgorithms_(possibleAlgorithmsIn) {
  std::map<std::string, interfaceType>::const_iterator iter;
  for(iter = this->possibleAlgorithms_.begin(); iter != this->possibleAlgorithms_.end(); iter++) {
   this->permittedValues_.push_back(iter->first);
  }
}

Argument<interfaceType>& Argument<interfaceType>::operator=(const Argument<interfaceType>& argIn) {
  if(this != &argIn) {
    this->shortName_ = argIn.shortName_;
    this->longName_ = argIn.longName_;
    this->description_ = argIn.description_;
    this->required_ = argIn.required_;
    this->hasDefaultValue_ = argIn.hasDefaultValue_;
    this->defaultValue_ = argIn.defaultValue_;
    this->permittedValues_.clear();
    this->permittedValues_.insert(this->permittedValues_.begin(), argIn.permittedValues_.begin(), argIn.permittedValues_.end());
    this->isSet_ = argIn.isSet_;
    this->possibleAlgorithms_.clear();
    this->possibleAlgorithms_.insert(argIn.possibleAlgorithms_.begin(), argIn.possibleAlgorithms_.end());
  }
  return *this;
}

interfaceType Argument<interfaceType>::getInterfaceType() const {
  std::map<std::string, interfaceType>::iterator iter;
  iter = this->possibleAlgorithms_.find(*(this->storage_));
  if(iter != this->possibleAlgorithms_.end()) {
    return iter->second;
  } else {
    std::cerr << "error: unknown algorithm" << std::endl;
    abort();
  }
}

template <typename T>
Argument<std::vector<T> >::Argument(std::vector<T>& storageIn, const std::string& shortNameIn,
      const std::string& longNameIn, const std::string& descriptionIn, const bool requiredIn)
      : Argument<std::string>(userInput_, shortNameIn, longNameIn, descriptionIn, requiredIn), storage_(&storageIn) {

}

template <typename T>
const std::string& Argument<std::vector<T> >::getVecFileLocation() const {
   return this->userInput_;
}

template <typename T>
std::vector<T>& Argument<std::vector<T> >::getValue() const {
   return *(this->storage_);
}

} // namespace interface

} // namespace opengm
#endif /* ARGUMENT_HXX_ */
