#pragma once
#ifndef OPENGM_SINGLESITEFUNCTION_HXX
#define OPENGM_SINGLESITEFUNCTION_HXX

#include <vector>
#include <algorithm>

#include "opengm/opengm.hxx"
#include "opengm/functions/function_registration.hxx"
#include "opengm/functions/function_properties_base.hxx"

namespace opengm {

/// \cond HIDDEN_SYMBOLS

template<class T, size_t SIZE>
class StackStorage {
public:
   StackStorage(){}
   ~StackStorage(){}
   template<class ITERATOR>
   const T &  operator()(ITERATOR iter) const {
      OPENGM_ASSERT(*iter<SIZE);
      return this->dataPointer_[*iter];
   }
   template<class ITERATOR>
   T & operator()(ITERATOR iter) {
      OPENGM_ASSERT(*iter<SIZE);
      return this->dataPointer_[*iter];
   }
   T & operator[](const size_t index) {
      OPENGM_ASSERT(index<SIZE);
      return this->dataPointer_[index];
   }
   const T & operator[](const size_t index) const {
      OPENGM_ASSERT(index<SIZE);
      return this->dataPointer_[index];
   }

private:
   T dataPointer_[SIZE];
};

template<class T, size_t SIZE>
class HeapStorage {
public:
   HeapStorage() {
      dataPointer_ = new T[SIZE];
   }
   ~HeapStorage(){
      delete[] dataPointer_;
   }
   T& operator[](const size_t index) {
      OPENGM_ASSERT(index<SIZE);
      return this->dataPointer_[index];
   }
   const T& operator[](const size_t index) const {
      OPENGM_ASSERT(index<SIZE);
      return this->dataPointer_[index];
   }
   template<class ITERATOR>
   const T& operator()(ITERATOR iter) const {
      OPENGM_ASSERT(*iter<SIZE);
      return this->dataPointer_[*iter];
   }
   template<class ITERATOR>
   T& operator()(ITERATOR iter) {
      OPENGM_ASSERT(*iter<SIZE);
      return this->dataPointer_[*iter];
   }

private:
   T* dataPointer_;
};

/// \endcond 

/// \brief Single site function whose size is fixed at compile time
///
/// \ingroup functions
template<class T, size_t SIZE, template<typename, size_t> class STORAGE>
class StaticSingleSiteFunction
:  STORAGE<T, SIZE>,
   public FunctionBase<StaticSingleSiteFunction<T, SIZE, STORAGE>, T, size_t, size_t>
{
public:
   typedef T ValueType;
   typedef T value_type;

   StaticSingleSiteFunction():STORAGE<T, SIZE>() {}
   StaticSingleSiteFunction(const StaticSingleSiteFunction& other)
      :  STORAGE<T, SIZE>()
      {
         for(size_t i=0;i<SIZE;++i){
            (*this).operator [](i)=other[i];
         }
      }

   StaticSingleSiteFunction& operator=(const StaticSingleSiteFunction & other)
      {
         if(this != &other) {
            for(size_t i=0;i<SIZE;++i) {
               (*this).operator [](i)=other[i];
            }
         }
         return *this;
      }

   template<class ITERATOR> const T& operator()(ITERATOR iter) const
      {
         OPENGM_ASSERT(*iter < SIZE);
         return (static_cast<const STORAGE<T, SIZE>&>(*this)).operator()(iter);
      }

   template<class ITERATOR> T& operator()(ITERATOR iter)
      {
         OPENGM_ASSERT(*iter<SIZE);
         return (static_cast<STORAGE<T, SIZE>&>(*this)).operator()(iter);
      }

   size_t size() const
      { return SIZE; }

   size_t dimension() const
      { return 1; }

   size_t shape(const size_t index) const
      {
         OPENGM_ASSERT(index == 0);
         return SIZE;
      }
};

/// Single site function with dynamic size
///
/// \ingroup functions
template<class T>
class DynamicSingleSiteFunction {
public:
   typedef T ValueType;
   typedef T value_type;

   DynamicSingleSiteFunction(const size_t size = 0)
      :  size_(size)
      {
         if(size_ != 0){
            dataPointer_ = new T[size];
         }
      }

   DynamicSingleSiteFunction(const size_t size, ValueType value)
      :  size_(size)
      {
         if(size_ != 0) {
            dataPointer_ = new T[size];
            std::fill(dataPointer_, dataPointer_+size_, value);
         }
      }

   ~DynamicSingleSiteFunction()
      {
         if(size_ != 0) {
            delete[] dataPointer_;
         }
      }

   DynamicSingleSiteFunction(const DynamicSingleSiteFunction & other)
      {
         if(other.size_ != 0) {
            dataPointer_ = new T[other.size_];
            size_ = other.size_;
            std::copy(other.dataPointer_, other.dataPointer_+size_, dataPointer_);
         }
         else {
            size_=0;
         }
      }

   DynamicSingleSiteFunction& operator=(const DynamicSingleSiteFunction& other)
      {
         if(this != &other) {
            if(other.size_ > size_) {
               delete[] dataPointer_;
               dataPointer_ = new T[other.size_];
               size_ = other.size_;
               std::copy(other.dataPointer_, other.dataPointer_ + size_, dataPointer_);
            }
            else if(other.size_ < size_) {
               delete[] dataPointer_;
               if(other.size_!= 0) {
                  dataPointer_= new T[other.size_];
                  size_ = other.size_;
                  std::copy(other.dataPointer_, other.dataPointer_+size_, dataPointer_);
               }
               else {
                  size_ = 0;
               }
            }
            if(other.size_ == size_) {
               std::copy(other.dataPointer_, other.dataPointer_+size_, dataPointer_);
            }
         }
         return *this;
      }

   void assign(const size_t size)
      {
         if(size_ != size){
            delete[] dataPointer_;
            if(size != 0){
               dataPointer_ = new T[size];
               size_=size;
            }
            else {
               size_ = 0;
            }
         }
      }

   void assign(const size_t size, const T value)
      {
         if(size_ != size){
            delete[] dataPointer_;
            if(size != 0) {
               dataPointer_ = new T[size];
               size_ = size;
               std::fill(dataPointer_, dataPointer_+size_, value);
            }
            else {
               size_=0;
            }
         }
         else {
            std::fill(dataPointer_, dataPointer_+size_, value);
         }
      }

   size_t size() const
      { return size_; }

   size_t dimension() const
      { return 1; }

   size_t shape(const size_t index) const
      {
         OPENGM_ASSERT(index==0);
         return size_;
      }

   T& operator[](const size_t index)
      {
         OPENGM_ASSERT(index < size_);
         return dataPointer_[index];
      }

   const T & operator[](const size_t index) const
      {
         OPENGM_ASSERT(index < size_);
         return dataPointer_[index];
      }

   template<class ITERATOR>
   const T& operator()(ITERATOR iter) const
      {
         OPENGM_ASSERT(*iter < size_);
         return dataPointer_[*iter];
      }

   template<class ITERATOR>
   T& operator()(ITERATOR iter)
      {
         OPENGM_ASSERT(*iter < size_);
         return dataPointer_[*iter];
      }

private:
   T* dataPointer_;
   size_t size_;
};

/// \cond HIDDEN_SYMBOLS
/// FunctionRegistration
template<class T, size_t SIZE, template < typename , size_t > class STORAGE>
struct FunctionRegistration< StaticSingleSiteFunction<T, SIZE, STORAGE> > {
   enum ID { Id = opengm::FUNCTION_TYPE_ID_OFFSET + 9 };
};

/// FunctionRegistration
template<class T>
struct FunctionRegistration< DynamicSingleSiteFunction<T> > {
   enum ID { Id = opengm::FUNCTION_TYPE_ID_OFFSET + 10 };
};

/// FunctionSerialization
template<class T, size_t SIZE, template < typename , size_t > class STORAGE>
class FunctionSerialization< StaticSingleSiteFunction<T, SIZE, STORAGE> > {
public:
   typedef typename StaticSingleSiteFunction<T, SIZE, STORAGE> ::ValueType ValueType;

   static size_t indexSequenceSize(const StaticSingleSiteFunction<T, SIZE, STORAGE> & f){
      return 0;
   }

   static size_t valueSequenceSize(const StaticSingleSiteFunction<T, SIZE, STORAGE> & f){
      return SIZE;
   }

   template<class INDEX_OUTPUT_ITERATOR, class VALUE_OUTPUT_ITERATOR >
      static void serialize(const StaticSingleSiteFunction<T, SIZE, STORAGE>  & f, INDEX_OUTPUT_ITERATOR ii, VALUE_OUTPUT_ITERATOR vi){
      for(size_t i=0;i<SIZE;++i){
         size_t c[]={i};
         *vi=f(c);
         ++vi;
      }
   }

   template<class INDEX_INPUT_ITERATOR , class VALUE_INPUT_ITERATOR>
      static void deserialize( INDEX_INPUT_ITERATOR ii, VALUE_INPUT_ITERATOR vi, StaticSingleSiteFunction<T, SIZE, STORAGE>  & f){
      for(size_t i=0;i<SIZE;++i){
         size_t c[]={i};
         f(c)=*vi;
         ++vi;
      }
   }
};

/// FunctionSerialization
template<class T>
class FunctionSerialization< DynamicSingleSiteFunction<T> > {
public:
   typedef typename DynamicSingleSiteFunction<T>::ValueType ValueType;

   static size_t indexSequenceSize(const  DynamicSingleSiteFunction<T> & f){
      return 1;
   }

   static size_t valueSequenceSize(const  DynamicSingleSiteFunction<T> & f){
      return f.size();
   }

   template<class INDEX_OUTPUT_ITERATOR, class VALUE_OUTPUT_ITERATOR >
      static void serialize(const  DynamicSingleSiteFunction<T>  & f, INDEX_OUTPUT_ITERATOR ii, VALUE_OUTPUT_ITERATOR vi){
      for(size_t i=0;i<f.size();++i){
         size_t c[]={i};
         *vi=f(c);
         ++vi;
      }
   }

   template<class INDEX_INPUT_ITERATOR , class VALUE_INPUT_ITERATOR>
      static void deserialize( INDEX_INPUT_ITERATOR ii, VALUE_INPUT_ITERATOR vi, DynamicSingleSiteFunction<T>   & f){
      const size_t size=*ii;
      f.assign(size);
      for(size_t i=0;i<size;++i){
         size_t c[]={i};
         f(c)=*vi;
         ++vi;
      }
   }
};
/// \endcond

} // namespace opengm

#endif // #ifndef OPENGM_SINGLESITEFUNCTION_HXX
