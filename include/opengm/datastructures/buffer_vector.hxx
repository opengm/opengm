#pragma once
#ifndef OPENGM_BUFFER_VECTOR_HXX
#define	OPENGM_BUFFER_VECTOR_HXX

#include <vector>
#include <algorithm>

#include "opengm/opengm.hxx"

/// \cond HIDDEN_SYMBOLS

namespace opengm{
   /// buffer vector to avoid allocatios
   ///
   /// a resize and clear does decrease the capacity of the vector.
   /// Only the destructor deallocates memory
   /// \ingroup datastructures
   template<class T>
   class BufferVector{
      public:
         typedef T ValueType;
         typedef T value_type;
         typedef T const * ConstIteratorType;
         typedef T const * const_iterator;
         typedef T * IteratorType;
         typedef T * iterator;
         BufferVector( );
         BufferVector(const size_t );
         BufferVector(const size_t ,const T & );
         BufferVector(const BufferVector<T> &);
         ~BufferVector( );
         BufferVector<T> & operator= (const BufferVector<T> &);
         size_t size() const ;
         T const * begin() const;
         T const * end() const;
         T * const begin();
         T * const end();
         
         T const * data() const;
         T * data();
         
         T const & operator[](const size_t)const;
         T & operator[](const size_t);
         void push_back(const T &);
         void resize(const size_t );
         void reserve(const size_t );
         void clear();
         bool empty()const;
         const T & front()const;
         const T & back()const;
         T & front();
         T & back();
         template<class ITERATOR>
         void assign(ITERATOR ,ITERATOR);
      private:
         size_t size_;
         size_t capacity_;
         T * data_;
   };
   /// empty constructor 
   template<class T>
   inline BufferVector<T>::BufferVector( )
   :  size_(0),
      capacity_(0),
      data_(NULL) {
   }
   
   /// constructor
   /// \param size of vector
   template<class T>
   inline BufferVector<T>::BufferVector
   (
      const size_t size
   )
   :  size_(size),
      capacity_(size) {
      if(size_!=0) {
      data_ = new T[size];
      }
   }
   
   /// constructor
   /// \param size
   /// \param value
   template<class T>
   inline BufferVector<T>::BufferVector
   (
      const size_t size,
      const T & value
   )
   :  size_(size),
      capacity_(size) {
      data_ = new T[size];
      std::fill(data_,data_+size_,value);
   }
   
   
   /// copy constructor
   /// \param other other vector
   template<class T>
   inline BufferVector<T>::BufferVector
   (
      const  BufferVector<T> & other
   )
   :  size_(other.size_),
      capacity_(other.size_)
   {
      if(size_!=0) {
         data_ = new T[size_];
         std::copy(other.data_,other.data_+size_,data_);
      }
   }
   
   /// get pointer to the data
   template<class T>
   inline T const * 
   BufferVector<T>::data() const{
      return data_;
   }
   
   /// get pointer to the data
   template<class T>
   inline T * 
   BufferVector<T>::data() {
      return data_;
   }
   
   /// destructor
   template<class T>
   inline BufferVector<T>::~BufferVector( ) {
      if(capacity_!=0) {
         OPENGM_ASSERT(data_!=NULL);
         delete[] data_;
      }
   }
   
   /// assignment operator
   template<class T>
   inline BufferVector<T> & BufferVector<T>::operator=
   (
      const  BufferVector<T> & other
   )
   {
      if(&other!=this) {
         size_=other.size_;
         if(size_>capacity_) {
            delete [] data_;
            data_ = new T[size_];
            capacity_=size_;
         }
         std::copy(other.data_,other.data_+size_,data_);
      }
      return *this;
   }
   
   /// size of the vector
   template<class T>
   inline size_t
   BufferVector<T>::size() const {
      OPENGM_ASSERT(data_!=NULL ||size_== 0 );
      return size_;
   }
   
   /// begin iterator
   template<class T>
   inline T const *
   BufferVector<T>::begin() const{
      OPENGM_ASSERT(data_!=NULL);
      return data_;
   }
   
   /// end iterator
   template<class T>
   inline T const *
   BufferVector<T>::end() const{
      OPENGM_ASSERT(data_!=NULL);
      return data_ + size_;
   }
   
   /// begin iterator
   template<class T>
   inline T * const
   BufferVector<T>::begin() {
      OPENGM_ASSERT(data_!=NULL);
      return data_;
   }
   
   /// end iterator
   template<class T>
   inline T * const
   BufferVector<T>::end() {
      OPENGM_ASSERT(data_!=NULL);
      return data_ + size_;
   }
   
   /// bracket operator to access values
   /// \param index index
   template<class T>
   inline T const &
   BufferVector<T>::operator[]
   (
      const size_t index
   )const{
      OPENGM_ASSERT(data_!=NULL);
      OPENGM_ASSERT(index<size_);
      return data_[index];
   }
   
   /// bracket operator to access values
   /// \param index index
   template<class T>
   inline T &
   BufferVector<T>::operator[]
   (
      const size_t index
   ) {
      OPENGM_ASSERT(index<size_);
      return data_[index];
   }
   
   /// push back a value
   /// \param value to push back
   template<class T>
   inline void
   BufferVector<T>::push_back
   (
      const T & value
   ) {
      OPENGM_ASSERT(size_<=capacity_);
      if(capacity_==size_) {
         if(size_!=0) {
            T * tmp=new T[capacity_*2];
            std::copy(data_,data_+size_,tmp);
            delete[] data_;
            capacity_*=2;
            data_=tmp;
         }
         else{
            T * tmp=new T[2];
            capacity_=2;
            data_=tmp;
         }
      }
      data_[size_]=value;
      ++size_;
      OPENGM_ASSERT(size_<=capacity_);
   }
   
   /// resize
   /// \param size new size of the array
   template<class T>
   inline void
   BufferVector<T>::resize
   (
      const size_t size
   ) {
      OPENGM_ASSERT(size_<=capacity_);
      if(size>capacity_) {
         if(size_!=0) {
            T * tmp=new T[size];
            std::copy(data_,data_+size_,tmp);
            delete[] data_;
            capacity_=size;
            data_=tmp;
         }
         else{
            data_=new T[size];
            capacity_=size;
         }
      }
      size_=size;
      OPENGM_ASSERT(size_<=capacity_);
   }
   
   /// reserve memory
   /// \param size to reserve
   template<class T>
   inline void
   BufferVector<T>::reserve
   (
      const size_t size
   ) {
      OPENGM_ASSERT(size_<=capacity_);
      if(size>capacity_) {
         if(size_!=0) {
            T * tmp=new T[size];
            std::copy(data_,data_+size_,tmp);
            delete[] data_;
            data_=tmp;
         }
         else{
            data_=new T[size];
         }
         size_=size;
         capacity_=size;
      }
      //size_=size;
      OPENGM_ASSERT(size_<=capacity_);
   }
   
   /// clear sets the size to zero but does/// bracket operator to access values
   /// \param index index no deallocations
   /// \warning  no deallocations are done
   template<class T>
   inline void
   BufferVector<T>::clear() {
      OPENGM_ASSERT(size_<=capacity_);
      size_=0;
   }
   
   /// is vector is empty
   template<class T>
   inline bool
   BufferVector<T>::empty()const{
      return size_==0 ? true:false;
   }
   
   /// assign data
   /// \param begin begin of values
   /// \param end end of values
   template<class T>
   template<class ITERATOR>
   inline void
   BufferVector<T>::assign(ITERATOR begin,ITERATOR end) {
      this->resize(std::distance(begin,end));
      std::copy(begin, end, data_);
   }
   
   /// reference to last element
   template<class T>
   inline const T &
   BufferVector<T>::back()const{
      return data_[size_-1];
   }

   /// reference to last element
   template<class T>
   inline T &
   BufferVector<T>::back() {
      return data_[size_-1];
   }
   
   /// reference to the first element
   template<class T>
   inline const T &
   BufferVector<T>::front()const{
      return data_[0];
   }
   
   /// reference to the first element
   template<class T>
   inline T &
   BufferVector<T>::front() {
      return data_[0];
   }
}

/// \endcond

#endif	//OPENGM_BUFFER_VECTOR_HXX
