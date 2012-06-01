#pragma once
#ifndef OPENGM_SPARSEMARRAY
#define OPENGM_SPARSEMARRAY

#include <stdexcept>
#include <limits>
#include <algorithm>
#include <vector>
#include <map>
#include <set>
#include <math.h>
#include  <functional>
#include  <iostream>

#include "opengm/utilities/metaprogramming.hxx"
#include "opengm/functions/function_properties_base.hxx"

namespace opengm {
   /// \cond HIDDEN_SYMBOLS
   //Runtime Error Handling
#ifdef NDEBUG
   const bool SPMA_NO_DEBUG = true; ///< General assertion testing disabled.
   const bool SPMA_NO_ARG_TEST = true; ///< Argument testing disabled.
#define SPMA_NO_DEBUG_DEF
#else
#ifdef SPMA_NO_DEBUG_DEF
#undef SPMA_NO_DEBUG_DEF
#endif
   const bool SPMA_NO_DEBUG = false; ///< General assertion testing enabled.
   const bool SPMA_NO_ARG_TEST = false; ///< Argument testing enabled.
#endif
   //Runtime Error with message and line and file information
#define SPMA_RUNTIME_ERROR(msg) \
	{ \
		std::stringstream s; \
		s<<"SparseMarray runtime-error: "<<"\n"<<static_cast<std::string>(msg)<<" in line [ "<< __LINE__ <<" ]" <<" in file [ "<< __FILE__ <<" ] "; \
		throw std::runtime_error (s.str().c_str()); \
	}
   //Runtime Error if assertion=true
#define SPMA_FORCE_ASSERT(assertion,msg) \
	if(!static_cast<bool> ( assertion ) ) \
	{ \
		std::stringstream s; \
		s<<"SparseMarray assertion failed: "<<"\n"<<static_cast<std::string>(msg)<<" in line [ "<< __LINE__ <<" ]" <<" in file [ "<< __FILE__ <<" ] "; \
		throw std::runtime_error (s.str().c_str()); \
	}
   //Runtime Error if assertion=true and NODEBUG is undef
#ifndef SPMA_NO_DEBUG_DEF
#define SPMA_ASSERT(assertion,msg) \
	if(!static_cast<bool> ( assertion ) ) \
	{ \
		std::stringstream s; \
		s<<"SparseMarray assertion failed: "<<"\n"<<static_cast<std::string>(msg)<<" in line [ "<< __LINE__ <<" ]" <<" in file [ "<< __FILE__ <<" ] "; \
		throw std::runtime_error (s.str().c_str()); \
	}
#else
#define SPMA_ASSERT(assertion,msg)
#endif

   //forward declarations:
   //sparsemarray and iterators
   template <class T_Container>
   class SparseMarray;
   template<class T>
   class AccessProxy;
   template<class T_SparseMarray, bool IS_CONST>
   class sparse_array_iterator;
   //meta-flags /structs

   struct DenseAssigment {
   };

   struct SparseAssigmentKey {
   };

   struct SparseAssigmentCoordinateTuple {
   };

   struct CoordinateCoordinateType {
   };

   struct CoordinateKeyType {
   };

   struct CoordinateFundamentalType {
   };

   struct CoordinateIteratorType {
   };

   struct CoordinateVectorType {
   };
   //copy container

   template<class T, class U>
   class CopyAssociativeContainer {
   public:
      inline void copy
      (
         T const & src,
         U & dest
      ) const {
         dest.clear();
         typename T::const_iterator iter = src.begin();
         while (iter != src.end()) {
            dest.insert(std::pair<typename U::key_type, typename U::mapped_type >
               (static_cast<typename U::key_type> (iter->first), static_cast<typename U::mapped_type> (iter->second)));
            ++iter;
         }
      }
   };

   template<class T>
   class CopyAssociativeContainer<T, T> {
      inline void
      copy
      (
         T const & src,
         T & dest
         ) const {
         dest = src;
      }
   };
   //equal with tolerance epsilon for floating point types
   //as float double and long double
   template<class T>
   struct FloatingPointEqualTo;

   template< >
   struct FloatingPointEqualTo<float> {

      bool operator()
      (
         float A,
         float B
         ) {
         return fabs(A - B) <= std::numeric_limits<float>::epsilon();
      };
   };

   template<>
   struct FloatingPointEqualTo<double> {

      bool operator()
      (
         double const A,
         double const B
         ) const {
         return fabs(A - B) <= std::numeric_limits<double>::epsilon();
      };
   };

   template<>
   struct FloatingPointEqualTo<long double> {

      bool operator()
      (
         long double const A,
         long double const B
         ) const {
         return fabs(A - B) <= std::numeric_limits<long double>::epsilon();
      };
   };

   /**
    * @class AccessProxy
    *
    * @brief  Proxy Objekt to differentiate between read and write acces
    *
    *  to differentiate between read and write access we cannot return just a const & in operator() or operator[],so we need this class to to this differentiation
    *
   <b> Usage:</b>
   \code
   #include <sparsemarray.hxx> \endcode
    *
   \code
   //lets say array3d is an existing 3-Dimensional int Sparse array with some size;
   int read=array2d(0,4,5); //array2d(0,4,5) returns a AccessProxy proxy objekt and will perform the read access
   array2d(0,4,5)=1; //array2d(0,4,5) returns a AccessProxy proxy objekt and will perform the write access
   //To make const member calls you can use code like this:
   //lets say array is an  Sparse array with some shape and with a non-Fundamental type Foo() as value_type
   //class Foo has  a NON-const member foo() and a const member bar();
   ( (Foo const &) array(0,4,7)  ).bar() //you will need the bracket bevor the dot.
                                //no object will be inserted at all into the sparsemarray like that
   \endcode
   @author Thorsten Beier
    *
    *   @date 10/23/2010
    */
   template<class T>
   class AccessProxy {
      typedef T SparseMarrayWrapperType;
      typedef typename SparseMarrayWrapperType::key_type key_type;
      typedef typename SparseMarrayWrapperType::const_reference_type const_reference_type;
      typedef typename SparseMarrayWrapperType::value_type value_type;
      typedef typename SparseMarrayWrapperType::const_assigned_assoziative_iterator const_assigned_assoziative_iterator;
   public:

      explicit inline AccessProxy(SparseMarrayWrapperType & opengm, key_type key) : mKey(key), refArray(opengm) {
      };

      inline operator   const_reference_type()const //this is the read function
      {
         if (refArray.mShape.size() == 0) {
            SPMA_ASSERT(mKey == 0, "key==0 is violated, for a scalar array the key/index must be 0 if operator() is used ");
            return refArray.mDefaultValue;
         } else {
            const_assigned_assoziative_iterator iter = refArray.mAssociativeContainer.find(mKey);
            if (iter == refArray.mAssociativeContainer.end()) {
               return refArray.mDefaultValue;
            } else return iter->second;
         }
      };

      inline void operator=(value_type value)const //this is the write fumction
      {
         if (refArray.mShape.size() == 0) {
            SPMA_ASSERT(mKey == 0, "key==0 is violated, for a scalar array the key/index must be 0 if operator() is used ");
            refArray.mDefaultValue = value;
         } else {
            refArray.mAssociativeContainer[mKey] = value;
         }
      };
      template<class U>
      friend std::ostream & operator<<(std::ostream & out, opengm::AccessProxy<U> & ap);
   private:
      key_type mKey;
      SparseMarrayWrapperType & refArray;
   public:

   };
   /// \endcond

   /**
    * @class SparseMarray
    *
    * @brief Sparse, runtime-flexible multi-dimensional array
    *
    * Multi-dimensional sparse arrays with runtime-flexible dimension and shape
    * \ingroup functions
    * \ingroup explicit_functions
    * \ingroup datastructures
    */
   template<class T_AssociativeContainer >
   class SparseMarray : public FunctionBase<
      SparseMarray<T_AssociativeContainer>,
      typename T_AssociativeContainer::mapped_type,
      size_t,
      size_t
   > {
   public:
      friend class opengm::AccessProxy<SparseMarray<T_AssociativeContainer> >;
      template<class >
      friend class opengm::AccessProxy;
   private:
      typedef SparseMarray<T_AssociativeContainer> SparseMarrayWrapperType;
   public:
      /**
       * @typedef associative_container_type
       *
       * @brief  type of T_AssociativeContainer
       */
      typedef T_AssociativeContainer associative_container_type;
      /**
       * @typedef mapped_type
       *
       * @brief mapped type of the  T_AssociativeContainer (= is usualy the same as value_type)
       */
      typedef typename associative_container_type::mapped_type mapped_type;
      /**
       * @typedef key_type
       *
       * @brief key type of the  T_AssociativeContainer
       */
      typedef typename associative_container_type::key_type key_type;
      typedef typename associative_container_type::key_type LabelType;
      typedef typename associative_container_type::key_type IndexType;
      /**
       * @typedef value_type
       *
       * @brief value type of the sparse array
       */
      typedef typename meta::CallTraits< mapped_type >::value_type value_type;
      typedef value_type ValueType;
      /**
       * @typedef reference_type
       *
       * @brief reference type of the sparse array
       */
      typedef typename meta::CallTraits<mapped_type>::reference reference_type;
      /**
       * @typedef const_reference_type
       *
       *
       * @brief const reference type of the sparse array
       */
      typedef typename meta::CallTraits<mapped_type>::const_reference const_reference_type;
      /**
       * @typedef param_type
       *
       * @brief parameter type of the sparse array
       */
      typedef typename meta::CallTraits<mapped_type>::param_type param_type;
      /**
       * @typedef coordinate_type
       *
       * @brief coordinate_type type of the sparse array
       */
      typedef key_type coordinate_type;
      /**
       * @typedef coordinate_tuple
       *
       * @brief coordinate_tuple type of the sparse array
       */
      typedef std::vector<coordinate_type> coordinate_tuple;
      /**
       * @typedef const_iterator
       *
       * @brief const_iterator type of the sparse array
       */
      typedef sparse_array_iterator< SparseMarray<T_AssociativeContainer>, true > const_iterator;
      /**
       * @typedef iterator
       *
       * @brief iterator type of the sparse array
       */
      typedef sparse_array_iterator< SparseMarray<T_AssociativeContainer>, false > iterator;
      /**
       * @typedef assigned_assoziative_iterator
       *
       * @brief assigned_assoziative_iterator is the iterator of the assigned assoziative container (std::map or a hashmap)
       */
      typedef typename associative_container_type::iterator assigned_assoziative_iterator;
      /**
       * @typedef const_assigned_assoziative_iterator
       *
       * @brief const_assigned_assoziative_iterator is the const_iterator of the assignedredundancy assoziative container(std::map or a hashmap)
       */
      typedef typename associative_container_type::const_iterator const_assigned_assoziative_iterator;
      typedef typename meta::If <
      meta::TypeInfo<value_type>::IsFloatingPoint::value,
      opengm::FloatingPointEqualTo<value_type>,
      std::equal_to<value_type>
      >::type value_comparator;
   public:
      //Construktors

      SparseMarray(param_type value) : mAssociativeContainer(), mShape(0), mDefaultValue(value) {
      };

      SparseMarray() : mAssociativeContainer(), mShape(0), mDefaultValue() {
      };
      template<typename InputIterator>
      SparseMarray(InputIterator shapeBegin, InputIterator ShapeEnd, param_type defaultValue);
      template<typename InputIteratorShape, typename InputIteratorData, class T_DefaultValue>
      SparseMarray(InputIteratorShape begin, InputIteratorShape end, InputIteratorData beginData, InputIteratorData endData, T_DefaultValue defaultValue);
      //Copy Constructors:
      template<class T_AC>
      SparseMarray(const SparseMarray<T_AC> & opengm);
      SparseMarray(const SparseMarray & opengm);
      //init
      template<class ShapeIter, class T_In>
      inline void init(ShapeIter begin, ShapeIter end, T_In defaultValue);
      //get shapes and sizes;
      inline void getShape(coordinate_tuple & shape) const;
      inline const coordinate_tuple & getShape() const;
      inline const coordinate_tuple & shape() const;

      inline size_t
      shape(const size_t shapeIndex) const {
         return this->size(shapeIndex);
      };
      inline size_t size(const size_t shapeIndex) const;
      inline size_t size() const;
      //get and set default Value
      template<class T_Value>
      //inline void setDefaultValue(typename meta::CallTraits<T_Value>::param_type defaultValue);
      inline void setDefaultValue(T_Value defaultValue);
      inline typename SparseMarray<T_AssociativeContainer>::const_reference_type getDefaultValue() const;
      //reshape
      template<typename InputIterator>
      void reshape(InputIterator begin, InputIterator end);
      //

      associative_container_type &
      getAssociativeContainer() {
         return mAssociativeContainer;
      };

      associative_container_type &
      getAssociativeContainer()const {
         return mAssociativeContainer;
      };

      associative_container_type &
      getAssociativeContainerReference() {
         return mAssociativeContainer;
      };

      const associative_container_type &
      getAssociativeContainerConstReference()const {
         return mAssociativeContainer;
      };

      template<class T_KeyType>
      inline const_assigned_assoziative_iterator
      find(T_KeyType key)const {
         return mAssociativeContainer.find(static_cast<key_type> (key));
      }

      template<class T_KeyType>
      inline assigned_assoziative_iterator
      find(T_KeyType key) {
         return mAssociativeContainer.find(static_cast<key_type> (key));
      }

      inline const_assigned_assoziative_iterator
      find(key_type key)const {
         return mAssociativeContainer.find(key);
      }

      inline assigned_assoziative_iterator
      find(key_type key) {
         return mAssociativeContainer.find(key);
      }

      size_t
      sizeOfAssociativeContainer()const {
         return mAssociativeContainer.size();
      }
      //coordinate to to key
      inline key_type coordinateToKey(coordinate_type index1) const;
      inline key_type coordinateToKey(coordinate_type index1, coordinate_type index2) const;
      inline key_type coordinateToKey(coordinate_type index1, coordinate_type index2, coordinate_type index3) const;
      inline key_type coordinateToKey(coordinate_type index1, coordinate_type index2, coordinate_type index3, coordinate_type index4) const;
      inline key_type coordinateToKey(coordinate_type index1, coordinate_type index2, coordinate_type index3, coordinate_type index4, coordinate_type index5) const;
      /**
      \cond HIDDEN_SYMBOLS
       */
      inline key_type coordinateToKey(coordinate_type index1, coordinate_type index2, coordinate_type index3, coordinate_type index4, coordinate_type index5,
         coordinate_type index6) const;
      inline key_type coordinateToKey(coordinate_type index1, coordinate_type index2, coordinate_type index3, coordinate_type index4, coordinate_type index5,
         coordinate_type index6, coordinate_type index7) const;
      inline key_type coordinateToKey(coordinate_type index1, coordinate_type index2, coordinate_type index3, coordinate_type index4, coordinate_type index5,
         coordinate_type index6, coordinate_type index7, coordinate_type index8) const;
      inline key_type coordinateToKey(coordinate_type index1, coordinate_type index2, coordinate_type index3, coordinate_type index4, coordinate_type index5,
         coordinate_type index6, coordinate_type index7, coordinate_type index8, coordinate_type index9) const;
      inline key_type coordinateToKey(coordinate_type index1, coordinate_type index2, coordinate_type index3, coordinate_type index4, coordinate_type index5,
         coordinate_type index6, coordinate_type index7, coordinate_type index8, coordinate_type index9, coordinate_type index10) const;

      inline void keyToCoordinate(key_type key, coordinate_tuple & coordinate) const;
      /**
      \endcond
       */
      template<class CoordinateInputType>
      inline key_type coordinateToKey(CoordinateInputType coordinate) const;
      template<class CoordinateInputType>
      inline key_type coordinateToKey(CoordinateInputType coordinate, CoordinateCoordinateType flag) const;
      template<class CoordinateInputType>
      inline key_type coordinateToKey(CoordinateInputType coordinate, CoordinateKeyType flag) const;
      template<class CoordinateInputType>
      inline key_type coordinateToKey(CoordinateInputType coordinate, CoordinateIteratorType flag) const;
      template<class CoordinateInputType>
      inline key_type coordinateToKey(CoordinateInputType coordinate, CoordinateVectorType flag) const;
      template<class CoordinateInputType>
      inline key_type coordinateToKey(CoordinateInputType coordinate, CoordinateFundamentalType flag) const;
      //index to coordinate

      coordinate_tuple
      keyToCoordinate(key_type key) const;
      inline typename SparseMarray<T_AssociativeContainer>::const_reference_type
      operator()(coordinate_type index1) const;
      inline typename SparseMarray<T_AssociativeContainer>::const_reference_type
      operator()(coordinate_type index1, coordinate_type index2) const;
      inline typename SparseMarray<T_AssociativeContainer>::const_reference_type
      operator()(coordinate_type index1, coordinate_type index2, coordinate_type index3) const;
      inline typename SparseMarray<T_AssociativeContainer>::const_reference_type
      operator()(coordinate_type index1, coordinate_type index2, coordinate_type index3, coordinate_type index4) const;
      inline typename SparseMarray<T_AssociativeContainer>::const_reference_type
      operator()(coordinate_type index1, coordinate_type index2, coordinate_type index3, coordinate_type index4, coordinate_type index5) const;
      /**
      \cond HIDDEN_SYMBOLS
       */
      inline typename SparseMarray<T_AssociativeContainer>::const_reference_type
      operator()(coordinate_type index1, coordinate_type index2, coordinate_type index3, coordinate_type index4, coordinate_type index5,
         coordinate_type index6) const;
      inline typename SparseMarray<T_AssociativeContainer>::const_reference_type
      operator()(coordinate_type index1, coordinate_type index2, coordinate_type index3, coordinate_type index4, coordinate_type index5,
         coordinate_type index6, coordinate_type index7) const;
      inline typename SparseMarray<T_AssociativeContainer>::const_reference_type
      operator()(coordinate_type index1, coordinate_type index2, coordinate_type index3, coordinate_type index4, coordinate_type index5,
         coordinate_type index6, coordinate_type index7, coordinate_type index8) const;
      inline typename SparseMarray<T_AssociativeContainer>::const_reference_type
      operator()(coordinate_type index1, coordinate_type index2, coordinate_type index3, coordinate_type index4, coordinate_type index5,
         coordinate_type index6, coordinate_type index7, coordinate_type index8, coordinate_type index9) const;
      inline typename SparseMarray<T_AssociativeContainer>::const_reference_type
      operator()(coordinate_type index1, coordinate_type index2, coordinate_type index3, coordinate_type index4, coordinate_type index5,
         coordinate_type index6, coordinate_type index7, coordinate_type index8, coordinate_type index9, coordinate_type index10) const;
      /**
      \endcond
       */
      template<class CoordinateInputType>
      inline typename SparseMarray<T_AssociativeContainer>::const_reference_type
      operator()(CoordinateInputType coordinate) const;
   private:
      //iterator call
      template<class Iter>
      inline typename SparseMarray<T_AssociativeContainer>::const_reference_type
      operator()(Iter coordinateBegin, opengm::CoordinateIteratorType flag) const;
      //fundamental call
      template<class CoordinateInputType>
      inline typename SparseMarray<T_AssociativeContainer>::const_reference_type
      operator()(CoordinateInputType coordinate, opengm::CoordinateFundamentalType flag) const;
      //keytype call
      inline typename SparseMarray<T_AssociativeContainer>::const_reference_type
      operator()(key_type coordinate, opengm::CoordinateKeyType flag) const;
      //coordinate call
      template<class CoordinateInputType>
      inline typename SparseMarray<T_AssociativeContainer>::const_reference_type
      operator()(coordinate_type coordinate, opengm::CoordinateCoordinateType flag) const;
      //vector call
      template<class CoordinateInputType>
      inline typename SparseMarray<T_AssociativeContainer>::const_reference_type
      operator()(const std::vector<CoordinateInputType> & coordinateBegin, opengm::CoordinateVectorType flag) const;
   public:

      /// \cond HIDDEN_SYMBOLS
      struct CoordinateVectorOfCoordianteType {
      };
      struct CoordinateVectorOfKeyType {
      };
      /// \endcond 

      inline typename SparseMarray<T_AssociativeContainer>::const_reference_type
      const_reference(coordinate_type index1) const;
      inline typename SparseMarray<T_AssociativeContainer>::const_reference_type
      const_reference(coordinate_type index1, coordinate_type index2) const;
      inline typename SparseMarray<T_AssociativeContainer>::const_reference_type
      const_reference(coordinate_type index1, coordinate_type index2, coordinate_type index3) const;
      inline typename SparseMarray<T_AssociativeContainer>::const_reference_type
      const_reference(coordinate_type index1, coordinate_type index2, coordinate_type index3, coordinate_type index4) const;
      inline typename SparseMarray<T_AssociativeContainer>::const_reference_type
      const_reference(coordinate_type index1, coordinate_type index2, coordinate_type index3, coordinate_type index4, coordinate_type index5) const;
      /**
      \cond HIDDEN_SYMBOLS
       */
      inline typename SparseMarray<T_AssociativeContainer>::const_reference_type
      const_reference(coordinate_type index1, coordinate_type index2, coordinate_type index3, coordinate_type index4, coordinate_type index5,
         coordinate_type index6) const;
      inline typename SparseMarray<T_AssociativeContainer>::const_reference_type
      const_reference(coordinate_type index1, coordinate_type index2, coordinate_type index3, coordinate_type index4, coordinate_type index5,
         coordinate_type index6, coordinate_type index7) const;
      inline typename SparseMarray<T_AssociativeContainer>::const_reference_type
      const_reference(coordinate_type index1, coordinate_type index2, coordinate_type index3, coordinate_type index4, coordinate_type index5,
         coordinate_type index6, coordinate_type index7, coordinate_type index8) const;
      inline typename SparseMarray<T_AssociativeContainer>::const_reference_type
      const_reference(coordinate_type index1, coordinate_type index2, coordinate_type index3, coordinate_type index4, coordinate_type index5,
         coordinate_type index6, coordinate_type index7, coordinate_type index8, coordinate_type index9) const;
      inline typename SparseMarray<T_AssociativeContainer>::const_reference_type
      const_reference(coordinate_type index1, coordinate_type index2, coordinate_type index3, coordinate_type index4, coordinate_type index5,
         coordinate_type index6, coordinate_type index7, coordinate_type index8, coordinate_type index9, coordinate_type index10) const;
      template<class Iter>
      inline typename SparseMarray<T_AssociativeContainer>::const_reference_type
      /**
      \endcond
       */
      const_reference(Iter coordinateBegin) const;
      inline typename SparseMarray<T_AssociativeContainer>::reference_type
      reference(coordinate_type index1);
      inline typename SparseMarray<T_AssociativeContainer>::reference_type
      reference(coordinate_type index1, coordinate_type index2);
      inline typename SparseMarray<T_AssociativeContainer>::reference_type
      reference(coordinate_type index1, coordinate_type index2, coordinate_type index3);
      inline typename SparseMarray<T_AssociativeContainer>::reference_type
      reference(coordinate_type index1, coordinate_type index2, coordinate_type index3, coordinate_type index4);
      inline typename SparseMarray<T_AssociativeContainer>::reference_type
      reference(coordinate_type index1, coordinate_type index2, coordinate_type index3, coordinate_type index4, coordinate_type index5);
      /**
      \cond HIDDEN_SYMBOLS
       */
      inline typename SparseMarray<T_AssociativeContainer>::reference_type
      reference(coordinate_type index1, coordinate_type index2, coordinate_type index3, coordinate_type index4, coordinate_type index5,
         coordinate_type index6);
      inline typename SparseMarray<T_AssociativeContainer>::reference_type
      reference(coordinate_type index1, coordinate_type index2, coordinate_type index3, coordinate_type index4, coordinate_type index5,
         coordinate_type index6, coordinate_type index7);
      inline typename SparseMarray<T_AssociativeContainer>::reference_type
      reference(coordinate_type index1, coordinate_type index2, coordinate_type index3, coordinate_type index4, coordinate_type index5,
         coordinate_type index6, coordinate_type index7, coordinate_type index8);
      inline typename SparseMarray<T_AssociativeContainer>::reference_type
      reference(coordinate_type index1, coordinate_type index2, coordinate_type index3, coordinate_type index4, coordinate_type index5,
         coordinate_type index6, coordinate_type index7, coordinate_type index8, coordinate_type index9);
      inline typename SparseMarray<T_AssociativeContainer>::reference_type
      reference(coordinate_type index1, coordinate_type index2, coordinate_type index3, coordinate_type index4, coordinate_type index5,
         coordinate_type index6, coordinate_type index7, coordinate_type index8, coordinate_type index9, coordinate_type index10);
      template<class CoordinateIter>
      /**
      \endcond
       */
      inline typename SparseMarray<T_AssociativeContainer>::reference_type
      reference(CoordinateIter coordinateBegin);
      typename SparseMarray<T_AssociativeContainer>::const_reference_type
      operator[](key_type key) const;
      //assignment
      template<class T_AC>
      SparseMarray & operator=(SparseMarray<T_AC> const & rhs);
      SparseMarray & operator=(const SparseMarray & rhs);
      template<class T_Value>
      SparseMarray & operator=(T_Value const & rhs);
      //arithmetic operators
      SparseMarray & operator++();
      SparseMarray operator++(int);
      SparseMarray & operator--();
      SparseMarray operator--(int);
      template<class T_Value>
      inline SparseMarray & operator+=(T_Value const & value);
      template<class T_Value>
      inline SparseMarray & operator-=(T_Value const & value);
      template<class T_Value>
      inline SparseMarray & operator*=(T_Value const & value);
      template<class T_Value>
      inline SparseMarray & operator/=(T_Value const & value);
      template<class T_Value>
      inline SparseMarray operator+(T_Value const & value)const;
      template<class T_Value>
      inline SparseMarray operator-(T_Value const & value)const;
      template<class T_Value>
      inline SparseMarray operator*(T_Value const & value)const;
      template<class T_Value>
      inline SparseMarray operator/(T_Value const & value)const;
      //to implement

      /// \cond HIDDEN_SYMBOLS

      template<class T_AC, class T_CO, class T_CP, class T_PL>
      SparseMarray & operator+=(SparseMarray<T_AC> const & array);
      template<class T_AC, class T_CO, class T_CP, class T_PL>
      SparseMarray & operator-=(SparseMarray<T_AC> const & array);
      template<class T_AC, class T_CO, class T_CP, class T_PL>
      SparseMarray & operator*=(SparseMarray<T_AC> const & array);
      template<class T_AC, class T_CO, class T_CP, class T_PL>
      SparseMarray & operator/=(SparseMarray<T_AC> const & array);
      /// \endcond

      //clear and delete data
      inline void clear();
      //iterators

      /**
       * @brief  begin
       *  returns a const_iterator pointing to the begin of the sparsemarray ,the iterator is first-Coordinate-Major-Order
       */
      inline const_iterator
      begin()const {
         return const_iterator(this, 0);
      };

      /**
       * @brief  end
       *  returns a const_iterator pointing to the end of the sparsemarray ,the iterator is first-Coordinate-Major-Order
       */
      inline const_iterator
      end()const {
         return const_iterator(this, this->size());
      };

      /**
       * @brief  begin
       *  returns a iterator pointing to the begin of the sparsemarray ,the iterator is first-Coordinate-Major-Order
       */
      inline iterator
      begin() {
         return iterator(this, 0);
      };

      /**
       * @brief  end
       *  returns a iterator pointing to the end of the sparsemarray ,the iterator is first-Coordinate-Major-Order
       */
      inline iterator
      end() {
         return iterator(this, this->size());
      };

      /**
       * @brief  assigned_assoziative_begin
       *  assigned_assoziative_iterator assigned_assoziative_begin (assigned_assoziative_iterator is the iterator of the assigned assoziative container(std::map or a hashmap) )
       */
      inline assigned_assoziative_iterator
      assigned_assoziative_begin() {
         return this->mAssociativeContainer.begin();
      };

      /**
       * @brief  assigned_assoziative_end
       *  assigned_assoziative_iterator assigned_assoziative_end  (assigned_assoziative_iterator is the iterator of the assigned assoziative container(std::map or a hashmap) )
       */
      inline assigned_assoziative_iterator
      assigned_assoziative_end() {
         return this->mAssociativeContainer.end();
      };

      /**
       * @brief  const_assigned_assoziative_begin
       *  const_assigned_assoziative_iterator assigned_assoziative_begin (const_assigned_assoziative_iterator is the const_iterator of the assigned assoziative container(std::map or a hashmap) )
       */
      inline const_assigned_assoziative_iterator
      assigned_assoziative_begin()const {
         return this->mAssociativeContainer.begin();
      };

      /**
       * @brief  assigned_assoziative_end
       *  assigned_assoziative_iterator assigned_assoziative_end (const_assigned_assoziative_iterator is the const_iterator of the assigned assoziative container(std::map or a hashmap) )
       */
      inline const_assigned_assoziative_iterator
      assigned_assoziative_end()const {
         return this->mAssociativeContainer.end();
      };

      inline size_t
      dimension()const {
         return this-> getDimension();
      };

      inline size_t
      getDimension()const {
         return mShape.size();
      };
   private:
      //private methods
      inline void insertAtKey(key_type key, typename SparseMarray<T_AssociativeContainer>::
         const_reference_type value);
      //all constructors must call this method:
      inline void computeShapeStrides();
      inline void computeShapeStrides(std::vector<key_type> & shapeStrides)const;
      //find elements in a range
      inline void findInRange(key_type keyStart, key_type keyEnd, std::vector<key_type> & inRange)const;
      inline bool isInRange(short dim, const coordinate_tuple & rangeStart, const coordinate_tuple & rangeEnd, const coordinate_tuple & coordinate)const;
      //assign dense (check if default value is insert)
      template<typename Iter>
      inline void assign(Iter begin, Iter end, opengm::DenseAssigment flag);
      template<typename Iter>
      inline void assign(Iter begin, Iter end, opengm::SparseAssigmentKey flag);
      template<typename Iter>
      inline void assign(Iter begin, Iter end, opengm::SparseAssigmentCoordinateTuple flag);
      //hash map to store data
      associative_container_type mAssociativeContainer;
      //size of the dimensions
      coordinate_tuple mShape;
      //default value
      value_type mDefaultValue;
      std::vector<key_type> mShapeStrides;

   public:
      /**
       * @typedef AccessProxyType
       *
       * @brief type of the AccessProxy
       */
      typedef AccessProxy<SparseMarrayWrapperType> AccessProxyType;
      //acces elements via coordinates
      AccessProxyType operator()(coordinate_type index1);
      AccessProxyType operator()(coordinate_type index1, coordinate_type index2);
      AccessProxyType operator()(coordinate_type index1, coordinate_type index2, coordinate_type index3);
      AccessProxyType operator()(coordinate_type index1, coordinate_type index2, coordinate_type index3, coordinate_type index4);
      AccessProxyType operator()(coordinate_type index1, coordinate_type index2, coordinate_type index3, coordinate_type index4, coordinate_type index5);
      /**
      \cond HIDDEN_SYMBOLS
       */
      AccessProxyType operator()(coordinate_type index1, coordinate_type index2, coordinate_type index3, coordinate_type index4, coordinate_type index5,
         coordinate_type index6);
      AccessProxyType operator()(coordinate_type index1, coordinate_type index2, coordinate_type index3, coordinate_type index4, coordinate_type index5,
         coordinate_type index6, coordinate_type index7);
      AccessProxyType operator()(coordinate_type index1, coordinate_type index2, coordinate_type index3, coordinate_type index4, coordinate_type index5,
         coordinate_type index6, coordinate_type index7, coordinate_type index8);
      AccessProxyType operator()(coordinate_type index1, coordinate_type index2, coordinate_type index3, coordinate_type index4, coordinate_type index5,
         coordinate_type index6, coordinate_type index7, coordinate_type index8, coordinate_type index9);
      AccessProxyType operator()(coordinate_type index1, coordinate_type index2, coordinate_type index3, coordinate_type index4, coordinate_type index5,
         coordinate_type index6, coordinate_type index7, coordinate_type index8, coordinate_type index9, coordinate_type index10);
      /**
      \endcond
       */
      template<class Iter>
      AccessProxyType operator()(Iter coordinateBegin);
      //iterator call
      template<class Iter>
      inline typename SparseMarray<T_AssociativeContainer>::AccessProxyType
      operator()(Iter coordinateBegin, opengm::CoordinateIteratorType flag);
      //fundamental call
      template<class CoordinateInputType>
      inline typename SparseMarray<T_AssociativeContainer>::AccessProxyType
      operator()(CoordinateInputType coordinate, opengm::CoordinateFundamentalType flag);
      //keytype call
      inline typename SparseMarray<T_AssociativeContainer>::AccessProxyType
      operator()(key_type coordinate, opengm::CoordinateKeyType flag);
      //coordinate call
      template<class CoordinateInputType>
      inline typename SparseMarray<T_AssociativeContainer>::AccessProxyType
      operator()(coordinate_type coordinate, opengm::CoordinateCoordinateType flag);
      //vector call
      template<class CoordinateInputType>
      inline typename SparseMarray<T_AssociativeContainer>::AccessProxyType
      operator()(const std::vector<CoordinateInputType> & coordinateVector, opengm::CoordinateVectorType flag);
      //acces via index
      inline typename SparseMarray<T_AssociativeContainer>::AccessProxyType
      operator[](key_type key);


   };

   /**
    *
   \brief Constructor
    *
   Copy Constructs
    *
   @param sparsemarraycc
   sparsemarray to copy
    *
   @author Thorsten BeierThorsten
    *
   @date 10/23/2010
    */
   template<class T_AssociativeContainer>
   SparseMarray<T_AssociativeContainer>
   ::SparseMarray(const SparseMarray & opengm)
   : mAssociativeContainer(opengm.mAssociativeContainer),
   mShape(opengm.mShape), mDefaultValue(opengm.mDefaultValue),
   mShapeStrides(opengm.mShapeStrides) {
   }

   template<class T_AssociativeContainer>
   template<class T_AC>
   SparseMarray<T_AssociativeContainer>
   ::SparseMarray(const SparseMarray<T_AC> & opengm)
   : mShape(opengm.mShape.begin(),
   opengm.mShape.end()),
   mDefaultValue(static_cast<value_type> (opengm.mDefaultValue)) {
      CopyAssociativeContainer<T_AC, T_AssociativeContainer> copyContainer;
      copyContainer.copy(opengm.mAssociativeContainer, mAssociativeContainer);
      mShapeStrides.assign(opengm.mShapeStrides.begin(), opengm.mShapeStrides.end());
   }

   /**
    *
   \brief Constructor
    *
   Constructs a spare array with a desired shape and  default value,
    *
    *
   @param shapeBegin
   begin of the shape iterator
    *
   @param shapeEnd
   end of the shape iterator
    *
   @param defaultValue
   default Value of the sparse array
    *
    Note that iterator_traits<InputIterator>::value_type must be convertable to size_t.
    *
   <b> Usage:</b>
   \code
   #include <sparsemarray.hxx>
   #include <map>
    \endcode \code
   size_t shape[3]={10,5,15};
   //3-Dimensional sparse array with shape(10,5,15) ,defaultValue=0
   sparsemarray::SparseMarrayWrapper< std::map<size_t,float> ,size_t > array(&shape[0],&shape[3],0);
   \endcode
   @author Thorsten Beier
    *
   @date 10/23/2010
    */
   template<class T_AssociativeContainer>
   template<typename InputIterator>
   SparseMarray<T_AssociativeContainer>
   ::SparseMarray(InputIterator shapeBegin, InputIterator shapeEnd, param_type defaultValue)
   : mAssociativeContainer(),
   mShape(shapeBegin, shapeEnd),
   mDefaultValue(defaultValue),
   mShapeStrides(mShape.size() - 1) {
      this->computeShapeStrides();
   }

   /**
    *
   \brief Constructor
    *
   Constructs a spare array with a desired shape,  default value,
   and fill it with values
    *
    *
   @param shapeBegin
   begin of the shape iterator
    *
   @param shapeEnd
   end of the shape iterator
    *
   @param dataBegin
   begin of the data iterator
    *
   @param dataEnd
   end of the data iterator
    *
   @param defaultValue
   default Value of the sparse array
    *
   <b> Usage:</b>
   <b> Dense Assignment:</b>
   \code
   #include <sparsemarray.hxx> \endcode \code
   size_t shape[2]={2,3};
   float data[2*3]={3,0,
                0,6,
                7,0};
   float defaultValue=0;
   //2-Dimensional sparse array with shape(2,3) defaultValue=0  , and 3,0,0,6,7,0 as data
   //there is a check if data[i]==defaultValue
   sparsemarray::SparseMarrayWrapper< std::map<size_t,float> ,size_t >  array(&shape[0],&shape[2],&data[0],&data[6],defaultValue);
   \endcode
   @author Thorsten Beier
    *
   @date 10/23/2010
    */
   template<class T_AssociativeContainer>
   template<typename InputIteratorShape, typename InputIteratorData, class T_DefaultValue>
   SparseMarray<T_AssociativeContainer>
   ::SparseMarray(InputIteratorShape shapeBegin, InputIteratorShape shapeEnd, InputIteratorData dataBegin, InputIteratorData dataEnd, T_DefaultValue defaultValue)
   : mAssociativeContainer(),
   mShape(shapeBegin, shapeEnd),
   mDefaultValue(static_cast<value_type> (defaultValue)),
   mShapeStrides(mShape.size() - 1) {
      SPMA_ASSERT(std::distance(shapeBegin, shapeEnd) != 0, "std::distance(shapeBegin, shapeEnd) != 0 is voilated,shape must be NOT empty");
      SPMA_ASSERT(std::distance(shapeBegin, shapeEnd) != 0, "std::distance(dataBegin, dataEnd) != 0 is voilated,data must be NOT empty");

      this->computeShapeStrides();
      typedef typename std::iterator_traits<InputIteratorData>::value_type DataType;
      typedef typename std::pair<key_type, mapped_type> PairTypeKey;
      typedef typename std::pair<key_type, coordinate_tuple> PairTypeCoordinateTuple;
      //meta-code to decide which implementation of this->assign has to be called
      // -if DataType == PairTypeKey =>sparse assignment with pair.first=key is called
      // -if DataType == PairTypeCoordinateTuple =>sparse assignment with this->cooridnateToKey(pair.first)=key is called
      // -Else ==> Dense Assigment in first coordinate major order
      typedef typename meta::TypeListGenerator
         <
         meta::SwitchCase<meta::Compare<DataType, PairTypeKey>::value, opengm::SparseAssigmentKey >,
         meta::SwitchCase<meta::Compare<DataType, PairTypeCoordinateTuple>::value, opengm::SparseAssigmentCoordinateTuple >
         >::type CaseList;
      typedef typename meta::Switch<CaseList, opengm::DenseAssigment>::type AssigmentFlag;
      this->assign<InputIteratorData > (dataBegin, dataEnd, AssigmentFlag());
   }
   /**
    *
    \brief Inititialization of an sparse array
    *
    set shape for an array
    *
    *
    @param begin
    begin of the shape of the sparse array
    *
    @param end
    end of the shape of the sparse array
    *
    @param defaultValue
    default Value
    *
    <b> Usage:</b>
    <b> Dense Assignment:</b>
    \code
    #include <sparsemarray.hxx> \endcode \code
   sparsemarray::SparseMarrayWrapper<float> array;++
    std::vector<size_t> shape(3,10);
    float defaultValue=0.0;
    //2-Dimensional sparse array with shape(10,10,10)
    array.init(shape,defaultValue);
    \endcode
   @author Thorsten Beier
    *
    @date 10/23/2010
    */
   //init

   template<class T_AssociativeContainer>
   template<class ShapeIter, class T_In>
   inline void SparseMarray<T_AssociativeContainer>
   ::init(ShapeIter shapeBegin, ShapeIter shapeEnd, T_In defaultValue) {
      if (std::distance(shapeBegin, shapeEnd) != 0) {
         typedef typename std::iterator_traits<ShapeIter>::value_type coordinate_type_in;
         mShape.assign(shapeBegin, shapeEnd);
         mDefaultValue = static_cast<value_type> (defaultValue);
         mShapeStrides.resize(mShape.size() - 1);
         this->computeShapeStrides();
         mAssociativeContainer.clear();
      } else {
         typedef typename std::iterator_traits<ShapeIter>::value_type coordinate_type_in;
         mShape.assign(shapeBegin, shapeEnd);
         mDefaultValue = static_cast<value_type> (defaultValue);
         mShapeStrides.clear();
         mAssociativeContainer.clear();
      }
   }
   //insert elements

   /**
    *
   \brief access Elements via operator()
    *
   access Elements of a 1-D sparse array via operator()
    *
    *
   @param index1
   first index of the sparse array
    *
   @return AccessProxy
   proxy Object to differentiate  between read and write access
    *
   <b> Usage:</b>
   \code
   #include <sparsemarray.hxx> \endcode \code
   //lets say array1d is an existing 1-Dimensional float Sparse array with some size;
   array1d(4)=1.0; //wirte access
   float read=array1d(4); //read access
   \endcode
   @author Thorsten Beier
    *
   @date 10/23/2010
    */
   template<class T_AssociativeContainer>
   typename SparseMarray<T_AssociativeContainer>::AccessProxyType
   SparseMarray<T_AssociativeContainer>::
   operator()(coordinate_type index1) {
      SPMA_ASSERT(index1<this->size(), "index1<this->size() is violated ");
      return AccessProxyType(*this, index1);
   }

   /**
    *
    \brief access Elements via operator()
    *
    access Elements of a 2-D sparse array via operator()
    *
    *
    @param index1
    first index of the sparse array
    *
    @param index2
    second index of the sparse array
    *
    @return AccessProxy
    proxy Object to differentiate  between read and write access
    *
    <b> Usage:</b>
    \code
    #include <sparsemarray.hxx> \endcode \code
    //lets say array2d is an existing 2-Dimensional int Sparse array with some size;
    array2d(0,4)=1; //write access at coordinate 0,4
    int read=array2d(0,4); //read access
    \endcode
    *
    @author Thorsten Beier
    *
    @date 10/23/2010
    */
   template<class T_AssociativeContainer>
   typename SparseMarray<T_AssociativeContainer>::AccessProxyType
   SparseMarray<T_AssociativeContainer>::
   operator()(coordinate_type index1, coordinate_type index2) {
      SPMA_ASSERT(this->getDimension() == 2, "this->getDimension()==2 is violated");
      SPMA_ASSERT((index1 < mShape[0] && index2 < mShape[1]), "index1<mShape[0] &&index2<mShape[1]  is violated");
      return AccessProxyType(*this, this->coordinateToKey(index1, index2));
   }

   /**
    *
    \brief access Elements via operator()
    *
    access Elements of a 3-D sparse array via operator()
    *
    *
    @param index1
    first index of the sparse array
    *
    @param index2
    second index of the sparse array
    *
    @param index3
    third index of the sparse array
    *
    @return AccessProxy
    proxy Object to differentiate  between read and write access
    *
    <b> Usage:</b>
    \code
    #include <sparsemarray.hxx>
    #include <map>
     \endcode \code
    //lets say array3d is an existing 3-Dimensional double Sparse array with some size;
    array3d(10,5,4)=1; //write access at coordinate 10,5,4
    int read=array3d(10,5,4); //read access at coordinate 10,5,4
    \endcode
    *
    @author Thorsten Beier
    *
    @date 10/24/2010
    */
   template<class T_AssociativeContainer>
   typename SparseMarray<T_AssociativeContainer>::AccessProxyType
   SparseMarray<T_AssociativeContainer>::
   operator()(coordinate_type index1, coordinate_type index2, coordinate_type index3) {
      SPMA_ASSERT(this->getDimension() == 3, "this->getDimension()==3 is violated");
      SPMA_ASSERT((index1 < mShape[0] && index2 < mShape[1] && index3 < mShape[2]), "index1<mShape[0] &&index2<mShape[1] && index3<mShape[2]  is violated");
      return AccessProxyType(*this, this->coordinateToKey(index1, index2, index3));
   }

   /**
    *
   \brief access Elements via operator()
    *
   access Elements of a 5-D sparse array via operator()
    *
    *
   @param index1
   first index of the sparse array
    *
   @param index2
   second index of the sparse array
    *
   @param index3
   third index of the sparse array
    *
   @param index4
   fourth index of the sparse array
    *
   @return AccessProxy
   proxy Object to differentiate  between read and write access
    *
   <b> Usage:</b>
   \code
   #include <sparsemarray.hxx> \endcode \code
   //lets say array4d is an existing 4-Dimensional bool Sparse array with some size;
   array4d(10,5,4,1)=true; //write access at coordinate 10,5,4,1
   bool read=array4d(10,5,4,1); //read access at coordinate 10,5,4,1
   \endcode
    *
   @author Thorsten Beier
    *
   @date 10/24/2010
    */
   template<class T_AssociativeContainer>
   typename SparseMarray<T_AssociativeContainer>::AccessProxyType
   SparseMarray<T_AssociativeContainer>::
   operator()(coordinate_type index1, coordinate_type index2, coordinate_type index3, coordinate_type index4) {
      SPMA_ASSERT(this->getDimension() == 4, "this->getDimension()==4 is violated");
      SPMA_ASSERT((index1 < mShape[0] && index2 < mShape[1] && index3 < mShape[2]), "index1<mShape[0] &&index2<mShape[1] && index3<mShape[2]  is violated");
      SPMA_ASSERT(index4 < mShape[3], "index4<mShape[3]  is violated");

      return AccessProxyType(*this, this->coordinateToKey(index1, index2, index3, index4));
   }

   /**
    *
\brief access Elements via operator()
    *
access Elements of a 5-D sparse array via operator()
    *
@param index1
first index of the sparse array
    *
@param index2
second index of the sparse array
    *
@param index3
third index of the sparse array
    *
@param index4
fourth index of the sparse array
    *
@param index5
fifth index of the sparse array
    *
@return AccessProxy
proxy Object to differentiate  between read and write access
    *
<b> Usage:</b>
\code
#include <sparsemarray.hxx> \endcode \code
//lets say array5d is an existing 5-Dimensional float Sparse array with some size;
array5d(10,5,4,1,6)=true; //write access at coordinate 10,5,4,1,6
bool read=array5d(10,5,4,1,6); //read access at coordinate 10,5,4,1,6
\endcode
    *
   @author Thorsten Beier
    *
@date 10/24/2010
    */
   template<class T_AssociativeContainer>
   typename SparseMarray<T_AssociativeContainer>::AccessProxyType
   SparseMarray<T_AssociativeContainer>::
   operator()(coordinate_type index1, coordinate_type index2, coordinate_type index3, coordinate_type index4, coordinate_type index5) {
      SPMA_ASSERT(this->getDimension() == 5, "this->getDimension()==5 is violated");
      SPMA_ASSERT((index1 < mShape[0] && index2 < mShape[1] && index3 < mShape[2]), "index1<mShape[0] &&index2<mShape[1] && index3<mShape[2]  is violated");
      SPMA_ASSERT(index4 < mShape[3], "index4<mShape[3]  is violated");
      SPMA_ASSERT(index5 < mShape[4], "index5<mShape[4]  is violated");
      return AccessProxyType(*this, this->coordinateToKey(index1, index2, index3, index4, index5));
   }

   /**
   \cond HIDDEN_SYMBOLS
    */
   template<class T_AssociativeContainer>
   typename SparseMarray<T_AssociativeContainer>::AccessProxyType
   SparseMarray<T_AssociativeContainer>::
   operator()(coordinate_type index1, coordinate_type index2, coordinate_type index3, coordinate_type index4, coordinate_type index5, coordinate_type index6) {
      SPMA_ASSERT(this->getDimension() == 6, "this->getDimension()==6 is violated");
      SPMA_ASSERT((index1 < mShape[0] && index2 < mShape[1] && index3 < mShape[2]), "index1<mShape[0] &&index2<mShape[1] && index3<mShape[2]  is violated");
      SPMA_ASSERT(index4 < mShape[3], "index4<mShape[3]  is violated");
      SPMA_ASSERT(index5 < mShape[4], "index5<mShape[4]  is violated");
      SPMA_ASSERT(index6 < mShape[5], "index6<mShape[5]  is violated");
      return AccessProxyType(*this, this->coordinateToKey(index1, index2, index3, index4, index5, index6));
   }

   template<class T_AssociativeContainer>
   typename SparseMarray<T_AssociativeContainer>::AccessProxyType
   SparseMarray<T_AssociativeContainer>::
   operator()(coordinate_type index1, coordinate_type index2, coordinate_type index3, coordinate_type index4, coordinate_type index5, coordinate_type index6, coordinate_type index7) {
      SPMA_ASSERT(this->getDimension() == 7, "this->getDimension()==7 is violated");
      SPMA_ASSERT((index1 < mShape[0] && index2 < mShape[1] && index3 < mShape[2]), "index1<mShape[0] &&index2<mShape[1] && index3<mShape[2]  is violated");
      SPMA_ASSERT(index4 < mShape[3], "index4<mShape[3]  is violated");
      SPMA_ASSERT(index5 < mShape[4], "index5<mShape[4]  is violated");
      SPMA_ASSERT(index6 < mShape[5], "index6<mShape[5]  is violated");
      SPMA_ASSERT(index7 < mShape[6], "index7<mShape[6]  is violated");
      return AccessProxyType(*this, this->coordinateToKey(index1, index2, index3, index4, index5, index6, index7));
   }

   template<class T_AssociativeContainer>
   typename SparseMarray<T_AssociativeContainer>::AccessProxyType
   SparseMarray<T_AssociativeContainer>::
   operator()(coordinate_type index1, coordinate_type index2, coordinate_type index3, coordinate_type index4, coordinate_type index5,
   coordinate_type index6, coordinate_type index7, coordinate_type index8) {
      SPMA_ASSERT(this->getDimension() == 8, "this->getDimension()==8 is violated");
      SPMA_ASSERT((index1 < mShape[0] && index2 < mShape[1] && index3 < mShape[2]), "index1<mShape[0] &&index2<mShape[1] && index3<mShape[2]  is violated");
      SPMA_ASSERT(index4 < mShape[3], "index4<mShape[3]  is violated");
      SPMA_ASSERT(index5 < mShape[4], "index5<mShape[4]  is violated");
      SPMA_ASSERT(index6 < mShape[5], "index6<mShape[5]  is violated");
      SPMA_ASSERT(index7 < mShape[6], "index7<mShape[6]  is violated");
      SPMA_ASSERT(index8 < mShape[7], "index8<mShape[7]  is violated");
      return AccessProxyType(*this, this->coordinateToKey(index1, index2, index3, index4, index5, index6, index7, index8));
   }

   template<class T_AssociativeContainer>
   typename SparseMarray<T_AssociativeContainer>::AccessProxyType
   SparseMarray<T_AssociativeContainer>::
   operator()(coordinate_type index1, coordinate_type index2, coordinate_type index3, coordinate_type index4, coordinate_type index5,
   coordinate_type index6, coordinate_type index7, coordinate_type index8, coordinate_type index9) {
      SPMA_ASSERT(this->getDimension() == 9, "this->getDimension()==9 is violated");
      SPMA_ASSERT((index1 < mShape[0] && index2 < mShape[1] && index3 < mShape[2]), "index1<mShape[0] &&index2<mShape[1] && index3<mShape[2]  is violated");
      SPMA_ASSERT(index4 < mShape[3], "index4<mShape[3]  is violated");
      SPMA_ASSERT(index5 < mShape[4], "index5<mShape[4]  is violated");
      SPMA_ASSERT(index6 < mShape[5], "index6<mShape[5]  is violated");
      SPMA_ASSERT(index7 < mShape[6], "index7<mShape[6]  is violated");
      SPMA_ASSERT(index8 < mShape[7], "index8<mShape[7]  is violated");
      SPMA_ASSERT(index9 < mShape[8], "index9<mShape[8]  is violated");
      return AccessProxyType(*this, this->coordinateToKey(index1, index2, index3, index4, index5, index6, index7, index8, index9));
   }

   template<class T_AssociativeContainer>
   typename SparseMarray<T_AssociativeContainer>::AccessProxyType
   SparseMarray<T_AssociativeContainer>::
   operator()(coordinate_type index1, coordinate_type index2, coordinate_type index3, coordinate_type index4, coordinate_type index5,
   coordinate_type index6, coordinate_type index7, coordinate_type index8, coordinate_type index9, coordinate_type index10) {
      SPMA_ASSERT(this->getDimension() == 10, "this->getDimension()==10 is violated");
      SPMA_ASSERT((index1 < mShape[0] && index2 < mShape[1] && index3 < mShape[2]), "index1<mShape[0] &&index2<mShape[1] && index3<mShape[2]  is violated");
      SPMA_ASSERT(index4 < mShape[3], "index4<mShape[3]  is violated");
      SPMA_ASSERT(index5 < mShape[4], "index5<mShape[4]  is violated");
      SPMA_ASSERT(index6 < mShape[5], "index6<mShape[5]  is violated");
      SPMA_ASSERT(index7 < mShape[6], "index7<mShape[6]  is violated");
      SPMA_ASSERT(index8 < mShape[7], "index8<mShape[7]  is violated");
      SPMA_ASSERT(index9 < mShape[8], "index9<mShape[8]  is violated");
      SPMA_ASSERT(index10 < mShape[9], "index10<mShape[9]  is violated");
      return AccessProxyType(*this, this->coordinateToKey(index1, index2, index3, index4, index5, index6, index7, index8, index9, index10));
   }
   /**
   \endcond
    */

   /**
    *
    \brief access Elements via operator()
    *
    access Elements of a N-D sparse array via operator()
    *
    @param coordinateBegin
    begin of coordinate tuple
    *
    @return AccessProxy
    proxy Object to differentiate  between read and write access
    *
    <b> Usage:</b>
    \code
    #include <sparsemarray.hxx>
    #include <vector>
    \endcode
   \code
    //500-D SparseMarray
    std::vector<size_t> highDimShape(500,500);
    std::vector<size_t> highDimCoordinateTuple(500,500);
    for(size_t i=0;i<500;i++)
    {
       highDimCoordinateTuple[i]=i;
    }
    sparsemarray::SparseMarrayWrapper<std::map<size_t,float> ,size_t> highDimArray(highDimShape.begin(),highDimShape.end(),0.0);
    //write to the Coordinate
    highDimArray(highDimCoordinateTuple.begin())=10.0;
    \endcode
    *
    @author Thorsten Beier
    *
    @date 03/02/2011
    */
   template<class T_AssociativeContainer>
   template<class CoordinateInputType>
   typename SparseMarray<T_AssociativeContainer>::AccessProxyType
   SparseMarray<T_AssociativeContainer>::
   operator()(CoordinateInputType coordinate) {
      //OVERLOAD HELPER:
      //generic compiletime switch-cases (is just a typelist witch SwitchCases ,a Switch Case is something like a pair of bool and a type/flag)
      typedef typename meta::TypeListGenerator
         <
         meta::SwitchCase< meta::Compare<key_type, CoordinateInputType >::value, CoordinateKeyType>,
         meta::SwitchCase< meta::Compare<coordinate_type, CoordinateInputType >::value, CoordinateCoordinateType>,
         meta::SwitchCase< meta::IsFundamental<CoordinateInputType >::value, CoordinateFundamentalType >
         >::type CaseList;
      //generic compiletime switch with struct CoordinateIterator as default flag
      typedef typename meta::Switch<CaseList, CoordinateIteratorType>::type CoordinateFlag;
      //call the best version of operator()(coordinate ,someFlag)
      //the operator needs to be implemented for all types of flags
      return this->operator()(coordinate, CoordinateFlag());
   }
   //iterator call

   template<class T_AssociativeContainer>
   template<class Iter>
   inline typename SparseMarray<T_AssociativeContainer>::AccessProxyType
   SparseMarray<T_AssociativeContainer>::
   operator()(Iter coordinateBegin, opengm::CoordinateIteratorType flag) {
      if (this->mShape.size() != 0) {
         return AccessProxyType(*this, this->coordinateToKey(coordinateBegin));
      } else {
         SPMA_ASSERT(*coordinateBegin == 0, "*coordinateBegin==0 violated, for a scalar array the index/key must be 0");
         return AccessProxyType(*this, 0);
      }
   }
   //fundamental call

   template<class T_AssociativeContainer>
   template<class CoordinateInputType>
   inline typename SparseMarray<T_AssociativeContainer>::AccessProxyType
   SparseMarray<T_AssociativeContainer>::
   operator()(CoordinateInputType coordinate, opengm::CoordinateFundamentalType flag) {
      return AccessProxyType(*this, this->coordinateToKey(static_cast<coordinate_type> (coordinate)));
   }
   //keytype call

   template<class T_AssociativeContainer>
   inline typename SparseMarray<T_AssociativeContainer>::AccessProxyType
   SparseMarray<T_AssociativeContainer>::
   operator()(key_type coordinate, opengm::CoordinateKeyType flag) {
      return AccessProxyType(*this, this->coordinateToKey(static_cast<coordinate_type> (coordinate)));
   }
   //coordinate call

   template<class T_AssociativeContainer>
   template<class CoordinateInputType>
   inline typename SparseMarray<T_AssociativeContainer>::AccessProxyType
   SparseMarray<T_AssociativeContainer>::
   operator()(coordinate_type coordinate, opengm::CoordinateCoordinateType flag) {
      return AccessProxyType(*this, this->coordinateToKey(coordinate));
   }
   //vector call

   template<class T_AssociativeContainer>
   template<class CoordinateInputType>
   inline typename SparseMarray<T_AssociativeContainer>::AccessProxyType
   SparseMarray<T_AssociativeContainer>::
   operator()(const std::vector<CoordinateInputType> & coordinateVector, opengm::CoordinateVectorType flag) {
      return AccessProxyType(*this, this->coordinateToKey(coordinateVector.begin()));
   }

   /**
    *
   \brief const access Elements via operator()
    *
   access Elements of a 1-D sparse array via operator()
    *
   @param index1
   first index of the sparse array
    *
   @return typename SparseMarrayWrapper<T_AssociativeContainer>::const_reference_type
    *
   <b> Usage:</b>
   \code
   #include <sparsemarray.hxx> \endcode \code
   //lets say array1d is an existing 1-Dimensional float Sparse array with some size;
   float read=array1d(4); //read access
   \endcode
   @author Thorsten Beier
    *
   @date 10/23/2010
    */
   template<class T_AssociativeContainer>
   typename SparseMarray<T_AssociativeContainer>::const_reference_type
   SparseMarray<T_AssociativeContainer>::
   operator()(coordinate_type index1) const {
      if (mShape.size() == 0) {
         SPMA_ASSERT(index1 == 0, "index1==0  is violated, for a scalar array the index/key must be 0");
         return mDefaultValue;
      } else {
         SPMA_ASSERT(index1<this->size(), "index1<this->size() is violated");
         typename associative_container_type::const_iterator iter = mAssociativeContainer.find(index1);
         if (iter == mAssociativeContainer.end()) {
            return mDefaultValue;
         } else return iter->second;
      }
   }

   /**
    *
    \brief const access Elements via operator()
    *
    access Elements of a 2-D sparse array via operator()
    *
    *
    @param index1
    first index of the sparse array
    *
    @param index2
    second index of the sparse array
    *
    @return const_reference_type
    *
    <b> Usage:</b>
    \code
    #include <sparsemarray.hxx> \endcode \code
    //lets say array2d is an existing 2-Dimensional char Sparse array with some size;
    char read=array2d(4,5); //read access
    \endcode
   @author Thorsten Beier
    *
    @date 10/23/2010
    */
   template<class T_AssociativeContainer>
   typename SparseMarray<T_AssociativeContainer>::const_reference_type
   SparseMarray<T_AssociativeContainer>
   ::operator()(coordinate_type index1, coordinate_type index2) const {
      SPMA_ASSERT(this->getDimension() == 2, "this->getDimension()==2 is violated");
      SPMA_ASSERT((index1 < mShape[0] && index2 < mShape[1]), "index1<mShape[0] &&index2<mShape[1] is violated");
      typename associative_container_type::const_iterator iter =
         mAssociativeContainer.find(this->coordinateToKey(index1, index2));
      if (iter == mAssociativeContainer.end()) {
         return mDefaultValue;
      } else return iter->second;
   }

   /**
    *
   \brief const access Elements via operator()
    *
   access Elements of a 3-D sparse array via operator()
    *
    *
   @param index1
   first index of the sparse array
    *
   @param index2
   second index of the sparse array
    *
   @param index3
   third index of the sparse array
    *
   @return const_reference_type
    *
   <b> Usage:</b>
   \code
   #include <sparsemarray.hxx> \endcode \code
   //lets say array3d is an existing 3-Dimensional double Sparse array with some size;
   double read=array3d(4,5,8); //read access
   \endcode
   @author Thorsten Beier
    *
   @date 10/23/2010
    */
   template<class T_AssociativeContainer>
   typename SparseMarray<T_AssociativeContainer>::const_reference_type
   SparseMarray<T_AssociativeContainer>
   ::operator()(coordinate_type index1, coordinate_type index2, coordinate_type index3) const {
      SPMA_ASSERT(this->getDimension() == 3, "this->getDimension()==3 is violated");
      SPMA_ASSERT((index1 < mShape[0] && index2 < mShape[1] && index3 < mShape[2]), "index1<mShape[0] &&index2<mShape[1] && index3<mShape[2]  is violated");
      typename associative_container_type::const_iterator iter =
         mAssociativeContainer.find(this->coordinateToKey(index1, index2, index3));
      if (iter == mAssociativeContainer.end()) {
         return mDefaultValue;
      } else return iter->second;
   }

   /**
    *
\brief const access Elements via operator()
    *
const access Elements of a 4-D sparse array via operator()
    *
    *
@param index1
first index of the sparse array
    *
@param index2
second index of the sparse array
    *
@param index3
third index of the sparse array
    *
@param index4
fourth index of the sparse array
    *
@return const_reference_type
    *
<b> Usage:</b>
\code
#include <sparsemarray.hxx> \endcode  \code
//lets say array4d is an existing 4-Dimensional int Sparse array with some size;
int read=array4d(4,11,5,8); //read access
\endcode
   @author Thorsten Beier
    *
@date 10/23/2010
    */
   template<class T_AssociativeContainer>
   typename SparseMarray<T_AssociativeContainer>::const_reference_type
   SparseMarray<T_AssociativeContainer>
   ::operator()(coordinate_type index1, coordinate_type index2, coordinate_type index3, coordinate_type index4) const {
      SPMA_ASSERT(this->getDimension() == 4, "this->getDimension()==4 is violated");
      SPMA_ASSERT((index1 < mShape[0] && index2 < mShape[1] && index3 < mShape[2]), "index1<mShape[0] &&index2<mShape[1] && index3<mShape[2]  is violated");
      SPMA_ASSERT(index4 < mShape[3], "index4<mShape[3]  is violated");
      typename associative_container_type::const_iterator iter =
         mAssociativeContainer.find(this->coordinateToKey(index1, index2, index3, index4));
      if (iter == mAssociativeContainer.end()) {
         return mDefaultValue;
      } else return iter->second;
   }

   /**
    *
   \brief const access Elements via operator()
    *
   const access Elements of a 5-D sparse array via operator()
    *
    *
   @param index1
   first index of the sparse array
    *
   @param index2
   second index of the sparse array
    *
   @param index3
   third index of the sparse array
    *
   @param index4
   fourth index of the sparse array
    *
   @param index5
   fifth index of the sparse array
    *
   @return typename SparseMarrayWrapper<T_AssociativeContainer>::const_reference_type
    *
   <b> Usage:</b>
   \code
   #include <sparsemarray.hxx> \endcode  \code
   //lets say array5d is an existing 5-Dimensional int Sparse array with some size;
   int read=array5d(4,11,5,8,1); //read access
   \endcode
   @author Thorsten Beier
    *
   @date 10/23/2010
    */
   template<class T_AssociativeContainer>
   typename SparseMarray<T_AssociativeContainer>::const_reference_type
   SparseMarray<T_AssociativeContainer>
   ::operator()(coordinate_type index1, coordinate_type index2, coordinate_type index3, coordinate_type index4, coordinate_type index5) const {
      SPMA_ASSERT(this->getDimension() == 5, "this->getDimension()==5 is violated");
      SPMA_ASSERT((index1 < mShape[0] && index2 < mShape[1] && index3 < mShape[2]), "index1<mShape[0] &&index2<mShape[1] && index3<mShape[2]  is violated");
      SPMA_ASSERT(index4 < mShape[3], "index4<mShape[3]  is violated");
      SPMA_ASSERT(index5 < mShape[4], "index5<mShape[4]  is violated");
      typename associative_container_type::const_iterator iter =
         mAssociativeContainer.find(this->coordinateToKey(index1, index2, index3, index4, index5));
      if (iter == mAssociativeContainer.end()) {
         return mDefaultValue;
      } else return iter->second;
   }

   /**
   \cond HIDDEN_SYMBOLS
    */
   template<class T_AssociativeContainer>
   typename SparseMarray<T_AssociativeContainer>::const_reference_type
   SparseMarray<T_AssociativeContainer>
   ::operator()(coordinate_type index1, coordinate_type index2, coordinate_type index3, coordinate_type index4, coordinate_type index5,
   coordinate_type index6) const {
      SPMA_ASSERT(this->getDimension() == 6, "this->getDimension()==6 is violated");
      SPMA_ASSERT((index1 < mShape[0] && index2 < mShape[1] && index3 < mShape[2]), "index1<mShape[0] &&index2<mShape[1] && index3<mShape[2]  is violated");
      SPMA_ASSERT(index4 < mShape[3], "index4<mShape[3]  is violated");
      SPMA_ASSERT(index5 < mShape[4], "index5<mShape[4]  is violated");
      SPMA_ASSERT(index6 < mShape[5], "index6<mShape[5]  is violated");
      typename associative_container_type::const_iterator iter =
         mAssociativeContainer.find(this->coordinateToKey(index1, index2, index3, index4, index5, index6));
      if (iter == mAssociativeContainer.end()) {
         return mDefaultValue;
      } else return iter->second;
   }

   template<class T_AssociativeContainer>
   typename SparseMarray<T_AssociativeContainer>::const_reference_type SparseMarray<T_AssociativeContainer>
   ::operator()(coordinate_type index1, coordinate_type index2, coordinate_type index3, coordinate_type index4, coordinate_type index5,
   coordinate_type index6, coordinate_type index7) const {
      SPMA_ASSERT(this->getDimension() == 7, "this->getDimension()==7 is violated");
      SPMA_ASSERT((index1 < mShape[0] && index2 < mShape[1] && index3 < mShape[2]), "index1<mShape[0] &&index2<mShape[1] && index3<mShape[2]  is violated");
      SPMA_ASSERT(index4 < mShape[3], "index4<mShape[3]  is violated");
      SPMA_ASSERT(index5 < mShape[4], "index5<mShape[4]  is violated");
      SPMA_ASSERT(index6 < mShape[5], "index6<mShape[5]  is violated");
      SPMA_ASSERT(index7 < mShape[6], "index7<mShape[6]  is violated");
      typename associative_container_type::const_iterator iter =
         mAssociativeContainer.find(this->coordinateToKey(index1, index2, index3, index4, index5, index6, index7));
      if (iter == mAssociativeContainer.end()) {
         return mDefaultValue;
      } else return iter->second;
   }

   template<class T_AssociativeContainer>
   typename SparseMarray<T_AssociativeContainer>::const_reference_type
   SparseMarray<T_AssociativeContainer>::
   operator()(coordinate_type index1, coordinate_type index2, coordinate_type index3, coordinate_type index4, coordinate_type index5,
   coordinate_type index6, coordinate_type index7, coordinate_type index8) const {
      SPMA_ASSERT(this->getDimension() == 8, "this->getDimension()==8 is violated");
      SPMA_ASSERT((index1 < mShape[0] && index2 < mShape[1] && index3 < mShape[2]), "index1<mShape[0] &&index2<mShape[1] && index3<mShape[2]  is violated");
      SPMA_ASSERT(index4 < mShape[3], "index4<mShape[3]  is violated");
      SPMA_ASSERT(index5 < mShape[4], "index5<mShape[4]  is violated");
      SPMA_ASSERT(index6 < mShape[5], "index6<mShape[5]  is violated");
      SPMA_ASSERT(index7 < mShape[6], "index7<mShape[6]  is violated");
      SPMA_ASSERT(index8 < mShape[7], "index8<mShape[7]  is violated");
      typename associative_container_type::const_iterator iter =
         mAssociativeContainer.find(this->coordinateToKey(index1, index2, index3, index4, index5, index6, index7, index8));
      if (iter == mAssociativeContainer.end()) {
         return mDefaultValue;
      } else return iter->second;
   }

   template<class T_AssociativeContainer>
   typename SparseMarray<T_AssociativeContainer>::const_reference_type
   SparseMarray<T_AssociativeContainer>
   ::operator()(coordinate_type index1, coordinate_type index2, coordinate_type index3, coordinate_type index4, coordinate_type index5,
   coordinate_type index6, coordinate_type index7, coordinate_type index8, coordinate_type index9) const {
      SPMA_ASSERT(this->getDimension() == 9, "this->getDimension()==9 is violated");
      SPMA_ASSERT((index1 < mShape[0] && index2 < mShape[1] && index3 < mShape[2]), "index1<mShape[0] &&index2<mShape[1] && index3<mShape[2]  is violated");
      SPMA_ASSERT(index4 < mShape[3], "index4<mShape[3]  is violated");
      SPMA_ASSERT(index5 < mShape[4], "index5<mShape[4]  is violated");
      SPMA_ASSERT(index6 < mShape[5], "index6<mShape[5]  is violated");
      SPMA_ASSERT(index7 < mShape[6], "index7<mShape[6]  is violated");
      SPMA_ASSERT(index8 < mShape[7], "index8<mShape[7]  is violated");
      SPMA_ASSERT(index9 < mShape[8], "index9<mShape[8]  is violated");
      typename associative_container_type::const_iterator iter =
         mAssociativeContainer.find(this->coordinateToKey(index1, index2, index3, index4, index5, index6, index7, index8, index9));
      if (iter == mAssociativeContainer.end()) {
         return mDefaultValue;
      } else return iter->second;
   }

   template<class T_AssociativeContainer>
   typename SparseMarray<T_AssociativeContainer>::const_reference_type
   SparseMarray<T_AssociativeContainer>
   ::operator()(coordinate_type index1, coordinate_type index2, coordinate_type index3, coordinate_type index4, coordinate_type index5,
   coordinate_type index6, coordinate_type index7, coordinate_type index8, coordinate_type index9, coordinate_type index10) const {
      SPMA_ASSERT(this->getDimension() == 10, "this->getDimension()==10 is violated");
      SPMA_ASSERT((index1 < mShape[0] && index2 < mShape[1] && index3 < mShape[2]), "index1<mShape[0] &&index2<mShape[1] && index3<mShape[2]  is violated");
      SPMA_ASSERT(index4 < mShape[3], "index4<mShape[3]  is violated");
      SPMA_ASSERT(index5 < mShape[4], "index5<mShape[4]  is violated");
      SPMA_ASSERT(index6 < mShape[5], "index6<mShape[5]  is violated");
      SPMA_ASSERT(index7 < mShape[6], "index7<mShape[6]  is violated");
      SPMA_ASSERT(index8 < mShape[7], "index8<mShape[7]  is violated");
      SPMA_ASSERT(index9 < mShape[8], "index9<mShape[8]  is violated");
      SPMA_ASSERT(index10 < mShape[9], "index10<mShape[9]  is violated");
      typename associative_container_type::const_iterator iter =
         mAssociativeContainer.find(this->coordinateToKey(index1, index2, index3, index4, index5, index6, index7, index8, index9, index10));
      if (iter == mAssociativeContainer.end()) {
         return mDefaultValue;
      } else return iter->second;
   }
   /**
   \endcond
    */

   /**
    *
    \brief access Elements via operator()
    *
    const access Elements of a N-D sparse array via operator()
    *
    @param coordinateBegin
    begin of coordinate tuple
    *
    @return AccessProxy
    proxy Object to differentiate  between read and write access
    *
    <b> Usage:</b>
    \code
    #include <sparsemarray.hxx>
    #include <vector>
    \endcode
   \code
    //500-D SparseMarray
    std::vector<size_t> highDimShape(500,500);
    std::vector<size_t> highDimCoordinateTuple(500,500);
    for(size_t i=0;i<500;i++)
    {
       highDimCoordinateTuple[i]=i;
    }
   //construct 500 D Array
    sparsemarray::SparseMarrayWrapper<std::map<size_t,float> ,size_t> highDimArray(highDimShape.begin(),highDimShape.end(),0.0);
    //read to the Coordinate
    float foo=highDimArray(highDimCoordinateTuple.begin());
    \endcode
    *
    @author Thorsten Beier
    *
    @date 03/02/2011
    */
   template<class T_AssociativeContainer>
   template<class CoordinateInputType>
   typename SparseMarray<T_AssociativeContainer>::const_reference_type
   SparseMarray<T_AssociativeContainer>::
   operator()(CoordinateInputType coordinate) const {
      //OVERLOAD HELPER:
      //generic compiletime switch-cases
      typedef typename meta::TypeListGenerator
         <
         meta::SwitchCase< meta::Compare<key_type, CoordinateInputType>::value, CoordinateKeyType>,
         meta::SwitchCase< meta::Compare<coordinate_type, CoordinateInputType>::value, CoordinateCoordinateType>,
         meta::SwitchCase< meta::IsFundamental<CoordinateInputType>::value, CoordinateFundamentalType >
         >::type CaseList;
      //generic compiletime switch with struct CoordinateIterator as default flag
      typedef typename meta::Switch<CaseList, CoordinateIteratorType>::type CoordinateFlag;
      //call the best version of operator()(coordinate ,someFlag)
      return this->operator()(coordinate, CoordinateFlag());
   }
   //iterator call

   template<class T_AssociativeContainer>
   template<class Iter>
   inline typename SparseMarray<T_AssociativeContainer>::const_reference_type
   SparseMarray<T_AssociativeContainer>::
   operator()(Iter coordinateBegin, opengm::CoordinateIteratorType flag) const {
      if (this->mShape.size() != 0) {
         typename associative_container_type::const_iterator iter = mAssociativeContainer.find(this->coordinateToKey(coordinateBegin));
         if (iter == mAssociativeContainer.end()) {
            return mDefaultValue;
         } else return iter->second;
      } else {
         SPMA_ASSERT(*coordinateBegin == 0, "*coordinateBegin==0 is violated, for a scalar array the key/index must be 0");
         return this->mDefaultValue;
      }
   }
   //fundamental call

   template<class T_AssociativeContainer>
   template<class CoordinateInputType>
   inline typename SparseMarray<T_AssociativeContainer>::const_reference_type
   SparseMarray<T_AssociativeContainer>::
   operator()(CoordinateInputType coordinate, opengm::CoordinateFundamentalType flag) const {
      SPMA_ASSERT(static_cast<coordinate_type> (coordinate)<this->size(), "static_cast<coordinate_type>(coordinate)<this->size() is violated");
      typename associative_container_type::const_iterator iter = mAssociativeContainer.find(this->coordinateToKey(static_cast<coordinate_type> (coordinate)));
      if (iter == mAssociativeContainer.end()) {
         return mDefaultValue;
      } else return iter->second;
   }
   //keytype call

   template<class T_AssociativeContainer>
   inline typename SparseMarray<T_AssociativeContainer>::const_reference_type
   SparseMarray<T_AssociativeContainer>::
   operator()(key_type coordinate, opengm::CoordinateKeyType flag) const {
      SPMA_ASSERT(static_cast<coordinate_type> (coordinate)<this->size(), "static_cast<coordinate_type>(coordinate)<this->size() is violated");
      typename associative_container_type::const_iterator iter = mAssociativeContainer.find(this->coordinateToKey(static_cast<coordinate_type> (coordinate)));
      if (iter == mAssociativeContainer.end()) {
         return mDefaultValue;
      } else return iter->second;
   }
   //coordinate call

   template<class T_AssociativeContainer>
   template<class CoordinateInputType>
   inline typename SparseMarray<T_AssociativeContainer>::const_reference_type
   SparseMarray<T_AssociativeContainer>::
   operator()(coordinate_type coordinate, opengm::CoordinateCoordinateType flag) const {
      SPMA_ASSERT(coordinate_type(coordinate)<this->size(), "coordinate<this->size() is violated");
      typename associative_container_type::const_iterator iter = mAssociativeContainer.find(this->coordinateToKey(coordinate));
      if (iter == mAssociativeContainer.end()) {
         return mDefaultValue;
      } else return iter->second;
   }
   //vector call

   template<class T_AssociativeContainer>
   template<class CoordinateInputType>
   inline typename SparseMarray<T_AssociativeContainer>::const_reference_type
   SparseMarray<T_AssociativeContainer>::
   operator()(const std::vector<CoordinateInputType> & coordinateVector, opengm::CoordinateVectorType flag) const {
      typename associative_container_type::const_iterator iter = mAssociativeContainer.find(this->coordinateToKey(coordinateVector.begin()));
      if (iter == mAssociativeContainer.end()) {
         return mDefaultValue;
      } else return iter->second;
   }

   /**
    *
    \brief  access Elements at index via operator[]
    *
     access Elements at key of an sparse array via operator[]
    *
    *
    @param key
    scalar integral key
    *
    @return AccessProxy
    proxy Object to differentiate  between read and write access
    *
    <b> Usage:</b>
    \code
    #include <sparsemarray.hxx> \endcode  \code
    //lets say sparseArray is an existing coordinate_type Sparse  array with shape(2,3,4);
    for(coordinate_type i=0;i<2*3*4;i++)
    {
       sparseArray[i]=i; //write access
    }
    for(coordinate_type i=0;i<2*3*4;i++)
    {
       coordinate_type read=sparseArray[i]; //read access
    }
   \endcode
   @author Thorsten Beier
    *
    @date 10/23/2010
    */
   template<class T_AssociativeContainer>
   typename SparseMarray<T_AssociativeContainer>::AccessProxyType
   SparseMarray<T_AssociativeContainer>::operator[](key_type key) {
      return AccessProxyType(*this, key);
   }

   /**
    *
   \brief const access Elements at index via operator[]
    *
   const access Elements at key of an sparse array via operator[]
    *
    *
   @param key
   scalar integral key
    *
   @return typename SparseMarrayWrapper<T_AssociativeContainer>::const_reference_type
    *
   <b> Usage:</b>
   \code
   #include <sparsemarray.hxx> \endcode  \code
   //lets say sparseArray is an existing  Sparse array with shape(2,3,4);
   for(coordinate_type i=0;i<2*3*4;i++)
   {
      int read=sparseArray[i]; //read access
   }
   \endcode
   @author Thorsten Beier
    *
   @date 10/23/2010
    */
   template<class T_AssociativeContainer>
   typename SparseMarray<T_AssociativeContainer>::const_reference_type
   SparseMarray<T_AssociativeContainer>::operator[](key_type key) const {
      typename associative_container_type::const_iterator iter = mAssociativeContainer.find(key);
      if (iter == mAssociativeContainer.end()) {
         return mDefaultValue;
      } else return iter->second;
   }

   template<class T_AssociativeContainer>
   inline void SparseMarray<T_AssociativeContainer>
   ::computeShapeStrides() {
         /**
            \cond HIDDEN_SYMBOLS
          */
         //unrolled loop?!?
         //TODO measure runtimes
#define SPARSEMARRAY_COMPUTE_SHAPE_STRIDE_LOOP(N)\
            for(size_t i=0;i<N-1;i++)\
            {\
                key_type factor=mShape[i];\
                for(size_t j=0;j<i;j++) {factor*=static_cast<key_type>(mShape[j]);}mShapeStrides[i]=factor;\
            }\
           /**
                \endcond
            */
         {
         size_t dim = this->getDimension();
         switch (dim) {
            case 1:SPARSEMARRAY_COMPUTE_SHAPE_STRIDE_LOOP(1);
               break;
            case 2:SPARSEMARRAY_COMPUTE_SHAPE_STRIDE_LOOP(2);
               break;
            case 3:SPARSEMARRAY_COMPUTE_SHAPE_STRIDE_LOOP(3);
               break;
            case 4:SPARSEMARRAY_COMPUTE_SHAPE_STRIDE_LOOP(4);
               break;
            case 5:SPARSEMARRAY_COMPUTE_SHAPE_STRIDE_LOOP(5);
               break;
            case 6:SPARSEMARRAY_COMPUTE_SHAPE_STRIDE_LOOP(6);
               break;
            case 7:SPARSEMARRAY_COMPUTE_SHAPE_STRIDE_LOOP(7);
               break;
            case 8:SPARSEMARRAY_COMPUTE_SHAPE_STRIDE_LOOP(8);
               break;
            case 9:SPARSEMARRAY_COMPUTE_SHAPE_STRIDE_LOOP(9);
               break;
            case 10:SPARSEMARRAY_COMPUTE_SHAPE_STRIDE_LOOP(10);
               break;
            default:SPARSEMARRAY_COMPUTE_SHAPE_STRIDE_LOOP(dim);
         }
      }
   }

   template<class T_AssociativeContainer>
   inline typename SparseMarray<T_AssociativeContainer>::const_reference_type
   SparseMarray<T_AssociativeContainer>
   ::const_reference(coordinate_type index1) const {
      if (mShape.size() == 0) {
         SPMA_ASSERT(index1 == 0, "index1==0 is violated, for a scalar array the key/index must be 0");
         return mDefaultValue;
      } else {
         SPMA_ASSERT(index1<this->size(), "index1==0 is violated, for a scalar array the key/index must be 0");
         return this->operator()(index1);
      }
   }

   template<class T_AssociativeContainer>
   inline typename SparseMarray<T_AssociativeContainer>::const_reference_type
   SparseMarray<T_AssociativeContainer>
   ::const_reference(coordinate_type index1, coordinate_type index2) const {
      return this->operator()(index1, index2);
   }

   template<class T_AssociativeContainer>
   inline typename SparseMarray<T_AssociativeContainer>::const_reference_type
   SparseMarray<T_AssociativeContainer>
   ::const_reference(coordinate_type index1, coordinate_type index2, coordinate_type index3) const {
      return this->operator()(index1, index2, index3);
   }

   template<class T_AssociativeContainer>
   inline typename SparseMarray<T_AssociativeContainer>::const_reference_type\
 SparseMarray<T_AssociativeContainer>
   ::const_reference(coordinate_type index1, coordinate_type index2, coordinate_type index3, coordinate_type index4) const {
      return this->operator()(index1, index2, index3, index4);
   }

   template<class T_AssociativeContainer>
   inline typename SparseMarray<T_AssociativeContainer>::const_reference_type
   SparseMarray<T_AssociativeContainer>
   ::const_reference(coordinate_type index1, coordinate_type index2, coordinate_type index3, coordinate_type index4, coordinate_type index5) const {
      return this->operator()(index1, index2, index3, index4, index5);
   }

   /**
          \cond HIDDEN_SYMBOLS
    */
   template<class T_AssociativeContainer>
   inline typename SparseMarray<T_AssociativeContainer>::const_reference_type
   SparseMarray<T_AssociativeContainer>
   ::const_reference(coordinate_type index1, coordinate_type index2, coordinate_type index3, coordinate_type index4, coordinate_type index5,
   coordinate_type index6) const {
      return this->operator()(index1, index2, index3, index4, index5, index6);
   }

   template<class T_AssociativeContainer>
   inline typename SparseMarray<T_AssociativeContainer>::const_reference_type
   SparseMarray<T_AssociativeContainer>
   ::const_reference(coordinate_type index1, coordinate_type index2, coordinate_type index3, coordinate_type index4, coordinate_type index5,
   coordinate_type index6, coordinate_type index7) const {
      return this->operator()(index1, index2, index3, index4, index5, index6, index7);
   }

   template<class T_AssociativeContainer>
   inline typename SparseMarray<T_AssociativeContainer>::const_reference_type
   SparseMarray<T_AssociativeContainer>
   ::const_reference(coordinate_type index1, coordinate_type index2, coordinate_type index3, coordinate_type index4, coordinate_type index5,
   coordinate_type index6, coordinate_type index7, coordinate_type index8) const {
      return this->operator()(index1, index2, index3, index4, index5, index6, index7, index8);
   }

   template<class T_AssociativeContainer>
   inline typename SparseMarray<T_AssociativeContainer>::const_reference_type
   SparseMarray<T_AssociativeContainer>
   ::const_reference(coordinate_type index1, coordinate_type index2, coordinate_type index3, coordinate_type index4, coordinate_type index5,
   coordinate_type index6, coordinate_type index7, coordinate_type index8, coordinate_type index9) const {
      return this->operator()(index1, index2, index3, index4, index5, index6, index7, index8, index9);
   }

   template<class T_AssociativeContainer>
   inline typename SparseMarray<T_AssociativeContainer>::const_reference_type
   SparseMarray<T_AssociativeContainer>
   ::const_reference(coordinate_type index1, coordinate_type index2, coordinate_type index3, coordinate_type index4, coordinate_type index5,
   coordinate_type index6, coordinate_type index7, coordinate_type index8, coordinate_type index9, coordinate_type index10) const {
      return this->operator()(index1, index2, index3, index4, index5, index6, index7, index8, index9, index10);
   }

   /**
         \endcond
    */
   template<class T_AssociativeContainer>
   template<class CoordinateInputType>
   inline typename SparseMarray<T_AssociativeContainer>::const_reference_type
   SparseMarray<T_AssociativeContainer>
   ::const_reference(CoordinateInputType coordinate) const {
      return this->operator()(coordinate);
   }

   template<class T_AssociativeContainer>
   inline typename SparseMarray<T_AssociativeContainer>::reference_type
   SparseMarray<T_AssociativeContainer>
   ::reference(coordinate_type index1) {
      if (mShape.size() == 0) {
         SPMA_ASSERT(index1 == 0, "index1==0 is violated");
         return mDefaultValue;
      }
      {
         SPMA_ASSERT(index1<this->size(), "index1<this->size() is violated");
         key_type key = this->coordinateToKey(index1);
         assigned_assoziative_iterator iter = mAssociativeContainer.find(key);
         if (iter == mAssociativeContainer.end()) {
            mAssociativeContainer[key] = mDefaultValue;
            return mAssociativeContainer[key];
         }
         return iter->second;
      }
   }

   template<class T_AssociativeContainer>
   inline typename SparseMarray<T_AssociativeContainer>::reference_type
   SparseMarray<T_AssociativeContainer>
   ::reference(coordinate_type index1, coordinate_type index2) {
      SPMA_ASSERT(this->getDimension() == 2, "this->getDimension()==2 is violated");
      SPMA_ASSERT((index1 < mShape[0] && index2 < mShape[1]), "index1<mShape[0] &&index2<mShape[1]  is violated");
      key_type key = this->coordinateToKey(index1, index2);
      assigned_assoziative_iterator iter = mAssociativeContainer.find(key);
      if (iter == mAssociativeContainer.end()) {
         mAssociativeContainer[key] = mDefaultValue;
         return mAssociativeContainer[key];
      }
      return iter->second;
   }

   template<class T_AssociativeContainer>
   inline typename SparseMarray<T_AssociativeContainer>::reference_type
   SparseMarray<T_AssociativeContainer>
   ::reference(coordinate_type index1, coordinate_type index2, coordinate_type index3) {
      SPMA_ASSERT(this->getDimension() == 3, "this->getDimension()==3 is violated");
      SPMA_ASSERT((index1 < mShape[0] && index2 < mShape[1] && index3 < mShape[2]), "index1<mShape[0] &&index2<mShape[1] && index3<mShape[2]  is violated");
      key_type key = this->coordinateToKey(index1, index2, index3);
      assigned_assoziative_iterator iter = mAssociativeContainer.find(key);
      if (iter == mAssociativeContainer.end()) {
         mAssociativeContainer[key] = mDefaultValue;
         return mAssociativeContainer[key];
      }
      return iter->second;
   }

   template<class T_AssociativeContainer>
   inline typename SparseMarray<T_AssociativeContainer>::reference_type
   SparseMarray<T_AssociativeContainer>
   ::reference(coordinate_type index1, coordinate_type index2, coordinate_type index3, coordinate_type index4) {
      SPMA_ASSERT(this->getDimension() == 4, "this->getDimension()==4 is violated");
      SPMA_ASSERT((index1 < mShape[0] && index2 < mShape[1] && index3 < mShape[2]), "index1<mShape[0] &&index2<mShape[1] && index3<mShape[2]  is violated");
      SPMA_ASSERT(index4 < mShape[3], "index4<mShape[3]  is violated");
      key_type key = this->coordinateToKey(index1, index2, index3, index4);
      assigned_assoziative_iterator iter = mAssociativeContainer.find(key);
      if (iter == mAssociativeContainer.end()) {
         mAssociativeContainer[key] = mDefaultValue;
         return mAssociativeContainer[key];
      }
      return iter->second;
   }

   template<class T_AssociativeContainer>
   inline typename SparseMarray<T_AssociativeContainer>::reference_type
   SparseMarray<T_AssociativeContainer>
   ::reference(coordinate_type index1, coordinate_type index2, coordinate_type index3, coordinate_type index4, coordinate_type index5) {
      SPMA_ASSERT(this->getDimension() == 5, "this->getDimension()==5 is violated");
      SPMA_ASSERT((index1 < mShape[0] && index2 < mShape[1] && index3 < mShape[2]), "index1<mShape[0] &&index2<mShape[1] && index3<mShape[2]  is violated");
      SPMA_ASSERT(index4 < mShape[3], "index4<mShape[3]  is violated");
      SPMA_ASSERT(index5 < mShape[4], "index5<mShape[4]  is violated");
      key_type key = this->coordinateToKey(index1, index2, index3, index4, index5);
      assigned_assoziative_iterator iter = mAssociativeContainer.find(key);
      if (iter == mAssociativeContainer.end()) {
         mAssociativeContainer[key] = mDefaultValue;
         return mAssociativeContainer[key];
      }
      return iter->second;
   }

   /**
    \cond HIDDEN_SYMBOLS
    */
   template<class T_AssociativeContainer>
   inline typename SparseMarray<T_AssociativeContainer>::reference_type
   SparseMarray<T_AssociativeContainer>
   ::reference(coordinate_type index1, coordinate_type index2, coordinate_type index3, coordinate_type index4, coordinate_type index5,
   coordinate_type index6) {
      SPMA_ASSERT(this->getDimension() == 6, "this->getDimension()==6 is violated");
      SPMA_ASSERT((index1 < mShape[0] && index2 < mShape[1] && index3 < mShape[2]), "index1<mShape[0] &&index2<mShape[1] && index3<mShape[2]  is violated");
      SPMA_ASSERT(index4 < mShape[3], "index4<mShape[3]  is violated");
      SPMA_ASSERT(index5 < mShape[4], "index5<mShape[4]  is violated");
      SPMA_ASSERT(index6 < mShape[5], "index6<mShape[5]  is violated");
      key_type key = this->coordinateToKey(index1, index2, index3, index4, index5, index6);
      assigned_assoziative_iterator iter = mAssociativeContainer.find(key);
      if (iter == mAssociativeContainer.end()) {
         mAssociativeContainer[key] = mDefaultValue;
         return mAssociativeContainer[key];
      }
      return iter->second;
   }

   template<class T_AssociativeContainer>
   inline typename SparseMarray<T_AssociativeContainer>::reference_type
   SparseMarray<T_AssociativeContainer>
   ::reference(coordinate_type index1, coordinate_type index2, coordinate_type index3, coordinate_type index4, coordinate_type index5,
   coordinate_type index6, coordinate_type index7) {
      SPMA_ASSERT(this->getDimension() == 7, "this->getDimension()==7 is violated");
      SPMA_ASSERT((index1 < mShape[0] && index2 < mShape[1] && index3 < mShape[2]), "index1<mShape[0] &&index2<mShape[1] && index3<mShape[2]  is violated");
      SPMA_ASSERT(index4 < mShape[3], "index4<mShape[3]  is violated");
      SPMA_ASSERT(index5 < mShape[4], "index5<mShape[4]  is violated");
      SPMA_ASSERT(index6 < mShape[5], "index6<mShape[5]  is violated");
      SPMA_ASSERT(index7 < mShape[6], "index7<mShape[6]  is violated");
      key_type key = this->coordinateToKey(index1, index2, index3, index4, index5, index6, index7);
      assigned_assoziative_iterator iter = mAssociativeContainer.find(key);
      if (iter == mAssociativeContainer.end()) {
         mAssociativeContainer[key] = mDefaultValue;
         return mAssociativeContainer[key];
      }
      return iter->second;
   }

   template<class T_AssociativeContainer>
   inline typename SparseMarray<T_AssociativeContainer>::reference_type
   SparseMarray<T_AssociativeContainer>
   ::reference(coordinate_type index1, coordinate_type index2, coordinate_type index3, coordinate_type index4, coordinate_type index5,
   coordinate_type index6, coordinate_type index7, coordinate_type index8) {
      SPMA_ASSERT(this->getDimension() == 8, "this->getDimension()==8 is violated");
      SPMA_ASSERT((index1 < mShape[0] && index2 < mShape[1] && index3 < mShape[2]), "index1<mShape[0] &&index2<mShape[1] && index3<mShape[2]  is violated");
      SPMA_ASSERT(index4 < mShape[3], "index4<mShape[3]  is violated");
      SPMA_ASSERT(index5 < mShape[4], "index5<mShape[4]  is violated");
      SPMA_ASSERT(index6 < mShape[5], "index6<mShape[5]  is violated");
      SPMA_ASSERT(index7 < mShape[6], "index7<mShape[6]  is violated");
      SPMA_ASSERT(index8 < mShape[7], "index8<mShape[7]  is violated");
      key_type key = this->coordinateToKey(index1, index2, index3, index4, index5, index6, index7, index8);
      assigned_assoziative_iterator iter = mAssociativeContainer.find(key);
      if (iter == mAssociativeContainer.end()) {
         mAssociativeContainer[key] = mDefaultValue;
         return mAssociativeContainer[key];
      }
      return iter->second;
   }

   template<class T_AssociativeContainer>
   inline typename SparseMarray<T_AssociativeContainer>::reference_type
   SparseMarray<T_AssociativeContainer>
   ::reference(coordinate_type index1, coordinate_type index2, coordinate_type index3, coordinate_type index4, coordinate_type index5,
   coordinate_type index6, coordinate_type index7, coordinate_type index8, coordinate_type index9) {
      SPMA_ASSERT(this->getDimension() == 9, "this->getDimension()==9 is violated");
      SPMA_ASSERT((index1 < mShape[0] && index2 < mShape[1] && index3 < mShape[2]), "index1<mShape[0] &&index2<mShape[1] && index3<mShape[2]  is violated");
      SPMA_ASSERT(index4 < mShape[3], "index4<mShape[3]  is violated");
      SPMA_ASSERT(index5 < mShape[4], "index5<mShape[4]  is violated");
      SPMA_ASSERT(index6 < mShape[5], "index6<mShape[5]  is violated");
      SPMA_ASSERT(index7 < mShape[6], "index7<mShape[6]  is violated");
      SPMA_ASSERT(index8 < mShape[7], "index8<mShape[7]  is violated");
      SPMA_ASSERT(index9 < mShape[8], "index9<mShape[8]  is violated");
      key_type key = this->coordinateToKey(index1, index2, index3, index4, index5, index6, index7, index8, index9);
      assigned_assoziative_iterator iter = mAssociativeContainer.find(key);
      if (iter == mAssociativeContainer.end()) {
         mAssociativeContainer[key] = mDefaultValue;
         return mAssociativeContainer[key];
      }
      return iter->second;
   }

   template<class T_AssociativeContainer>
   inline typename SparseMarray<T_AssociativeContainer>::reference_type
   SparseMarray<T_AssociativeContainer>
   ::reference(coordinate_type index1, coordinate_type index2, coordinate_type index3, coordinate_type index4, coordinate_type index5,
   coordinate_type index6, coordinate_type index7, coordinate_type index8, coordinate_type index9, coordinate_type index10) {
      SPMA_ASSERT(this->getDimension() == 10, "this->getDimension()==10 is violated");
      SPMA_ASSERT((index1 < mShape[0] && index2 < mShape[1] && index3 < mShape[2]), "index1<mShape[0] &&index2<mShape[1] && index3<mShape[2]  is violated");
      SPMA_ASSERT(index4 < mShape[3], "index4<mShape[3]  is violated");
      SPMA_ASSERT(index5 < mShape[4], "index5<mShape[4]  is violated");
      SPMA_ASSERT(index6 < mShape[5], "index6<mShape[5]  is violated");
      SPMA_ASSERT(index7 < mShape[6], "index7<mShape[6]  is violated");
      SPMA_ASSERT(index8 < mShape[7], "index8<mShape[7]  is violated");
      SPMA_ASSERT(index9 < mShape[8], "index9<mShape[8]  is violated");
      SPMA_ASSERT(index10 < mShape[9], "index10<mShape[9]  is violated");
      key_type key = this->coordinateToKey(index1, index2, index3, index4, index5, index6, index7, index8, index9, index10);
      assigned_assoziative_iterator iter = mAssociativeContainer.find(key);
      if (iter == mAssociativeContainer.end()) {
         mAssociativeContainer[key] = mDefaultValue;
         return mAssociativeContainer[key];
      }
      return iter->second;
   }

   /**
   \endcond
    */
   template<class T_AssociativeContainer>
   template<class CoordinateIter>
   inline typename SparseMarray<T_AssociativeContainer>::reference_type
   SparseMarray<T_AssociativeContainer>
   ::reference(CoordinateIter coordinateBegin) {
      if (this->mShape.size() != 0) {
         const key_type key = this->coordinateToKey(coordinateBegin);
         assigned_assoziative_iterator iter = mAssociativeContainer.find(key);
         if (iter == mAssociativeContainer.end()) {
            mAssociativeContainer[key] = mDefaultValue;
            return mAssociativeContainer[key];
         }
         return iter->second;
      } else {
         const key_type key = 0;
         assigned_assoziative_iterator iter = mAssociativeContainer.find(key);
         if (iter == mAssociativeContainer.end()) {
            mAssociativeContainer[key] = mDefaultValue;
            return mAssociativeContainer[key];
         }
         return iter->second;
      }
   }

   template<class T_AssociativeContainer>
   inline void SparseMarray<T_AssociativeContainer>
   ::computeShapeStrides(std::vector<key_type> & shapeStrides)const {
      /**
         \cond HIDDEN_SYMBOLS
       */
      //unrolled loop?!?
      //TODO measure runtimes
#define SPARSEMARRAY_COMPUTE_SHAPE_STRIDE_LOOP_2(N)\
            for(size_t i=0;i<N-1;i++)\
            {\
                key_type factor=mShape[i];\
                for(size_t j=0;j<i;j++) {factor*=static_cast<key_type>(mShape[j]);}shapeStrides[i]=factor;\
            }\
           /**
                \endcond
            */

         size_t dimension = this->getDimension();
         switch (dimension) {
            case 1:SPARSEMARRAY_COMPUTE_SHAPE_STRIDE_LOOP_2(1);
               break;
            case 2:SPARSEMARRAY_COMPUTE_SHAPE_STRIDE_LOOP_2(2);
               break;
            case 3:SPARSEMARRAY_COMPUTE_SHAPE_STRIDE_LOOP_2(3);
               break;
            case 4:SPARSEMARRAY_COMPUTE_SHAPE_STRIDE_LOOP_2(4);
               break;
            case 5:SPARSEMARRAY_COMPUTE_SHAPE_STRIDE_LOOP_2(5);
               break;
            case 6:SPARSEMARRAY_COMPUTE_SHAPE_STRIDE_LOOP_2(6);
               break;
            case 7:SPARSEMARRAY_COMPUTE_SHAPE_STRIDE_LOOP_2(7);
               break;
            case 8:SPARSEMARRAY_COMPUTE_SHAPE_STRIDE_LOOP_2(8);
               break;
            case 9:SPARSEMARRAY_COMPUTE_SHAPE_STRIDE_LOOP_2(9);
               break;
            case 10:SPARSEMARRAY_COMPUTE_SHAPE_STRIDE_LOOP_2(10);
               break;
            default:SPARSEMARRAY_COMPUTE_SHAPE_STRIDE_LOOP_2(dimension);
         }
   }

   /**
    *
    \brief coordinateToKey
    *
    converts a coordinate tuple to a key
    *
    *
    @param index1
    scalar integral index
    *
    @return typename SparseMarrayWrapper<T_AssociativeContainer>::key_type
    *
    <b> Usage:</b>
    \code
    #include <sparsemarray.hxx> \endcode  \code
    //lets say sparseArray is an existing  Sparse array with shape(2);
    size_t key=array.coordinateToKey(1);
   \endcode
   @author Thorsten Beier
    *
    @date 10/23/2010
    */
   template<class T_AssociativeContainer>
   inline typename SparseMarray<T_AssociativeContainer>::key_type
   SparseMarray<T_AssociativeContainer>
   ::coordinateToKey(coordinate_type index1) const {

      SPMA_ASSERT(index1 < this->size(), "index1<mShape[0] is violated");
      return static_cast<key_type> (index1);
   }

   /**
    *
   \brief coordinateToKey
    *
   converts a coordinate tuple to a key
    *
    *
   @param index1
   scalar integral index
    *
   @param index2
   scalar integral index
    *
   @return typename SparseMarrayWrapper<T_AssociativeContainer>::key_type
    *
   <b> Usage:</b>
   \code
   #include <sparsemarray.hxx> \endcode  \code
   //lets say sparseArray is an existing  Sparse array with shape(2,3);
   size_t key=array.coordinateToKey(0,2);
   \endcode
   @author Thorsten Beier
    *
   @date 10/23/2010
    */
   template<class T_AssociativeContainer>
   inline typename SparseMarray<T_AssociativeContainer>::key_type
   SparseMarray<T_AssociativeContainer>
   ::coordinateToKey(coordinate_type index1, coordinate_type index2) const {
      SPMA_ASSERT(this->getDimension() == 2, "this->getDimension()==2 is violated");
      SPMA_ASSERT((index1 < mShape[0] && index2 < mShape[1]), "index1<mShape[0] &&index2<mShape[1]  is violated");
      return static_cast<key_type> (index1) + mShapeStrides[0] * static_cast<key_type> (index2);
   }

   /**
    *
   \brief coordinateToKey
    *
   converts a coordinate tuple to a key
    *
    *
   @param index1
   scalar integral index
    *
   @param index2
   scalar integral index
    *
   @param index3
   scalar integral index
    *
   @return typename SparseMarrayWrapper<T_AssociativeContainer>::key_type
    *
   <b> Usage:</b>
   \code
   #include <sparsemarray.hxx> \endcode  \code
   //lets say sparseArray is an existing  Sparse array with shape(10,10,10);
   size_t key=array.coordinateToKey(5,2,8);
   \endcode
   @author Thorsten Beier
    *
   @date 10/23/2010
    */
   template<class T_AssociativeContainer>
   inline typename SparseMarray<T_AssociativeContainer>::key_type
   SparseMarray<T_AssociativeContainer>
   ::coordinateToKey(coordinate_type index1, coordinate_type index2, coordinate_type index3) const {
      SPMA_ASSERT(this->getDimension() == 3, "this->getDimension()==3 is violated");
      SPMA_ASSERT((index1 < mShape[0] && index2 < mShape[1] && index3 < mShape[2]), "index1<mShape[0] &&index2<mShape[1] && index3<mShape[2]  is violated");
      return static_cast<key_type> (index1) + mShapeStrides[0] * static_cast<key_type> (index2) + static_cast<key_type> (index3) * mShapeStrides[1];
   }

   /**
    *
   \brief coordinateToKey
    *
   converts a coordinate tuple to a key
    *
    *
   @param index1
   scalar integral index
    *
   @param index2
   scalar integral index
    *
   @param index3
   scalar integral index
    *
   @param index4
   scalar integral index
    *
   @return typename SparseMarrayWrapper<T_AssociativeContainer>::key_type
    *
   <b> Usage:</b>
   \code
   #include <sparsemarray.hxx> \endcode  \code
   //lets say sparseArray is an existing  Sparse array with shape(100,10,200,200);
   size_t key=array.coordinateToKey(99,5,150,158);
   \endcode
   @author Thorsten Beier
    *
   @date 10/23/2010
    */
   template<class T_AssociativeContainer>
   inline typename SparseMarray<T_AssociativeContainer>::key_type
   SparseMarray<T_AssociativeContainer>
   ::coordinateToKey(coordinate_type index1, coordinate_type index2, coordinate_type index3, coordinate_type index4) const {
      SPMA_ASSERT(this->getDimension() == 4, "this->getDimension()==4 is violated");
      SPMA_ASSERT((index1 < mShape[0] && index2 < mShape[1] && index3 < mShape[2]), "index1<mShape[0] &&index2<mShape[1] && index3<mShape[2]  is violated");
      SPMA_ASSERT(index4 < mShape[3], "index4<mShape[3]  is violated");
      return static_cast<key_type> (index1) + mShapeStrides[0] * static_cast<key_type> (index2) + static_cast<key_type> (index3) * mShapeStrides[1] +
      static_cast<key_type> (index4) * mShapeStrides[2];
   }

   /**
    *
   \brief coordinateToKey
    *
   converts a coordinate tuple to a key
    *
    *
   @param index1
   scalar integral index
    *
   @param index2
   scalar integral index
    *
   @param index3
   scalar integral index
    *
   @param index4
   scalar integral index
    *
   @param index5
   scalar integral index
    *
   @return typename SparseMarrayWrapper<T_AssociativeContainer>::key_type
    *
   <b> Usage:</b>
   \code
   #include <sparsemarray.hxx> \endcode  \code
   //lets say sparseArray is an existing  Sparse array with shape(100,10,200,10,20);
   size_t key=array.coordinateToKey(99,5,150,6,17);
   \endcode
   @author Thorsten Beier
    *
   @date 10/23/2010
    */
   template<class T_AssociativeContainer>
   inline typename SparseMarray<T_AssociativeContainer>::key_type
   SparseMarray<T_AssociativeContainer>
   ::coordinateToKey(coordinate_type index1, coordinate_type index2, coordinate_type index3, coordinate_type index4, coordinate_type index5) const {
      SPMA_ASSERT(this->getDimension() == 5, "this->getDimension()==5 is violated");
      SPMA_ASSERT((index1 < mShape[0] && index2 < mShape[1] && index3 < mShape[2]), "index1<mShape[0] &&index2<mShape[1] && index3<mShape[2]  is violated");
      SPMA_ASSERT(index4 < mShape[3], "index4<mShape[3]  is violated");
      SPMA_ASSERT(index5 < mShape[4], "index5<mShape[4]  is violated");
      return static_cast<key_type> (index1) + mShapeStrides[0] * static_cast<key_type> (index2) + static_cast<key_type> (index3) * mShapeStrides[1] +
         static_cast<key_type> (index4) * mShapeStrides[2] + static_cast<key_type> (index5) * mShapeStrides[3];

   }

   /**
   \cond HIDDEN_SYMBOLS
    */
   template<class T_AssociativeContainer>
   inline typename SparseMarray<T_AssociativeContainer>::key_type
   SparseMarray<T_AssociativeContainer>
   ::coordinateToKey(coordinate_type index1, coordinate_type index2, coordinate_type index3, coordinate_type index4, coordinate_type index5,
   coordinate_type index6) const {
      SPMA_ASSERT(this->getDimension() == 6, "this->getDimension()==6 is violated");
      SPMA_ASSERT((index1 < mShape[0] && index2 < mShape[1] && index3 < mShape[2]), "index1<mShape[0] &&index2<mShape[1] && index3<mShape[2]  is violated");
      SPMA_ASSERT(index4 < mShape[3], "index4<mShape[3]  is violated");
      SPMA_ASSERT(index5 < mShape[4], "index5<mShape[4]  is violated");
      SPMA_ASSERT(index6 < mShape[5], "index6<mShape[5]  is violated");

      return static_cast<key_type> (index1) + mShapeStrides[0] * static_cast<key_type> (index2) + static_cast<key_type> (index3) * mShapeStrides[1] +
         static_cast<key_type> (index4) * mShapeStrides[2] + static_cast<key_type> (index5) * mShapeStrides[3]
         + static_cast<key_type> (index6) * mShapeStrides[4];
      
   }

   template<class T_AssociativeContainer>
   inline typename SparseMarray<T_AssociativeContainer>::key_type
   SparseMarray<T_AssociativeContainer>
   ::coordinateToKey(coordinate_type index1, coordinate_type index2, coordinate_type index3, coordinate_type index4, coordinate_type index5,
   coordinate_type index6, coordinate_type index7) const {
      SPMA_ASSERT(this->getDimension() == 7, "this->getDimension()==7 is violated");
      SPMA_ASSERT((index1 < mShape[0] && index2 < mShape[1] && index3 < mShape[2]), "index1<mShape[0] &&index2<mShape[1] && index3<mShape[2]  is violated");
      SPMA_ASSERT(index4 < mShape[3], "index4<mShape[3]  is violated");
      SPMA_ASSERT(index5 < mShape[4], "index5<mShape[4]  is violated");
      SPMA_ASSERT(index6 < mShape[5], "index6<mShape[5]  is violated");
      SPMA_ASSERT(index7 < mShape[6], "index7<mShape[6]  is violated");

      return static_cast<key_type> (index1) + mShapeStrides[0] * static_cast<key_type> (index2) + static_cast<key_type> (index3) * mShapeStrides[1] +
         static_cast<key_type> (index4) * mShapeStrides[2] + static_cast<key_type> (index5) * mShapeStrides[3]
         + static_cast<key_type> (index6) * mShapeStrides[4] + static_cast<key_type> (index7) * mShapeStrides[5];
       
   }

   template<class T_AssociativeContainer>
   inline typename SparseMarray<T_AssociativeContainer>::key_type
   SparseMarray<T_AssociativeContainer>
   ::coordinateToKey(coordinate_type index1, coordinate_type index2, coordinate_type index3, coordinate_type index4, coordinate_type index5,
   coordinate_type index6, coordinate_type index7, coordinate_type index8) const {
      SPMA_ASSERT(this->getDimension() == 8, "this->getDimension()==8 is violated");
      SPMA_ASSERT((index1 < mShape[0] && index2 < mShape[1] && index3 < mShape[2]), "index1<mShape[0] &&index2<mShape[1] && index3<mShape[2]  is violated");
      SPMA_ASSERT(index4 < mShape[3], "index4<mShape[3]  is violated");
      SPMA_ASSERT(index5 < mShape[4], "index5<mShape[4]  is violated");
      SPMA_ASSERT(index6 < mShape[5], "index6<mShape[5]  is violated");
      SPMA_ASSERT(index7 < mShape[6], "index7<mShape[6]  is violated");
      SPMA_ASSERT(index8 < mShape[7], "index8<mShape[7]  is violated");
      return static_cast<key_type> (index1) + mShapeStrides[0] * static_cast<key_type> (index2) + static_cast<key_type> (index3) * mShapeStrides[1] +
         static_cast<key_type> (index4) * mShapeStrides[2] + static_cast<key_type> (index5) * mShapeStrides[3]
         + static_cast<key_type> (index6) * mShapeStrides[4] + static_cast<key_type> (index7) * mShapeStrides[5] + static_cast<key_type> (index8) * mShapeStrides[6];

   }

   template<class T_AssociativeContainer>
   inline typename SparseMarray<T_AssociativeContainer>::key_type
   SparseMarray<T_AssociativeContainer>
   ::coordinateToKey(coordinate_type index1, coordinate_type index2, coordinate_type index3, coordinate_type index4, coordinate_type index5,
   coordinate_type index6, coordinate_type index7, coordinate_type index8, coordinate_type index9) const {
      SPMA_ASSERT(this->getDimension() == 9, "this->getDimension()==9 is violated");
      SPMA_ASSERT((index1 < mShape[0] && index2 < mShape[1] && index3 < mShape[2]), "index1<mShape[0] &&index2<mShape[1] && index3<mShape[2]  is violated");
      SPMA_ASSERT(index4 < mShape[3], "index4<mShape[3]  is violated");
      SPMA_ASSERT(index5 < mShape[4], "index5<mShape[4]  is violated");
      SPMA_ASSERT(index6 < mShape[5], "index6<mShape[5]  is violated");
      SPMA_ASSERT(index7 < mShape[6], "index7<mShape[6]  is violated");
      SPMA_ASSERT(index8 < mShape[7], "index8<mShape[7]  is violated");
      SPMA_ASSERT(index9 < mShape[8], "index9<mShape[8]  is violated");
      return static_cast<key_type> (index1) + mShapeStrides[0] * static_cast<key_type> (index2) + static_cast<key_type> (index3) * mShapeStrides[1] +
         static_cast<key_type> (index4) * mShapeStrides[2] + static_cast<key_type> (index5) * mShapeStrides[3]
         + static_cast<key_type> (index6) * mShapeStrides[4] + static_cast<key_type> (index7) * mShapeStrides[5] + static_cast<key_type> (index8) * mShapeStrides[6] +
         static_cast<key_type> (index9) * mShapeStrides[7];
       
   }

   template<class T_AssociativeContainer>
   inline typename SparseMarray<T_AssociativeContainer>::key_type
   SparseMarray<T_AssociativeContainer>::coordinateToKey(coordinate_type index1, coordinate_type index2, coordinate_type index3, coordinate_type index4, coordinate_type index5,
   coordinate_type index6, coordinate_type index7, coordinate_type index8, coordinate_type index9, coordinate_type index10) const {
      SPMA_ASSERT(this->getDimension() == 10, "this->getDimension()==10 is violated");
      SPMA_ASSERT((index1 < mShape[0] && index2 < mShape[1] && index3 < mShape[2]), "index1<mShape[0] &&index2<mShape[1] && index3<mShape[2]  is violated");
      SPMA_ASSERT(index4 < mShape[3], "index4<mShape[3]  is violated");
      SPMA_ASSERT(index5 < mShape[4], "index5<mShape[4]  is violated");
      SPMA_ASSERT(index6 < mShape[5], "index6<mShape[5]  is violated");
      SPMA_ASSERT(index7 < mShape[6], "index7<mShape[6]  is violated");
      SPMA_ASSERT(index8 < mShape[7], "index8<mShape[7]  is violated");
      SPMA_ASSERT(index9 < mShape[8], "index9<mShape[8]  is violated");
      SPMA_ASSERT(index10 < mShape[9], "index10<mShape[9]  is violated");

      return static_cast<key_type> (index1) + mShapeStrides[0] * static_cast<key_type> (index2) + static_cast<key_type> (index3) * mShapeStrides[1] +
         static_cast<key_type> (index4) * mShapeStrides[2] + static_cast<key_type> (index5) * mShapeStrides[3]
         + static_cast<key_type> (index6) * mShapeStrides[4] + static_cast<key_type> (index7) * mShapeStrides[5] + static_cast<key_type> (index8) * mShapeStrides[6] +
         static_cast<key_type> (index9) * mShapeStrides[7] + static_cast<key_type> (index10) * mShapeStrides[8];

   }

   template<class T_AssociativeContainer>
   template<class CoordinateInputType>
   inline typename SparseMarray<T_AssociativeContainer>::key_type
   SparseMarray<T_AssociativeContainer>
   ::coordinateToKey(CoordinateInputType coordinate) const {
      //OVERLOAD HELPER:
      //generic compiletime switch-cases (is just a typelist witch SwitchCases ,a Switch Case is something like a pair of bool and a type/flag)
      typedef typename meta::TypeListGenerator
         <
         meta::SwitchCase< meta::Compare<key_type, CoordinateInputType >::value, CoordinateKeyType>,
         meta::SwitchCase< meta::Compare<coordinate_type, CoordinateInputType >::value, CoordinateCoordinateType>,
         meta::SwitchCase< meta::IsFundamental<CoordinateInputType >::value, CoordinateFundamentalType >
         >::type CaseList;
      //generic compiletime switch with struct CoordinateIterator as default flag
      typedef typename meta::Switch<CaseList, CoordinateIteratorType>::type CoordinateFlag;
      //call the best version of operator()(coordinate ,someFlag)
      //the operator needs to be implemented for all types of flags
      return this->coordinateToKey(coordinate, CoordinateFlag());
   }

   template<class T_AssociativeContainer>
   template<class CoordinateInputType>
   inline typename SparseMarray<T_AssociativeContainer>::key_type
   SparseMarray<T_AssociativeContainer>
   ::coordinateToKey(CoordinateInputType coordinate, CoordinateCoordinateType flag) const {
      return this->coordinateToKey(coordinate);
   }

   template<class T_AssociativeContainer>
   template<class CoordinateInputType>
   inline typename SparseMarray<T_AssociativeContainer>::key_type
   SparseMarray<T_AssociativeContainer>
   ::coordinateToKey(CoordinateInputType coordinate, CoordinateKeyType flag) const {
      return this->coordinateToKey(static_cast<coordinate_type> (coordinate));
   }

   template<class T_AssociativeContainer>
   template<class CoordinateInputType>
   inline typename SparseMarray<T_AssociativeContainer>::key_type
   SparseMarray<T_AssociativeContainer>
   ::coordinateToKey(CoordinateInputType coordinate, CoordinateFundamentalType flag) const {
      return this->coordinateToKey(static_cast<coordinate_type> (coordinate));
   }

   template<class T_AssociativeContainer>
   template<class CoordinateInputType>
   inline typename SparseMarray<T_AssociativeContainer>::key_type
   SparseMarray<T_AssociativeContainer>
   ::coordinateToKey(CoordinateInputType coordinate, CoordinateVectorType flag) const {
      return this->coordinateToKey(coordinate.begin());
   }
   /**
    \endcond
    */

   /**
    *
    \brief coordinateToKey
    *
    converts a coordinate tuple to a key
    *
    *
    @param coordinateBegin
    begin of coordinte tuple
    *
    @return typename SparseMarrayWrapper<T_AssociativeContainer>::key_type
    *
    <b> Usage:</b>
    \code
    #include <sparsemarray.hxx> \endcode  \code
    //lets say sparseArray is an existing  Sparse array with shape(100,100,100,100,100);
    size_t coordinateTupe[]={55,0,1,44,98}
    size_t key=array.coordinateToKey(coordinateTupe,coordinateTupe+5);
    \endcode
    @author Thorsten Beier
    *
    @date 10/23/2010
    */
   template<class T_AssociativeContainer>
   template<class CoordinateIter>
   inline typename SparseMarray<T_AssociativeContainer>::key_type
   SparseMarray<T_AssociativeContainer>
   ::coordinateToKey(CoordinateIter coordinateBegin, CoordinateIteratorType flag) const {
      /**
         \cond HIDDEN_SYMBOLS
       */
#define SPARSEMARRAY_COORDINATE_TO_KEY_ITERATOR_LOOP(N) \
        { \
            typedef typename std::iterator_traits<CoordinateIter>::value_type iter_ValueType; \
            key_type key=static_cast<key_type>(*coordinateBegin);\
            for(size_t i=0;i<N-1;i++) \
            { \
                coordinateBegin++; \
                key+=static_cast<key_type>(*coordinateBegin)*mShapeStrides[i]; \
            } \
            return key;\
        }
      /**
          \endcond
       */

         size_t dimension = this->getDimension();
         switch (dimension) {
            case 1: return static_cast<key_type> (*coordinateBegin);
            case 2: SPARSEMARRAY_COORDINATE_TO_KEY_ITERATOR_LOOP(2);
            case 3: SPARSEMARRAY_COORDINATE_TO_KEY_ITERATOR_LOOP(3);
            case 4: SPARSEMARRAY_COORDINATE_TO_KEY_ITERATOR_LOOP(4);
            case 5: SPARSEMARRAY_COORDINATE_TO_KEY_ITERATOR_LOOP(5);
            case 6: SPARSEMARRAY_COORDINATE_TO_KEY_ITERATOR_LOOP(6);
            case 7: SPARSEMARRAY_COORDINATE_TO_KEY_ITERATOR_LOOP(7);
            case 8: SPARSEMARRAY_COORDINATE_TO_KEY_ITERATOR_LOOP(8);
            case 9: SPARSEMARRAY_COORDINATE_TO_KEY_ITERATOR_LOOP(9);
            case 10: SPARSEMARRAY_COORDINATE_TO_KEY_ITERATOR_LOOP(10);
            default: SPARSEMARRAY_COORDINATE_TO_KEY_ITERATOR_LOOP(dimension);
         }

   }

   /**
    *
    \brief get coordinate tuple from key
    *
    *@param key
    * key to convert in coordinateTuple
    *
    *
    @param[out] coordinateTuple
    coorindate tuple to that key
    *
    <b> Usage:</b>
    \code
    #include <sparsemarray.hxx> \endcode  \code
    //lets say sparseArray is an existing  sparse array with some shape;
    std::vector<size_t> coordinate_tuple;
    sparseArray.keyToCoordinate(3,coordinate_tuple);
    \endcode
   @author Thorsten Beier
    *
    @date 03/02/2011
    */
   template<class T_AssociativeContainer>
   inline void SparseMarray<T_AssociativeContainer>
   ::keyToCoordinate(key_type key, coordinate_tuple & coordinateTuple)const {
      /**
          \cond HIDDEN_SYMBOLS
       */
      
#define SPARSEMARRAY_KEY_TO_COORDINATE_LOOP(N) \
        for(size_t i=0;i<N-2;i++) \
        { \
            coordinateTuple[N-1-i]=key/mShapeStrides[N-2-i]; \
            key=key-coordinateTuple[N-1-i]*mShapeStrides[N-2-i]; \
        } \
        coordinateTuple[1]=key/mShapeStrides[0]; \
        coordinateTuple[0]=key-coordinateTuple[1]*mShapeStrides[0]; \
       /**
            \endcond
        */
      
      size_t dimension = this->getDimension();
      coordinateTuple.resize(dimension);
      switch (dimension) {
         case 1:
         {
            coordinateTuple[0] = key;
            break;
         }
         case 2:
         {
            key_type tkey = key / mShapeStrides[0];
            coordinateTuple[1] = tkey;
            coordinateTuple[0] = key - tkey * mShapeStrides[0];
            break;
         }
         case 3:
         {
            SPARSEMARRAY_KEY_TO_COORDINATE_LOOP(3);
            break;
         }
         case 4:
         {
            SPARSEMARRAY_KEY_TO_COORDINATE_LOOP(4);
            break;
         }
         case 5:
         {
            SPARSEMARRAY_KEY_TO_COORDINATE_LOOP(5);
            break;
         }
         case 6:
         {
            SPARSEMARRAY_KEY_TO_COORDINATE_LOOP(6);
            break;
         }
         case 7:
         {
            SPARSEMARRAY_KEY_TO_COORDINATE_LOOP(7);
            break;
         }
         case 8:
         {
            SPARSEMARRAY_KEY_TO_COORDINATE_LOOP(8);
            break;
         }
         case 9:
         {
            SPARSEMARRAY_KEY_TO_COORDINATE_LOOP(9);
            break;
         }
         case 10:
         {
            SPARSEMARRAY_KEY_TO_COORDINATE_LOOP(10);
            break;
         }
         default:
         {
            SPARSEMARRAY_KEY_TO_COORDINATE_LOOP(dimension);
         }
      }
   }

   /**
    *
    \brief get coordinate tuple from key
    *
     @param key
    *
    *
    @return coordinate_tuple
    coorindate tuple to that key
    *
    <b> Usage:</b>
    \code
    #include <sparsemarray.hxx> \endcode  \code
    //lets say sparseArray is an existing  sparse array with some shape;
    std::vector<size_t> coordinate_tuple=sparseArray.keyToCoordinate(3);
    \endcode
   @author Thorsten Beier
    *
    @date 03/02/2011
    */
   template<class T_AssociativeContainer>
   inline typename SparseMarray<T_AssociativeContainer>::coordinate_tuple
   SparseMarray<T_AssociativeContainer>
   ::keyToCoordinate(key_type key)const {
      SPMA_ASSERT(key < this->size(), "key < this->size() is violated");
      coordinate_tuple coordinate;
      this->keyToCoordinate(key, coordinate);
      return coordinate;
   }

   /**
    *
    \brief get shape of a sparse array
    *
    get Shape  of a sparse array as an std::vector<size_t>
    *
    *
    @param[out] shape
    shape of the sparse array
    *
    <b> Usage:</b>
    \code
    #include <sparsemarray.hxx> \endcode  \code
    //lets say sparseArray is an existing  sparse array with some shape;
    std::vector<size_t> shape;
    sparseArray.getShape(shape);  //shape of the sparse array is now stored in std::vector<size_t> shape;
    \endcode
   @author Thorsten Beier
    *
    @date 10/23/2010
    */
   template<class T_AssociativeContainer>
   inline void SparseMarray<T_AssociativeContainer>
   ::getShape(coordinate_tuple & shape) const {
      shape = mShape;
   }

   /**
    *
   \brief get shape of a sparse array
    *
   get Shape  of a sparse array as an std::vector<size_t>
    *
    *
   @return shape of the sparse array
    *
   <b> Usage:</b>
   \code
   #include <sparsemarray.hxx> \endcode  \code
   //lets say sparseArray is an existing  Sparse array with some shape and size_t as coordinate_type;
   std::vector<size_t> shape=sparseArray.getShape();  //shape of the sparse array is now stored in std::vector<size_t> shape;
   \endcode
   @author Thorsten Beier
    *
   @date 10/23/2010
    */
   template<class T_AssociativeContainer>
   inline const typename SparseMarray<T_AssociativeContainer>::coordinate_tuple &
   SparseMarray<T_AssociativeContainer>
   ::getShape() const {
      return mShape;
   }

   template<class T_AssociativeContainer>
   inline const typename SparseMarray<T_AssociativeContainer>::coordinate_tuple &
   SparseMarray<T_AssociativeContainer>
   ::shape() const {
      return mShape;
   }

   /**
    *
   \brief get size of one dimension
    *
   get size  of one dimension of the sparse array
    *
    *
   @param shapeIndex index of the Dimension you want to get the size of
    *
   @return size of the Dimension
    *
   <b> Usage:</b>
   \code
   #include <sparsemarray.hxx> \endcode  \code
   //lets say sparseArray is an existing  Sparse array with  shape(2,4,5);
   size_t nrOfElements=sparseArray.size(); //==2*4*5=40;
   size_t s1=sparseArray.size(0); //==2;
   size_t s2=sparseArray.size(1); //==4;
   size_t s3=sparseArray.size(2); //==5;
   \endcode
   @author Thorsten Beier
    *
   @date 10/23/2010
    */
   template<class T_AssociativeContainer>
   size_t SparseMarray<T_AssociativeContainer>
   ::size(const size_t shapeIndex) const {
      SPMA_ASSERT(shapeIndex<this->getDimension(), "shapeIndex<this->getDimension() is violated");
      return mShape[shapeIndex];
   }

   /**
    *
    \brief get size of a sparse array
    *
    get size  of a sparse array (total number of elements which can be stored in the sparse array)
    *
    *
    @return size of the sparse array
    *
    <b> Usage:</b>
    \code
    #include <sparsemarray.hxx> \endcode  \code
    //lets say sparseArray is an existing  Sparse array with some shape;
    size_t nrOfElements=sparseArray.size();
   \endcode
   @author Thorsten Beier
    *
    @date 10/23/2010
    */
   template<class T_AssociativeContainer>
   size_t SparseMarray<T_AssociativeContainer>
   ::size() const {
      size_t nrOfElements = 1;
      size_t dimension = (this->getDimension());
      if (dimension == 0) {
         nrOfElements = 1;
      } else {
         for (size_t i = 0; i < dimension; i++) {
            nrOfElements *= mShape[i];
         }
      }
      return nrOfElements;
   }

   template<class T_AssociativeContainer>
   template<typename Iter>
   inline void SparseMarray<T_AssociativeContainer>
   ::assign(Iter beginData, Iter endData, opengm::DenseAssigment flag) {
      SPMA_ASSERT(size_t(std::distance(beginData, endData)) == size_t(this->size()), "std::distance(beginData,endData)==this->size() is violated");
      size_t index = 0;
      value_comparator comperator;
      while (beginData != endData) {
         if (comperator(static_cast<value_type> (*beginData), mDefaultValue) == false) {
            mAssociativeContainer.insert(std::make_pair(index, *beginData));
         }
         index++;
         beginData++;
      }
   }

   template<class T_AssociativeContainer>
   template<typename Iter>
   inline void SparseMarray<T_AssociativeContainer>
   ::assign(Iter beginData, Iter endData, opengm::SparseAssigmentKey flag) {
      SPMA_ASSERT(std::distance(beginData, endData) != 0, "std::distance(beginData,endData)!=0 is violated");
      while (beginData != endData) {
         if (beginData->second != mDefaultValue) {
            mAssociativeContainer.insert(*beginData);
         }
         beginData++;
      }
   }

   template<class T_AssociativeContainer>
   template<typename Iter>
   inline void SparseMarray<T_AssociativeContainer>
   ::assign(Iter beginData, Iter endData, opengm::SparseAssigmentCoordinateTuple flag) {
      SPMA_ASSERT(std::distance(beginData, endData) != 0, "std::distance(beginData,endData)!=0 is violated");
      while (beginData != endData) {
         if (beginData->second != mDefaultValue) {
            mAssociativeContainer.insert(std::make_pair(this->coordinateToKey(beginData->first), beginData->second));
         }
         beginData++;
      }
   }

   template<class T_AssociativeContainer>
   template<class T_AC>
   SparseMarray<T_AssociativeContainer> &
   SparseMarray<T_AssociativeContainer>::
   operator=(SparseMarray<T_AC> const & rhs) {
      if (this == &rhs) {
         return *this;
      }
      CopyAssociativeContainer<T_AC, T_AssociativeContainer> copyContainer;
      copyContainer.copy(rhs.mAssociativeContainer, mAssociativeContainer);
      mShape.assign(rhs.mShape.begin(), rhs.mShape.end());
      mDefaultValue = static_cast<value_type> (rhs.mDefaultValue);
      mShapeStrides.assign(rhs.mShapeStrides.begin(), rhs.mShapeStrides.end());

      
      return *this;
   }

   template<class T_AssociativeContainer>
   SparseMarray<T_AssociativeContainer> &
   SparseMarray<T_AssociativeContainer>::
   operator=(const SparseMarray & rhs) {
      if (this == &rhs) {
         return *this;
      }
      mAssociativeContainer = rhs.mAssociativeContainer;
      mShape = rhs.mShape;
      mShapeStrides = rhs.mShapeStrides;
      mDefaultValue = rhs.mDefaultValue;
      return *this;
   }

   template<class T_AssociativeContainer>
   template<class T_Value>
   SparseMarray<T_AssociativeContainer> &
   SparseMarray<T_AssociativeContainer>::
   operator=(T_Value const & rhs) {
      mAssociativeContainer.clear();
      mDefaultValue = static_cast<value_type> (rhs);
   }

   /**
    *
    \brief postfix operator++
    *
    operator++ postfix
    *
    <b> Usage:</b>
    \code
    #include <sparsemarray.hxx> \endcode \code
   //lets say array is an existing  Sparse array with some size;
    array++; // increase all values of the array by one (inclusive the defaul value)
    \endcode
   @author Thorsten Beier
    *
    @date 10/23/2010
    */
   template<class T_AssociativeContainer>
   SparseMarray
   <
   T_AssociativeContainer
   > &
   SparseMarray<T_AssociativeContainer>
   ::operator++() {
      typename associative_container_type::iterator iter = this->mAssociativeContainer.begin();
      while (iter != mAssociativeContainer.end()) {
         iter->second++;
         iter++;
      }
      mDefaultValue++;
      return *this;
   }

   /**
    *
   \brief prefix operator++
    *
   operator++ prefix
    *
   <b> Usage:</b>
   \code
   #include <sparsemarray.hxx> \endcode
   \code
   //lets say array is an existing  Sparse array with some size;
   ++array; // increase all values of the array by one (inclusive the defaul value)
   \endcode
   @author Thorsten Beier
    *
   @date 10/23/2010
    */
   template<class T_AssociativeContainer>
   SparseMarray
   <
   T_AssociativeContainer
   >
   SparseMarray<T_AssociativeContainer>
   ::operator++(int) {
      SparseMarrayWrapperType aCopy = this;
      typename associative_container_type::iterator iter = aCopy.mAssociativeContainer.begin();
      while (iter != aCopy.mAssociativeContainer.end()) {
         iter->second++;
         iter++;
      }
      aCopy.mDefaultValue++;
      return aCopy;
   }

   /**
    *
   \brief postfix operator--
    *
   operator-- postfix
    *
    *
   <b> Usage:</b>
   \code
   #include <sparsemarray.hxx> \endcode \code
   //lets say array is an existing  Sparse array with some size;
   array--; // drecrease all values of the array by one (inclusive the defaul value)
   \endcode
   @author Thorsten Beier
    *
   @date 10/23/2010
    */
   template<class T_AssociativeContainer>
   SparseMarray
   <
   T_AssociativeContainer
   > &
   SparseMarray<T_AssociativeContainer>
   ::operator--() {
      typename associative_container_type::iterator iter = this->mAssociativeContainer.begin();
      while (iter != mAssociativeContainer.end()) {
         iter->second--;
         iter++;
      }
      mDefaultValue--;
      return *this;
   }

   /**
    *
   \brief prefix operator--
    *
   operator-- prefix
    *
   <b> Usage:</b>
   \code
   #include <sparsemarray.hxx> \endcode
   \code
   //lets say array is an existing  Sparse array with some size;
   --array; // drecrease all values of the array by one (inclusive the defaul value)
   \endcode
   @author Thorsten Beier
    *
   @date 10/23/2010
    */
   template<class T_AssociativeContainer>
   SparseMarray
   <
      T_AssociativeContainer
   > SparseMarray<T_AssociativeContainer>::operator--(int) {
      SparseMarrayWrapperType aCopy = this;
      typename associative_container_type::iterator iter = aCopy.mAssociativeContainer.begin();
      while (iter != aCopy.mAssociativeContainer.end()) {
         iter->second--;
         iter++;
      }
      aCopy.mDefaultValue--;
      return aCopy;
   }

   /**
    *
   \brief operator+=
    *
   operator+=
    *
   <b> Usage:</b>
   \code
   #include <sparsemarray.hxx> \endcode
   \code
   //lets say array is an existing  Sparse array with some size;
   array+=1.0; // increases all values of the array by 1.0
   \endcode
   @author Thorsten Beier
    *
   @date 10/23/2010
    */
   template<class T_AssociativeContainer>
   template<class T_Value>
   inline SparseMarray<T_AssociativeContainer> &
   SparseMarray<T_AssociativeContainer>
   ::operator+=(T_Value const & value) {
      typename associative_container_type::iterator iter = this->mAssociativeContainer.begin();
      while (iter != mAssociativeContainer.end()) {
         iter->second += static_cast<mapped_type> (value);
         iter++;
      }
      mDefaultValue += static_cast<mapped_type> (value);
      return *this;
   }

   /**
    *
   \brief operator-=
    *
   operator-=
    *
   <b> Usage:</b>
   \code
   #include <sparsemarray.hxx> \endcode
   \code
   //lets say array is an existing  Sparse array with some size;
   array-=1.0; // drecrease all values of the array by 1.0
   \endcode
   @author Thorsten Beier
    *
   @date 10/23/2010
    */
   template<class T_AssociativeContainer>
   template<class T_Value>
   inline SparseMarray<T_AssociativeContainer> &
   SparseMarray<T_AssociativeContainer>
   ::operator-=(T_Value const & value) {
      typename associative_container_type::iterator iter = this->mAssociativeContainer.begin();
      while (iter != mAssociativeContainer.end()) {
         iter->second -= static_cast<mapped_type> (value);
         iter++;
      }
      mDefaultValue -= static_cast<mapped_type> (value);
      return *this;
   }

   /**
    *
   \brief operator*=
    *
   operator*=
    *
   <b> Usage:</b>
   \code
   #include <sparsemarray.hxx> \endcode
   \code
   //lets say array is an existing  Sparse array with some size;
   array*=2.0; // multiply all values with 2.0
   \endcode
   @author Thorsten Beier
    *
   @date 10/23/2010
    */
   template<class T_AssociativeContainer>
   template<class T_Value>
   inline SparseMarray<T_AssociativeContainer> &
   SparseMarray<T_AssociativeContainer>
   ::operator*=(T_Value const & value) {
      typename associative_container_type::iterator iter = this->mAssociativeContainer.begin();
      while (iter != mAssociativeContainer.end()) {
         iter->second *= static_cast<mapped_type> (value);
         iter++;
      }
      mDefaultValue *= static_cast<mapped_type> (value);
      return *this;
   }

   /**
    *
   \brief operator/=
    *
   operator/=
    *
   <b> Usage:</b>
   \code
   #include <sparsemarray.hxx>
   \endcode
    *
   \code
   //lets say array is an existing  Sparse array with some size;
   array*=2.0; // divide all values with 2.0
   \endcode
   @author Thorsten Beier
    *
   @date 10/23/2010
    */
   template<class T_AssociativeContainer>
   template<class T_Value>
   inline SparseMarray<T_AssociativeContainer> &
   SparseMarray<T_AssociativeContainer>
   ::operator/=(T_Value const & value) {
      typename associative_container_type::iterator iter = this->mAssociativeContainer.begin();
      while (iter != mAssociativeContainer.end()) {
         iter->second /= static_cast<mapped_type> (value);
         iter++;
      }
      mDefaultValue /= static_cast<mapped_type> (value);
      return *this;
   }

   template<class T_AssociativeContainer>
   template<class T_Value>
   inline SparseMarray<T_AssociativeContainer>
   SparseMarray<T_AssociativeContainer>
   ::operator+(T_Value const & value)const {
      return SparseMarray(*this) += value;
   }

   template<class T_AssociativeContainer>
   template<class T_Value>
   inline SparseMarray<T_AssociativeContainer>
   SparseMarray<T_AssociativeContainer>
   ::operator-(T_Value const & value)const {
      return SparseMarray(*this) -= value;
   }

   template<class T_AssociativeContainer>
   template<class T_Value>
   inline SparseMarray<T_AssociativeContainer>
   SparseMarray<T_AssociativeContainer>
   ::operator*(T_Value const & value)const {
      return SparseMarray(*this) *= value;
   }

   template<class T_AssociativeContainer>
   template<class T_Value>
   inline SparseMarray<T_AssociativeContainer>
   SparseMarray<T_AssociativeContainer>
   ::operator/(T_Value const & value)const {
      return SparseMarray(*this) /= value;
   }

   /**
    *
   \brief set default value
    *
   set default value of the sparse array
    *
    *
   @param defaultValue
   default value of the sparse Array
    *
   <b> Usage:</b>
   \code
   #include <sparsemarray.hxx> \endcode \code
   //lets say array is an existing float sparse array with some size;
   float defaultValue=0.0f;
   array.setDefaultValue(defaultValue);  //defaultValue of the sparse array is now 0.0
   \endcode
    *
   @author Thorsten Beier
    *
   @date 10/24/2010
    */
   template<class T_AssociativeContainer>
   template<class T_Value>
   inline void SparseMarray<T_AssociativeContainer>::setDefaultValue
   //(typename meta::CallTraits<T_Value>::param_type defaultValue)
   (T_Value defaultValue) {
      mDefaultValue = static_cast<value_type> (defaultValue);
   }

   /**
    *
    \brief get default value
    *
    get default value of the sparse array
    *
    *
    @return  default value of the sparse Array
    *
    <b> Usage:</b>
    \code
    #include <sparsemarray.hxx> \endcode \code
    //lets say array is an existing float sparse array with some size;
    float defaultValue=array.getDefaultValue();
    \endcode
    *
    @author Thorsten Beier
    *
    @date 10/24/2010
    */
   template<class T_AssociativeContainer>
   inline typename SparseMarray<T_AssociativeContainer>::const_reference_type SparseMarray<T_AssociativeContainer>
   ::getDefaultValue() const {
      return mDefaultValue;
   }

   /**
    *
   \brief reshape the array
    *
   reshape the array to a new shape
    *
   <b> Usage:</b>
   \code
   #include <sparsemarray.hxx> \endcode \code
   //lets say array is an existing float sparse array with some size an shape(3,4,2,2);
   size_t newshape[3]={2,2,12};
   array.reshape(newshape,newshape+3);
   \endcode
    *
   @author Thorsten Beier
    *
   @date 10/24/2010
    */
   template<class T_AssociativeContainer>
   template<typename InputIterator>
   void SparseMarray<T_AssociativeContainer>
   ::reshape(InputIterator beginShape, InputIterator endShape) {
      SPMA_ASSERT(std::distance(beginShape, endShape) != 0, "std::distance(beginShape,endShape)!=0 is violated");
      if (opengm::SPMA_NO_ARG_TEST == false) {
         std::vector<coordinate_type> tmpShape(beginShape, endShape);
         key_type nrOfElements = 1;
         for (size_t i = 0; i < tmpShape.size(); i++) {
            nrOfElements *= tmpShape[i];
         }
         SPMA_ASSERT(nrOfElements == this->size(), "nrOfElements==this->size() is violated")
         mShape = tmpShape;
      } else {
         mShape.assign(beginShape, endShape);
      }
      mShapeStrides.resize(mShape.size() - 1);
      this->computeShapeStrides();
   }
   //clear and delete data

   /**
    *
   \brief clear all data
    *
   clears/delets all data from the sparse array
    *
   <b> Usage:</b>
   \code
   #include <sparsemarray.hxx> \endcode \code
   //lets say array is an existing float sparse array;
   array.clear();  //clears all data from the array;
   \endcode
    *
   @author Thorsten Beier
    *
   @date 10/24/2010
    */
   template<class T_AssociativeContainer>
   inline void SparseMarray<T_AssociativeContainer>
   ::clear() {
      mAssociativeContainer.clear();
      mShape.clear();
      mShapeStrides.clear();
   }

   template<class T_AssociativeContainer>
   inline void SparseMarray<T_AssociativeContainer>
   ::findInRange(key_type keyStart, key_type keyEnd, std::vector<key_type> & inRange)const {
      inRange.clear();
      {
         size_t dim = this->getDimension();
         coordinate_tuple coordinateStart(dim);
         coordinate_tuple coordinateEnd(dim);
         coordinate_tuple coordinate(dim);
         typename associative_container_type::const_iterator iter = mAssociativeContainer.begin();
         typename associative_container_type::const_iterator endIter = mAssociativeContainer.end();
         key_type key;
         while (iter != endIter) {
            key = iter->first;
            if (key >= keyStart && key <= keyEnd) {
               this->keyToCoordinate(keyStart, coordinateStart);
               this->keyToCoordinate(keyEnd, coordinateEnd);
               this->keyToCoordinate(key, coordinate);
               if (this->isInRange(dim, coordinateStart, coordinateEnd, coordinate)) {
                  inRange.push_back(key);
               }
            }
            iter++;
         }
      }
   }

   template<class T_AssociativeContainer>
   inline bool SparseMarray<T_AssociativeContainer>
   ::isInRange(short dim, const coordinate_tuple & rangeStart, const coordinate_tuple & rangeEnd, const coordinate_tuple & coordinate)const {
      for (size_t i = 0; i < dim; i++) {
         if (coordinate[i] < rangeStart[i] || coordinate[i] > rangeEnd[i]) {
            return false;
         }
      }
      return true;
   }

   //namepsace beier

   template<class U>
   std::ostream & operator<<(std::ostream & out, const opengm::AccessProxy<U> & ap) {
      typedef U SPMA;
      out << static_cast<typename SPMA::value_type> (ap);
      return out;
   }

   /// \cond HIDDEN_SYMBOLS
   /**
    * @class sparse_array_iterator
    *
    * @brief iterator class
    *
    * iterator class for the sparsemarray class
    */
   template<class T_SparseMarray, bool IS_CONST>
   class sparse_array_iterator;

   template<class T_SparseMarray>
   class sparse_array_iterator<T_SparseMarray, true > {
      friend class sparse_array_iterator<T_SparseMarray, false >;
   protected:
      typedef T_SparseMarray SparseMarrayType;
      typedef typename SparseMarrayType::value_type T;
   public:
      typedef typename SparseMarrayType::value_type value_type;
      typedef typename SparseMarrayType::reference_type reference_type;
      typedef typename SparseMarrayType::const_reference_type const_reference_type;
      typedef typename SparseMarrayType::param_type param_type;
      typedef const SparseMarrayType * ptr_sparse_array_type;
      typedef size_t distance_type;
   protected:
      ptr_sparse_array_type pSparseMarray;
      size_t index;\
       public:

      sparse_array_iterator() : pSparseMarray(NULL) {
      };

      sparse_array_iterator(ptr_sparse_array_type opengm, size_t _index) : pSparseMarray(opengm), index(_index) {
      };

      sparse_array_iterator(const sparse_array_iterator<T_SparseMarray, false > & iter) {
         pSparseMarray = iter.pSparseMarray;
         index = iter.index;
      };

      sparse_array_iterator operator=(const sparse_array_iterator<T_SparseMarray, false > & iter) {
         index = iter.index;
         pSparseMarray = iter.pSparseMarray;
         return *this;
      };

      sparse_array_iterator & operator++() {
         index++;
         return *this;
      }

      sparse_array_iterator operator++(int) {
         index++;
         return sparse_array_iterator(*this);
      }

      sparse_array_iterator & operator--() {
         this.index--;
         return *this;
      }

      sparse_array_iterator operator--(int) {
         index--;
         return sparse_array_iterator(*this);
      }

      sparse_array_iterator & operator+=(sparse_array_iterator const & iter) {
         index += iter.index;
         return *this;
      };

      sparse_array_iterator & operator-=(sparse_array_iterator const & iter) {
         index -= iter.index;
         return *this;
      };

      sparse_array_iterator operator+(distance_type distance) {
         return sparse_array_iterator(index + distance);
      };

      sparse_array_iterator operator-(distance_type distance) {
         return sparse_array_iterator(index - distance);
      };

      bool operator==(sparse_array_iterator const & iter) {
         return index == iter.index ? true : false;
      };

      bool operator!=(sparse_array_iterator const & iter) {
         return index != iter.index ? true : false;
      };

      const_reference_type operator [] (distance_type distance)const {
         return this->pSparseMarray[index + distance];
      };

      const_reference_type operator *() {
         return (this->pSparseMarray->operator[](index));
      };
   };

   template<class T_SparseMarray>
   class sparse_array_iterator<T_SparseMarray, false > {
      friend class sparse_array_iterator<T_SparseMarray, true >;
   protected:
      typedef T_SparseMarray SparseMarrayType;
      typedef typename SparseMarrayType::value_type T;
   public:
      typedef typename T_SparseMarray::value_type value_type;
      typedef typename T_SparseMarray::reference_type reference_type;
      typedef typename T_SparseMarray::const_reference_type const_reference_type;
      typedef typename T_SparseMarray::param_type param_type;
      typedef SparseMarrayType * ptr_sparse_array_type; // constant pointer to changeable sparse array
      typedef typename SparseMarrayType::AccesProxyType AccesProxyType;
      typedef size_t distance_type;
   protected:
      ptr_sparse_array_type pSparseMarray;
      size_t index;
   public:

      sparse_array_iterator() {
      };

      sparse_array_iterator(ptr_sparse_array_type opengm, size_t _index) : pSparseMarray(opengm), index(_index) {
      };

      sparse_array_iterator(sparse_array_iterator<T_SparseMarray, true > & iter) {
         pSparseMarray = iter.pSparseMarray;
         index = iter.index;
      };

      sparse_array_iterator & operator=(sparse_array_iterator<T_SparseMarray, true > & iter) {
         index = iter.index;
         pSparseMarray = iter.pSparseMarray;
         return *this;
      };

      sparse_array_iterator & operator++() {
         index++;
         return *this;
      }

      sparse_array_iterator operator++(int) {
         index++;
         return sparse_array_iterator(*this);
      }

      sparse_array_iterator & operator--() {
         this.index--;
         return *this;
      }

      sparse_array_iterator operator--(int) {
         index--;
         return sparse_array_iterator(*this);
      }

      sparse_array_iterator & operator+=(sparse_array_iterator const & iter) {
         index += iter.index;
         return *this;
      };

      sparse_array_iterator & operator-=(sparse_array_iterator const & iter) {
         index -= iter.index;
         return *this;
      };

      sparse_array_iterator operator+(distance_type distance) {
         return sparse_array_iterator(index + distance);
      };

      sparse_array_iterator operator-(distance_type distance) {
         return sparse_array_iterator(index - distance);
      };

      bool operator==(sparse_array_iterator const & iter) {
         return index == iter.index ? true : false;
      };

      bool operator!=(sparse_array_iterator const & iter) {
         return index != iter.index ? true : false;
      };

      AccesProxyType operator [] (distance_type distance)const {
         return this->pSparseMarray[index + distance];
      };

      AccesProxyType operator *() {
         return this->pSparseMarray[index];
      };
   };
   /// \endcond
}
#endif
