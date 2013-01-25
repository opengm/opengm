#pragma once
#ifndef OPENGM_FUNCTORS_HXX
#define OPENGM_FUNCTORS_HXX

#include<cmath>

namespace opengm {
   
/// \cond HIDDEN_SYMBOLS

   /// functor to accumulate
   template<class ACC,class VALUE_TYPE>
   class AccumulationFunctor{
      public:
         AccumulationFunctor(const VALUE_TYPE v)
         :accValue_(v) {
         }
         /// constructor
         AccumulationFunctor()
         :accValue_(ACC::template neutral<VALUE_TYPE>()) {
         }
         /// do accumulation for one value
         /// \param v value to accumulate
         void operator()(const VALUE_TYPE v) {
            ACC::op(v,accValue_);
         }
         /// get accumulation value
         VALUE_TYPE value() {
            return accValue_;
         }
      private:
         VALUE_TYPE accValue_;
   };   
      
   /// functor to compute minimum and maximum at once
   template<class T>
   class MinMaxFunctor{
   public:
      /// constructor
      typedef T ValueType;
      MinMaxFunctor()
      :  first_(true),
         min_(T()),
         max_(T()) {
      }
      MinMaxFunctor(T min,T max)
      :  first_(false),
         min_(min),
         max_(max) {
      }
      /// get min
      ValueType min() {
         return min_;
      }
      /// get max
      ValueType max() {
         return max_;
      }
      /// check if value is min or max
      void operator()(const ValueType v) {
         if(first_) {
            min_=v;
            max_=v;
            first_=false;
         }
         else{
            if(v<min_) {
               min_=v;
            }
            if(v>max_) {
               max_=v;
            }
         }
      }
   private:
      bool first_;
      ValueType min_;
      ValueType max_;
   };
   
   /// functor to compute power
   template<class T>
   struct PowFunctor {
      /// constructor
      /// \param w take some value to the power of w
      template<class T_In>
      PowFunctor(T_In w)
      :  w_(w) {
      }
      /// \copy constructor
      PowFunctor(const PowFunctor& other)
      :  w_(other.w_) {
      }
      /// compute w's power of a value
      /// \value value to compute power of
      template<class T_In>
      const T operator()(T_In value) {
         return std::pow(value, w_);
      }
      T w_;
   };

   /// swap arguments of a binary functor
   template<class T_ReturnType, class T_Functor>
   class SwapArgumemtFunctor {
   public:
      ///  constructor
      /// \param other functor to swap arguments
      SwapArgumemtFunctor(const SwapArgumemtFunctor& other)
      :  functor_(other.functor_) {
      }
      /// constructor
      /// \param other functor to swap arguments
      SwapArgumemtFunctor(T_Functor functor)
      :  functor_(functor) {
      }
      /// operator
      /// \param a operant a
      /// \param b operant b
      template<class T_A, class T_B>
      T_ReturnType operator()(T_A a, T_B b) {
         return functor_(b, a);
      } 
   private:
      T_Functor functor_;
   };
   
   /// convert a binary functor to a unary functor
   template<class T_Scalar, class T_Functor, bool ScalarLeft>
   class BinaryToUnaryFunctor;
   
   /// convert a binary functor to a unary functor (if scalar is left operant)
   template<class T_Scalar, class T_Functor>
   class BinaryToUnaryFunctor<T_Scalar, T_Functor, true> {
   public:
      BinaryToUnaryFunctor(const BinaryToUnaryFunctor& other)
      :  functor_(other.functor_),
         scalar_(other.scalar_) {
      }
      BinaryToUnaryFunctor(const T_Scalar& scalar, T_Functor& functor)
      :functor_(functor),  
      scalar_(scalar) 
          {
      }
      template<class TIN>
      T_Scalar operator()(const TIN in) {
         return functor_(scalar_, in);
      }
   private:
      T_Functor functor_;
      T_Scalar scalar_;
      
   };
   
   /// convert a binary functor to a unary functor (if scalar is right operant)
   template<class T_Scalar, class T_Functor>
   class BinaryToUnaryFunctor<T_Scalar, T_Functor, false> {
   public:
      BinaryToUnaryFunctor(const BinaryToUnaryFunctor& other)
      :  functor_(other.functor_),
         scalar_(other.scalar_) {
      }
      BinaryToUnaryFunctor(const T_Scalar& scalar, T_Functor& functor)
      :functor_(functor),
       scalar_(scalar){
      }
      template<class TIN>
      T_Scalar operator()(const TIN in) {
         return functor_(in, scalar_);
      }
   private:
      T_Functor functor_;
      T_Scalar scalar_;
      
   };

   // a copy functor
   template<class OUT_ITERATOR>
   class CopyFunctor{
   public:
      CopyFunctor(OUT_ITERATOR iterator):outIterator_(iterator){}
      template<class T>
      void operator()(const T & value){
         (*outIterator_)=value;
         ++outIterator_;
      }
   private:
      OUT_ITERATOR outIterator_;
   };


/// \endcond

} // namespace opengm

#endif // #ifndef OPENGM_FUNCTORS_HXX
