#pragma once
#ifndef OPENGM_MESSAGE_PASSING_BUFFER_HXX
#define OPENGM_MESSAGE_PASSING_BUFFER_HXX

/// \cond HIDDEN_SYMBOLS

namespace opengm {

   template<class ARRAY>
   class MessageBuffer {
   public:
      typedef ARRAY ArrayType;
      typedef typename ARRAY::ValueType ValueType;

      // construction and assignment
      MessageBuffer();
      template<class SHAPE_ITERATOR>
      MessageBuffer(SHAPE_ITERATOR, SHAPE_ITERATOR, const ValueType& = ValueType());
      template<class SHAPE_ITERATOR>
      void assign(SHAPE_ITERATOR, SHAPE_ITERATOR, const ValueType& = ValueType()); 
      template<class SHAPE>
      void assign(SHAPE, const ValueType& = ValueType());
/*
      template<class GRAPHICAL_MODEL, class INDEX_ITERATOR>
      MessageBuffer(const GRAPHICAL_MODEL& , INDEX_ITERATOR, INDEX_ITERATOR, const ValueType& = ValueType());
      template<class GRAPHICAL_MODEL, class INDEX_ITERATOR>
      void assign(const GRAPHICAL_MODEL& , INDEX_ITERATOR, INDEX_ITERATOR, const ValueType& = ValueType());
*/
      // query
      const ARRAY& current() const;
      const ARRAY& old() const;
      template<class DIST>
      ValueType dist() const; // distance between current and old

      ARRAY& current();
      ARRAY& old();

      // manipulation
      void toggle();

   private:
      bool flag_;
      ARRAY buffer1_;
      ARRAY buffer2_;
   };


   //**********************
   // IMPLEMENTATION
   //**********************

   template<class ARRAY>
   inline MessageBuffer<ARRAY>::MessageBuffer()
   {}

   template<class ARRAY>
   template<class SHAPE_ITERATOR>
   inline  MessageBuffer<ARRAY>::MessageBuffer
   (
      SHAPE_ITERATOR begin,
      SHAPE_ITERATOR end,
      const typename ARRAY::ValueType& constant
   ) {
      assign(begin, end, constant);
   }
/*
   template<class ARRAY>
   template<class GRAPHICAL_MODEL, class INDEX_ITERATOR>
   inline
   MessageBuffer<ARRAY>::MessageBuffer
   (
      const GRAPHICAL_MODEL& gm,
      INDEX_ITERATOR begin,
      INDEX_ITERATOR end,
      const typename ARRAY::ValueType& constant
   ) {
      assign(gm, begin, end, constant);
   }
*/
   template<class ARRAY>
   template<class SHAPE_ITERATOR>
   inline void  MessageBuffer<ARRAY>::assign
   (
      SHAPE_ITERATOR begin,
      SHAPE_ITERATOR end,
      const typename ARRAY::ValueType& constant
   )
   {
      if(begin == end) {
         buffer1_ = constant;
         buffer2_ = constant;
      }
      else {
         buffer1_.assign(begin, end, constant);
         buffer2_.assign(begin, end, constant);
      }
      flag_ = false;
   } 

   template<class ARRAY>
   template<class SHAPE>
   inline void  MessageBuffer<ARRAY>::assign
   (
      SHAPE shape,
      const typename ARRAY::ValueType& constant
   )
   {
      if(shape == 0) {
         buffer1_ = constant;
         buffer2_ = constant;
      }
      else {
         buffer1_.resize(&shape, &shape+1, constant);
         buffer2_.resize(&shape, &shape+1, constant);
      }
      flag_ = false;
   }
/*
   template<class ARRAY>
   template<class GRAPHICAL_MODEL, class INDEX_ITERATOR>
   inline void  MessageBuffer<ARRAY>::assign
   (
      const GRAPHICAL_MODEL& gm,
      INDEX_ITERATOR begin,
      INDEX_ITERATOR end,
      const typename ARRAY::ValueType& constant
   )
   {
      if(begin == end) {
         buffer1_.assign(constant);
         buffer2_.assign(constant);
      }
      else {
         buffer1_.assign(gm, begin, end, constant);
         buffer2_.assign(gm, begin, end, constant);
      }
      flag_ = false;
   }
*/

   template<class ARRAY>
   inline ARRAY& MessageBuffer<ARRAY>::current() {
      return flag_ ? buffer1_ : buffer2_;
   }

   template<class ARRAY>
   inline const ARRAY& MessageBuffer<ARRAY>::current() const {
      return flag_ ? buffer1_ : buffer2_;
   }

   template<class ARRAY>
   inline ARRAY& MessageBuffer<ARRAY>::old() {
      return flag_ ? buffer2_ : buffer1_;
   }

   template<class ARRAY>
   inline const ARRAY& MessageBuffer<ARRAY>::old() const {
      return flag_ ? buffer2_ : buffer1_;
   }

   template<class ARRAY>
   inline void MessageBuffer<ARRAY>::toggle() {
      flag_ = flag_ ? false : true;
   }

   template<class ARRAY>
   template<class DIST>
   inline typename ARRAY::ValueType MessageBuffer<ARRAY>::dist() const {
      return DIST::op(buffer1_, buffer2_);
   }
}

/// \endcond

#endif // #ifndef OPENGM_MESSAGE_PASSING_BUFFER_HXX
