#pragma once
#ifndef OPENGM_INDEXING_HXX
#define OPENGM_INDEXING_HXX

#include "opengm/opengm.hxx"
#include "opengm/datastructures/fast_sequence.hxx"

namespace opengm {
   /// \cond HIDDEN_SYMBOLS
   template<class VECTOR>
   bool isEqualValueVector(const VECTOR vector) {
      for(size_t i=0;i<vector.size();++i) {
         if(vector[0]!=vector[i]) {
            return false;
         }
      }
      return true;
   }
   
   /// walk over a factor / function , and some variables can be  fixed
   template<class Iterator,class FIXED_COORDINATE_INDEX_CONTAINER,class FIXED_COORDINATE_VALUE_CONTAINER>
   class SubShapeWalker{
   public:
      /// constructor
      /// \param shapeBegin begin of the shape of the function
      /// \param dimension size of the function / shape
      /// \param fixedCoordinateIndex container/ random access iterator which contains the position of the variables to fix
      /// \param fixedCoordinateValue container/ random access iterator which contains the values of the variables which are fixed
      SubShapeWalker
      (
         Iterator shapeBegin,
         const size_t dimension,
         const FIXED_COORDINATE_INDEX_CONTAINER & fixedCoordinateIndex,
         const FIXED_COORDINATE_VALUE_CONTAINER & fixedCoordinateValue
      )
      : shapeBegin_(shapeBegin),
      coordinateTuple_(dimension,0),
      fixedCoordinateValue_(fixedCoordinateValue),
      fixedCoordinateIndex_(fixedCoordinateIndex),
      dimension_(dimension) {
         for(size_t d = 0; d < fixedCoordinateIndex_.size(); ++d) {
            coordinateTuple_[fixedCoordinateIndex_[d]] = fixedCoordinateValue_[d];
         }
      };

      /// reset coordinate to zeros
      void  resetCoordinate() {
         for(size_t i = 0; i < dimension_; ++i) {
            coordinateTuple_[i] = static_cast<size_t>(0);
         }
         for(size_t i = 0; i < fixedCoordinateIndex_.size(); ++i) {
            coordinateTuple_[fixedCoordinateIndex_[i]] = fixedCoordinateValue_[i];
         }
      };

      /// increment coordinate in last - coordinate major order  (therefore the first coordinate is incremented first)
      inline SubShapeWalker & operator++() {
         size_t counter = 0;
         for(size_t d = 0; d < dimension_; ++d) {
            bool atFixedValue = false;
            for(size_t i = counter; i < fixedCoordinateIndex_.size(); ++i) {
               if(d == fixedCoordinateIndex_[i]) {
                  atFixedValue = true;
                  ++counter;
               }
            }
            if(atFixedValue == false) {
               if(coordinateTuple_[d] != shapeBegin_[d] - 1) {
                  coordinateTuple_[d]++;
                  break;
               }
               else {
                  if(d != dimension_ - 1) {
                     coordinateTuple_[d] = 0;
                  }
                  else {
                     coordinateTuple_[d]++;
                     break;
                  }
               }
            }
         }
         return *this;
      };

      /// get the coordinate tuple
      const opengm::FastSequence<size_t> & coordinateTuple()const {
            return coordinateTuple_;
      };
      
      /// get the number of elements over which one wants to walk
      size_t subSize() {
         size_t result = 1;
         size_t counter = 0;
         for(size_t d = 0; d < dimension_; ++d) {
            bool fixedVal = false;
            for(size_t i = counter; i < fixedCoordinateIndex_.size(); ++i) {
               if(d == fixedCoordinateIndex_[i]) ///??? replace i with counter for old version
               {
                  fixedVal = true;
                  counter++;
                  break;
               }
            }
            if(fixedVal == false) {
               result *= shapeBegin_[d];
            }
         }
         return result;
      }

   private:
      Iterator  shapeBegin_;
      opengm::FastSequence<size_t> coordinateTuple_;
      const FIXED_COORDINATE_VALUE_CONTAINER & fixedCoordinateValue_;
      const FIXED_COORDINATE_INDEX_CONTAINER & fixedCoordinateIndex_;
      const size_t dimension_;
      //size_t subSize_;
   };

   template<class Iterator>
   class ShapeWalker
   {
   public:
      ShapeWalker(Iterator shapeBegin,size_t dimension)
      : shapeBegin_(shapeBegin),
      coordinateTuple_(dimension, 0),
      dimension_(dimension) { }
      ShapeWalker & operator++() {
         for(size_t d = 0; d < dimension_; ++d) {
            if( size_t(coordinateTuple_[d]) != (size_t(shapeBegin_[d]) - size_t(1)) ) {
               ++coordinateTuple_[d];
               OPENGM_ASSERT(coordinateTuple_[d]<shapeBegin_[d]);
               break;
            }
            else {
               if(d != dimension_ - 1) {
                  coordinateTuple_[d] = 0;
               }
               else {
                  coordinateTuple_[d]++;
                  break;
               }
            }
         }
         return *this;
      };
      const opengm::FastSequence<size_t> & coordinateTuple()const {
            return coordinateTuple_;
      };

   private:
      Iterator shapeBegin_;
      opengm::FastSequence<size_t> coordinateTuple_;
      const size_t dimension_;
   };
   
   template<class Iterator>
   class ShapeWalkerSwitchedOrder
   {
   public:
      ShapeWalkerSwitchedOrder(Iterator shapeBegin,size_t dimension)
      : shapeBegin_(shapeBegin),
      coordinateTuple_(dimension, 0),
      dimension_(dimension) { }
      ShapeWalkerSwitchedOrder & operator++() {
         for(size_t d = dimension_-1; true; --d) {
            if( size_t(coordinateTuple_[d]) != (size_t(shapeBegin_[d]) - size_t(1)) ) {
               ++coordinateTuple_[d];
               OPENGM_ASSERT(coordinateTuple_[d]<shapeBegin_[d]);
               break;
            }
            else {
               if(d != 0) {
                  coordinateTuple_[d] = 0;
               }
               else {
                  coordinateTuple_[d]++;
                  break;
               }
            }
            //if(d==0){
            //   break;
            //}
         }
         return *this;
      };
      const opengm::FastSequence<size_t> & coordinateTuple()const {
            return coordinateTuple_;
      };

   private:
      Iterator shapeBegin_;
      opengm::FastSequence<size_t> coordinateTuple_;
      const size_t dimension_;
   };

   template<class SHAPE_AB_ITERATOR>
   class TripleShapeWalker{
   public:
      template<class VI_AB,class VI_A,class VI_B>
      TripleShapeWalker
      (
         SHAPE_AB_ITERATOR  shapeABBegin,
         const size_t dimAB,
         const VI_AB & viAB,
         const VI_A & viA,
         const VI_B & viB
      ): shapeABBegin_(shapeABBegin),
      dimensionAB_(dimAB),
      coordinateTupleAB_(viAB.size(), 0),
      coordinateTupleA_(viA.size(), 0),
      coordinateTupleB_(viB.size(), 0),
      viMatchA_(viAB.size(), false),
      viMatchB_(viAB.size(), false),
      viMatchIndexA_(viAB.size()),
      viMatchIndexB_(viAB.size()) {
         OPENGM_ASSERT(dimAB == viAB.size());
         OPENGM_ASSERT( viA.size() != 0);
         OPENGM_ASSERT( viB.size() != 0);
         //vi matching:
         size_t counterA = 0;
         size_t counterB = 0;
         for(size_t d = 0; d < dimensionAB_; ++d) {
            if(counterA<viA.size()) {
               if(viAB[d] == viA[counterA]) {
                  viMatchA_[d] = true;
                  viMatchIndexA_[d] = counterA;
                  counterA++;
               }
            }
            if(counterB<viB.size()) {
               if(viAB[d] == viB[counterB]) {
                  viMatchB_[d] = true;
                  viMatchIndexB_[d] = counterB;
                  counterB++;
               }
            }
         }
      }

      TripleShapeWalker & operator++() {
         for(size_t d = 0; d < dimensionAB_; ++d) {
            if( int (coordinateTupleAB_[d]) != int( int(shapeABBegin_[d]) - int(1))) {
               coordinateTupleAB_[d]++;
               if(viMatchA_[d]) {
                  coordinateTupleA_[viMatchIndexA_[d]]++;
               }
               if(viMatchB_[d]) {
                  coordinateTupleB_[viMatchIndexB_[d]]++;
               }
               break;
            }
            else {
               coordinateTupleAB_[d] = 0;
               if(viMatchA_[d]) {
                  coordinateTupleA_[viMatchIndexA_[d]] = 0;
               }
               if(viMatchB_[d]) {
                  coordinateTupleB_[viMatchIndexB_[d]] = 0;
               }
            }
         }
         return *this;
      };

      const opengm::FastSequence<size_t> &coordinateTupleA()const {
         return coordinateTupleA_;
      };

      const opengm::FastSequence<size_t> & coordinateTupleB()const {
         return coordinateTupleB_;
      };

      const opengm::FastSequence<size_t> & coordinateTupleAB()const {
         return coordinateTupleAB_;
      };
   private:
      SHAPE_AB_ITERATOR shapeABBegin_;
      const size_t dimensionAB_;
      opengm::FastSequence<size_t> coordinateTupleAB_;
      opengm::FastSequence<size_t> coordinateTupleA_;
      opengm::FastSequence<size_t> coordinateTupleB_;
      opengm::FastSequence<bool> viMatchA_;
      opengm::FastSequence<bool> viMatchB_;
      opengm::FastSequence<size_t> viMatchIndexA_;
      opengm::FastSequence<size_t> viMatchIndexB_;
   };

   template<class SHAPE_AB_ITERATOR>
   class DoubleShapeWalker {
   public:
      template<class VI_A,class VI_B>
      DoubleShapeWalker
      (
         SHAPE_AB_ITERATOR shapeABbegin,
         const size_t dimAb,
         const VI_A & viAB,
         const VI_B & viA
      )
      :shapeABbegin_(shapeABbegin),
      dimensionAB_(dimAb),
      coordinateTupleAB_(dimensionAB_, 0),
      coordinateTupleA_(viA.size(), 0),
      viMatchA_(dimensionAB_, false),
      viMatchIndexA_(dimensionAB_) {
         //vi matching:
         size_t counterA = 0;
         for(size_t d = 0; d < dimensionAB_; ++d) {
            for(size_t i = counterA; i < viA.size(); ++i) {
               if(viAB[d] == viA[i]) {
                  viMatchA_[d] = true;
                  viMatchIndexA_[d] = i;
                  ++counterA;
               }
            }
         }
      }

      DoubleShapeWalker & operator++() {
         for(size_t d = 0; d < dimensionAB_; ++d) {
            if(coordinateTupleAB_[d] != shapeABbegin_[d] - 1) {
               coordinateTupleAB_[d]++;
               if(viMatchA_[d] == true) {
                  coordinateTupleA_[viMatchIndexA_[d]]++;
               }
               break;
            }
            else {
               coordinateTupleAB_[d] = 0;
               if(viMatchA_[d] == true) {
                  coordinateTupleA_[viMatchIndexA_[d]] = 0;
               }
            }
         }
         return *this;
      };

      const opengm::FastSequence<size_t> & coordinateTupleA()const {
         return coordinateTupleA_;
      };

      const opengm::FastSequence<size_t> & coordinateTupleAB()const {
            return coordinateTupleAB_;
      };

   private:
      SHAPE_AB_ITERATOR shapeABbegin_;
      const size_t dimensionAB_;
      opengm::FastSequence<size_t> coordinateTupleAB_;
      opengm::FastSequence<size_t> coordinateTupleA_;
      opengm::FastSequence<bool> viMatchA_;
      opengm::FastSequence<size_t> viMatchIndexA_;
   };

} // namespace opengm

#endif // #ifndef OPENGM_INDEXING_HXX
