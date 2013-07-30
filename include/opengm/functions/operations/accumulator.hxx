#pragma once
#ifndef OPENGM_FUNCTION_LEVEL_ACCUMULATOR_HXX
#define OPENGM_FUNCTION_LEVEL_ACCUMULATOR_HXX

#include "opengm/utilities/accumulation.hxx"
#include "opengm/utilities/accessor_iterator.hxx"
#include "opengm/utilities/shape_accessor.hxx"
#include "opengm/utilities/indexing.hxx"

namespace opengm {

/// \cond HIDDEN_SYMBOLS

/// accumulate over all variables of a function
template<class A, class B, class ACC>
class AccumulateAllImpl {
   typedef typename A::LabelType LabelType;
   typedef typename A::ValueType ValueType;

public:
   static void op(const A&, B&);
   static void op(const A&, B&, std::vector<LabelType>& state);
};

/// accumulate over some variables of a function
template<class A, class B, class ACC>
class AccumulateSomeImpl {
   typedef typename A::LabelType LabelType;
   typedef typename A::IndexType IndexType;
   typedef typename A::ValueType ValueType;
   typedef std::vector<IndexType> ViSequenceType;

public:
   template<class Iterator,class VIS_A,class VIS_B>
   static void op(const A&, const VIS_A &, Iterator, Iterator, B&, VIS_B&);
};

/// accumulate inplace over some variables of a function
template<class A, class ACC>
class AccumulateSomeInplaceImpl {
   typedef typename A::LabelType LabelType;
   typedef typename A::IndexType IndexType;
   typedef typename A::ValueType ValueType;
   typedef std::vector<IndexType> ViSequenceType;

public:
   template<class Iterator>
      static void op(A&, ViSequenceType&, Iterator, Iterator);
};

/// accumulate over all variables
template<class A, class B, class ACC>
void AccumulateAllImpl<A, B, ACC>::op
(
   const A& a,
   B& b
) {
   OPENGM_ASSERT(a.dimension() != 0 || (a.dimension() == 0 && a.size() == 1));
   opengm::Accumulation<ValueType, LabelType, ACC> acc;
   const size_t dimA = a.dimension();
   const size_t numElement = a.size();
   if(dimA != 0) {
      typedef opengm::AccessorIterator<opengm::FunctionShapeAccessor<A>, true> ShapeIterType;
      ShapeIterType shapeABegin(a, 0);
      opengm::ShapeWalker< ShapeIterType> shapeWalker(shapeABegin, dimA);
      const opengm::FastSequence<size_t> & coordinate = shapeWalker.coordinateTuple();
      for(size_t i=0; i<numElement ; ++i) {
         acc(a(coordinate.begin()));
         ++shapeWalker;
      }
   }
   else {
      size_t indexSequenceToScalar[] = {0};
      acc(a(indexSequenceToScalar));
   }
   b = static_cast<B>(acc.value());
}

/// accumulate over all variables also processing labels
template<class A, class B, class ACC>
void AccumulateAllImpl<A, B, ACC>::op
(
   const A& a,
   B& b,
   std::vector<typename AccumulateAllImpl<A, B, ACC>::LabelType>& state
) {
   OPENGM_ASSERT(a.dimension() != 0 || (a.dimension() == 0 && a.size() == 1));
   opengm::Accumulation<ValueType, LabelType, ACC> acc;
   const size_t dimA = a.dimension();
   const size_t numElement = a.size();
   if(dimA != 0) {
      state.resize(dimA);
      typedef opengm::AccessorIterator<opengm::FunctionShapeAccessor<A>, true> ShapeIterType;
      ShapeIterType shapeABegin(a, 0);
      opengm::ShapeWalker< ShapeIterType> shapeWalker(shapeABegin, dimA);
      const opengm::FastSequence<size_t> & coordinate = shapeWalker.coordinateTuple();
      for(size_t i=0; i<numElement; ++i) {
         acc(a(coordinate.begin()), coordinate);
         ++shapeWalker;
      }
      acc.state(state);
   }
   else {
      size_t indexSequenceToScalar[] = {0};
      acc(a(indexSequenceToScalar));
      state.clear();
   }
   b = static_cast<B>(acc.value());
}

/// accumulate over all variables
template<class A, class B, class ACC>
template<class Iterator,class VIS_A,class VIS_B>
void AccumulateSomeImpl<A, B, ACC>::op
(
   const A& a,
   const VIS_A & viA,
   Iterator viAccBegin,
   Iterator viAccEnd,
   B& b,
   VIS_B & viB
) {
   OPENGM_ASSERT(a.dimension() == viA.size());
   OPENGM_ASSERT(a.dimension() != 0 || (a.dimension() == 0 && a.size() == 1));
   const size_t dimA = a.dimension();
   viB.clear();
   b.assign();
   if(dimA != 0) {
      size_t rawViSize = std::distance(viAccBegin, viAccEnd);
      opengm::FastSequence<size_t> viAcc;
      opengm::FastSequence<size_t> shapeAcc;
      opengm::FastSequence<size_t> shapeNotAcc;
      opengm::FastSequence<size_t> notAccPosition;
      for(size_t i=0; i <dimA; ++i) {
         bool found = false;
         for(size_t j=0; j < rawViSize; ++j) {
            if(static_cast<opengm::UInt64Type>(viA[i]) == static_cast<opengm::UInt64Type>(viAccBegin[j])) {
               viAcc.push_back(viAccBegin[j]);
               shapeAcc.push_back(a.shape(i));
               found = true;
               break;
            }
         }
         if(!found) {
            viB.push_back(viA[i]);
            shapeNotAcc.push_back(a.shape(i));
            notAccPosition.push_back(i);
         }
      }
      if(shapeAcc.size() == dimA) {
         // accumulate over all variables ???
         ValueType scalarAccResult;
         AccumulateAllImpl<A, ValueType, ACC>::op(a, scalarAccResult);
         size_t indexSequenceToScalar[] = {0};
         size_t shapeToScalarArray[] = {0};
         b.resize(shapeToScalarArray, shapeToScalarArray);
         b(indexSequenceToScalar) = scalarAccResult;
      }
      else if(shapeAcc.size() == 0) {
         // accumulate over no variable
         // ==> copy function
         b.resize(shapeNotAcc.begin(), shapeNotAcc.end());
         opengm::ShapeWalker< opengm::FastSequence<size_t>::const_iterator> shapeWalker(shapeNotAcc.begin(), dimA);
         const opengm::FastSequence<size_t> & coordinate = shapeWalker.coordinateTuple();
         for(size_t i=0; i <a.size() ; ++i) {
            b(coordinate.begin()) = a(coordinate.begin());
            ++shapeWalker;
         }
         viB.assign(viA.begin(),viA.end());
      }
      else {
         // resize dstFactor
         b.resize(shapeNotAcc.begin(), shapeNotAcc.end());
         // create a shapeWalker to walk over NOTACC:
         opengm::ShapeWalker< opengm::FastSequence<size_t>::const_iterator> walker(shapeNotAcc.begin(), shapeNotAcc.size());
         const opengm::FastSequence<size_t> & coordinateNotAcc = walker.coordinateTuple();
         // create a subshape walker to walker over ACC
         typedef opengm::AccessorIterator<opengm::FunctionShapeAccessor<A>, true> ShapeIterType;
         ShapeIterType shapeABegin(a, 0);
         SubShapeWalker< ShapeIterType ,opengm::FastSequence<size_t>,opengm::FastSequence<size_t> >
         subWalker(shapeABegin, dimA, notAccPosition, coordinateNotAcc);
         const size_t subSizeAcc = subWalker.subSize();
         // loop over variables we don't want to accumulate over
         for(size_t i=0; i<b.size(); ++i) {
            // loop over the variables we want to accumulate over
            // create an accumulator
            Accumulation<ValueType, LabelType, ACC> acc;
            subWalker.resetCoordinate();
            for(size_t j=0; j < subSizeAcc; ++j) {
               acc(a( subWalker.coordinateTuple().begin()));
               ++subWalker;
            }
            b(coordinateNotAcc.begin()) = acc.value();
            ++walker;
         }
      }
   }
   else {
      Accumulation<ValueType, LabelType, ACC> acc;
      size_t indexSequenceToScalar[] = {0};
      ValueType accRes=static_cast<ValueType>(0.0);
      acc(accRes);
      size_t shapeToScalarArray[] = {0};
      b.resize(shapeToScalarArray, shapeToScalarArray);
      b(indexSequenceToScalar) = acc.value();
   }
   OPENGM_ASSERT(b.dimension() == viB.size());
   OPENGM_ASSERT(b.dimension() != 0 || (b.dimension() == 0 && b.size() == 1));
}

/// accumulate inplace over some variables
template<class A, class ACC>
template<class Iterator>
void AccumulateSomeInplaceImpl<A, ACC>::op
(
   A& a,
   ViSequenceType& viA,
   Iterator viAccBegin,
   Iterator viAccEnd
) {
   OPENGM_ASSERT(a.dimension() == viA.size());
   OPENGM_ASSERT(a.dimension() != 0 || (a.dimension() == 0 && a.size() == 1));
   const size_t dimA = a.dimension();
   opengm::FastSequence<size_t> viB;
   if(dimA != 0) {
      const size_t rawViSize = std::distance(viAccBegin, viAccEnd);
      opengm::FastSequence<size_t> viAcc;
      opengm::FastSequence<size_t> shapeAcc;
      opengm::FastSequence<size_t> shapeNotAcc;
      opengm::FastSequence<size_t> notAccPosition;
      for(size_t i = 0; i <dimA; ++i) {
         bool found = false;
         for(size_t j=0; j < rawViSize; ++j) {
            if( static_cast<UInt64Type>(viA[i]) == static_cast<UInt64Type>(viAccBegin[j])) {
               viAcc.push_back(viAccBegin[j]);
               shapeAcc.push_back(a.shape(i));
               found = true;
               break;
            }
         }
         if(!found) {
            viB.push_back(viA[i]);
            shapeNotAcc.push_back(a.shape(i));
            notAccPosition.push_back(i);
         }
      }
      if(shapeAcc.size() == dimA) {
         // accumulate over all variables
         ValueType scalarAccResult;
         AccumulateAllImpl<A, ValueType, ACC>::op(a, scalarAccResult);
         a.assign();
         size_t shapeToScalarArray[] = {0};
         a.resize(shapeToScalarArray, shapeToScalarArray);
         a(shapeToScalarArray) = scalarAccResult;
         viA.clear();
      }
      else if(shapeAcc.size() == 0) {
         // accumulate over no variable
         // do nothing
      }
      else {
         // resize dstFactor
         A b;
         b.resize(shapeNotAcc.begin(), shapeNotAcc.end());
         // create a shapeWalker to walk over NOTACC:
         opengm::ShapeWalker< typename opengm::FastSequence<size_t>::const_iterator> walker(shapeNotAcc.begin(), shapeNotAcc.size());
         const opengm::FastSequence<size_t> & coordinateNotAcc = walker.coordinateTuple();
         // create a subshape walker to walker over ACC
         typedef opengm::AccessorIterator<opengm::FunctionShapeAccessor<A>, true> ShapeIterType;
         ShapeIterType shapeABegin(a, 0);
         SubShapeWalker< ShapeIterType,opengm::FastSequence<size_t> ,opengm::FastSequence<size_t> >
         subWalker(shapeABegin, dimA, notAccPosition, coordinateNotAcc);
         const size_t subSizeAcc = subWalker.subSize();
         // loop over variables  we don't want to accumulate
         for(size_t i=0; i < b.size(); ++i) {
            // loop over the variables  we want to accumulate
            // create an accumulator
            Accumulation<ValueType, LabelType, ACC> acc;
            subWalker.resetCoordinate();
            for(size_t j=0; j<subSizeAcc; ++j) {
               acc(a(subWalker.coordinateTuple().begin()));
               ++subWalker;
            }
            b(coordinateNotAcc.begin()) = acc.value();
            ++walker;
         }
         a = b;
         viA.assign(viB.begin(),viB.end());
      }
   }
   else {
      Accumulation<ValueType, LabelType, ACC> acc;
      ValueType accRes=static_cast<ValueType>(0.0);
      acc(accRes);
      a.assign();
      a(0) = acc.value();
   }
   OPENGM_ASSERT(a.dimension() == viA.size());
   OPENGM_ASSERT(a.dimension() != 0 || (a.dimension() == 0 && a.size() == 1));
}

} // namespace opengm

/// \endcond exclude from reference documentation

#endif // OPENGM_FUNCTION_LEVEL_ACCUMULATOR_HXX
