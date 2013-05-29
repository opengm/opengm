#pragma once
#ifndef OPENGM_FUNCTION_LEVEL_OPERATOR_HXX
#define OPENGM_FUNCTION_LEVEL_OPERATOR_HXX

#include <vector>
#include <set>

#include "opengm/opengm.hxx"
#include "opengm/utilities/accessor_iterator.hxx"
#include "opengm/utilities/shape_accessor.hxx"
#include "opengm/utilities/indexing.hxx"

namespace opengm {

/// \cond HIDDEN_SYMBOLS

/// compute the shape and variable indices sequence of the result of a binary
/// operation on two functions
class ComputeViAndAShape {
public:
   typedef std::vector<size_t> ViSequenceType;

   template<class A, class B ,class VI_A,class VI_B,class VI_C,class SHAPE_C>
   static inline void computeViandShape
   (
      const VI_A & via,
      const VI_B & vib,
      VI_C & vic,
      const A& a,
      const B& b ,
      SHAPE_C & shapeC
   ) {
      OPENGM_ASSERT(a.dimension() == via.size());
      OPENGM_ASSERT(a.dimension() !=  0 || (a.dimension() == 0 && a.size() == 1));
      OPENGM_ASSERT(b.dimension() == vib.size());
      OPENGM_ASSERT(b.dimension() !=  0 || (b.dimension() == 0 && b.size() == 1));
      shapeC.clear();
      vic.clear();
      const size_t dimA = via.size();
      const size_t dimB = vib.size();
      vic.reserve(dimA+dimB);
      shapeC.reserve(dimA+dimB);
      if(via.size() == 0 && vib.size() == 0) {
      }
      else if(via.size() == 0) {
         vic.assign(vib.begin(),vib.end());
         for(size_t i=0;i<dimB;++i) {
            shapeC.push_back(b.shape(i));
         }
      }
      else if(vib.size() == 0) {
         vic.assign(via.begin(),via.end());
         for(size_t i=0;i<dimA;++i) {
            shapeC.push_back(a.shape(i));
         }
      }
      else {
         size_t ia = 0;
         size_t ib = 0;
         bool  first = true;
         while(ia<dimA || ib<dimB) {
            if(first==true) {
               first=false;
               if(via[ia]<=vib[ib]) {
                  vic.push_back(via[ia]);
                  shapeC.push_back(a.shape(ia));
                  ++ia;
               }
               else{
                  vic.push_back(vib[ib]);
                  shapeC.push_back(b.shape(ib));
                  ++ib;
               }
            }
            else if(ia>=dimA) {
               if(vic.back()!=vib[ib]) {
                  vic.push_back(vib[ib]);
                  shapeC.push_back(b.shape(ib));
               }
               ++ib;
            }
            else if(ib>=dimB) {
               if(vic.back()!=via[ia]) {
                  vic.push_back(via[ia]);
                  shapeC.push_back(a.shape(ia));
               }
               ++ia;
            }
            else if(via[ia]<=vib[ib]) {
               if(vic.back()!=via[ia] ) {
                  vic.push_back(via[ia]);
                  shapeC.push_back(a.shape(ia));
               }
               ++ia;
            }
            else{
               if(vic.back()!=vib[ib] ) {
                  vic.push_back(vib[ib]);
                  shapeC.push_back(b.shape(ib));
               }
               ++ib;
            }
         }
         OPENGM_ASSERT(ia == dimA);
         OPENGM_ASSERT(ib == dimB);
      }
   }
};

/// binary operation on two functions
template<class A, class B, class C, class OP>
class BinaryOperationImpl
{
public:
   template<class VI_A,class VI_B,class VI_C>
   static void op(const A& , const B& , C& , const VI_A& , const VI_B & , VI_C & , OP);
};

/// binary inplace operation on two functions
template<class A, class B, class OP>
class BinaryOperationInplaceImpl
{
public:
   template<class VI_A,class VI_B>
   static void op(A& , const B& , VI_A& , const VI_B&  , OP);
};

/// unary operation on a function
template<class A, class B, class OP>
class UnaryOperationImpl {
public:
   static void op(const A& a, B& b, OP);
};

/// unary inplace operation on a function
template<class A, class OP>
class UnaryOperationInplaceImpl {
public:
   static void op(A& a, OP);
};

/// binary operation on two functions
template<class A, class B, class C, class OP>
template<class VI_A,class VI_B,class VI_C>
void BinaryOperationImpl<A, B, C, OP>::op
(
   const A& a,
   const B& b,
   C& c,
   const VI_A & via,
   const VI_B & vib,
   VI_C & vic,
   OP op
) {
   OPENGM_ASSERT(a.dimension() == via.size());
   OPENGM_ASSERT(a.dimension() !=  0 || (a.dimension() == 0 && a.size() == 1));
   OPENGM_ASSERT(b.dimension() == vib.size());
   OPENGM_ASSERT(b.dimension() !=  0 || (b.dimension() == 0 && b.size() == 1));
   // clear c
   c.assign();
   // compute output vi's and shape of c
   opengm::FastSequence<typename C::LabelType> shapeC;
   ComputeViAndAShape::computeViandShape(via, vib, vic, a, b, shapeC);
   OPENGM_ASSERT(shapeC.size() == vic.size());
   // reshape c
   c.resize(shapeC.begin(), shapeC.end());
   // get dimensions and number of Elements in c
   const size_t dimA = a.dimension();
   const size_t dimB = b.dimension();
   //const size_t dimC = c.dimension();
   const size_t numElemmentC = c.size();
   typedef typename opengm::FastSequence<typename C::LabelType>::ConstIteratorType FIterType;
   if(dimA !=  0 && dimB !=  0) {
      opengm::TripleShapeWalker<FIterType > shapeWalker(shapeC.begin(),shapeC.size(), vic, via, vib);
      for(size_t i=0; i<numElemmentC; ++i) {
         OPENGM_ASSERT(a.dimension() == shapeWalker.coordinateTupleA().size());
         OPENGM_ASSERT(b.dimension() == shapeWalker.coordinateTupleB().size());
         OPENGM_ASSERT(c.dimension() == shapeWalker.coordinateTupleAB().size());
         c(shapeWalker.coordinateTupleAB().begin()) = op(a(shapeWalker.coordinateTupleA().begin()), b(shapeWalker.coordinateTupleB().begin()));
         ++shapeWalker;
      }
   }
   else if(dimA == 0 && dimB == 0) {
      const size_t scalarIndex=0;
      c.resize(&scalarIndex, &scalarIndex+1);
      c(&scalarIndex) = op(a(&scalarIndex), b(&scalarIndex));
   }
   else if(dimA == 0) {
      opengm::ShapeWalker<FIterType > shapeWalker(shapeC.begin(),shapeC.size());
      const size_t scalarIndex=0;
      for(size_t i=0; i<numElemmentC; ++i) {
         c(shapeWalker.coordinateTuple().begin()) = op(a(&scalarIndex), b(shapeWalker.coordinateTuple().begin()));
         ++shapeWalker;
      }
   }
   else { // DimB == 0
      opengm::ShapeWalker<FIterType > shapeWalker(shapeC.begin(),shapeC.size());
      const size_t scalarIndex=0;
      for(size_t i=0; i<numElemmentC; ++i) {
         c(shapeWalker.coordinateTuple().begin()) = op(a(shapeWalker.coordinateTuple().begin()), b(&scalarIndex));
         ++shapeWalker;
      }
   }
   OPENGM_ASSERT(a.dimension() == via.size());
   OPENGM_ASSERT(a.dimension() !=  0 || (a.dimension() == 0 && a.size() == 1));
   OPENGM_ASSERT(b.dimension() == vib.size());
   OPENGM_ASSERT(b.dimension() !=  0 || (b.dimension() == 0 && b.size() == 1));
   OPENGM_ASSERT(c.dimension() == vic.size());
   OPENGM_ASSERT(c.dimension() !=  0 || (c.dimension() == 0 && c.size() == 1));
}

/// binary inplace  operation on two functions
template<class A, class B, class OP>
template<class VI_A,class VI_B>
void BinaryOperationInplaceImpl<A, B, OP>::op
(
   A& a,
   const B& b,
   VI_A & via,
   const VI_B & vib,
   OP op
)
{
   OPENGM_ASSERT(a.dimension() == via.size());
   OPENGM_ASSERT(a.dimension() !=  0 || (a.dimension() == 0 && a.size() == 1));
   OPENGM_ASSERT(b.dimension() == vib.size());
   OPENGM_ASSERT(b.dimension() !=  0 || (b.dimension() == 0 && b.size() == 1));
   // compute output vi's and shape of a(new)
   opengm::FastSequence<size_t> shapeANew;
   opengm::FastSequence<size_t> viaNew;
   ComputeViAndAShape::computeViandShape(via, vib, viaNew, a, b, shapeANew);
   OPENGM_ASSERT(shapeANew.size() == viaNew.size());
   // in-place
   if(viaNew.size() == via.size()) {
      if(viaNew.size() !=  0) {
         if(vib.size() !=  0) {
            const size_t numElementInA = a.size();
            opengm::DoubleShapeWalker<opengm::FastSequence<size_t>::const_iterator > shapeWalker(shapeANew.begin(),shapeANew.size(), viaNew, vib);
            for(size_t i=0; i<numElementInA; ++i) {
               a(shapeWalker.coordinateTupleAB().begin()) = op(a(shapeWalker.coordinateTupleAB().begin()), b(shapeWalker.coordinateTupleA().begin()));
               ++shapeWalker;
            }
         }
         else {
            const size_t numElementInA = a.size();
            opengm::DoubleShapeWalker<opengm::FastSequence<size_t>::const_iterator > shapeWalker(shapeANew.begin(),shapeANew.size(), viaNew, vib);            const size_t scalarIndex = 0;
            for(size_t i=0; i<numElementInA; ++i) {
               a(shapeWalker.coordinateTupleAB().begin()) = op(a(shapeWalker.coordinateTupleAB().begin()), b(&scalarIndex));
               ++shapeWalker;
            }
         }
      }
      else {
         const size_t scalarIndex=0;
         a.resize(&scalarIndex, &scalarIndex+1);
         a(&scalarIndex) = op(a(&scalarIndex), b(&scalarIndex));
         via.assign(viaNew.begin(),viaNew.end());
      }
   }
   // not inplace
   else {
      A aNew;
      BinaryOperationImpl<A, B, A, OP>::op(a, b, aNew, via, vib, viaNew, op);
      a = aNew;
      via.assign(viaNew.begin(),viaNew.end());
   }
   OPENGM_ASSERT(a.dimension() == via.size());
   OPENGM_ASSERT(a.dimension() !=  0 || (a.dimension() == 0 && a.size() == 1));
   OPENGM_ASSERT(b.dimension() == vib.size());
   OPENGM_ASSERT(b.dimension() !=  0 || (b.dimension() == 0 && b.size() == 1));
}

/// unary operation on a function
template<class A, class B, class OP>
void UnaryOperationImpl<A, B, OP>::op
(
   const A& a,
   B& b,
   OP op
)
{
   OPENGM_ASSERT(a.dimension() !=  0 || (a.dimension() == 0 && a.size() == 1));
   // clear b
   b.assign();
   // get dimensions and number of Elements in b
   const size_t dimA = a.dimension();
   if(dimA !=  0) {
      // reshape b
      typedef opengm::AccessorIterator<opengm::FunctionShapeAccessor<A>, true> ShapeIterType;
      ShapeIterType shapeABegin(a, 0);
      ShapeIterType shapeAEnd(a, dimA);
      b.resize(shapeABegin, shapeAEnd);
      opengm::ShapeWalker< ShapeIterType> shapeWalker(shapeABegin, dimA);
      const size_t numElemmentA = a.size();
      for(size_t i=0; i<numElemmentA; ++i) {
         b(shapeWalker.coordinateTuple().begin()) = op(a(shapeWalker.coordinateTuple().begin()));
         ++shapeWalker;
      }
   }
   else {
      const size_t scalarIndex=0;
      b.resize(&scalarIndex, &scalarIndex+1);
      b(&scalarIndex) = op(a(&scalarIndex));
   }
}

/// unary inplace operation on a function
template<class A, class OP>
void UnaryOperationInplaceImpl<A, OP>::op
(
   A& a,
   OP op
)
{
   OPENGM_ASSERT(a.dimension() !=  0 || (a.dimension() == 0 && a.size() == 1));
   const size_t dimA = a.dimension();
   if(dimA !=  0) {
      typedef opengm::AccessorIterator<opengm::FunctionShapeAccessor<A>, true> ShapeIterType;
      ShapeIterType shapeABegin(a, 0);
      opengm::ShapeWalker< ShapeIterType> shapeWalker(shapeABegin, dimA);
      const size_t numElemmentA = a.size();
      for(size_t i=0; i<numElemmentA; ++i) {
         a(shapeWalker.coordinateTuple().begin()) = op(a(shapeWalker.coordinateTuple().begin()));
         ++shapeWalker;
      }
   }
   else {
      const size_t scalarIndex=0;
      a(&scalarIndex) = op(a(&scalarIndex));
   }
}

/// \endcond

} // namespace opengm

#endif // #ifndef OPENGM_FUNCTION_LEVEL_OPERATOR_HXX
