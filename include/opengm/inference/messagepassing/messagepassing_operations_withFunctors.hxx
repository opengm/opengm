#pragma once
#ifndef OPENGM_MESSAGEPASSING_OPERATIONS_HXX
#define OPENGM_MESSAGEPASSING_OPERATIONS_HXX

#include <opengm/opengm.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/operations/multiplier.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/operations/maximizer.hxx>

/// \cond HIDDEN_SYMBOLS

namespace opengm {
   namespace messagepassingOperations {

// out = M(M.shape, OP:neutral)
/// \todo unroll loop
      template<class OP, class M>
      inline void clean(M& out) {
         for(size_t n=0; n<out.size(); ++n ) {
            OP::neutral(out(n));
         }
      }
      
      template<class OP, class ACC, class M>
      inline void normalize
      (
         M& out
         ) {
         typename M::ValueType v;
         ACC::neutral(v);
         for(size_t n=0; n<out.size(); ++n)
            ACC::op(out(n),v);
         
         if( opengm::meta::Compare<OP,opengm::Multiplier>::value && v <= 0.00001)
            return;
         if(opengm::meta::Compare<OP,opengm::Multiplier>::value)
            OPENGM_ASSERT(v > 0.00001); // ??? this should be checked in released code   
         for(size_t n=0; n<out.size();++n ) {
            OP::iop(v,out(n));      
         }
      }

/// out = op( hop(in1,alpha), hop(in2,1-alpha) )
      template<class OP, class M, class T>
      inline void weightedMean
      (
         const M& in1,
         const M& in2, 
         const T alpha, 
         M& out
         ) { 
         /// TODO
         /// Speedup
         T v1,v2;
         const T oneMinusAlpha=static_cast<T>(1)-alpha;
        
         for(size_t n=0; n<out.size();++n ) {
            OP::hop(in1(n),alpha,  v1);
            OP::hop(in2(n),oneMinusAlpha,v2);
            OP::op(v1,v2,out(n));
         }
      }
      
/// out = op(vec[0].current, ..., vec[n].current ) 
      template<class OP, class BUFFER, class M>
      inline void operate
      (
         const std::vector<BUFFER>& vec,
         M& out
         ) {
         /// ???
         /// switch order of loops ?
         clean<OP>(out);
         for(size_t j = 0; j < vec.size(); ++j) {
            const typename BUFFER::ArrayType& b = vec[j].current();
            OPENGM_ASSERT(b.size()==out.size());
            for(size_t n=0; n<out.size(); ++n) 
               OP::op(b(n), out(n));
         }
      }  
      
/// out = op( out , op_i( hop(vec[i],rho[i]) ) )
      template<class GM, class BUFFER, class M>
      inline void operateW
      (
         const std::vector<BUFFER>& vec,
         const std::vector<typename GM::ValueType>& rho,
         M& out
         ) {
         typedef typename GM::OperatorType OP;
         clean<OP>(out);
         /// ???
         /// switch order of loops ?
         /// => loop over out?
         for(size_t j = 0; j < vec.size(); ++j) {
            const typename BUFFER::ArrayType& b = vec[j].current();
            typename GM::ValueType e = rho[j];
            typename GM::ValueType v;
            for(size_t n=0; n<out.size(); ++n) {
               OP::hop(b(n),e,v);
               OP::op(v,out(n));
            }
         }
      }
   
/// out = op( vec[0].current, ..., vec[i-1].current,vec[i+1].current, ... , vec[n].current ) 
      template<class OP, class BUFVEC, class M, class INDEX>
      inline void operate
      (
         const BUFVEC& vec,
         const INDEX i, 
         M& out
         ) {
         clean<OP>(out); 
         /// TODO
         /// switch order of loops ? (loop over out?)
         /// => clean could be inside the loop over the result
         for(size_t j = 0; j < i; ++j) {
            const M& f = vec[j].current();
            for(size_t n=0; n<out.size(); ++n) 
               OP::op(f(n), out(n));
         }
         for(size_t j = i+1; j < vec.size(); ++j) {
            const M& f = vec[j].current();
            for(size_t n=0; n<out.size(); ++n) 
               OP::op(f(n), out(n));        
         }
      }
 
/// out = op( hop(vec[i],rho[i]-1), op_j/i( hop(vec[j],rho[j]) ) )
      template<class GM, class BUFVEC, class M, class INDEX>
      inline void operateW
      (
         const BUFVEC& vec, 
         const INDEX i, 
         const std::vector<typename GM::ValueType>& rho,
         M& out
         ) {  
         typedef typename GM::OperatorType OP;
         OPENGM_ASSERT(vec[i].current().size()==out.size());
         typename GM::ValueType v;
         const typename GM::ValueType e = rho[i]-1;
         const M& b = vec[i].current();
         for(size_t n=0; n<out.size(); ++n) {
            //OP::hop(b(n),e,v);
            //OP::op(v,out(n));
            OP::hop(b(n),e,out(n));
         }
         
         for(size_t j = 0; j < i; ++j) {
            const M& b = vec[j].current();
            const typename GM::ValueType e = rho[j];
            OPENGM_ASSERT(b.size()==out.size());
            for(size_t n=0; n<out.size(); ++n) {
               OP::hop(b(n),e,v);
               OP::op(v,out(n));
            }
         } 
         for(size_t j = i+1; j < vec.size(); ++j) {
            const M& b = vec[j].current();
            const typename GM::ValueType e = rho[j];
            OPENGM_ASSERT(b.size()==out.size());
            for(size_t n=0; n<out.size(); ++n) {
               OP::hop(b(n),e,v);
               OP::op(v,out(n));
            }
         }       
      }
      
/// out = acc( op(f, vec[0].current, ..., vec[n].current ), -i) 

      template<class GM, class ACC, class BUFVEC, class ARRAY ,class INDEX>
      struct OperateF_Functor{
         OperateF_Functor(
            const BUFVEC & vec,
            const INDEX i,
            ARRAY & out
            )
            : vec_(vec),
              i_(i),
              out_(out){
         }

         template<class FUNCTION>
         void operator()(const FUNCTION & f){
            typedef typename GM::OperatorType OP;
            if(f.dimension()==2) {
               size_t count[2];
               typename GM::ValueType v;
               for(size_t n=0; n<out_.size(); ++n)
                  ACC::neutral(out_(n));
               if(i_==0){
                  for(count[0]=0;count[0]<f.shape(0);++count[0]){
                     for(count[1]=0;count[1]<f.shape(1);++count[1]) {
                        v = f(count);
                        OP::op(vec_[1].current()(count[1]), v);
                        ACC::op(v,out_(count[0]));
                     }
                  }
               }else{ 
                  for(count[0]=0;count[0]<f.shape(0);++count[0]){
                     for(count[1]=0;count[1]<f.shape(1);++count[1]) {
                        v = f(count);
                        OP::op(vec_[0].current()(count[0]), v);
                        ACC::op(v,out_(count[1]));
                     } 
                  }
               }
            }
            else{
               // accumulation over all variables except x
               typedef typename GM::IndexType IndexType;
               typedef typename GM::LabelType LabelType;
               // neutral initialization of output
               for(size_t n=0; n<f.shape(i_); ++n)
                  ACC::neutral(out_(n));
               // factor shape iterator
               typedef typename FUNCTION::FunctionShapeIteratorType FunctionShapeIteratorType;
               opengm::ShapeWalker<FunctionShapeIteratorType> shapeWalker(f.functionShapeBegin(),f.dimension());
               for(IndexType scalarIndex=0;scalarIndex<f.size();++scalarIndex,++shapeWalker) {
                  // loop over the variables
                  // initialize output value with value of the factor at this coordinate
                  // operate j=[0,..i-1]
                  typename GM::ValueType value=f(shapeWalker.coordinateTuple().begin());
                  for(IndexType j=0;j<static_cast<typename GM::IndexType>(i_);++j) {
                     const LabelType label=static_cast<LabelType>(shapeWalker.coordinateTuple()[j]);
                     OP::op(vec_[j].current()(label),value);
                  }
                  // operate j=[i+1,..,vec.size()]
                  for(IndexType j=i_+1;j< vec_.size();++j) {
                     const LabelType label=static_cast<LabelType>(shapeWalker.coordinateTuple()[j]);
                     OP::op(vec_[j].current()(label),value);
                  }
                  // accumulate
                  ACC::op(value,out_(shapeWalker.coordinateTuple()[i_]));
               }
            }
         }


         const BUFVEC & vec_;
         const INDEX i_;
         ARRAY & out_;
      };

      template<class GM, class ACC, class BUFVEC, class ARRAY, class INDEX>
      inline void operateF
      (
         const typename GM::FactorType& f,
         const BUFVEC& vec,
         const INDEX i,
         ARRAY& out
         ) {
         OperateF_Functor<GM,ACC,BUFVEC,ARRAY,INDEX> functor(vec,i,out);
         f.callFunctor(functor);
      }


/// out = acc_-i( op( ihop(f,rho), op_j/i( vec[j] ) ) )
      template<class GM, class ACC, class BUFVEC, class M ,class INDEX>
      struct OperateWF_Functor{
         typedef typename GM::IndexType IndexType;
         typedef typename GM::LabelType LabelType; 
         typedef typename GM::ValueType ValueType;
         typedef typename GM::OperatorType OP;

         OperateWF_Functor(const ValueType rho, const BUFVEC & vec, const INDEX i,M & out)
            :  rho_(rho), vec_(vec), i_(i), out_(out){}

         template<class FUNCTION>
         void operator()(const FUNCTION & f){
            // neutral initialization of output
            for(size_t n=0; n<f.shape(i_); ++n)
               ACC::neutral(out_(n));
            // factor shape iterator 
            typedef typename FUNCTION::FunctionShapeIteratorType FunctionShapeIteratorType;
            opengm::ShapeWalker<FunctionShapeIteratorType> shapeWalker(f.functionShapeBegin(),f.dimension());
            for(IndexType scalarIndex=0;scalarIndex<f.size();++scalarIndex,++shapeWalker) {
               // loop over the variables
               // initialize output value with value of the factor at this coordinate
               // operate j=[0,..i-1]
               ValueType value;
               OP::ihop(f(shapeWalker.coordinateTuple().begin()),rho_,value);
               for(IndexType j=0;j<static_cast<typename GM::IndexType>(i_);++j) {
                  const LabelType label=static_cast<LabelType>(shapeWalker.coordinateTuple()[j]);
                  OP::op(vec_[j].current()(label),value);
               }
               // operate j=[i+1,..,vec.size()]
               for(IndexType j=i_+1;j< vec_.size();++j) {
                  const LabelType label=static_cast<LabelType>(shapeWalker.coordinateTuple()[j]);
                  OP::op(vec_[j].current()(label),value);
               }
               // accumulate 
               ACC::op(value,out_(shapeWalker.coordinateTuple()[i_]));
            }
         } 
   
         const ValueType rho_;
         const BUFVEC & vec_;
         const INDEX i_;
         M & out_;
      };

      template<class GM, class ACC, class BUFVEC, class M, class INDEX>
      inline void operateWF
      (
         const typename GM::FactorType& f, 
         const typename GM::ValueType rho, 
         const BUFVEC& vec, 
         const INDEX i, 
         M& out
         ) {
         OperateWF_Functor<GM,ACC,BUFVEC,M,INDEX> functor(rho,vec,i,out);
         f.callFunctor(functor);
      }
 

/// out = op(f, vec[0].current, ..., vec[n].current )
      template<class GM, class BUFVEC>
      struct OperatorF2_Functor{ 
         typedef typename GM::IndexType IndexType;
         typedef typename GM::LabelType LabelType;
         typedef typename GM::ValueType ValueType;
         typedef typename GM::OperatorType OP;
         OperatorF2_Functor(const BUFVEC& vec, typename GM::IndependentFactorType& out):vec_(vec), out_(out){}
  
         template<class FUNCTION>
         void operator()(const FUNCTION & f){
            OPENGM_ASSERT(out_.numberOfVariables()!=0);
            // shape iterator
            typedef typename FUNCTION::FunctionShapeIteratorType FunctionShapeIteratorType;
            opengm::ShapeWalker<FunctionShapeIteratorType> shapeWalker(f.functionShapeBegin(),f.dimension());
            for(IndexType scalarIndex=0;scalarIndex<f.size();++scalarIndex,++shapeWalker) {
               // loop over the variables
               ValueType value=f(shapeWalker.coordinateTuple().begin());
               for(IndexType j=0;j<static_cast<typename GM::IndexType>(vec_.size());++j) {
                  const LabelType label=static_cast<LabelType>(shapeWalker.coordinateTuple()[j]);
                  OP::op(vec_[j].current()(label),value);
               }
               out_(scalarIndex)=value;
            }
         }

         const BUFVEC& vec_; 
         typename GM::IndependentFactorType& out_;
      };
      template<class GM, class BUFVEC>
      inline void operateF
      (
         const typename GM::FactorType& f, 
         const BUFVEC& vec, 
         typename GM::IndependentFactorType& out
         )
      {
         OperatorF2_Functor<GM, BUFVEC> functor(vec,out);
         f.callFunctor(functor);
      }

/// out = op( ihop(f,rho), op_j(vec[j]) )
      template<class GM, class BUFVEC>
      struct OperatorWF2_Functor{ 
         typedef typename GM::IndexType IndexType;
         typedef typename GM::LabelType LabelType;
         typedef typename GM::ValueType ValueType;
         typedef typename GM::OperatorType OP;
         OperatorWF2_Functor(ValueType rho, const BUFVEC& vec, typename GM::IndependentFactorType& out) : rho_(rho), vec_(vec), out_(out){}
  
         template<class FUNCTION>
         void operator()(const FUNCTION & f){
            // shape iterator
            typedef typename FUNCTION::FunctionShapeIteratorType FunctionShapeIteratorType;
            opengm::ShapeWalker<FunctionShapeIteratorType> shapeWalker(f.functionShapeBegin(),f.dimension());
            for(IndexType scalarIndex=0;scalarIndex<f.size();++scalarIndex,++shapeWalker) { // loop over the variables
               ValueType value;
               OP::ihop(f(shapeWalker.coordinateTuple().begin()),rho_,value);
               for(IndexType j=0;j<static_cast<typename GM::IndexType>(vec_.size());++j) {
                  const LabelType label=static_cast<LabelType>(shapeWalker.coordinateTuple()[j]);
                  OP::op(vec_[j].current()(label),value);
               }
               out_(scalarIndex)=value;
            }
         }

         const ValueType rho_;
         const BUFVEC& vec_; 
         typename GM::IndependentFactorType& out_;
      };

      template<class GM, class BUFVEC>
      inline void operateWF
      (
         const typename GM::FactorType& f, 
         const typename GM::ValueType rho, 
         const BUFVEC& vec, 
         typename GM::IndependentFactorType& out
         ) {
         OperatorWF2_Functor<GM, BUFVEC> functor(rho,vec,out);
         f.callFunctor(functor);           
      }

   } // namespace messagepassingOperations
} // namespace opengm

/// \endcond

#endif
