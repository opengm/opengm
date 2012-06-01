#pragma once
#ifndef OPENGM_SHAPE_ACCESSOR_HXX
#define OPENGM_SHAPE_ACCESSOR_HXX

namespace opengm {

/// \cond HIDDEN_SYMBOLS

   template<class FUNCTION>
   class FunctionShapeAccessor {
   public:
      typedef size_t value_type;
      typedef const value_type reference;
      typedef const value_type* pointer;
      typedef const FUNCTION& factor_reference;
      typedef const FUNCTION* factor_pointer;

      FunctionShapeAccessor(factor_pointer f = NULL)
         : factor_(f) 
         {}
      FunctionShapeAccessor(factor_reference f)
         : factor_(&f) 
         {}
      size_t size() const { 
         return factor_ == NULL ? 0 : factor_->dimension();
      }
      value_type operator[](const size_t j) { 
         OPENGM_ASSERT(j<factor_->dimension());
         return factor_->shape(j); 
      }
      const value_type operator[](const size_t j) const { 
         OPENGM_ASSERT(j<factor_->dimension());
         return factor_->shape(j); 
      }
      bool operator==(const FunctionShapeAccessor<FUNCTION> & other) const 
         { return factor_ == other.factor_; }
   
   private:
      factor_pointer factor_;
   };

   template<class FACTOR>
   class FactorShapeAccessor {
   public:
      typedef size_t value_type;
      typedef const value_type reference;
      typedef const value_type* pointer;
      typedef const FACTOR& factor_reference;
      typedef const FACTOR* factor_pointer;

      FactorShapeAccessor(factor_pointer f = 0)
         : factor_(f) 
         {}
      FactorShapeAccessor(factor_reference f)
         : factor_(&f) 
         {}
      size_t size() const 
         { return factor_ == 0 ? 0 : factor_->numberOfVariables(); }
      reference operator[](const size_t j) 
         { return factor_->numberOfLabels(j); }
      const value_type operator[](const size_t j) const 
         { return factor_->numberOfLabels(j); }
      bool operator==(const FactorShapeAccessor<FACTOR> & other) const 
         { return factor_ == other.factor_; }
   
   private:
      factor_pointer factor_;
   };
   
   template<class FACTOR>
   class FactorVariablesAccessor {
   public:
      typedef typename FACTOR::IndexType IndexType;
      typedef IndexType value_type;
      typedef const value_type reference;
      typedef const value_type* pointer;
      typedef const FACTOR& factor_reference;
      typedef const FACTOR* factor_pointer;

      FactorVariablesAccessor(factor_pointer f = 0)
         : factor_(f) 
         {}
      FactorVariablesAccessor(factor_reference f)
         : factor_(&f) 
         {}
      IndexType size() const 
         { return factor_ == 0 ? 0 : factor_->numberOfVariables(); }
      reference operator[](const size_t j) 
         { return factor_->numberOfLabels(j); }
      const value_type operator[](const size_t j) const 
         { return factor_->variableIndex(j); }
      bool operator==(const FactorVariablesAccessor<FACTOR> & other) const 
         { return factor_ == other.factor_; }
   
   private:
      factor_pointer factor_;
   };

/// \endcond

} // namespace opengm

#endif // #ifndef OPENGM_SHAPE_ACCESSOR_HXX

