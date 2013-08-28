#ifndef SHAPEHOLDER_HXX
#define	SHAPEHOLDER_HXX
#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <opengm/python/opengmpython.hxx>
#include <opengm/python/converter.hxx>
#include <opengm/python/numpyview.hxx>
#include <opengm/python/pythonfunction.hxx>

#include <algorithm>
#include <iterator>
#include <iterator>
#include <string>
#include <string>
#include <sstream>
#include <stddef.h>
#include <opengm/graphicalmodel/graphicalmodel.hxx>

#include <opengm/python/opengmpython.hxx>
#include <opengm/python/converter.hxx>
#include <opengm/python/numpyview.hxx>
#include <opengm/python/pythonfunction.hxx>


template<class FACTOR>
class FactorShapeHolder {
public:
   
   typedef typename FACTOR::ValueType ValueType;
   typedef typename FACTOR::IndexType IndexType;
   typedef typename FACTOR::ShapeIteratorType IteratorType;
   typedef IteratorType const_iterator;
   typedef IteratorType iterator;

   FactorShapeHolder() : factor_(NULL) {
   }
   FactorShapeHolder(const FACTOR & factor) : factor_(& factor) {
   }

   IndexType operator[](const size_t index) const {
      return factor_->shape(index);
   }

   size_t size()const {
      return factor_->numberOfVariables();
   }

   boost::python::list toList()const {
      return opengm::python::iteratorToList(factor_->shapeBegin(),this->size());
   }

   boost::python::numeric::array toNumpy()const {
      return opengm::python::iteratorToNumpy(factor_->shapeBegin(),this->size());
   }
   boost::python::tuple toTuple()const {
      return  opengm::python::iteratorToTuple(factor_->shapeBegin(),this->size());
   }
   
   const_iterator begin() const  {return factor_->shapeBegin();}
   const_iterator end()   const  {return factor_->shapeEnd();}
   iterator begin()              {return factor_->shapeBegin();}
   iterator end()                {return factor_->shapeEnd();}

   std::string asString() const {
      std::stringstream ss;
      ss << "[";
      for (size_t i = 0; i<this->size(); ++i) {
            ss << factor_->shape(i) << ", ";
      }
      ss << "]";
      return ss.str();
   }
private:
   //const FACTOR & factor_;
   FACTOR const * factor_;
};


template<class FACTOR>
class FactorViHolder {
public:
   
   typedef typename FACTOR::ValueType ValueType;
   typedef typename FACTOR::IndexType IndexType;
   typedef typename FACTOR::VariablesIteratorType IteratorType;
   typedef IteratorType const_iterator;
   typedef IteratorType iterator;

   const_iterator begin() const  {return factor_->variableIndicesBegin();}
   const_iterator end()   const  {return factor_->variableIndicesEnd();}
   iterator begin()              {return factor_->variableIndicesBegin();}
   iterator end()                {return factor_->variableIndicesEnd();}

   FactorViHolder() : factor_(NULL) {
   }
   FactorViHolder(const FACTOR & factor) : factor_(& factor) {
   }

   IndexType operator[](const size_t index) const {
      return factor_->variableIndex(index);
   }

   size_t size()const {
      return factor_->numberOfVariables();
   }

   boost::python::list toList()const {
      return opengm::python::iteratorToList(factor_-> variableIndicesBegin(),this->size());
   }

   boost::python::numeric::array toNumpy()const {
      return opengm::python::iteratorToNumpy(factor_->variableIndicesBegin(),this->size());
   }
   boost::python::tuple toTuple()const {
      return  opengm::python::iteratorToTuple(factor_->variableIndicesBegin(),this->size());
   }
   
   

   std::string asString() const {
      std::stringstream ss;
      ss << "[";
      for (size_t i = 0; i<this->size(); ++i) {
            ss << factor_-> variableIndex(i) << ", ";
      }
      ss << "]";
      return ss.str();
   }
private:
   //const FACTOR & factor_;
   FACTOR const * factor_;
};


template<class GM>
class FactorsOfVariableHolder {
public:
   
   typedef typename GM::IndexType ValueType;
   typedef typename GM::IndexType IndexType;
   FactorsOfVariableHolder() : gm_(NULL),variableIndex_(0) {
   }
   FactorsOfVariableHolder(const GM & gm,const size_t variableIndex) : gm_(& gm),variableIndex_(variableIndex) {
   }

   IndexType operator[](const size_t factorIndex) const {
      return gm_->factorOfVariable(variableIndex_,factorIndex);
   }

   size_t size()const {
      return gm_->numberOfFactors(variableIndex_);
   }

   boost::python::list toList()const {
      return opengm::python::iteratorToList(gm_-> factorsOfVariableBegin(variableIndex_),this->size());
   }

   boost::python::numeric::array toNumpy()const {
      return opengm::python::iteratorToNumpy(gm_->factorsOfVariableBegin(variableIndex_),this->size());
   }
   boost::python::tuple toTuple()const {
      return opengm::python::iteratorToTuple(gm_->factorsOfVariableBegin(variableIndex_),this->size());
   }
   
   

   std::string asString() const {
      std::stringstream ss;
      ss << "[";
      for (size_t i = 0; i<this->size(); ++i) {
            ss << gm_-> factorOfVariable(variableIndex_,i) << ", ";
      }
      ss << "]";
      return ss.str();
   }
private:
   GM const * gm_;
   size_t variableIndex_;
};

#endif	/* SHAPEHOLDER_HXX */

