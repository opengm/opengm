

#ifndef SHAPEHOLDER_HXX
#define	SHAPEHOLDER_HXX
#include <algorithm>
#include <iterator>
#include <iterator>
#include <string>
#include "../converter.hxx"
#include <string>
#include <sstream>
#include <stddef.h>
#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <opengm/graphicalmodel/graphicalmodel.hxx>


template<class FACTOR>
class FactorShapeHolder {
public:
   
   typedef typename FACTOR::ValueType ValueType;
   typedef typename FACTOR::IndexType IndexType;
   FactorShapeHolder() : factor_(NULL) {
   }
   FactorShapeHolder(const FACTOR & factor) : factor_(& factor) {
   }

   ValueType operator[](const size_t index) const {
      return factor_->shape(index);
   }

   size_t size()const {
      return factor_->numberOfVariables();
   }

   boost::python::list toList()const {
      return iteratorToList(factor_->shapeBegin(),this->size());
   }

   boost::python::numeric::array toNumpy()const {
      return iteratorToNumpy(factor_->shapeBegin(),this->size());
   }
   boost::python::tuple toTuple()const {
      return iteratorToTuple(factor_->shapeBegin(),this->size());
   }
   
   

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
   FactorViHolder() : factor_(NULL) {
   }
   FactorViHolder(const FACTOR & factor) : factor_(& factor) {
   }

   ValueType operator[](const size_t index) const {
      return factor_->variableIndex(index);
   }

   size_t size()const {
      return factor_->numberOfVariables();
   }

   boost::python::list toList()const {
      return iteratorToList(factor_-> variableIndicesBegin(),this->size());
   }

   boost::python::numeric::array toNumpy()const {
      return iteratorToNumpy(factor_->variableIndicesBegin(),this->size());
   }
   boost::python::tuple toTuple()const {
      return iteratorToTuple(factor_->variableIndicesBegin(),this->size());
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

#endif	/* SHAPEHOLDER_HXX */

