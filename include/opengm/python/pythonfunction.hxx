#pragma once
#ifndef OPENGM_PYTHON_FUNCTION_INCL_HXX
#define OPENGM_PYTHON_FUNCTION_INCL_HXX

#include <algorithm>
#include <vector>
#include <cmath>

#include "opengm/opengm.hxx"
#include "opengm/functions/function_registration.hxx"
#include "opengm/functions/function_properties_base.hxx"

#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>
//#include "../export_typedes.hxx"
//#include "../converter.hxx"


namespace opengm{
namespace python{

using namespace boost::python;

template<class T, class I , class L>
class PythonFunction
: public opengm::FunctionBase<PythonFunction<T, I, L>, T, I, L>
{
public:
   typedef T ValueType;
   typedef L LabelType;
   typedef I IndexType;

   //empty
   PythonFunction(){

   }

   // copy constructor
   PythonFunction(const PythonFunction & other)
   :
   gilEnsure_(other.gilEnsure_),
   functionObj_(other.functionObj_),
   labelVector_(other.labelVector_),
   coordinateBuffer_(other.coordinateBuffer_),
   shape_(other.shape_),
   size_(other.size_){

   }

   // regular constructor
   PythonFunction(boost::python::object functionObj,boost::python::object shapeObj,const bool gilEnsure=true)
   :gilEnsure_(gilEnsure),
   functionObj_(functionObj),
   labelVector_(),
   coordinateBuffer_(),
   shape_(),
   size_(){
      stl_input_iterator<int> shapeBegin(shapeObj), shapeEnd;
      shape_.assign(shapeBegin,shapeEnd);
      labelVector_.resize(shape_.size());
      size_=1;
      for(size_t d=0;d<shape_.size();++d){
         size_*=shape_[d];
      }
      //std::cout<<"construct donen\n";
   }

   // assigment operator
   PythonFunction & operator=(const PythonFunction & other){
      if(&other!=this){
         gilEnsure_=other.gilEnsure_;
         coordinateBuffer_=other.coordinateBuffer_;
         shape_=other.shape_;
         functionObj_=other.functionObj_;
         size_=other.size_;
         labelVector_=other.labelVector_;
      }
      return *this;
   }

   // destructor
   ~PythonFunction(){
      if(shape_.size()!=0){
         //delete numpyCoordinates_;
        // delete[] coordinateBuffer_;
      }
   }

   LabelType shape(const size_t i) const{
      return shape_[i];
   }
   size_t size() const{
      return size_;
   }
   size_t dimension() const{
      return shape_.size();
   }

   template<class ITERATOR> 
   ValueType operator()(ITERATOR labeling) const{
      std::copy(labeling,labeling+shape_.size(),labelVector_.begin());
      ValueType returnValue;
      if(gilEnsure_){
         PyGILState_STATE gstate;
         gstate = PyGILState_Ensure ();
         returnValue = boost::python::extract<ValueType> ( functionObj_( labelVector_) );
         PyGILState_Release (gstate);
      }
      else{
         returnValue = boost::python::extract<ValueType> ( functionObj_( labelVector_) );
      }
      return returnValue;
   }

private:
   bool gilEnsure_;
   boost::python::object functionObj_;
   mutable std::vector<LabelType> labelVector_;
   //mutable boost::python::numeric::array  numpyCoordinates_;
   mutable  LabelType * coordinateBuffer_;
   std::vector<LabelType> shape_;
   size_t size_;
   

};

}
}

namespace opengm{
   template<class T,class I,class L>
   class FunctionSerialization<python::PythonFunction<T, I, L> >{
   public:
      typedef typename python::PythonFunction<T, I, L>::ValueType ValueType;

      static size_t indexSequenceSize(const python::PythonFunction<T, I, L>&){
         throw RuntimeError("Cannot save/load gm with a pure python function: Pure python function cannot be serialized / deserialized");
         return 0;
      }
      static size_t valueSequenceSize(const python::PythonFunction<T, I, L>&){
         throw RuntimeError("Cannot save/load gm with a pure python function: Pure Python Function cannot be serialized / deserialized");
         return 0;
      }
      template<class INDEX_OUTPUT_ITERATOR, class VALUE_OUTPUT_ITERATOR>
      static void serialize(const python::PythonFunction<T, I, L>&, INDEX_OUTPUT_ITERATOR, VALUE_OUTPUT_ITERATOR){
         throw RuntimeError("Cannot save/load gm with a pure python function: Pure Python Function cannot be serialized / deserialized");
      }
      template<class INDEX_INPUT_ITERATOR, class VALUE_INPUT_ITERATOR>
      static void deserialize( INDEX_INPUT_ITERATOR, VALUE_INPUT_ITERATOR, python::PythonFunction<T, I, L>&){
         throw RuntimeError(" Cannot save/load gm with a pure python function: Pure Python Function cannot be serialized / deserialized");
      }
   };
}

namespace opengm{
   template<class F>
   struct FunctionRegistration;
   /// \cond HIDDEN_SYMBOLS
   /// FunctionRegistration
   template<class T, class I, class L>
   struct FunctionRegistration<python::PythonFunction<T, I, L> > {
      enum ID {
         Id = opengm::FUNCTION_TYPE_ID_OFFSET + 100
      };
   };
}



#endif