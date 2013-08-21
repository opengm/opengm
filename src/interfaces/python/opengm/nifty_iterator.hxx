
#ifndef NIFTY_ITERATOR_HXX
#define	NIFTY_ITERATOR_HXX




#include <boost/python.hpp>


#include <stddef.h>

#include <opengm/opengm.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>

//#include <numpy/noprefix.h>

template<class T, bool isConst>
class PythonFundamentalListAccessor {
public:
   typedef T value_type;
   typedef value_type reference;
   typedef const value_type* pointer;
   
   PythonFundamentalListAccessor(boost::python::list const * listPtr = NULL)
      : listPtr_(listPtr) {}
   PythonFundamentalListAccessor(const boost::python::list & listRef)
      : listPtr_(&listRef) {}
   size_t size() const { 
      return listPtr_ == NULL ? size_t(0) : size_t(boost::python::len(*listPtr_)); 
   }
   
   value_type helper(const size_t j)const{
      if(opengm::meta::IsFundamental<value_type>::value==true){
         // If value_type is an integral type
         if(opengm::meta::IsFloatingPoint<value_type>::value ==false){
            {
               boost::python::extract<T> extractor((*listPtr_)[j]);
               if(extractor.check()){
                  return static_cast<T>(extractor());
               }
            }
            if(opengm::meta::Compare<T,opengm::Int32Type>::value==false){
               boost::python::extract<opengm::Int32Type> extractor((*listPtr_)[j]);
               if(extractor.check()){
                  return static_cast<T>(extractor());
               }
            }
            if(opengm::meta::Compare<T,opengm::Int64Type>::value==false){
               boost::python::extract<opengm::Int64Type> extractor((*listPtr_)[j]);
               if(extractor.check()){
                  return static_cast<T>(extractor());
               }
            }
            if(opengm::meta::Compare<T,opengm::UInt32Type>::value==false){
               boost::python::extract<opengm::UInt32Type> extractor((*listPtr_)[j]);
               if(extractor.check()){
                  return static_cast<T>(extractor());
               }
            }
            if(opengm::meta::Compare<T,opengm::UInt64Type>::value==false){
               boost::python::extract<opengm::UInt64Type> extractor((*listPtr_)[j]);
               if(extractor.check()){
                  return static_cast<T>(extractor());
               }
            }
            std::string perror_str = "python list has non integral values";
            std::cout << "Error in Python OpenGM: " << perror_str << std::endl;
            throw opengm::RuntimeError("python list has non integral values");
         }
         // If value_type is a floating point type
         else{
            {
               boost::python::extract<T> extractor((*listPtr_)[j]);
               if(extractor.check()){
                  return static_cast<T>(extractor());
               }
            }
            if(opengm::meta::Compare<T,opengm::Float32Type>::value==false){
               boost::python::extract<opengm::Float32Type> extractor((*listPtr_)[j]);
               if(extractor.check()){
                  return static_cast<T>(extractor());
               }
            }
            if(opengm::meta::Compare<T,opengm::Float64Type>::value==false){
               boost::python::extract<opengm::Float64Type> extractor((*listPtr_)[j]);
               if(extractor.check()){
                  return static_cast<T>(extractor());
               }
            }
            std::string perror_str = "python list has non floating point values";
            std::cout << "Error in Python OpenGM: " << perror_str << std::endl;
            throw opengm::RuntimeError("python list has non floating point  values");
         }
      }
      else{
         std::string perror_str = "python list has non fundamental values";
         std::cout << "Error in Python OpenGM: " << perror_str << std::endl;
         throw opengm::RuntimeError("python list has non fundamental values");
      }
   }
   
   reference operator[](const size_t j){ 
      return helper(j);
   }
   const value_type& operator[](const size_t j) const { 
      return helper(j);
   }
   template<bool isConstLocal>
      bool operator==(const PythonFundamentalListAccessor<T, isConstLocal>& other) const
         { return listPtr_ == other.listPtr_; }

private:
   boost::python::list const * listPtr_;
};


template<class T, bool isConst>
class PythonIntListAccessor {
public:
   typedef T value_type;
   typedef value_type reference;
   typedef const value_type* pointer;


   PythonIntListAccessor(boost::python::list const * listPtr = NULL)
      : listPtr_(listPtr) {}
   PythonIntListAccessor(const boost::python::list & listRef)
      : listPtr_(&listRef) {}
   size_t size() const { 
      return listPtr_ == NULL ? size_t(0) : size_t(boost::python::len(*listPtr_)); 
   }
   
   value_type helper(const size_t j)const{
      {
         boost::python::extract<T> extractor((*listPtr_)[j]);
         if(extractor.check()){
            return static_cast<T>(extractor());
         }
      }
      if(opengm::meta::Compare<T,opengm::Int32Type>::value==false){
         boost::python::extract<opengm::Int32Type> extractor((*listPtr_)[j]);
         if(extractor.check()){
            return static_cast<T>(extractor());
         }
      }
      if(opengm::meta::Compare<T,opengm::Int64Type>::value==false){
         boost::python::extract<opengm::Int64Type> extractor((*listPtr_)[j]);
         if(extractor.check()){
            return static_cast<T>(extractor());
         }
      }
      if(opengm::meta::Compare<T,opengm::UInt32Type>::value==false){
         boost::python::extract<opengm::UInt32Type> extractor((*listPtr_)[j]);
         if(extractor.check()){
            return static_cast<T>(extractor());
         }
      }
      if(opengm::meta::Compare<T,opengm::UInt64Type>::value==false){
         boost::python::extract<opengm::UInt64Type> extractor((*listPtr_)[j]);
         if(extractor.check()){
            return static_cast<T>(extractor());
         }
      }
      
      std::string perror_str = "python list has non integral values";
      std::cout << "Error in Python OpenGM: " << perror_str << std::endl;
      throw opengm::RuntimeError("python list has non integral values");
   }
   
   reference operator[](const size_t j){ 
      return helper(j);
   }
   const value_type& operator[](const size_t j) const { 
      return helper(j);
   }
   template<bool isConstLocal>
      bool operator==(const PythonIntListAccessor<T, isConstLocal>& other) const
         { return listPtr_ == other.listPtr_; }

private:
   boost::python::list const * listPtr_;
};



template<class T, bool isConst>
class PythonIntTupleAccessor {
public:
   typedef T value_type;
   typedef value_type reference;
   typedef const value_type* pointer;


   PythonIntTupleAccessor(boost::python::tuple const * tuplePtr = NULL)
      : tuplePtr_(tuplePtr) {}
   PythonIntTupleAccessor(const boost::python::tuple & tupleRef)
      : tuplePtr_(&tupleRef) {}
   size_t size() const { 
      return tuplePtr_ == NULL ? size_t(0) : size_t(boost::python::len(*tuplePtr_)); 
   }
   
   value_type helper(const size_t j)const{
      {
         boost::python::extract<T> extractor((*tuplePtr_)[j]);
         if(extractor.check()){
            return static_cast<T>(extractor());
         }
      }
      if(opengm::meta::Compare<T,opengm::Int32Type>::value==false){
         boost::python::extract<opengm::Int32Type> extractor((*tuplePtr_)[j]);
         if(extractor.check()){
            return static_cast<T>(extractor());
         }
      }
      if(opengm::meta::Compare<T,opengm::Int64Type>::value==false){
         boost::python::extract<opengm::Int64Type> extractor((*tuplePtr_)[j]);
         if(extractor.check()){
            return static_cast<T>(extractor());
         }
      }
      if(opengm::meta::Compare<T,opengm::UInt32Type>::value==false){
         boost::python::extract<opengm::UInt32Type> extractor((*tuplePtr_)[j]);
         if(extractor.check()){
            return static_cast<T>(extractor());
         }
      }
      if(opengm::meta::Compare<T,opengm::UInt64Type>::value==false){
         boost::python::extract<opengm::UInt64Type> extractor((*tuplePtr_)[j]);
         if(extractor.check()){
            return static_cast<T>(extractor());
         }
      }
      
      std::string perror_str = "python tuple has non integral values";
      std::cout << "Error in Python OpenGM: " << perror_str << std::endl;
      throw opengm::RuntimeError("python tuple has non integral values");
      

   }
   
   reference operator[](const size_t j){ 
      return helper(j);
   }
   const value_type& operator[](const size_t j) const { 
      return helper(j);
   }
   template<bool isConstLocal>
      bool operator==(const PythonIntTupleAccessor<T, isConstLocal>& other) const
         { return tuplePtr_ == other.tuplePtr_; }

private:
   boost::python::tuple const * tuplePtr_;
};

/*
template<class T, bool isConst>
class Python1dNumpyIntegralArrayAccessor {
public:
   typedef T value_type;
   typedef value_type reference;
   typedef const value_type* pointer;
   Python1dNumpyIntegralArrayAccessor(boost::python::numeric::array const * npaPtr = NULL)
      : numpyArrayPtr_(npaPtr) {
      if(numpyArrayPtr_!=NULL){
         const boost::python::tuple &shape = boost::python::extract<boost::python::tuple > ((*numpyArrayPtr_).attr("shape"));
         size_t dimension = boost::python::len(shape);
         if(dimension!=1){
            throw opengm::RuntimeError("error, numpy array dimension must be 1");
         }
         size_=static_cast<size_t>(boost::python::extract<int>(shape[0]));
      }
   }
   Python1dNumpyIntegralArrayAccessor(const boost::python::numeric::array & npa)
      : numpyArrayPtr_(&npa) {
      const boost::python::tuple &shape = boost::python::extract<boost::python::tuple > (npa.attr("shape"));
      size_t dimension = boost::python::len(shape);
      if(dimension!=1){
         throw opengm::RuntimeError("error, numpy array dimension must be 1");
      }
      size_=static_cast<size_t>(boost::python::extract<int>(shape[0]));
   }
   size_t size() const { 
      return numpyArrayPtr_ == NULL ? size_t(0) : size_; 
   }
   
    value_type helper(const size_t j)const{
      {
         boost::python::extract<T> extractor((*numpyArrayPtr_)[boost::python::make_tuple(int(j))]);
         if(extractor.check()){
            return static_cast<T>(extractor());
         }
      }      
      if(opengm::meta::Compare<T,uint>::value==false){
         boost::python::extract<uint> extractor((*numpyArrayPtr_)[boost::python::make_tuple(int(j))]);
         if(extractor.check()){
            return static_cast<T>(extractor());
         }
      }
      if(opengm::meta::Compare<T,opengm::UInt32Type>::value==false){
         boost::python::extract<opengm::UInt32Type> extractor((*numpyArrayPtr_)[boost::python::make_tuple(int(j))]);
         if(extractor.check()){
            return static_cast<T>(extractor());
         }
      }
      if(opengm::meta::Compare<T,opengm::Int64Type>::value==false){
         boost::python::extract<opengm::Int64Type> extractor((*numpyArrayPtr_)[boost::python::make_tuple(int(j))]);
         if(extractor.check()){
            return static_cast<T>(extractor());
         }
      }
      
      if(opengm::meta::Compare<T,opengm::UInt64Type>::value==false){
         boost::python::extract<opengm::UInt64Type> extractor((*numpyArrayPtr_)[boost::python::make_tuple(int(j))]);
         if(extractor.check()){
            return static_cast<T>(extractor());
         }
      }
      std::string perror_str = "python numpyarray has non integral values";
      std::cout << "Error in Python OpenGM: " << perror_str << std::endl;
      throw opengm::RuntimeError("python numpyarray has non integral values");
   }
   
   reference operator[](const size_t j){ 
      return helper(j);
   }
   const value_type& operator[](const size_t j) const { 
      return helper(j);
   }
   template<bool isConstLocal>
      bool operator==(const Python1dNumpyIntegralArrayAccessor<T, isConstLocal>& other) const
         { return numpyArrayPtr_ == other.numpyArrayPtr_; }

private:
   boost::python::numeric::array const * numpyArrayPtr_;
   size_t size_;
};
*/

template<class ACCESSOR>
class IteratorHolder{
   public:
   typedef opengm::AccessorIterator<ACCESSOR,true> Iterator;
   template<class PY_OBJECT>
   IteratorHolder(const PY_OBJECT & object):accessor_(object){
      
   }
   Iterator begin( )const{
      return Iterator(accessor_,0);
   }
   Iterator end( )const{
      return Iterator(accessor_,accessor_.size());
   }
   private:
   ACCESSOR accessor_;
};



#endif	/* NIFTY_ITERATOR_HXX */

