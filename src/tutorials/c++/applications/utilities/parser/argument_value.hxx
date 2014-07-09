#ifndef ARGUMENT_VALUE_HXX
#define	ARGUMENT_VALUE_HXX
#include "parser_error.hxx"
#include "string_utilities.hxx"
namespace parser{
   
   template<class T>
   class BoundCheck{
      
   };
   
   
   template<class T>
   class ArgValue{
   public:
      //constructors
      template<class ITERATOR>
      ArgValue(T & value,ITERATOR allowedValuesBegin, ITERATOR allowedValuesEnd );
      ArgValue(T & value,const std::vector<T> & allowedValues=std::vector<T>());
      template<class ITERATOR>
      ArgValue(T & value,const T & defaultValue,ITERATOR allowedValuesBegin, ITERATOR allowedValuesEnd);
      ArgValue(T & value,const T & defaultValue,const std::vector<T> & allowedValues =std::vector<T>());
      
      void setValue(const T & value);
      const T & value()const;
      T & value();
      const T & defaultValue()const;
      bool hasDefaultValue()const;
      const std::vector<T> & allowedValues()const;
   private:
      T * value_;
      T defaultValue_;
      bool hasDefaultValue_;
      std::vector<T> allowedValues_;
   };
   
   
   template<class T>
   template<class ITERATOR>
   ArgValue<T>::ArgValue
   (
      T & value,
      ITERATOR allowedValuesBegin, 
      ITERATOR allowedValuesEnd 
   )
      :value_(&value),defaultValue_(T()),hasDefaultValue_(false),allowedValues_(allowedValuesBegin,allowedValuesEnd){
   }
   
   template<class T>
   ArgValue<T>::ArgValue
   (
      T & value,
      const std::vector<T> & allowedValues
   )
      :value_(&value),defaultValue_(T()),hasDefaultValue_(false),allowedValues_(allowedValues){
   }
   
   template<class T>
   template<class ITERATOR>
   ArgValue<T>::ArgValue
   (
      T & value,
      const T & defaultValue,
      ITERATOR allowedValuesBegin, 
      ITERATOR allowedValuesEnd
   )
      :value_(&value),defaultValue_(defaultValue),hasDefaultValue_(true),allowedValues_(allowedValuesBegin,allowedValuesEnd){
   }
   
   template<class T>
   ArgValue<T>::ArgValue
   (
      T & value,
      const T & defaultValue,
      const std::vector<T> & allowedValues
   )
      :value_(&value),defaultValue_(defaultValue),hasDefaultValue_(true),allowedValues_(allowedValues){
   }
   
   
   template<class T>
   void ArgValue<T>::setValue(
      const T & value
   ){
      *value_=value;
   }
   
   template<class T>
   const T & ArgValue<T>::value()const{
      return *value_;
   }
   
   template<class T>
   T & ArgValue<T>::value(){
      return *value_;
   }
   
   template<class T>
   const T & ArgValue<T>::defaultValue()const{
      return defaultValue_;
   }
   
   template<class T>
   bool ArgValue<T>::hasDefaultValue()const{
      return hasDefaultValue_;
   }
   
   template<class T>
   const std::vector<T> & ArgValue<T>::allowedValues()const{
      return allowedValues_;
   }
}
#endif	/* ARGUMENT_VALUE_HXX */

