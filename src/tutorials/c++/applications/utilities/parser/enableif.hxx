#ifndef ENABLE_IF_HXX
#define	ENABLE_IF_HXX

#include <string>
namespace parser{
   
   struct Enabled{
      bool operator()()const{
         return true;
      }
      bool alwaysEnabled()const{return true;}
      std::string description()const {return "-";}
      std::string ifParentValue()const{
         return  "";
      }
   };
   
   template<class PARENT_ARGUMENT_TYPE,class VALUE_TYPE>
   struct EnabledIfParentHasValue{
      typedef VALUE_TYPE ValueType;
      EnabledIfParentHasValue(PARENT_ARGUMENT_TYPE * parent = NULL,const ValueType & valueToEnable = ValueType())
      :enableIfParentHasThisValue_(valueToEnable),parentPtr_(parent){
      }
      bool operator()()const{
         std::string valueAsString = parentPtr_->valueToString();
         ValueType valueAsNumericValue;
         Converter<ValueType>::fromString(valueAsString,valueAsNumericValue);
         if(valueAsNumericValue==enableIfParentHasThisValue_)
            return true;
         else 
            return false;
      }
      
      std::string description()const {
         if( parentPtr_ != NULL ){
            std::string tmp="argument ";
            std::string tmp2;
            Stringify<ValueType>::toString(enableIfParentHasThisValue_,tmp2); 
            tmp.append(qoute(parentPtr_->longName())).append(" = ").append(tmp2);
            return tmp;
         }
         else {
            return "-";
         }
      }
      std::string ifParentValue()const{
         std::string tmp;
         Stringify<ValueType>::toString(enableIfParentHasThisValue_,tmp); 
         return tmp;
      }
      std::string parentLongName()const{
         return  parentPtr_->longName();
      }
      bool alwaysEnabled()const{return  parentPtr_ == NULL ? true : false;}
      PARENT_ARGUMENT_TYPE const * parentPtr_;
      ValueType enableIfParentHasThisValue_;
   };
}
#endif	/* ENABLE_IF_HXX */

