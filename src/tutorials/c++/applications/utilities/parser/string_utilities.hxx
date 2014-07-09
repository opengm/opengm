#ifndef STRINGSTUFF_HXX
#define	STRINGSTUFF_HXX

#include <string>
#include <sstream>
#include <cmath>
#include <limits>
namespace parser{

std::string colorString(const std::string text,const int textcolor){
   #ifndef __GNUC__
   std::stringstream ss;
   ss<< "\033[1;" << textcolor << "m"<<text<<"\033[0m";
   return ss.str();
   #else
   return text;
   #endif
}

template<class T>
struct Name{
   std::string name(){return "unknown-type";}
};

template<class T>
std::string nameOfType(){
   Name<T> n;
   return n.name();
}

template<class T>
struct Name< std::vector<T> >{
   std::string name(){return nameOfType<T>().append("[ ]");}
};

#define NAME_GEN_MACRO( TYPE ,NAME) template< > \
struct Name<TYPE>{ \
   std::string name(){return NAME;} \
}


NAME_GEN_MACRO(bool , "bool");
NAME_GEN_MACRO(unsigned int, "unsigned int");
NAME_GEN_MACRO(int, "int");
NAME_GEN_MACRO(unsigned long, "unsigned long");
NAME_GEN_MACRO(long, "long");
NAME_GEN_MACRO(float , "float");
NAME_GEN_MACRO(double , "double");
NAME_GEN_MACRO(char ,"char");
NAME_GEN_MACRO(std::string , "string");


class Splitter {
   //! Contains the split tokens
   std::vector<std::string> _tokens;
   public:
   //! Subscript type for use with operator[]
   typedef std::vector<std::string>::size_type size_type;
   public:
   Splitter ( const std::string& src, const std::string& delim ){
      reset ( src, delim );
   }
   std::string& operator[] ( size_type i ){
      return _tokens.at ( i );
   }
   size_type size() const{
      return _tokens.size();
   }
   void reset ( const std::string& src, const std::string& delim ){
      std::vector<std::string> tokens;
      std::string::size_type start = 0;
      std::string::size_type end;

      for ( ; ; ) {
      end = src.find ( delim, start );
      tokens.push_back ( src.substr ( start, end - start ) );

      // We just copied the last token
      if ( end == std::string::npos )
      break;

      // Exclude the delimiter in the next search
      start = end + delim.size();
      }
      _tokens.swap ( tokens );
   }
};

bool onlyWhitespaces(const std::string &value){
   for(size_t i=0;i<value.size();++i)
      if(value[i]!=' ')
         return false;
   return true;
}

bool hasNoneOf(const std::string &value ,const char f){
   for(size_t i=0;i<value.size();++i)
      if(value[i]==f)
         return false;
   return true;
}

inline size_t 
numberOfWords(const std::string & value,const std::string & seperator =std::string(" ")){
   Splitter splitter(value," ");
   return splitter.size();
}

inline std::string 
getWord(const std::string  & value, const size_t i,const std::string & seperator =std::string(" ") ){
   Splitter splitter(value,seperator);
   return splitter[i];
}

inline bool 
hasOnlySingleWhiteSpaces(const std::string & value){
   if(value.size()<1){
      for(size_t i=0;i<value.size();++i){
         if(value[i]==' '){
            if(i<value.size()-1){
               if(value[i+1]==' '){
                  return false;
               }
            }
         }
      }
   }
   return true;
}
   
   
inline bool 
hasEnd
(
   const std::string & value,
   const std::string & end
){
   if(value.size()>=end.size()){
      for(size_t i=0;i<end.size();++i)
         if(end[end.size()-1-i]!=value[value.size()-1-i])
            return false;
      return true;
   }
   return false;
}

inline bool 
hasBegin
(
   const std::string & value,
   const std::string & begin
){
   if(value.size()>=begin.size()){
      for(size_t i=0;i<begin.size();++i)
         if(begin[i]!=value[i])
            return false;
      return true;
   }
   return false;
}

inline std::string 
removeBeginAndEndMultible(const std::string & value ,const std::string & begin ,const std::string & end){
   std::string tmp=value;
   while(hasBegin(tmp,begin) && tmp.size()!=0){
      tmp=tmp.substr(begin.size(),tmp.size()-begin.size());
   }
   if(tmp.size()>0){
      while(hasEnd(tmp,end) && tmp.size()!=0){
         tmp=tmp.substr(0,tmp.size()-end.size());
      }
   }
   return tmp;
}

inline std::string 
removeBeginAndEndSingle(const std::string & value ,const std::string & begin ,const std::string & end){
   std::string tmp=value;
   if(hasBegin(value,begin) && tmp.size()!=0){
      tmp=tmp.substr(begin.size(),tmp.size()-begin.size());
   }
   if(tmp.size()>0){
      if(hasEnd(value,end) && tmp.size()!=0){
         tmp=tmp.substr(0,tmp.size()-end.size());
      }
   }
   return tmp;
}


inline std::string 
removeBeginAndEndWhitespaces(const std::string & value){
   return removeBeginAndEndMultible(value," "," ");
}

inline std::string 
removeBeginAndEndCompoundSymbols(const std::string & value){
   return removeBeginAndEndMultible(value,"[","]");
}

inline std::string 
isolteCompoundSymbols( const std::string & value ){
   std::string tmp;
   tmp.reserve(value.size()+8);
   for(size_t i=0;i<value.size();++i){
      if(value[i]=='{' || value[i]=='}'){
         if(i==0){
            tmp.push_back(value[i]);
            if(value[i+1]!=' ')
               tmp.push_back(' ');
         }
         else if(i==value.size()-1){
            if(value[i-1]!=' ')
              tmp.push_back(' ');
            tmp.push_back(value[i]);
         }
         else {
            if(value[i-1]!=' ')
               tmp.push_back(' ');
            tmp.push_back(value[i]);
            if(value[i+1]!=' ')
               tmp.push_back(' ');
         }
      }
      else{
         tmp.push_back(value[i]);
      }
   }
   return removeBeginAndEndWhitespaces(tmp);
}
inline bool 
hasBeginAndEnd
(
   const std::string & value,
   const std::string & begin,
   const std::string & end
){
   return hasBegin(value,begin) && hasEnd(value,end);
}


inline bool 
isCompounded
(
   const std::string & value
){
   return hasBegin(value,"[") && hasEnd(value,"]");
}

   
template<class T1,class T2>
std::string strcat
(
   const T1 & s1,
   const T2 & s2
){
   return static_cast<std::string>(s1)+
      static_cast<std::string>(s2);
}

template<class T1,class T2,class T3>
std::string strcat
(
   const T1 & s1,
   const T2 & s2,
   const T3 & s3
){
   return static_cast<std::string>(s1)+
      static_cast<std::string>(s2)+
      static_cast<std::string>(s3);
}

template<class T1,class T2,class T3,class T4>
std::string strcat
(
   const T1 & s1,
   const T2 & s2,
   const T3 & s3,
   const T4 & s4
){
   return static_cast<std::string>(s1)+
      static_cast<std::string>(s2)+
      static_cast<std::string>(s3)+
      static_cast<std::string>(s4);
}

template<class T1,class T2,class T3,class T4,class T5>
std::string strcat
(
   const T1 & s1,
   const T2 & s2,
   const T3 & s3,
   const T4 & s4,
   const T5 & s5
){
   return static_cast<std::string>(s1)+
      static_cast<std::string>(s2)+
      static_cast<std::string>(s3)+
      static_cast<std::string>(s4)+
      static_cast<std::string>(s5);
}

template<class T1,class T2,class T3,class T4,class T5,class T6>
std::string strcat
(
   const T1 & s1,
   const T2 & s2,
   const T3 & s3,
   const T4 & s4,
   const T5 & s5,
   const T6 & s6
){
   return static_cast<std::string>(s1)+
      static_cast<std::string>(s2)+
      static_cast<std::string>(s3)+
      static_cast<std::string>(s4)+
      static_cast<std::string>(s5)+
      static_cast<std::string>(s6);
}

template<class T1,class T2,class T3,class T4,class T5,class T6,class T7>
std::string strcat
(
   const T1 & s1,
   const T2 & s2,
   const T3 & s3,
   const T4 & s4,
   const T5 & s5,
   const T6 & s6,
   const T7 & s7
){
   return static_cast<std::string>(s1)+
      static_cast<std::string>(s2)+
      static_cast<std::string>(s3)+
      static_cast<std::string>(s4)+
      static_cast<std::string>(s5)+
      static_cast<std::string>(s6)+
      static_cast<std::string>(s7);
}

template<class T1,class T2,class T3,class T4,class T5,class T6,class T7,class T8>
std::string strcat
(
   const T1 & s1,
   const T2 & s2,
   const T3 & s3,
   const T4 & s4,
   const T5 & s5,
   const T6 & s6,
   const T7 & s7,
   const T8 & s8
){
   return static_cast<std::string>(s1)+
      static_cast<std::string>(s2)+
      static_cast<std::string>(s3)+
      static_cast<std::string>(s4)+
      static_cast<std::string>(s5)+
      static_cast<std::string>(s6)+
      static_cast<std::string>(s7)+
      static_cast<std::string>(s8);
}

template<class T1,class T2,class T3,class T4,class T5,class T6,class T7,class T8,class T9>
std::string strcat
(
   const T1 & s1,
   const T2 & s2,
   const T3 & s3,
   const T4 & s4,
   const T5 & s5,
   const T6 & s6,
   const T7 & s7,
   const T8 & s8,
   const T9 & s9
){
   return static_cast<std::string>(s1)+
      static_cast<std::string>(s2)+
      static_cast<std::string>(s3)+
      static_cast<std::string>(s4)+
      static_cast<std::string>(s5)+
      static_cast<std::string>(s6)+
      static_cast<std::string>(s7)+
      static_cast<std::string>(s8)+
      static_cast<std::string>(s9);
}

template<class T1,class T2,class T3,class T4,class T5,class T6,class T7,class T8,class T9,class T10>
std::string strcat
(
   const T1 & s1,
   const T2 & s2,
   const T3 & s3,
   const T4 & s4,
   const T5 & s5,
   const T6 & s6,
   const T7 & s7,
   const T8 & s8,
   const T9 & s9,
   const T10 & s10
){
   return static_cast<std::string>(s1)+
      static_cast<std::string>(s2)+
      static_cast<std::string>(s3)+
      static_cast<std::string>(s4)+
      static_cast<std::string>(s5)+
      static_cast<std::string>(s6)+
      static_cast<std::string>(s7)+
      static_cast<std::string>(s8)+
      static_cast<std::string>(s9)+
      static_cast<std::string>(s10);
}
   

enum TextColor{
   Red=31,
   Green=32,
   Yellow=33,
   Blue=34,
   Purple = 35,
   LightBlue=37
};

      
void printSpace(const size_t numberOfSpace){
   for(size_t i=0;i<numberOfSpace;++i)
      std::cout<<" ";
}

std::string getSpace(const size_t numberOfSpace){
   std::string tmp;
   for(size_t i=0;i<numberOfSpace;++i)
      tmp.append(" ");
   return tmp;
}

std::string getLine(const size_t numberOfSpace){
   std::string tmp;
   for(size_t i=0;i<numberOfSpace;++i)
      tmp.append("-");
   return tmp;
}


 template<class T>
   bool stringToBuildInType(const std::string stringValue,T & value){
      std::stringstream ss;
      ss<<stringValue;
      return ss>>value;
   }
    
   bool isNumber(const std::string & stringValue){
      float number;
      return stringToBuildInType(stringValue,number);
   }
   
   bool isInteger(const std::string & stringValue){
      if(isNumber(stringValue)){
         int asInt;float asFloat;
         if(stringToBuildInType(stringValue,asInt) && stringToBuildInType(stringValue,asFloat)  )
            return (std::fabs(asFloat-float(asInt))<0.00001);
      }
      return false;
   }
 
   bool isUnsignedInteger(const std::string & stringValue){
      if(isInteger(stringValue)){
         int asInt;
         if(stringToBuildInType(stringValue,asInt))
            return (asInt>=0);
      }
      return false;
   }
   
   bool isChar(const std::string & stringValue){
      char asChar;
      if(isNumber(stringValue)==false){
         if(stringValue.size()==1)
            return stringToBuildInType(stringValue,asChar);
      }
      else{
         if(isInteger(stringValue))
            return stringToBuildInType(stringValue,asChar);
      }
      return false;
   }
   bool isUnsignedChar(const std::string & stringValue){
      unsigned char asUChar;
      if(isNumber(stringValue)==false){
         if(stringValue.size()==1)
            return stringToBuildInType(stringValue,asUChar);
      }
      else{
         if(isUnsignedInteger(stringValue))
            return stringToBuildInType(stringValue,asUChar);
      }
      return false;
   }
   
   bool isBool(const std::string & stringValue){
      if(isUnsignedInteger(stringValue)==false){
         std::string allowedValues[]={"true","false","True","False", "TRUE","FALSE", "o", "x" };
         for(size_t i=0;i<8;++i)
            if(stringValue==allowedValues[i])
               return true;
         return false;
      }
      else{
         unsigned int asUInt;
         bool asBool;
         if(stringToBuildInType(stringValue,asUInt))
            return (asUInt==0 || asUInt==1) && stringToBuildInType(stringValue,asBool);
         return false;
      }
   }
     
   template<class T>
   struct Converter{
      static std::string toString(const T & value){
         std::stringstream ss;
         ss<<value;
         return ss.str();
      }
      static void fromString(const std::string & stringValue,T & value){
         // check type    
         if(std::numeric_limits<T>::is_specialized){
            parserAssert(isNumber(stringValue),stringValue +std::string(" is no numeric value"));
            if(std::numeric_limits<T>::is_integer){
               parserAssert(isInteger(stringValue),stringValue +std::string(" is no integral value"));
               if(!std::numeric_limits<T>::is_signed){
                  parserAssert(isUnsignedInteger(stringValue),stringValue +std::string(" is not a UNSIGNED integral value"));
               }
            }
         }
         // fallback parsing (should be fine for the build in types
         std::stringstream ss;
         ss<<stringValue;
         if(!(ss>>value)){
            std::string errorMsg="cannot parse \" ";
            errorMsg.append(stringValue).append(" \" into a ").append(nameOfType<T>()).append(" .");
            throw RuntimeError(errorMsg);
         } 
      }
   };
   
   
   
   template< >
   struct Converter<bool>{
      static std::string toString(const bool & value){
         std::stringstream ss;
         ss<<value;
         return ss.str();
      }
      static void fromString(const std::string & stringValue,bool & value){
         parserAssert(isBool(stringValue),stringValue +std::string(" is no bool"));
         if(isUnsignedInteger(stringValue)==false){
            std::string allowedValuesTrue[]={"true","True","TRUE","o"};
            std::string allowedValuesFalse[]={"false","False","FALSE", "x" };
            for(size_t i=0;i<4;++i){
               if(stringValue==allowedValuesTrue[i]){ value = true; break;}
               if(stringValue==allowedValuesFalse[i]){value = false;break;}
            }
         }
         else 
            parserAssert(stringToBuildInType(stringValue,value),stringValue +std::string(" is no bool"));
      }
   };
      
   template< >
   struct Converter<char>{
      static std::string toString(const char & value){
         std::stringstream ss;
         ss<<value;
         return ss.str();
      }
      static void fromString(const std::string & stringValue,char & value){
         parserAssert(isChar(stringValue),stringValue +std::string(" is no char"));
         parserAssert(stringToBuildInType(stringValue,value),stringValue +std::string(" is no char"));
      }
   };
   
   template< >
   struct Converter<unsigned char>{
      static std::string toString(const unsigned char & value){
         std::stringstream ss;
         ss<<value;
         return ss.str();
      }
      static void fromString(const std::string & stringValue,unsigned char & value){
         parserAssert(isUnsignedChar(stringValue),stringValue +std::string(" is no unsigned char"));
         parserAssert(stringToBuildInType(stringValue,value),stringValue +std::string(" is no  unsigned char"));
      }
   };
   
}

#endif	/* STRINGSTUFF_HXX */

