#ifndef STRINGIFY_HXX
#define	STRINGIFY_HXX
#include <string>
#include <sstream>
#include <iomanip>

#define E_SIZE 17

namespace parser{

         
   std::string qoute(const std::string & s){
      return strcat("\"",s,"\"");
   }
   
   template<class T>
   class Stringify;
    
   
  
   
   template<class T>
   class Stringify  {
   public:
      typedef T ValueType;
      static std::string nameOfType(){
         return "UNKOWN";
      }
      static std::string ebnfRoules(){
         std::stringstream ss;
         ss<<"< "<<Stringify<T>::nameOfType()<<" > = "<<"< UNKNOWN >";
      }
      // to string
      static void toString(const ValueType & value,std::string & stringValue) {  
          stringValue = Converter<ValueType>::toString(value);
      }
      // from string 
      static void fromString(const std::string & stringValue,ValueType & value) { 
         Converter<ValueType>::fromString(stringValue,value);
      }
   };
   
      
   #define FUNDAMENTA_STRINGIFY_GENERATOR_MACRO(TYPE,NAME,EBNF) template< > \
   class Stringify< TYPE  >   { \
   public: \
      typedef TYPE ValueType; \
      static std::string nameOfType(){ return NAME ; } \
      static std::string ebnfRoules(){  \
         std::stringstream ss; \
         std::string name=strcat("< ",Stringify<ValueType>::nameOfType()," >"); \
         ss<<std::setw(E_SIZE)<<std::left<<name<<"=  "<< EBNF ;return ss.str(); } \
      static void toString(const ValueType & value,std::string & stringValue) {  stringValue = Converter<ValueType>::toString(value);} \
      static void fromString(const std::string & stringValue,ValueType & value) {  Converter<ValueType>::fromString(stringValue,value);} \
   }
    
   //floats 
   
   FUNDAMENTA_STRINGIFY_GENERATOR_MACRO(float,"float","< Floating-Point-Number>");
   FUNDAMENTA_STRINGIFY_GENERATOR_MACRO(double,"double","< Floating-Point-Number >"); 
   
   // signed integers
   FUNDAMENTA_STRINGIFY_GENERATOR_MACRO(char,"char"," < Char > | < Char-Integral-Number >");
   FUNDAMENTA_STRINGIFY_GENERATOR_MACRO(short,"short","< Short-Integral-Number >"); 
   FUNDAMENTA_STRINGIFY_GENERATOR_MACRO(int,"int","< Integral-Number >"); 
   FUNDAMENTA_STRINGIFY_GENERATOR_MACRO(long,"long","< Long-Integral-Number >"); 
   
   // unsigned integers
   FUNDAMENTA_STRINGIFY_GENERATOR_MACRO(unsigned char,"uchar"," < Char > | <  Unsigned-Char-Integral-Number >");
   FUNDAMENTA_STRINGIFY_GENERATOR_MACRO(unsigned short,"ushort","< Unsigned-Short-Integral-Number >"); 
   FUNDAMENTA_STRINGIFY_GENERATOR_MACRO(unsigned int,"uint","< Unsigned-Integral-Number >"); 
   FUNDAMENTA_STRINGIFY_GENERATOR_MACRO(unsigned long,"ulong","< Unsigned-Long-Integral-Number >"); 
   // bool
   FUNDAMENTA_STRINGIFY_GENERATOR_MACRO(bool,"bool","< true > | < false > ;\n< true >         =  \"\" | \"1\" | \"true\" | \"True\" | \"TRUE\" ;\n< false >        =  \"0\" | \"false\" | \"False\" | \"FALSE\" ;");
   
   template< >
   class Stringify< std::string  >  { 
   public: 
      typedef std::string  ValueType; 
      static std::string nameOfType(){ 
         return "string" ; 
      } 
      static std::string ebnfRoules(){  
         std::stringstream ss;
         std::string name=strcat("< ",Stringify<ValueType>::nameOfType()," >");
         ss<<std::setw(E_SIZE)<<std::left<<name<<"= "; 
         ss<< " < Whitepace-Free-String > |\n                    "<<qoute("[")<<" { < Whitepace-Free-Strings >  "<<qoute(" ")<< "  } "<<qoute("]")<<"\n";
         return ss.str();
      } 
      // to string
      static void toString(const ValueType & value,std::string & stringValue) {  
         std::string tmp=removeBeginAndEndCompoundSymbols(value);
         if(hasNoneOf(stringValue,' '))
            stringValue=tmp;
         else
            stringValue=strcat("[ ",tmp," ]");
      }
      // from string 
      static void fromString(const std::string & stringValue,ValueType & value) { 
         if(isCompounded(stringValue))
            value=removeBeginAndEndWhitespaces(removeBeginAndEndCompoundSymbols(stringValue));
         
         else{
            parserAssert(hasNoneOf(stringValue,' '),"internal parser error");
            value=stringValue;
         }
      }
   };
   
   
   
   
   
   // fallback ,works for all fundamentals,std::vector of fundamentals
   // and all Types which have a specialization of Stringify
   
   template<class T >
   class Stringify< std::vector<T> >    {
   public:
      typedef std::vector<T> ValueType;
      
      static std::string ebnfRoules(){
         std::stringstream ss; 
         std::string name=strcat("< ",Stringify<ValueType>::nameOfType()," >");
         ss<<std::setw(E_SIZE)<<std::left<<name<<"=  ";
         ss<<qoute("[")<<" " ;
         ss<<"{ < "<<Stringify<T>::nameOfType()<<" >  "<<qoute(" ")<<" "<<" }";
         ss<<" "<<qoute("]")<<"\n";
         ss<<Stringify<T>::ebnfRoules();
         return ss.str();
      }
      static std::string nameOfType(){
         return Stringify<T>::nameOfType()+std::string("[ ]");
      }
      // to string
      static void toString(const ValueType & value,std::string & stringValue) {  
         stringValue.clear();
         stringValue.append("[");
         std::string tmp;
         for(size_t i=0;i<value.size();++i){
            tmp.clear();
            Stringify<T>::toString(value[i],tmp);
            stringValue.append(tmp).append(" ");
         }
         stringValue.append("]");
      }
      // from string 
      static void fromString(const std::string & stringValue,ValueType & value) {   
         std::string tmp=removeBeginAndEndWhitespaces(stringValue);
         parserAssert(isCompounded(tmp),strcat( qoute(stringValue)," is not a ","< ",Stringify<ValueType>::nameOfType()," > \nEBNF:\n",Stringify<ValueType>::ebnfRoules()));
         Splitter splitter(removeBeginAndEndWhitespaces(removeBeginAndEndCompoundSymbols(tmp))," ");
         value.resize(splitter.size());
         for(size_t i=0;i<splitter.size();++i)
            Stringify<T>::fromString(splitter[i],value[i]);
      }
   };
   
   
   
   template<class T,class U>
   class Stringify< std::pair<T,U> >   {
   public:
      typedef std::pair<T,U> ValueType;
      static std::string nameOfType(){
         return Stringify<T>::nameOfType()+std::string("-")+Stringify<U>::nameOfType();
      }
      static std::string ebnfRoules(){
         std::stringstream ss; ss<<"< "<<Stringify<ValueType>::nameOfType()<<"  >  =  ";
         ss<<qoute("[") ;
         ss<<" < "<<Stringify<T>::nameOfType()<<" >  "<<qoute(" ")<<"  <"<<Stringify<U>::nameOfType()<<" > ";
         ss<<qoute("]")<<"\n";
         ss<<Stringify<T>::ebnfRoules()<<"\n";
         ss<<Stringify<U>::ebnfRoules();
         return ss.str();
      }
      // to string
      static void toString(const ValueType & value,std::string & stringValue) {  
         stringValue.clear();
         stringValue.append("{ ");
         std::string f,s;
         Stringify<T>::toString(value.first,f);
         Stringify<U>::toString(value.second,s);
         stringValue.append(f).append(" ").append(s);
         stringValue.append("]");
      }
      // from string 
      static void fromString(const std::string & stringValue,ValueType & value) { 
         std::string tmp=removeBeginAndEndWhitespaces(stringValue);
         //parserAssert(isCompounded(tmp),strcat(qoute(stringValue),"is not a ","< ",Stringify<ValueType>::nameOfType()," > \n EBNF:\n",Stringify<ValueType>::ebnfRoules()));
         Splitter splitter(removeBeginAndEndCompoundSymbols(tmp)," ");
         parserAssert(splitter.size()==2,strcat(qoute(stringValue),"is not a ","< ",Stringify<ValueType>::nameOfType()," > \n EBNF:\n",Stringify<ValueType>::ebnfRoules()));
         Stringify<T>::fromString(splitter[0],value.first);
         Stringify<U>::fromString(splitter[1],value.second);
      }  
   };
}

#endif	/* STRINGIFY_HXX */

