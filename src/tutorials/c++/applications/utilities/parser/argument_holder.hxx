#ifndef ARGUMENT_PREPROCESSOR_HXX
#define	ARGUMENT_PREPROCESSOR_HXX

#include <set>
#include <vector>
#include "parser_error.hxx"
#include "argument_name.hxx"
#include "argument_value.hxx"
#include "stringify.hxx"
namespace parser{
   
   
   class ArgContainer{
      class ArgValuePair{
      public:
         ArgValuePair(const std::string & arg,const std::string & value):arg_(arg),value_(value){}
         const std::string & arg()const{return arg_; }
         void setArg(const std::string & arg){arg_ =arg; }
         const std::string & value()const{return value_; }
         void setValue(const std::string & value){ value_ =value;}
      private:
         std::string arg_;
         std::string value_;
      };
      public:
         ArgContainer(int argc,char ** argv,const std::string & paramPrefix="-",const std::set<std::string > & ignore=std::set<std::string >( ) );
         template<class T>
         void parseArg(const ArgName & argName,ArgValue<T> & argValue )const;
         bool hasArgument(const ArgName & )const;
         const ArgValuePair & operator[](const size_t) const;
         size_t size()const;
      private:
         bool isCompoundArg(const size_t)const;
         std::string getCompundArg(const size_t start,size_t & end);
         bool isArg(const std::string & arg)const;
         bool isCompoundStart(const std::string & arg)const;
         bool isCompoundEnd(const std::string & arg)const;
         std::string paramPrefix_;
         std::string compoundValueStart_;
         std::string compoundValueEnd_;
         std::vector<ArgValuePair  > argValues_;
         std::vector<std::string> argV_;
   };
   size_t ArgContainer::size()const{
      return argValues_.size();
   }
   const ArgContainer::ArgValuePair & ArgContainer::operator[](const size_t i) const{
      return argValues_[i];
   }
   
   bool ArgContainer::hasArgument(const ArgName & argName)const{
      bool fL=false,fS=false;
      for (size_t i = 0; i <  argValues_.size(); ++i) {
         if(argValues_[i].arg()==argName.longName()){
            fL=true;
            if(!argName.hasShortName())
               break;
         }
         else if(argName.hasShortName() && argValues_[i].arg()==argName.shortName()){
            fS=true;
         }
         parserAssert(!fL || !fS,strcat(" long name \" ", argName.longName()," \" long name \" ",argName.shortName() ," \""," are both set ."));
      }
      return fL || fS ;
   }
   
   template<class T>
   void ArgContainer::parseArg
   (
      const ArgName & argName,
      ArgValue<T> & argValue 
   )const{
      size_t pL,pS;
      bool fL=false,fS=false;
      for (size_t i = 0; i <  argValues_.size(); ++i) {
         if(argValues_[i].arg()==argName.longName()){
            pL=i;
            fL=true;
            if(!argName.hasShortName())
               break;
         }
         else if(argName.hasShortName() && argValues_[i].arg()==argName.shortName()){
            pS=i;
            fS=true;
         }
         parserAssert(!fL || !fS,strcat(" long name \" ", argName.longName()," \" long name \" ",argName.shortName() ," \""," are both set ."));
         
      }
      if(fL || fS){
         // bool as flag
         if(typeid(T).name() == typeid(bool).name() && argValues_[ fL ? pL :pS ].value().size()==0){
            Stringify<T>::fromString(std::string("1"),argValue.value());
         }
         else{
            try{
               Stringify<T>::fromString(argValues_[ fL ? pL :pS ].value(),argValue.value());
            }
            catch(const RuntimeError & rtE){
               throw RuntimeError(strcat("argument ",qoute(argName.longName()),": could not parse \" ",argValues_[ fL ? pL :pS ].value(),
                  " \" into a \" ",Stringify<T>::nameOfType()," \"\nReason:\n",rtE.what()));
            }
            catch(...){
               throw RuntimeError(strcat("argument ",qoute(argName.longName()),"could not parse \" ",argValues_[ fL ? pL :pS ].value(),
                  " \" into a \" ",Stringify<T>::nameOfType()," \"\nReason:\n","unknown error"));
            }
         }
      }
      else{
         if(argName.hasShortName())
            parserAssert(argValue.hasDefaultValue(),strcat("Required argument missing: long-name \" ",
               argName.longName()," \" and short-name \" ", argName.shortName(), " \""," are both NOT set but argument is required."));
         else
            parserAssert(argValue.hasDefaultValue(),strcat("Required argument missing: long-name \" ",
               argName.shortName()," \""," is NOT set but argument is required."));
         // take default value
         argValue.setValue(argValue.defaultValue());
      }
   }
   
   inline bool 
   ArgContainer::isCompoundArg(const size_t argIndex)const{
      const std::string & value=argValues_[argIndex].value();
      if(value.size() >= this->compoundValueStart_.size()){
         for(size_t i=0;i<compoundValueStart_.size();++i)
            if(value[i]!=compoundValueStart_[i])
               return false;
         return true;
      }
      return false;
   }
   
   ArgContainer::ArgContainer
   (
      int argc,
      char ** argv,
      const std::string & paramPrefix,
      const std::set<std::string > & ignore
   ) : paramPrefix_(paramPrefix),compoundValueStart_("["),compoundValueEnd_("]"){
      for (size_t i = 1; i < argc; ++i) {
         //std::cout<<"raw raw :"<<argv[i]<<"\n";
         if (ignore.find(std::string(argv[i])) == ignore.end()){
            //std::cout<<"raw raw :"<<argv[i]<<"\n";
            if(onlyWhitespaces(argv[i])==false){
               std::string  rawVal=removeBeginAndEndWhitespaces(std::string(argv[i]));
               rawVal=isolteCompoundSymbols(rawVal);
               Splitter splitter(rawVal," ");
               for(size_t i=0;i<splitter.size();++i){
                  //std::cout<<"splitter val :"<<removeBeginAndEndWhitespaces(splitter[i])<<"\n";
                  argV_.push_back( removeBeginAndEndWhitespaces(splitter[i]) );
               }
            }
         }
      }
      if (argV_.size() != 0) {
         for (size_t i = 0; i < argV_.size(); ++i) {
            std::string argVal = argV_[i];
            parserAssert(isArg(argVal),strcat(qoute(argVal)," is not a valid argument but should an argument ( or forgotten ",qoute("[")," could be the reason )"));
       
            if(isArg(argVal)==true) {
               size_t j = (i + 1);
               if(j<argV_.size()){
                  std::string stringValue(argV_[j]);        
                  if (isCompoundEnd(stringValue))
                     parserAssert(i < argV_.size(), std::string("missing \"[\""));
                  else if (isCompoundStart(stringValue)) {
                     size_t end;
                     std::string compoundStringValue = getCompundArg(j, end);
                     argValues_.push_back(ArgValuePair(argVal, compoundStringValue));
                     i = end;
                  }
                  else if (isArg(stringValue)==true) {
                     argValues_.push_back(ArgValuePair(argVal, ""));
                  }
                  else {
                     parserAssert(hasNoneOf(stringValue,' ' ) ,strcat("whitepsaces in uncompounded argument value ",qoute(stringValue)," detected"));
                     argValues_.push_back(ArgValuePair(argVal, stringValue));
                     i=j;
                  }
               }
               else{
                  argValues_.push_back(ArgValuePair(argVal, ""));
               }
            }
         }
      }
   }

   std::string
   ArgContainer::getCompundArg
   (
      const size_t start, 
      size_t & end
   ) {
      // assertions
      parserAssert(isCompoundStart(argV_[start]), std::string("internal error"));
      parserAssert(start + 2 <= argV_.size() - 1, std::string(" unbalanced compound symbol detected"));
      parserAssert(isCompoundEnd(argV_[start + 1]) == false, std::string("empty compound value") + compoundValueStart_ + std::string(" ") + compoundValueEnd_);
      size_t openSymbols = 1, closeSymbols = 0, i = start + 1;
      std::string compoundValue = argV_[start];
      while (openSymbols != closeSymbols) {
         parserAssert(i < argV_.size(), std::string("unbalanced compound symbol detected"));
         parserAssert(openSymbols>closeSymbols, std::string("unbalanced compound symbol detected"));
         if (isCompoundStart(argV_[i]))
            ++openSymbols;
         else if (isCompoundEnd(argV_[i]))
            ++closeSymbols;
         compoundValue.push_back(' ');
         compoundValue.append(argV_[i]);
            ++i;
      }
      end = i-1;
      parserAssert(end >= start + 2, std::string("internal error"));
      parserAssert(isCompoundEnd(argV_[i - 1]), std::string("internal error"));
      return compoundValue;
   }

   inline bool
   ArgContainer::isArg(const std::string & arg)const {
      //parserAssert(arg != paramPrefix_, arg + std::string(" is not a valid argument"));
      if (arg.size() < paramPrefix_.size())
         return false;
      for (size_t i = 0; i < paramPrefix_.size(); ++i) {
         if (arg[i] != paramPrefix_[i])
            return false;
      }
      return true;
   }

   inline bool
   ArgContainer::isCompoundStart
   (
      const std::string & arg
   )const {
      return arg == compoundValueStart_;
   }
   

   inline bool
   ArgContainer::isCompoundEnd
   (
      const std::string & arg
   )const {
      return arg == compoundValueEnd_;
   }
}
#endif	/* ARGUMENT_PREPROCESSOR_HXX */

