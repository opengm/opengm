#ifndef PARSING2_HXX
#define PARSING2_HXX

#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <typeinfo>
#include <set>
#include <cstdlib>
#include <iostream>

#include "argument_holder.hxx"
#include "string_utilities.hxx"
#include "enableif.hxx"
#include "argument.hxx"
#include "restrictions.hxx"
#include "argument_root.hxx"
#include "argument_base.hxx"


namespace parser{

class ProgressPrinter{
public:
   ProgressPrinter(const size_t size,const size_t nth,const std::string & name):size_(size),nth_(nth),name_(name){
   }
   void operator()(const size_t i)const{
      if(i==0 || i==size_-1 || i%nth_==0){
         if(i==0){
            std::cout.precision(2);
            std::cout<<"\r"<<name_<<" 0   "<<" % "<<" [ "<<std::setw(10)<<std::left<<i+1<<" / "<<size_<<" ]          ";
            std::cout<<"\r                                     ";
         }
         if(i==size_-1){
            std::cout.precision(2);
            std::cout<<"\r                                                         ";
            std::cout<<"\r"<<name_<<" 100 "<<" % "<<" [ "<<std::setw(10)<<std::left<<i+1<<" / "<<size_<<" ]           "<<"\n";
         }
         if(i<size_-2){
            std::cout.precision(2);
            std::cout.clear();
            //std::cout<<"\r                                                         ";
            std::cout<<"\r"<<name_<<std::setw(5)<<std::left<<float(i+1)/float(size_)*100.0<<" % "<<" [ "<<std::setw(10)<<std::left<<i+1<<" / "<<size_<<" ]            ";
         }
      }
   }
   size_t size_;
   size_t nth_;
   std::string name_;
};
   
class ArgumentBase;

class Arg{
   friend class ArgumentBase;
   friend class CmdParser;
public:
   Arg(ArgumentBase * ptr =NULL):argPtr_(ptr){ 
   }
   ArgumentBase * argPointer(){
      return argPtr_;
   }
   ArgumentBase * argPointer()const{
      return argPtr_;
   }
private:
   ArgumentBase * argPtr_;
};
template<class T>
class IfParentArg{
public:
   IfParentArg(const Arg & arg, const T value)
   :arg_(arg),value_(value){

   }
   const Arg & arg()const{return arg_;}
   const T & value()const{return value_;}
   Arg & arg(){return arg_;}
   T & value(){return value_;}
private:
   Arg arg_;
   T value_;
};   
   
class CmdParser{
public:
   
    CmdParser(int argc, char **argv,const std::string & name,const std::string & desc,const std::string & version,const std::string & author);
   
   void error(const std::string msg)const;
   void helpAndExit()const;
   void versionAndExit()const;
   void authorAndExit()const;
  
   void print()const;
   void parse();
   // add scalar arg
   template<class T>
   Arg addArg(const ArgName & argName,const ArgValue<T> & argValue);
   template<class T,class U>
   Arg addArg(const ArgName & argName,const ArgValue<T> & argValue,const IfParentArg<U> & ifArgValue );
   // add vector arg
   template<class T>
   Arg addArg(const ArgName & argName,const ArgValue<std::vector<T> > & argValue);
   template<class T,class U>
   Arg addArg(const ArgName & argName,const ArgValue<std::vector<T> > & argValue,const IfParentArg<U> & ifArgValue );

private:
   std::string argumentIdentifier_;
   size_t tabSize_;
   bool coloredHelp_;
   std::set<std::string> igonre_;
   ArgumentBase * rootArg_;
   ArgContainer argContainer_;
   std::set<std::string> addedArgumentNames_;
   
   std::string author_,version_;
};



 void CmdParser::error (const std::string msg)const{
   std::string errorMsg="\nCMD-Paser Error:\n";
   errorMsg.append(msg);
   errorMsg.append("\ntry \" -help \" for help");
   throw std::runtime_error(errorMsg);
}
   
void CmdParser::helpAndExit()const{
   this->rootArg_->print(0);
   exit(0);
}


void CmdParser::versionAndExit()const{
   std::cout<<rootArg_->longName()<<" : \n";
   std::cout<<"Version : "<<version_<<"\n";
   exit(0);
}

void CmdParser::authorAndExit()const{
   std::cout<<rootArg_->longName()<<" : \n";
   std::cout<<"Author(s) : "<<author_<<"\n";
   exit(0);
}

   
   
CmdParser::CmdParser
(
   int argc,
   char **argv,
   const std::string & name,
   const std::string & desc,
   const std::string & version,
   const std::string & author
):argContainer_(argc,argv),version_(version),author_(author){
   //std::cout<<"number of arguments:"<<argc<<"\n";
   bool printOther=false;
   rootArg_ = new RootArgument(name,desc);
   if(
      argContainer_.hasArgument(ArgName("-help","")) ||
      argContainer_.hasArgument(ArgName("-version","")) ||
      argContainer_.hasArgument(ArgName("-author","")) 
      ){
      printOther=true;
   }  
   if(argc<1){
      this->error("wrong number of arguments");
   }
}
   
void CmdParser::print()const{
   rootArg_->print(0);
}
void CmdParser::parse(){
   if(argContainer_.hasArgument(ArgName("-help","")))
      this->helpAndExit();
   else if(argContainer_.hasArgument(ArgName("-author","")))
      this->authorAndExit();
   else if(argContainer_.hasArgument(ArgName("-version","")))
      this->versionAndExit();
   
   // check if there are any options which are unknown
   for(size_t i=0;i<argContainer_.size();++i){
      if(this->addedArgumentNames_.find(argContainer_[i].arg())==addedArgumentNames_.end())
         throw RuntimeError(std::string("unknown argument ") + std::string(argContainer_[i].arg()));
   }
   rootArg_->parse(argContainer_);
}


template<class T>
Arg CmdParser::addArg
(
   const ArgName & argName,
   const ArgValue<T> & argValue
){
   if(addedArgumentNames_.find(argName.longName())!=addedArgumentNames_.end())
      throw RuntimeError(std::string("the parameter name ") + std::string(argName.longName())+std::string(" is taken"));
   else
      addedArgumentNames_.insert(argName.longName());
   if(argName.hasShortName()){
      if(addedArgumentNames_.find(argName.shortName())!=addedArgumentNames_.end())
         throw RuntimeError(std::string("the parameter name ") + std::string(argName.shortName())+std::string(" is taken"));
      else
         addedArgumentNames_.insert(argName.shortName());
   }
   ArgumentBase * newArg = new Argument<T,RestrictedToAllowedValues<T>,Enabled,true>(
   RestrictedToAllowedValues<T>(argValue.allowedValues()),Enabled(),argName,argValue
   );
   rootArg_->addChild(newArg);
   return Arg(newArg);
}

template<class T,class U>
Arg CmdParser::addArg
(
   const ArgName & argName,
   const ArgValue<T> & argValue,
   const IfParentArg<U> & ifArgValue
){
   if(addedArgumentNames_.find(argName.longName())!=addedArgumentNames_.end())
      throw RuntimeError(std::string("the parameter name ") + std::string(argName.longName())+std::string(" is taken"));
   else
      addedArgumentNames_.insert(argName.longName());
   if(argName.hasShortName()){
      if(addedArgumentNames_.find(argName.shortName())!=addedArgumentNames_.end())
         throw RuntimeError(std::string("the parameter name ") + std::string(argName.shortName())+std::string(" is taken"));
      else
         addedArgumentNames_.insert(argName.shortName());
   }
   ArgumentBase * newArg =  new Argument<T,RestrictedToAllowedValues<T>,EnabledIfParentHasValue<ArgumentBase,U>,true>(
      RestrictedToAllowedValues<T>(argValue.allowedValues()),EnabledIfParentHasValue<ArgumentBase,U>(ifArgValue.arg().argPointer()   ,ifArgValue.value()),argName,argValue
   );
   ifArgValue.arg().argPointer()->addChild(newArg);
   return Arg(newArg);
}


template<class T>
Arg CmdParser::addArg
(
   const ArgName & argName,
   const ArgValue<std::vector<T> > & argValue
){
   typedef std::vector<T> ArgValueType;
   if(addedArgumentNames_.find(argName.longName())!=addedArgumentNames_.end())
      throw RuntimeError(std::string("the parameter name ") + std::string(argName.longName())+std::string(" is taken"));
   else
      addedArgumentNames_.insert(argName.longName());
   if(argName.hasShortName()){
      if(addedArgumentNames_.find(argName.shortName())!=addedArgumentNames_.end())
         throw RuntimeError(std::string("the parameter name ") + std::string(argName.shortName())+std::string(" is taken"));
      else
         addedArgumentNames_.insert(argName.shortName());
   }
   ArgumentBase * newArg = new Argument<ArgValueType,RestrictedToAllowedValues<ArgValueType>,Enabled,false>(
   RestrictedToAllowedValues<ArgValueType>(argValue.allowedValues()),Enabled(),argName,argValue
   );
   rootArg_->addChild(newArg);
   return Arg(newArg);
}

template<class T,class U>
Arg CmdParser::addArg
(
   const ArgName & argName,
   const ArgValue<std::vector<T> > & argValue,
   const IfParentArg<U> & ifArgValue
){
   typedef std::vector<T> ArgValueType;
   if(addedArgumentNames_.find(argName.longName())!=addedArgumentNames_.end())
      throw RuntimeError(std::string("the parameter name ") + std::string(argName.longName())+std::string(" is taken"));
   else
      addedArgumentNames_.insert(argName.longName());
   if(argName.hasShortName()){
      if(addedArgumentNames_.find(argName.shortName())!=addedArgumentNames_.end())
         throw RuntimeError(std::string("the parameter name ") + std::string(argName.shortName())+std::string(" is taken"));
      else
         addedArgumentNames_.insert(argName.shortName());
   }
   ArgumentBase * newArg =  new Argument<ArgValueType,RestrictedToAllowedValues<ArgValueType>,EnabledIfParentHasValue<ArgumentBase,U>,true>(
      RestrictedToAllowedValues<T>(argValue.allowedValues()),EnabledIfParentHasValue<ArgumentBase,U>(ifArgValue.arg().argPointer()   ,ifArgValue.value()),argName,argValue
   );
   ifArgValue.arg().argPointer()->addChild(newArg);
   return Arg(newArg);
}



} //namespace opengm
#endif

