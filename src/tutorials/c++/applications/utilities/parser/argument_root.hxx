#ifndef ARGUMENT_ROOT_HXX
#define	ARGUMENT_ROOT_HXX

#include <vector>
#include <string>
#include "argument_name.hxx"
#include "argument_base.hxx"
#include "stringify.hxx"

namespace parser{
   
   class RootArgument 
   : public ArgumentBase {
   public:
      RootArgument(const std::string & name ,const std::string & description);
      std::string valueToString()const;
      std::string ifParentValue()const;
      bool hasDefault() const;
      bool isScalarArg()const;
      void parse( const ArgContainer & parser );
      void printBrief(const size_t level)const;
      void print(const size_t level)const;
      virtual void collectTypesEbnfs(std::set<std::string> & ebfn)const;
      virtual size_t depth(const size_t current)const;
   };
   
   void RootArgument::collectTypesEbnfs(std::set<std::string> & ebfn)const{
      for(size_t i=0;i<this->numberOfChildren();++i){
         this->children(i)->collectTypesEbnfs(ebfn);
      }
   }
   
   size_t RootArgument::depth(const size_t current)const{
      size_t c=current;
      for(size_t i=0;i<this->numberOfChildren();++i){
         c=c<this->children(i)->depth(c+1) ? this->children(i)->depth(c+1):c;
      }
      return c;
   }
   
   RootArgument::RootArgument
   (
      const std::string & name,
      const std::string & description
   ):ArgumentBase(ArgName(name,"",description),NULL){

   }
   
   std::string RootArgument::valueToString()const{
      return "0";
   }
   
   std::string RootArgument::ifParentValue()const{
      return "0";
   }
   
   bool RootArgument::hasDefault() const {
      return true;
   }
   
   bool  RootArgument::isScalarArg()const {
      return true;
   }
   
   void RootArgument::parse
   (
      const ArgContainer & argContainer 
   ){ 
      for(size_t i=0;i<this->numberOfChildren();++i)
         this->children(i)->parse(argContainer);
   }
   
   void RootArgument::print
   (
      const size_t level
   )const { 
      
      std::set<std::string > ebfns;
      this->collectTypesEbnfs(ebfns);
      std::cout<<getLine(80)<<"\n";
      std::cout<<"CMD Arguments Types of "<<this->longName()<<"\n";
      for(std::set<std::string >::const_iterator begin=ebfns.begin();begin!=ebfns.end();++begin){
         std::cout<<getLine(80)<<"\n";
         std::cout<<*begin<<"\n";
        
      }
      std::cout<<getLine(80)<<"\n";
      
      std::cout<<"CMD-Arguments of "<<this->longName()<<"\n";
      std::cout<<getLine(80)<<"\n";
      for(size_t i=0;i<this->numberOfChildren();++i){
         this->children(i)->print(level+1);
         std::cout<<"\n";
      }
      std::cout<<getLine(80)<<"\n";
   }
}

   
#endif	/* ARGUMENT_ROOT_HXX */

