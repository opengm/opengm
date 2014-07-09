#ifndef ARGUMENT_BASE_HXX
#define	ARGUMENT_BASE_HXX

#include <vector>

#include "argument_holder.hxx"
#include "argument_name.hxx"
#include "parser_error.hxx"

namespace parser{

   
   
class ArgumentBase{
public:
   // constructor
   ArgumentBase(const ArgName & ,ArgumentBase * =NULL,const std::vector<ArgumentBase * >  & =std::vector<ArgumentBase * >());
   // virtual with default implementation
   virtual std::string longName()const;
   virtual std::string shortName()const;
   virtual bool hasShortName()const;
   virtual std::string description()const;
   virtual bool isParsed()const;
   virtual void setAsParsed();
   virtual void setAsUnparsedParsed();
   // virtual graph function with default implementation
   virtual size_t numberOfChildren()const;
   virtual ArgumentBase * children(const size_t);
   virtual ArgumentBase const * children(const size_t)const;
   virtual ArgumentBase * parent();
   virtual ArgumentBase const * parent()const;
   virtual void addChild(ArgumentBase * );
   virtual void setParent(ArgumentBase *);

   // pure virtual functions WITHOUT implementation
   virtual std::string valueToString()const =0;
   virtual std::string ifParentValue()const =0;
   virtual bool hasDefault() const  =0;
   virtual bool  isScalarArg()const =0;
   virtual void parse( const ArgContainer &  )=0;
   virtual void print(const size_t )const=0;
   virtual void collectTypesEbnfs(std::set<std::string> & ebnfs)const=0;
   virtual size_t depth(const size_t current)const=0;
protected:
   ArgName argName_;
   bool isParsed_;
   ArgumentBase * parent_;
   std::vector<ArgumentBase * > children_;
};


ArgumentBase::ArgumentBase
(
   const ArgName & argName,
   ArgumentBase * parent,
   const std::vector<ArgumentBase * >  & children
)
:argName_(argName),parent_(parent),children_(children){
}

// virtual with default implementation
std::string ArgumentBase::longName()const{
   return argName_.longName();
}

std::string ArgumentBase::shortName()const{
   return argName_.shortName();
}

bool ArgumentBase::hasShortName()const{
   return argName_.hasShortName();
}

std::string ArgumentBase::description()const{
   return argName_.description();
}

bool ArgumentBase::isParsed()const{
   return isParsed_;
}

void ArgumentBase::setAsParsed(){
   isParsed_=true;
}

void ArgumentBase::setAsUnparsedParsed(){
   isParsed_=false;
}

size_t ArgumentBase::numberOfChildren()const{
   return children_.size(); 
}

ArgumentBase * ArgumentBase::children
(
   const size_t i
){
   return children_[i];
}

ArgumentBase const * ArgumentBase::children
(
   const size_t i
)const{
   return children_[i];
}

ArgumentBase * ArgumentBase::parent(){
   return parent_;
}

ArgumentBase  const * ArgumentBase::parent()const{
   return parent_;
}

void ArgumentBase::addChild
(
   ArgumentBase * child
){
   children_.push_back(child);
   child->setParent(this);
}

void ArgumentBase::setParent(
ArgumentBase * parent
){
   parent_=parent;
}

}
#endif	/* ARGUMENT_BASE_HXX */

