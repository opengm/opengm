#ifndef ARGUMENTS_HXX
#define	ARGUMENTS_HXX

#include "argument_base.hxx"
#include "argument_name.hxx"
#include "argument_value.hxx"
#include "stringify.hxx"
#include <iomanip>

namespace parser{
   
   template< class T,class RESTRICTIONS,class ENABLE_IF,bool SCALAR >
   class Argument : public ArgumentBase {
   public:
      typedef T ValueType ;
      Argument(const RESTRICTIONS & ,const ENABLE_IF & ,const ArgName & ,const ArgValue<T> & ,ArgumentBase * =NULL);
      // pure virtual interface
      bool hasDefault() const;
      bool  isScalarArg()const ;
      virtual std::string valueToString()const;
      // template specific implementation (not part of virtual interface)
      const T &  value() const;
      virtual std::string ifParentValue()const;
      virtual void parse( const ArgContainer & );
      virtual void print(const size_t level)const;
      virtual void collectTypesEbnfs(std::set<std::string> & )const;
      virtual size_t depth(const size_t current)const;
   protected:      
      ArgValue<T> argValue_;
      RESTRICTIONS restriction_;
      ENABLE_IF enableIf_;
   };
   
   template< class T,class RESTRICTIONS,class ENABLE_IF,bool SCALAR >
   void Argument<T,RESTRICTIONS,ENABLE_IF,SCALAR>::collectTypesEbnfs(std::set<std::string> & ebfn)const{
      ebfn.insert(Stringify<T>::ebnfRoules());
      for(size_t i=0;i<this->numberOfChildren();++i){
         this->children(i)->collectTypesEbnfs(ebfn);
      }
   }
   
   template< class T,class RESTRICTIONS,class ENABLE_IF,bool SCALAR >
   size_t Argument<T,RESTRICTIONS,ENABLE_IF,SCALAR>::depth(const size_t current)const{
      size_t c=current;
      for(size_t i=0;i<this->numberOfChildren();++i){
         c=c<this->children(i)->depth(c+1) ? this->children(i)->depth(c+1):c;
      }
      return c;
   }
   
   template< class T,class RESTRICTIONS,class ENABLE_IF,bool SCALAR >
   Argument<T,RESTRICTIONS,ENABLE_IF,SCALAR>::Argument
   (
      const RESTRICTIONS & restriction,
      const ENABLE_IF & enableIf,
      const ArgName & argName,
      const ArgValue<T> & argValue,
      ArgumentBase * parent
   )
   :ArgumentBase(argName,parent),argValue_(argValue),restriction_(restriction),enableIf_(enableIf){
   }
   
   template< class T,class RESTRICTIONS,class ENABLE_IF,bool SCALAR >
   bool 
   Argument<T,RESTRICTIONS,ENABLE_IF,SCALAR>::hasDefault() const {
      return argValue_.hasDefaultValue();
   }
   
   template< class T,class RESTRICTIONS,class ENABLE_IF,bool SCALAR >
   bool  
   Argument<T,RESTRICTIONS,ENABLE_IF,SCALAR>::isScalarArg()const {
      return SCALAR;
   }
   
   template< class T,class RESTRICTIONS,class ENABLE_IF,bool SCALAR >
   std::string 
   Argument<T,RESTRICTIONS,ENABLE_IF,SCALAR>::valueToString()const{
      if(this->isParsed()==false){
         throw RuntimeError("error");
      }
      std::string tmp;
      Stringify<T>::toString(argValue_.value(),tmp);
      return tmp;
   }
   
   template< class T,class RESTRICTIONS,class ENABLE_IF,bool SCALAR >
   const T &  
   Argument<T,RESTRICTIONS,ENABLE_IF,SCALAR>::value()     const {
      return argValue_.value();
   }
   
   template< class T,class RESTRICTIONS,class ENABLE_IF,bool SCALAR >
   std::string 
   Argument<T,RESTRICTIONS,ENABLE_IF,SCALAR>::ifParentValue()const{
      return enableIf_.ifParentValue();
   }
   
   template< class T,class RESTRICTIONS,class ENABLE_IF,bool SCALAR >
   void 
   Argument<T,RESTRICTIONS,ENABLE_IF,SCALAR>::parse
   ( 
      const ArgContainer & argContainer 
   ){ 
      this->setAsUnparsedParsed();
      if(enableIf_()==true){
         argContainer.parseArg(argName_,argValue_);
         if(restriction_(argValue_.value())==false){
            std::stringstream ss;
            std::string asString;
            std::cout<<"to string for arg "<<argName_.longName()<<" \n";
            Stringify<T>::toString(argValue_.value(),asString);
            std::string errorMsg="the value ";
            errorMsg.append(qoute(asString)).append(" is not allowed for the argument ").append(this->longName()).append(":").append(restriction_.restrictionDescription());
            throw RuntimeError(errorMsg);
         }
         else{
            this->setAsParsed();
            // parse children
            for(size_t i=0;i<this->numberOfChildren();++i){
               this->children(i)->parse(argContainer);
            }
         }
      }
      else if(argContainer.hasArgument(argName_)){
            std::string errorMsg ="Error unneeded argument is set :\nThe argument ";
            errorMsg.append(argName_.longName()).append(" is only enabled if : ").append(enableIf_.description()).append(" .");
            throw RuntimeError(errorMsg);
      }
   }
   
   template< class T,class RESTRICTIONS,class ENABLE_IF,bool SCALAR >
   void 
   Argument<T,RESTRICTIONS,ENABLE_IF,SCALAR>::print
   (
      const size_t level
   )const { 
       std::cout<<getSpace((level-1)*3);
       if(this->hasDefault())
          std::cout<<"[ ";
       
       std::cout<<qoute(colorString(this->longName() ,Green ));
       if(this->hasShortName())
          std::cout<<" | "<<qoute(colorString(this->shortName() ,Green ));
       if(restriction_.canBeRestricted()){
         std::cout<<" ="<<restriction_.restrictionDescription2()<<" ";
       }
       else{
          std::cout<<" < "<<colorString(Stringify<T>::nameOfType(),Yellow)<<" > ";
       }
       if(this->hasDefault()){
          std::string tmp;
          Stringify<T>::toString(this->argValue_.defaultValue(),tmp);
          std::cout<<" ( = "<<qoute(colorString(tmp,Red))<<" )";
       }
       if(this->hasDefault())
          std::cout<<" ] ";
       std::cout<<"\n";
       std::cout<<getSpace((level-1)*3);
       std::cout<<this->description()<<"\n";
       std::vector<bool> isPrinted(this->numberOfChildren(),false);
       std::vector<std::string> enableIfParentValue(this->numberOfChildren());
       for(size_t i=0;i<this->numberOfChildren();++i){
          enableIfParentValue[i]=this->children(i)->ifParentValue();
       }
       for(size_t i=0;i<this->numberOfChildren();++i){
          if(isPrinted[i]==false){
             std::cout<<"\n"<<getSpace((level-1)*3);
             std::cout<<colorString("if",Blue)<<" "<<qoute(colorString(this->longName() ,Green ))<<colorString(" == ",Blue)<<qoute(colorString(enableIfParentValue[i],Purple))<<"\n\n";
             this->children(i)->print(level+1);
             isPrinted[i]=true;
             for(size_t j=i+1;j<this->numberOfChildren();++j){
                if(isPrinted[j]==false && enableIfParentValue[j]==enableIfParentValue[i]){
                   std::cout<<"\n";
                   this->children(j)->print(level+1);
                   isPrinted[j]=true;

                }
             }
          }
       }
       std::cout<<" ";
    }
}
#endif	/* ARGUMENTS_HXX */

