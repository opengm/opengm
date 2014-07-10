#ifndef ARGUMENT_NAME_HXX
#define	ARGUMENT_NAME_HXX

#include <vector>
#include "parser_error.hxx"

namespace parser{

   
class ArgName{
public:
   ArgName(const std::string & longName,const std::string & shortName,const std::string & description);
   ArgName(const std::string & longName,const std::string & description);

   bool hasShortName()const;
   std::string longName()const;
   std::string description()const;
   std::string shortName()const;
private:
   std::string longName_,shortName_,description_;
};


ArgName::ArgName(
   const std::string & longName,
   const std::string & shortName,
   const std::string & description
)
:longName_(longName),shortName_(shortName),description_(description){
}

ArgName::ArgName
(
   const std::string & longName,
   const std::string & description
)
:longName_(longName),shortName_(""),description_(description){
}

bool ArgName::hasShortName()const{
   return shortName_.size()>0;
}
std::string ArgName::longName()const{
   return longName_;
}
std::string ArgName::description()const{
   return description_;
}
std::string ArgName::shortName()const{
   return shortName_;
}
   
}

   
#endif	/* ARGUMENT_NAME_HXX */

