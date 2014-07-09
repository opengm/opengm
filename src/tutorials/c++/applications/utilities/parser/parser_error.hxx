/* 
 * File:   parser_error.hxx
 * Author: tbeier
 *
 * Created on March 21, 2012, 11:08 AM
 */

#ifndef PARSER_ERROR_HXX
#define	PARSER_ERROR_HXX
#include <stdexcept>
namespace parser{

struct RuntimeError
: public std::runtime_error
{
   typedef std::runtime_error base;

   RuntimeError(const std::string& message)
   :  base(std::string("CMD-Parser error: \n ") + message \
    +std::string("\ntry\n\" -help \"    for help")\
    +std::string("\n\" -version \" for the version number")\
    +std::string("\n\" -author \"  for author information")\
   ) {}
};

void parserAssert(const bool exp,const std::string & errorMsg){
   if(!exp){
      throw RuntimeError(errorMsg);
   }
}
}

#endif	/* PARSER_ERROR_HXX */

