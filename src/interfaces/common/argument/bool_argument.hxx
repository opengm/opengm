#ifndef BOOL_ARGUMENT_HXX_
#define BOOL_ARGUMENT_HXX_

#include "argument_base.hxx"

namespace opengm {

namespace interface {

/*********************
 * class definitions *
 *********************/

class BoolArgument : public ArgumentBase<bool> {
public:
   BoolArgument(bool& storageIn, const std::string& shortNameIn,
       const std::string& longNameIn, const std::string& descriptionIn);
};

/***********************
 * class documentation *
 ***********************/
//TODO add documentation

/******************
 * implementation *
 ******************/

BoolArgument::BoolArgument(bool& storageIn, const std::string& shortNameIn, const std::string& longNameIn, const std::string& descriptionIn)
   : ArgumentBase<bool>(storageIn, shortNameIn, longNameIn, descriptionIn) {
   ;
}

} // namespace interface

} // namespace opengm

#endif /* BOOL_ARGUMENT_HXX_ */
