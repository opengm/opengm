#ifndef OPENGM_FUNCTION_GEN_BASE
#define OPENGM_FUNCTION_GEN_BASE

#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>

#include <stdexcept>
#include <stddef.h>
#include <vector>
#include <map>

#include "opengm/utilities/functors.hxx"
#include "opengm/functions/explicit_function.hxx"
#include "opengm/functions/absolute_difference.hxx"
#include "opengm/functions/potts.hxx"
#include "opengm/functions/pottsn.hxx"
#include "opengm/functions/pottsg.hxx"
#include "opengm/functions/squared_difference.hxx"
#include "opengm/functions/truncated_absolute_difference.hxx"
#include "opengm/functions/truncated_squared_difference.hxx"
#include "opengm/functions/sparsemarray.hxx"
#include <opengm/python/opengmpython.hxx>
#include <opengm/python/converter.hxx>
#include <opengm/python/numpyview.hxx>
#include <opengm/python/pythonfunction.hxx>


template<class GM_ADDER,class GM_MULT>
class FunctionGeneratorBase{
public:

    FunctionGeneratorBase(){}
    virtual ~FunctionGeneratorBase(){}

    virtual std::vector<typename GM_ADDER::FunctionIdentifier> * addFunctions(GM_ADDER & gm) const=0;
    virtual std::vector<typename GM_MULT::FunctionIdentifier>  * addFunctions(GM_MULT & gm) const=0;
private:
    //
};



#endif //OPENGM_FUNCTION_GEN_BASE
