#ifndef INTERFACESTYPEDEFS_HXX_
#define INTERFACESTYPEDEFS_HXX_

#include <opengm/utilities/metaprogramming.hxx>

//operations
#include <opengm/operations/adder.hxx>
#include <opengm/operations/multiplier.hxx>
#include <opengm/operations/and.hxx>
#include <opengm/operations/or.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/operations/maximizer.hxx>
#include <opengm/operations/integrator.hxx>

//algorithms
#include "../caller/icm_caller.hxx"
#include "../caller/bruteforce_caller.hxx"
#include "../caller/messagepassing_bp_caller.hxx"

//functions
#include <opengm/functions/potts.hxx>
#include <opengm/functions/pottsn.hxx>

namespace opengm {

namespace interface {

template <typename T>
struct getNameFromType;

#define REGISTER_PARSE_TYPE(TYPE, NAME) \
   template <> struct getNameFromType<TYPE> { \
   static const std::string name; \
   }; \
   const std::string getNameFromType<TYPE>::name = NAME;

template <class VALUETYPE>
struct functionTypeList {
   typedef typename opengm::meta::TypeListGenerator<
      opengm::PottsFunction<VALUETYPE>,
      opengm::PottsNFunction<VALUETYPE>
   >::type list;
};

REGISTER_PARSE_TYPE(double, "DOUBLE");
REGISTER_PARSE_TYPE(float, "FLOAT");
REGISTER_PARSE_TYPE(int, "INT");

typedef opengm::meta::TypeListGenerator <
   double,
   float,
   int
>::type ValueTypeList;


REGISTER_PARSE_TYPE(Adder, "ADDER");
REGISTER_PARSE_TYPE(Multiplier, "MULTIPLIER");
//REGISTER_PARSE_TYPE(And, "AND");
//REGISTER_PARSE_TYPE(Or, "OR");

typedef opengm::meta::TypeListGenerator <
   Adder,
   Multiplier/*,
   And,
   Or*/
>::type OperatorTypeList;

typedef opengm::meta::TypeList<opengm::meta::ListEnd, opengm::meta::ListEnd> AccumulatorTypeListBegin;
REGISTER_PARSE_TYPE(Minimizer, "MIN");
REGISTER_PARSE_TYPE(Maximizer, "MAX");
REGISTER_PARSE_TYPE(Integrator, "INT");

typedef opengm::meta::TypeListGenerator <
   Minimizer,
   Maximizer,
   Integrator
>::type AccumulatorTypeList;

typedef opengm::meta::Bind3<ICMCaller> icmtest;
REGISTER_PARSE_TYPE(icmtest, "ICM");
typedef opengm::meta::Bind3<BruteforceCaller> bruteforcetest;
REGISTER_PARSE_TYPE(bruteforcetest, "BRUTEFORCE");
typedef opengm::meta::Bind3<MessagepassingBPCaller> beliefpropagationtest;
REGISTER_PARSE_TYPE(beliefpropagationtest, "BELIEFPROPAGATION");

typedef opengm::meta::TypeListGenerator <
   icmtest,
   bruteforcetest,
   beliefpropagationtest
>::type InferenceTypeList;

} //namespace interface

} // namespace opengm
#endif /* INTERFACESTYPEDEFS_HXX_ */
