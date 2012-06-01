#include <opengm/opengm.hxx>
#include <opengm/unittests/test.hxx>

#define OPENGM_TEST_TYPE_SIZE( type, size) \
OPENGM_TEST(sizeof( type )==size_t( size )||opengm::meta::IsInvalidType< type >::value)

int main() {

	typedef opengm::meta::TypeListGenerator<float,int,size_t>::type TA;
	typedef opengm::meta::TypeListGenerator<double,char,bool>::type TB;

	typedef opengm::meta::MergeTypeLists<TA,TB>::type TAB;
	
   OPENGM_TEST_TYPE_SIZE(opengm::UInt8Type,1);
   OPENGM_TEST_TYPE_SIZE(opengm::UInt16Type,2);
   OPENGM_TEST_TYPE_SIZE(opengm::UInt32Type,4);
   OPENGM_TEST_TYPE_SIZE(opengm::UInt64Type,8);
   OPENGM_TEST_TYPE_SIZE(opengm::Int8Type,1);
   OPENGM_TEST_TYPE_SIZE(opengm::Int16Type,2);
   OPENGM_TEST_TYPE_SIZE(opengm::Int32Type,4);
   OPENGM_TEST_TYPE_SIZE(opengm::Int64Type,8);
   OPENGM_TEST_TYPE_SIZE(opengm::Float32Type,4);
   OPENGM_TEST_TYPE_SIZE(opengm::Float64Type,8);
   OPENGM_TEST(sizeof(opengm::UInt8Type)+sizeof(opengm::UInt16Type)+sizeof(opengm::UInt32Type)+sizeof(opengm::UInt64Type) > 0);
   OPENGM_TEST(sizeof(opengm::Int8Type)+sizeof(opengm::Int16Type)+sizeof(opengm::Int32Type)+sizeof(opengm::Int64Type) > 0);
   return 0;
}
