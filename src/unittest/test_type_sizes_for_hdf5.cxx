#include <opengm/opengm.hxx>
#include <opengm/unittests/test.hxx>

#define OPENGM_TEST_TYPE_SIZE(type,size) \
OPENGM_TEST(!opengm::meta::IsInvalidType< type >::value ); \
OPENGM_TEST_EQUAL(sizeof( type ),size_t( size ))

int main() {
   OPENGM_TEST_TYPE_SIZE(opengm::UInt64Type,8);
   OPENGM_TEST_TYPE_SIZE(opengm::Int64Type,8);
   OPENGM_TEST_TYPE_SIZE(opengm::detail_types::Float,4);
   OPENGM_TEST_TYPE_SIZE(opengm::detail_types::Double,8);
   return 0;
}
