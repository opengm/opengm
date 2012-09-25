
#ifndef ITERATORTOTUPLE_HXX
#define	ITERATORTOTUPLE_HXX


#include <boost/python.hpp>
#include <opengm/graphicalmodel/graphicalmodel.hxx>

using namespace boost::python;

#define MAX_ITERATOR_TO_TUPLE_SIZE 10

template<class ITERATOR, class CAST_TO, class IF_ZERO>
boost::python::tuple iteratorToTuple
(
ITERATOR it,
const size_t size,
const IF_ZERO  ifZeroValue
) {
   typedef CAST_TO V;
   switch (size) {
      case 0:
         return make_tuple(static_cast<IF_ZERO> (ifZeroValue));
      case 1:
         return make_tuple(static_cast<V>(it[0]));
      case 2:
         return make_tuple(static_cast<V> (it[0]), static_cast<V> (it[1]));
      case 3:
         return make_tuple(static_cast<V> (it[0]), static_cast<V> (it[1]), static_cast<V> (it[2]));
      case 4:
         return make_tuple(static_cast<V> (it[0]), static_cast<V> (it[1]), static_cast<V> (it[2]),
            static_cast<V> (it[3]));
      case 5:
         return make_tuple(static_cast<V> (it[0]), static_cast<V> (it[1]), static_cast<V> (it[2]),
            static_cast<V> (it[3]), static_cast<V> (it[4]));
      case 6:
         return make_tuple(static_cast<V> (it[0]), static_cast<V> (it[1]), static_cast<V> (it[2]),
            static_cast<V> (it[3]), static_cast<V> (it[4]), static_cast<V> (it[5]));
      case 7:
         return make_tuple(static_cast<V> (it[0]), static_cast<V> (it[1]), static_cast<V> (it[2]),
            static_cast<V> (it[3]), static_cast<V> (it[4]), static_cast<V> (it[5]), static_cast<V> (it[6]));
      case 8:
         return make_tuple(static_cast<V> (it[0]), static_cast<V> (it[1]), static_cast<V> (it[2]),
            static_cast<V> (it[3]), static_cast<V> (it[4]), static_cast<V> (it[5]), static_cast<V> (it[6]),
            static_cast<V> (it[7]));
      case 9:
         return make_tuple(static_cast<V> (it[0]), static_cast<V> (it[1]), static_cast<V> (it[2]),
            static_cast<V> (it[3]), static_cast<V> (it[4]), static_cast<V> (it[5]), static_cast<V> (it[6]),
            static_cast<V> (it[7]), static_cast<V> (it[8]));
      case 10:
         return make_tuple(static_cast<V> (it[0]), static_cast<V> (it[1]), static_cast<V> (it[2]),
            static_cast<V> (it[3]), static_cast<V> (it[4]), static_cast<V> (it[5]), static_cast<V> (it[6]),
            static_cast<V> (it[7]), static_cast<V> (it[8]), static_cast<V> (it[9]));
      case 11:
         return make_tuple(static_cast<V> (it[0]), static_cast<V> (it[1]), static_cast<V> (it[2]),
            static_cast<V> (it[3]), static_cast<V> (it[4]), static_cast<V> (it[5]), static_cast<V> (it[6]),
            static_cast<V> (it[7]), static_cast<V> (it[8]), static_cast<V> (it[9]), static_cast<V> (it[10]));
      case 12:
         return make_tuple(static_cast<V> (it[0]), static_cast<V> (it[1]), static_cast<V> (it[2]),
            static_cast<V> (it[3]), static_cast<V> (it[4]), static_cast<V> (it[5]), static_cast<V> (it[6]),
            static_cast<V> (it[7]), static_cast<V> (it[8]), static_cast<V> (it[9]), static_cast<V> (it[10]),
            static_cast<V> (it[11]));
      case 13:
         return make_tuple(static_cast<V> (it[0]), static_cast<V> (it[1]), static_cast<V> (it[2]),
            static_cast<V> (it[3]), static_cast<V> (it[4]), static_cast<V> (it[5]), static_cast<V> (it[6]),
            static_cast<V> (it[7]), static_cast<V> (it[8]), static_cast<V> (it[9]), static_cast<V> (it[10]),
            static_cast<V> (it[11]), static_cast<V> (it[12]));
//      case 14:
//         return make_tuple(static_cast<V> (it[0]), static_cast<V> (it[1]), static_cast<V> (it[2]),
//            static_cast<V> (it[3]), static_cast<V> (it[4]), static_cast<V> (it[5]), static_cast<V> (it[6]),
//            static_cast<V> (it[7]), static_cast<V> (it[8]), static_cast<V> (it[9]), static_cast<V> (it[10]),
//            static_cast<V> (it[11]), static_cast<V> (it[12]), static_cast<V> (it[13]));
//      case 15:
//         return make_tuple(static_cast<V> (it[0]), static_cast<V> (it[1]), static_cast<V> (it[2]),
//            static_cast<V> (it[3]), static_cast<V> (it[4]), static_cast<V> (it[5]), static_cast<V> (it[6]),
//            static_cast<V> (it[7]), static_cast<V> (it[8]), static_cast<V> (it[9]), static_cast<V> (it[10]),
//            static_cast<V> (it[11]), static_cast<V> (it[12]), static_cast<V> (it[13]), static_cast<V> (it[14]));
//      case 16:
//         return make_tuple(static_cast<V> (it[0]), static_cast<V> (it[1]), static_cast<V> (it[2]),
//            static_cast<V> (it[3]), static_cast<V> (it[4]), static_cast<V> (it[5]), static_cast<V> (it[6]),
//            static_cast<V> (it[7]), static_cast<V> (it[8]), static_cast<V> (it[9]), static_cast<V> (it[10]),
//            static_cast<V> (it[11]), static_cast<V> (it[12]), static_cast<V> (it[13]), static_cast<V> (it[14]),
//            static_cast<V> (it[15]));
//      case 17:
//         return make_tuple(static_cast<V> (it[0]), static_cast<V> (it[1]), static_cast<V> (it[2]),
//            static_cast<V> (it[3]), static_cast<V> (it[4]), static_cast<V> (it[5]), static_cast<V> (it[6]),
//            static_cast<V> (it[7]), static_cast<V> (it[8]), static_cast<V> (it[9]), static_cast<V> (it[10]),
//            static_cast<V> (it[11]), static_cast<V> (it[12]), static_cast<V> (it[13]), static_cast<V> (it[14]),
//            static_cast<V> (it[15]), static_cast<V> (it[16]));
//      case 18:
//         return make_tuple(static_cast<V> (it[0]), static_cast<V> (it[1]), static_cast<V> (it[2]),
//            static_cast<V> (it[3]), static_cast<V> (it[4]), static_cast<V> (it[5]), static_cast<V> (it[6]),
//            static_cast<V> (it[7]), static_cast<V> (it[8]), static_cast<V> (it[9]), static_cast<V> (it[10]),
//            static_cast<V> (it[11]), static_cast<V> (it[12]), static_cast<V> (it[13]), static_cast<V> (it[14]),
//            static_cast<V> (it[15]), static_cast<V> (it[16]), static_cast<V> (it[17]));
//      case 19:
//         return make_tuple(static_cast<V> (it[0]), static_cast<V> (it[1]), static_cast<V> (it[2]),
//            static_cast<V> (it[3]), static_cast<V> (it[4]), static_cast<V> (it[5]), static_cast<V> (it[6]),
//            static_cast<V> (it[7]), static_cast<V> (it[8]), static_cast<V> (it[9]), static_cast<V> (it[10]),
//            static_cast<V> (it[11]), static_cast<V> (it[12]), static_cast<V> (it[13]), static_cast<V> (it[14]),
//            static_cast<V> (it[15]), static_cast<V> (it[16]), static_cast<V> (it[17]), static_cast<V> (it[18]));
//      case 20:
//         return make_tuple(static_cast<V> (it[0]), static_cast<V> (it[1]), static_cast<V> (it[2]),
//            static_cast<V> (it[3]), static_cast<V> (it[4]), static_cast<V> (it[5]), static_cast<V> (it[6]),
//            static_cast<V> (it[7]), static_cast<V> (it[8]), static_cast<V> (it[9]), static_cast<V> (it[10]),
//            static_cast<V> (it[11]), static_cast<V> (it[12]), static_cast<V> (it[13]), static_cast<V> (it[14]),
//            static_cast<V> (it[15]), static_cast<V> (it[16]), static_cast<V> (it[17]), static_cast<V> (it[18]),
//            static_cast<V> (it[19]));

      default:
      {
         throw opengm::RuntimeError("size of iterator is to big to return to tuple");
         return make_tuple(static_cast<V> (it[0]));
      }
   }
}


#endif	/* ITERATORTOTUPLE_HXX */

