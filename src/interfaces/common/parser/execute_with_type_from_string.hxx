#ifndef EXECUTE_WITH_TYPE_FROM_STRING_HXX_
#define EXECUTE_WITH_TYPE_FROM_STRING_HXX_

//#include "../helper/interfacesTypedefs.hxx"

namespace opengm {

namespace interface {

/********************
 * class definition *
 ********************/

template<class OP, class TYPELIST, size_t IX, size_t DX, bool END>
struct executeWithTypeFromString;

template<class OP, class TYPELIST, size_t IX, size_t DX>
struct executeWithTypeFromString<OP, TYPELIST, IX, DX, false> {
   static void execute(const std::string& typeName);
};

template<class OP, class TYPELIST, size_t IX, size_t DX>
struct executeWithTypeFromString<OP, TYPELIST, IX, DX, true> {
   static void execute(const std::string& typeName);
};

/***********************
 * class documentation *
 ***********************/
//TODO add documentation

/******************
 * implementation *
 ******************/

template<class OP, class TYPELIST, size_t IX, size_t DX>
inline void executeWithTypeFromString<OP, TYPELIST, IX, DX, true>::execute(const std::string& typeName) {
   std::string error("Unknown type: ");
   error += typeName;
   throw opengm::RuntimeError(error);
}

template<class OP, class TYPELIST, size_t IX, size_t DX>
inline void executeWithTypeFromString<OP, TYPELIST, IX, DX, false>::execute(const std::string& typeName) {
   typedef typename opengm::meta::TypeAtTypeList<TYPELIST, IX>::type currentType;
   if(typeName == (std::string)currentType::name_) {
      OP::template execute< currentType >();
      //return IX;
   } else {
      // proceed with next type
      typedef typename opengm::meta::Increment<IX>::type NewIX;
      executeWithTypeFromString<OP, TYPELIST, NewIX::value, DX, opengm::meta::EqualNumber<NewIX::value, DX>::value >::execute(typeName);
   }
}

} // namespace interface

} // namespace opengm
#endif /* EXECUTE_WITH_TYPE_FROM_STRING_HXX_ */
