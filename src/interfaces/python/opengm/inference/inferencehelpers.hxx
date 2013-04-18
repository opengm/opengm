#ifndef OPENGM_PYTHON_INF_HELPERS
#define	OPENGM_PYTHON_INF_HELPERS

#include <boost/python.hpp>
#include <stdexcept>
#include <stddef.h>
#include <string>
#include <sstream>
#include <vector>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/inference/inference.hxx>
#include <opengm/inference/icm.hxx>
#include "opengm_helpers.hxx"
#include "copyhelper.hxx"
#include "nifty_iterator.hxx"
#include "converter.hxx"
#include "pyVisitor.hxx"
using namespace boost::python;



#define OPENGM_PYTHON_VERBOSE_VISITOR_EXPORTER( CLASS_TYPE ,P_ALG_NAME,P_IMPL_NAME ) \
   class_<CLASS_TYPE > ("Verbose_" #P_ALG_NAME "_" #P_IMPL_NAME,"TODO DOC_STRING", init<size_t,bool>()) \
   .def(init<size_t>()) \
   .def(init< >())

#define OPENGM_PYTHON_PY_VISITOR_EXPORTER( CLASS_TYPE ,P_ALG_NAME,P_IMPL_NAME ) \
   class_<CLASS_TYPE > ("Python_" #P_ALG_NAME "_" #P_IMPL_NAME,"TODO DOC_STRING", init<boost::python::object,const size_t>()) 

template<class T>
struct InfDescriptor{

};

#define append_subnamespace_inf(SCOPE_NAME) \
    const std::string pySubNamespaceName(SCOPE_NAME); \
    scope current; \
    std::string currentScopeName(extract<const char*>(current.attr("__name__"))); \
    std::string submoduleName = currentScopeName + std::string(".") + pySubNamespaceName; \
    object submodule(borrowed(PyImport_AddModule(submoduleName.c_str()))); \
    current.attr(pySubNamespaceName.c_str()) = submodule; \
    submodule.attr("__package__") = submoduleName.c_str(); \
    scope submoduleScope = submodule




  //.def("F", (void (C::*)(int))&C::F)  // Note missing staticmethod call!
#define DESC_NAME(P_ALG_NAME,P_IMPL_NAME) InfDescriptor##P_ALG_NAME##P_IMPL_NAME
#define DESC_NAME_STR(P_ALG_NAME,P_IMPL_NAME) "InfDescriptor" #P_ALG_NAME #P_IMPL_NAME

#define OPENGM_PYTHON_INFERENCE_NO_RESET_EXPORTER(INF_CLASS_TYPE,P_ALG_NAME,P_IMPL_NAME,ALG_NAME_L,ALG_NAME_S,IMPL_NAME_L,IMPL_NAME_S,DOC_STRING) \
   {\
   append_subnamespace_inf("visitor");\
   OPENGM_PYTHON_VERBOSE_VISITOR_EXPORTER(typename INF_CLASS_TYPE::VerboseVisitorType ,P_ALG_NAME,P_IMPL_NAME); \
   OPENGM_PYTHON_PY_VISITOR_EXPORTER(PythonVisitor<INF_CLASS_TYPE> ,P_ALG_NAME,P_IMPL_NAME); \
   }\
   class_<INF_CLASS_TYPE > (#P_ALG_NAME "_" #P_IMPL_NAME,\
   DOC_STRING,\
   init<const  typename INF_CLASS_TYPE::GraphicalModelType & >()) \
   .def(init<const typename INF_CLASS_TYPE::GraphicalModelType &, const  typename  INF_CLASS_TYPE::Parameter &>()) \
   .def("setStartingPoint",&pyinf::setStartPyNumpy<INF_CLASS_TYPE>,(arg("startingPoint")),\
   "Set a starting labeling as start point for inference. Warm started inference might lead to better results."\
   ) \
   .def("setStartingPoint",&pyinf::setStartPyList<INF_CLASS_TYPE,int>,(arg("startingPoint")),\
   "Set a starting labeling as start point for inference. Warm started inference might lead to better results."\
   ) \
   .def("infer", &pyinf::inferVerboseVisitorPy< INF_CLASS_TYPE >,"infer with python visitor" ) \
   .def("infer", &pyinf::inferPythonVisitorPy< INF_CLASS_TYPE >,"infer with a verbose visitor" ) \
   .def("infer", &pyinf::inferEmptyVisitorPy< INF_CLASS_TYPE >,"infer without any visitor")\
   .def("bound",&INF_CLASS_TYPE::bound,\
   "get the current bound"\
   ) \
   .def("__str__",&pyinf::getNamePy<INF_CLASS_TYPE>) \
   .def("_arg_cpp", &pyinf::arg1PyVector<INF_CLASS_TYPE>,\
   "get the inference result ``.infer`` has to be called bevore ``arg``"\
   ) \
   .def("arg2", &pyinf::arg2PyNumpy<INF_CLASS_TYPE>, \
   "get the inference result ``.infer`` has to be called bevore ``arg``"\
   )\
   .def("gm",& INF_CLASS_TYPE::graphicalModel,return_internal_reference<>())\
   .def("graphicalModel",& INF_CLASS_TYPE::graphicalModel,return_internal_reference<>())\
   .def("pythonVisitor",&pyinf::getPythonVisitor< INF_CLASS_TYPE >,return_value_policy<manage_new_object>(),(arg("callbackObject"),arg("visitNth")=1)) \
   .def("verboseVisitor",&pyinf::getVerboseVisitor< INF_CLASS_TYPE >,return_value_policy<manage_new_object>(),(arg("printNth")=1,arg("multiline")=true) ) \
   .def("_getDefaultParameter",&pyinf::getDefaultParameter< INF_CLASS_TYPE >,( arg("_param")= typename INF_CLASS_TYPE::Parameter()  ),"dont call this method").staticmethod("_getDefaultParameter") \
   .def("_getAlgNameLong",    &pyinf::getAString< INF_CLASS_TYPE >,( arg("_name")=std::string( ALG_NAME_L  ) ),"dont call this method").staticmethod("_getAlgNameLong") \
   .def("_getAlgNameShort",   &pyinf::getAString< INF_CLASS_TYPE >,( arg("_name")=std::string( ALG_NAME_S  ) ),"dont call this method").staticmethod("_getAlgNameShort") \
   .def("_getImplNameLong",   &pyinf::getAString< INF_CLASS_TYPE >,( arg("_name")=std::string( IMPL_NAME_L ) ),"dont call this method").staticmethod("_getImplNameLong") \
   .def("_getImplNameShort",  &pyinf::getAString< INF_CLASS_TYPE >,( arg("_name")=std::string( IMPL_NAME_S ) ),"dont call this method").staticmethod("_getImplNameShort") \
   .def("graphicalModel",&INF_CLASS_TYPE::graphicalModel,return_internal_reference<>(),\
   "get a const reference of the graphical model"\
   ) 


#define OPENGM_PYTHON_INFERENCE_EXPORTER(INF_CLASS_TYPE,P_ALG_NAME,P_IMPL_NAME,ALG_NAME_L,ALG_NAME_S,IMPL_NAME_L,IMPL_NAME_S,DOC_STRING) \
   OPENGM_PYTHON_INFERENCE_NO_RESET_EXPORTER(INF_CLASS_TYPE,P_ALG_NAME,P_IMPL_NAME,ALG_NAME_L,ALG_NAME_S,IMPL_NAME_L,IMPL_NAME_S,DOC_STRING).def("reset",&INF_CLASS_TYPE::reset) 
   

           


namespace pyinf {

   template<class INF>
   typename INF::Parameter getDefaultParameter
   (
   //const INF & inf,
   const typename INF::Parameter & param
   ) {
      return param;
   }

   template<class INF>
   std::string getAString
   (
   //const INF & inf,
   const std::string & astring
   ) {
      return astring;
   }


   template<class INF>
   void arg1PyVector(const INF & inf,std::vector<typename INF::LabelType> & arg) {
      if(arg.size()<inf.graphicalModel().numberOfVariables())
         arg.resize(inf.graphicalModel().numberOfVariables());
      inf.arg(arg);
   }

   template<class INF>
   inline boost::python::numeric::array arg2PyNumpy(const INF & inf, const size_t argnr) {
      std::vector<typename INF::LabelType> arg;
      inf.arg(arg, argnr);
      return iteratorToNumpy(arg.begin(),arg.size());
   }
   
   template<class INF>
   opengm::InferenceTermination inferMaybeVerbose
   (
      INF & inf,
      const bool verbose,
      const size_t printNthStep,
      bool multiline     
   ) {
      if(!verbose)
         return inf.infer();
      else{
         typename INF::VerboseVisitorType visitor(printNthStep,multiline);
         return inf.infer(visitor);
      }
   }
   
   template<class INF>
   opengm::InferenceTermination inferEmptyVisitorPy(INF & inf) {
      return inf.infer();
   }


   template<class INF>
   opengm::InferenceTermination inferVerboseVisitorPy(INF & inf, typename INF::VerboseVisitorType & visitor) {
      return inf.infer(visitor);
   }






   template<class INF>
   opengm::InferenceTermination inferPythonVisitorPy(INF & inf, PythonVisitor< INF  >  & visitor) {
      return inf.infer(visitor);
   }

   template<class INF, class VISITOR>
   opengm::InferenceTermination inferVisitorPy(INF & inf, VISITOR & visitor) {
      return inf.infer(visitor);
   }

   template<class INF, class VALUE_TYPE>
   void setStartPyList
   (
      INF & inf,
      const boost::python::list & startingPoint
   ) {
      typedef PythonIntListAccessor<VALUE_TYPE, true > Accessor;
      IteratorHolder< Accessor> holder(startingPoint);      
      std::vector<typename INF::LabelType > sp(holder.begin(),holder.end());
      inf.setStartingPoint(sp.begin());
   }
   
   template<class INF>
   PythonVisitor<INF> * getPythonVisitor
   (
      INF & inf,
      boost::python::object obj,
      const size_t visitNth
   ) {
      return new PythonVisitor<INF>(obj,visitNth);
   }

   template<class INF>
   typename INF::VerboseVisitorType * getVerboseVisitor
   (
      INF & inf,
      const size_t visitNth,
      const bool multiline
   ) {
      return new typename INF::VerboseVisitorType(visitNth,multiline);
   }

   template<class INF>
   void setStartPyNumpy
   (
      INF & inf,
      NumpyView<typename INF::IndexType,1> startingPoint
   ) {
      std::vector<typename INF::LabelType > sp(startingPoint.begin1d(),startingPoint.end1d());
      inf.setStartingPoint(sp.begin());
   }

   template<class INF>
   std::string getNamePy
   (
   const INF & inf
   ) {
      std::string accname;
      std::string opname;
      typedef opengm::Adder Adder;
      typedef opengm::Multiplier Multiplier;
      typedef opengm::Integrator Integrator;
      typedef opengm::Minimizer Minimizer;
      typedef opengm::Maximizer Maximizer;
      typedef typename INF::AccumulationType Acc;
      typedef typename INF::OperatorType Opt;
      if (opengm::meta::Compare<Acc, Minimizer>::value)
         accname = "Minimizer";
      else if (opengm::meta::Compare<Acc, Maximizer>::value)
         accname = "Maximizer";
      else if (opengm::meta::Compare<Acc, Integrator>::value)
         accname = "Integrator";
      if (opengm::meta::Compare<Opt, Adder>::value)
         opname = "Adder";
      else if (opengm::meta::Compare<Opt, Multiplier>::value)
         opname = "Multiplier";
      std::stringstream ss;
      ss<< inf.name() << " " << accname << "," << opname;
      return ss.str();
   }

}

#endif
