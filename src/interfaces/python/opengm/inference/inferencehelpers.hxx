#ifndef OPENGM_PYTHON_INTERFACE
#define OPENGM_PYTHON_INTERFACE 1
#endif

#ifndef OPENGM_PYTHON_INF_HELPERS
#define	OPENGM_PYTHON_INF_HELPERS

#include <stdexcept>
#include <stddef.h>
#include <string>
#include <sstream>
#include <vector>
#include <boost/python.hpp>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/inference/inference.hxx>
#include <opengm/inference/icm.hxx>
#include "opengm_helpers.hxx"
#include "copyhelper.hxx"
#include "nifty_iterator.hxx"
#include "converter.hxx"
using namespace boost::python;

#define OPENGM_PYTHON_VERBOSE_VISITOR_EXPORTER( CLASS_TYPE ,  CLASS_STRING ) \
   class_<CLASS_TYPE > (CLASS_STRING, init<size_t,bool>()) \
   .def(init<size_t>()) \
   .def(init< >())

#define OPENGM_PYTHON_INFERENCE_EXPORTER(INF_CLASS_TYPE,CLASS_STRING) \
   class_<INF_CLASS_TYPE > (CLASS_STRING, init<const  typename INF_CLASS_TYPE::GraphicalModelType & >()) \
   .def(init<const typename INF_CLASS_TYPE::GraphicalModelType &, const  typename  INF_CLASS_TYPE::Parameter &>()) \
   .def("setStartingPoint",&pyinf::setStartPyNumpy<INF_CLASS_TYPE>) \
   .def("setStartingPoint",&pyinf::setStartPyList<INF_CLASS_TYPE,int>) \
   .def("graphicalModel",&INF_CLASS_TYPE::graphicalModel,return_internal_reference<>()) \
   .def ("infer", &pyinf::inferMaybeVerbose<INF_CLASS_TYPE>, \
            (arg("verbose")=false, arg("printNth")=size_t(1),boost::python::arg("multiline")=true) \
   ) \
   .def("reset",&INF_CLASS_TYPE::reset) \
   .def("bound",&INF_CLASS_TYPE::bound) \
   .def("__str__",&pyinf::getNamePy<INF_CLASS_TYPE>) \
   .def("arg", &pyinf::arg1PyNumpy<INF_CLASS_TYPE>) \
   .def("arg", &pyinf::arg2PyNumpy<INF_CLASS_TYPE>) 
           
   //.def("infer", &pyinf::inferEmptyVisitorPy<INF_CLASS_TYPE>) \
   //.def("infer", &pyinf::inferVisitorPy<INF_CLASS_TYPE, typename INF_CLASS_TYPE::VerboseVisitorType>) \
           
#define OPENGM_PYTHON_INFERENCE_NO_RESET_EXPORTER(INF_CLASS_TYPE,CLASS_STRING) \
   class_<INF_CLASS_TYPE > (CLASS_STRING, init<const  typename INF_CLASS_TYPE::GraphicalModelType & >()) \
   .def(init<const typename INF_CLASS_TYPE::GraphicalModelType &, const  typename  INF_CLASS_TYPE::Parameter &>()) \
   .def("setStartingPoint",&pyinf::setStartPyNumpy<INF_CLASS_TYPE>) \
   .def("setStartingPoint",&pyinf::setStartPyList<INF_CLASS_TYPE,int>) \
   .def("graphicalModel",&INF_CLASS_TYPE::graphicalModel,return_internal_reference<>()) \
   .def ("infer", &pyinf::inferMaybeVerbose<INF_CLASS_TYPE>, \
            (arg("verbose")=false, arg("printNth")=size_t(1),boost::python::arg("multiline")=true) \
   ) \
   .def("bound",&INF_CLASS_TYPE::bound) \
   .def("__str__",&pyinf::getNamePy<INF_CLASS_TYPE>) \
   .def("arg", &pyinf::arg1PyNumpy<INF_CLASS_TYPE>) \
   .def("arg", &pyinf::arg2PyNumpy<INF_CLASS_TYPE>) 


namespace pyinf {

   template<class INF>
   inline boost::python::numeric::array arg1PyNumpy(const INF & inf) {
      std::vector<typename INF::LabelType> arg;
      inf.arg(arg);
      return iteratorToNumpy(arg.begin(),arg.size());
   }
   
   template<class INF>
   inline boost::python::list arg1PyList(const INF & inf) {
      std::vector<typename INF::LabelType> arg;
      inf.arg(arg);
      return iteratorToList(arg.begin(),arg.size());
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
      ss<< inf.name() << "< " << accname << " , " << opname<<" >";
      return ss.str();
   }

}

#endif