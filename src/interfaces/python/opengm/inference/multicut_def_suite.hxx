#ifndef MULTICUT_DEF_VISITOR
#define MULTICUT_DEF_VISITOR

#include <boost/python.hpp>
#include <sstream>
#include <string>
#include <boost/python/def_visitor.hpp>
#include "gil.hxx"
#include "../converter.hxx"
#include "visitor_def_visitor.hxx"

#include <opengm/inference/inference.hxx>
#include <opengm/opengm.hxx>

template<class INF>
class MulticutInferenceSuite: public boost::python::def_visitor<MulticutInferenceSuite<INF> >{
public:
   friend class def_visitor_access;
   typedef typename INF::IndexType IndexType;
   typedef typename INF::LabelType LabelType;
   typedef typename INF::ValueType ValueType;
   
   MulticutInferenceSuite(){
   }

   template <class classT>
   void visit(classT& c) const{ 
      c
         .def("_getEdgeLabeling", &getEdgeLabelingNumpy)
      ;
   }

   static boost::python::object getEdgeLabelingNumpy
   (
       INF & inf
   )
   {
       std::vector<double> ret = inf.getEdgeLabeling();
       return iteratorToNumpy(ret.begin(), ret.size());
   }

};

#endif // MULTICUT_DEF_VISITOR