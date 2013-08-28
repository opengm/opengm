#ifndef PARTIAL_OPTIMALITY_DEF_VISITOR
#define PARTIAL_OPTIMALITY_DEF_VISITOR

#include <boost/python.hpp>
#include <sstream>
#include <string>
#include <boost/python/def_visitor.hpp>
#include "gil.hxx"
#include <opengm/python/opengmpython.hxx>
#include <opengm/python/converter.hxx>
#include <opengm/python/numpyview.hxx>
#include <opengm/python/pythonfunction.hxx>


#include <opengm/inference/inference.hxx>
#include <opengm/opengm.hxx>



template<class INF>
class PartialOptimalitySuite: public boost::python::def_visitor<PartialOptimalitySuite<INF> >{
public:
    friend class boost::python::def_visitor_access;
    typedef typename INF::IndexType IndexType;
    typedef typename INF::LabelType LabelType;

    template <class classT>
    void visit(classT& c) const{ 
        c
            .def("_partialOptimality", &partialOptimality)
        ;
    }

    static  boost::python::object partialOptimality(const INF & inf){
        boost::python::object obj = opengm::python::get1dArray<bool>(inf.graphicalModel().numberOfVariables());
        bool * castedPtr = opengm::python::getCastedPtr<bool>(obj);
        std::vector<bool> optVar(inf.graphicalModel().numberOfVariables());
        inf.partialOptimality(optVar);
        std::copy(optVar.begin(),optVar.end(),castedPtr);
        return obj;
    }
};


template<class INF>
class PartialOptimalitySuite2: public boost::python::def_visitor<PartialOptimalitySuite2<INF> >{
public:
    friend class boost::python::def_visitor_access;
    typedef typename INF::IndexType IndexType;
    typedef typename INF::LabelType LabelType;
    
    template <class classT>
    void visit(classT& c) const{ 
        c
            .def("_partialOptimality", &partialOptimality)
        ;
    }

    static  boost::python::object partialOptimality(const INF & inf){
        boost::python::object obj = opengm::python::get1dArray<bool>(inf.graphicalModel().numberOfVariables());
        bool * castedPtr = opengm::python::getCastedPtr<bool>(obj);

        LabelType label=0;
        for(IndexType vi=0;vi<inf.graphicalModel().numberOfVariables();++vi){
            if(inf.partialOptimality(vi,label)){
                castedPtr[vi]=true;
            }
            else{
                castedPtr[vi]=false;
            }
        }        
        return obj;
    }
};





#endif // PARTIAL_OPTIMALITY_DEF_VISITOR