#ifndef VISITOR_DEF_VISITOR
#define VISITOR_DEF_VISITOR


#include <boost/python.hpp>
#include <boost/python/def_visitor.hpp>
#include <sstream>
#include <string>
#include "gil.hxx"
#include <opengm/inference/inference.hxx>
#include "pyVisitor.hxx"



#include <opengm/python/opengmpython.hxx>
#include <opengm/python/converter.hxx>
#include <opengm/python/numpyview.hxx>
#include <opengm/python/pythonfunction.hxx>



///////////////////////////////////
// VERBOSE VISITOR
///////////////////////////////////

template<class INF,bool HAS_VISITOR>
class InfVerboseVisitorSuite;

template<class INF>
class InfVerboseVisitorSuite<INF,true> : public boost::python::def_visitor<InfVerboseVisitorSuite<INF,true> >{
public:
    friend class boost::python::def_visitor_access;
    typedef typename INF::VerboseVisitorType VisitorType;
    typedef typename INF::TimingVisitorType  TimingVisitor;

    InfVerboseVisitorSuite(const std::string & className)
    :className_(className),
    timingClassName_(className+std::string("Timing")){
    }

    const std::string className_;
    const std::string timingClassName_;

    template <class classT>
    void visit(classT& c) const{ 

        boost::python::class_<VisitorType > (className_.c_str() , boost::python::init<size_t,bool>(
                (
                    boost::python::arg("printNth")=1,
                    boost::python::arg("multiline")=true
                )
            )
        )
        ;

        boost::python::class_<TimingVisitor > (timingClassName_.c_str() , boost::python::init<const size_t,size_t,bool,bool>(
                (
                    boost::python::arg("visitNth")=1,
                    boost::python::arg("reserve")=0,
                    boost::python::arg("verbose")=true,
                    boost::python::arg("multiline")=true,
                    boost::python::arg("timeLimit")=std::numeric_limits<double>::infinity()
                )
            )
        )
        .def("getTimes",        &getTimes,      "get rumtimes for each visit step")
        .def("getValues",       &getValues,     "get value for each visit step")
        .def("getBounds",       &getBounds,     "get bound for each visit step")
        .def("getIterations",   &getIterations, "get iteration number for each visit step")
        ;


        
        //const size_t visitNth,
        //size_t reserve,
        //bool verbose,
        //bool multilineCout


        c
            // will return a verbose visitor
            .def("verboseVisitor", &verboseVisitor,boost::python::return_value_policy<boost::python::manage_new_object>(),
                (
                    boost::python::arg("printNth")=1,
                    boost::python::arg("multiline")=true
                )
            )
            // will return a timing visitor
            .def("timingVisitor", &timingVisitor,boost::python::return_value_policy<boost::python::manage_new_object>(),
                (
                    boost::python::arg("visitNth")=1,
                    boost::python::arg("reserve")=0,
                    boost::python::arg("verbose")=true,
                    boost::python::arg("multiline")=true,
                    boost::python::arg("timeLimit")=std::numeric_limits<double>::infinity()
                )
            )
            .def("_infer", &inferVerbose,
                (
                    boost::python::arg("visitor"),
                    boost::python::arg("releaseGil")=true
                )
            )
            .def("_infer", &inferTiming,
                (
                    boost::python::arg("visitor"),
                    boost::python::arg("releaseGil")=true
                )
            ) 
        ;
    }

    static TimingVisitor * timingVisitor(const INF & inf,const size_t visitNth,const size_t reserve,const bool verbose,const bool printMultiLine,const double timeLimit) {
        return new TimingVisitor(visitNth,reserve,verbose,printMultiLine,timeLimit);
    }

    // get results of timing 
    static boost::python::object getTimes(const TimingVisitor & tv){
        return opengm::python::iteratorToNumpy(tv.getTimes().begin(),tv.getTimes().size());
    }
    static boost::python::object getValues(const TimingVisitor & tv){
        return opengm::python::iteratorToNumpy(tv.getValues().begin(),tv.getValues().size());
    }
    static boost::python::object getBounds(const TimingVisitor & tv){
        return opengm::python::iteratorToNumpy(tv.getBounds().begin(),tv.getBounds().size());
    }
    static boost::python::object getIterations(const TimingVisitor & tv){
        return opengm::python::iteratorToNumpy(tv.getIterations().begin(),tv.getIterations().size());
    }



    static VisitorType * verboseVisitor(const INF & inf,const size_t printNth,const bool printMultiLine){
        return new VisitorType(printNth,printMultiLine);
    }

    static opengm::InferenceTermination inferVerbose(INF & inf,VisitorType & visitor,const bool releaseGil){
        opengm::InferenceTermination result;
        {
            if(releaseGil){
                releaseGIL rgil;
                result= inf.infer(visitor);
            }
            else{
                result= inf.infer(visitor);
            }
        }
        return result;
    }
    static opengm::InferenceTermination inferTiming(INF & inf,TimingVisitor & visitor,const bool releaseGil){
        opengm::InferenceTermination result;
        {
            if(releaseGil){
                releaseGIL rgil;
                result= inf.infer(visitor);
            }
            else{
                result= inf.infer(visitor);
            }
        }
        return result;
    }
};
template<class INF>
class InfVerboseVisitorSuite<INF,false> : public boost::python::def_visitor<InfVerboseVisitorSuite<INF,false> >{
public:
    friend class def_visitor_access;

    InfVerboseVisitorSuite(const std::string & className){
        //(void)className;
    }

    template <class classT>
    void visit(classT& c) const{
    } 
};



///////////////////////////////////
// PYTHON VISITOR
///////////////////////////////////

template<class INF,bool HAS_VISITOR>
class InfPythonVisitorSuite;

template<class INF>
class InfPythonVisitorSuite<INF,true> : public boost::python::def_visitor<InfPythonVisitorSuite<INF,true> >{
public:
    friend class boost::python::def_visitor_access;
    typedef PythonVisitor<INF> VisitorType;
    InfPythonVisitorSuite(const std::string & className)
    :className_(className){
    }

    const std::string className_;

    template <class classT>
    void visit(classT& c) const{ 

        boost::python::class_<VisitorType > (className_.c_str() , boost::python::init<boost::python::object,const size_t>(
                (
                    boost::python::arg("callbackObject"),
                    boost::python::arg("multiline")=1,
                    boost::python::arg("ensureGilState")=true
                )
            )
        )
        ;




        c
            .def("pythonVisitor", &pythonVisitor,boost::python::return_value_policy<boost::python::manage_new_object>(),
                (
                    boost::python::arg("callbackObject"),
                    boost::python::arg("visitNth")=1)
                )
            .def("_infer", &infer,
                (
                    boost::python::arg("visitor"),
                    boost::python::arg("releaseGil")=true
                )
            ) 
        ;
    }

    static VisitorType * pythonVisitor(const INF & inf,boost::python::object f,const size_t visitNth){
        return new VisitorType(f,visitNth);
    }

    static opengm::InferenceTermination infer(INF & inf,VisitorType & visitor,const bool releaseGil){
        visitor.setGilEnsure(releaseGil);
        opengm::InferenceTermination result;
        {
            if(releaseGil){
                releaseGIL rgil;
                result= inf.infer(visitor);
            }
            else{
                result= inf.infer(visitor);
            }
        }
        return result;
    }
};
template<class INF>
class InfPythonVisitorSuite<INF,false> : public boost::python::def_visitor<InfPythonVisitorSuite<INF,false> >{
public:
    friend class boost::python::def_visitor_access;

    InfPythonVisitorSuite(const std::string & className){
        //(void)className;
    }

    
    template <class classT>
    void visit(classT& c) const{
    } 
};


#endif // VISITOR_DEF_VISITOR