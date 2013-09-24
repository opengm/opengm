#ifndef INF_DEF_VISITOR
#define INF_DEF_VISITOR




#include <boost/python.hpp>
#include <sstream>
#include <string>
#include <boost/python/def_visitor.hpp>

#include <opengm/python/opengmpython.hxx>
#include <opengm/python/converter.hxx>
#include <opengm/python/numpyview.hxx>
#include <opengm/python/pythonfunction.hxx>


#include "gil.hxx"
#include "visitor_def_visitor.hxx"

#include <opengm/inference/inference.hxx>
#include <opengm/opengm.hxx>
#include <opengm/python/opengmpython.hxx>
#include <opengm/python/converter.hxx>
#include <opengm/python/numpyview.hxx>
#include <opengm/python/pythonfunction.hxx>





///////////////////////////////////
// RESET?????
///////////////////////////////////
template<class INF,bool HAS_RESET>
class InfResetSuite ;

template<class INF>
class InfResetSuite<INF,true> : public boost::python::def_visitor<InfResetSuite<INF,true> >{
public:
    friend class boost::python::def_visitor_access;
    template <class classT>
    void visit(classT& c) const{ 
        c.def("reset", &reset) ;
    }

    static void reset(INF & inf){
        inf.reset();
    }
};

template<class INF>
class InfResetSuite<INF,false> : public boost::python::def_visitor<InfResetSuite<INF,false> >{
public:
    friend class def_visitor_access;
    template <class classT>
    void visit(classT& c) const{
    }
};



template<class OP,class ACC>
inline std::string semiRingName(){
   std::string opString,accString;
   if(opengm::meta::Compare<OP,opengm::Adder>::value){
      opString="Adder_";
   }
   else if(opengm::meta::Compare<OP,opengm::Multiplier>::value){
      opString="Multiplier_";
   }
   if(opengm::meta::Compare<ACC,opengm::Minimizer>::value){
      opString="Minimizer";
   }
   else if(opengm::meta::Compare<ACC,opengm::Maximizer>::value){
      opString="Maximizer";
   }
   return opString+accString;
}



typedef std::vector<std::string> StringVector;

class InfSetup{
public:
    InfSetup()
    :
        cite(),
        guarantees(),
        limitations("None, this algorithm can be used for any graphical model"),
        notes(),
        algType(),
        dependencies("None, this inference algorithms is implemented in OpenGM by default."),
        hyperParameterKeyWords(),
        hyperParameters(),
        hyperParametersDoc(),
        isDefault(true),
        isParameterFree(false),
        hasInterchangeableParameter(true)
    {

    }
    std::string cite,guarantees,limitations,notes,algType,dependencies,examples;
    StringVector hyperParameterKeyWords,hyperParameters,hyperParametersDoc;
    bool isDefault,isParameterFree,hasInterchangeableParameter;
};





template<
    class INF,
    bool HAS_RESET=true,
    bool HAS_VERBOSE_VISITOR=true,
    bool HAS_PYTHON_VISITOR=true
>
class InfSuite : public  boost::python::def_visitor<InfSuite<INF,HAS_RESET,HAS_VERBOSE_VISITOR,HAS_PYTHON_VISITOR> >
{
public:
    friend class boost::python::def_visitor_access;

    typedef typename INF::GraphicalModelType                    GraphicalModelType;
    typedef typename GraphicalModelType::IndependentFactorType  IndependentFactorType;
    typedef typename INF::Parameter                             ParameterType;
    typedef typename GraphicalModelType::IndexType              IndexType;
    typedef typename GraphicalModelType::LabelType              LabelType;
    typedef typename GraphicalModelType::ValueType              ValueType;
    InfSuite(
        const std::string & algName,                                               // "Icm , Gibbs, LazyFlipper,GraphCut, AlphaBetaSwap"
        const InfSetup & infSetup
    )
    :   algName_(algName),
        infSetup_(infSetup){
    }   
    const std::string algName_;
    const InfSetup infSetup_;
    const std::string extraName_;

    template <class classT>
    void visit(classT& c) const{
        std::string className = std::string("_")+algName_;
        for(size_t hp=0;hp<infSetup_.hyperParameters.size();++hp){
            className+=std::string("_");
            className+=infSetup_.hyperParameters[hp];
        }
        const std::string verboseVisitorClassName =std::string("_")+className +std::string("VerboseVisitor");
        const std::string pythonVisitorClassName  =std::string("_")+className + std::string("PythonVisitor");
        c
            // BASIC INTEFACE
            .def( boost::python::init<const GraphicalModelType & ,const ParameterType & >(
                (
                    boost::python::arg("gm"),
                    boost::python::arg("parameter")
                )
            ))
            .def("_arg",&infArg,(
                boost::python::arg("out"),
                boost::python::arg("argNumber")=1)
            )
            .def("_setStartingPoint",&setStartingPoint,(boost::python::arg("labels")))
            .def("_infer_no_visitor", &infer,
                (
                    boost::python::arg("releaseGil")=true
                )
            ) 
            .def("bound",&bound)
            .def("value",&value)
            .def("graphicalModel",&graphicalModel,boost::python::return_internal_reference<>())
            // OPTIONAL INTERFACE
            // has reset??
            .def(InfResetSuite<INF,HAS_RESET>())
            // visitors
            .def(InfVerboseVisitorSuite<INF,HAS_VERBOSE_VISITOR>(verboseVisitorClassName))
            .def(InfPythonVisitorSuite <INF,HAS_PYTHON_VISITOR>(pythonVisitorClassName) )

 
            // STRING
            .def("_algName",     &stringFromArg,(boost::python::arg("_dont_use_me_")=algName_ )).staticmethod("_algName")
            // DOCSTRING RELATED 
            .def("_cite",        &stringFromArg,(boost::python::arg("_dont_use_me_")=infSetup_.cite         )).staticmethod("_cite")
            .def("_guarantees",  &stringFromArg,(boost::python::arg("_dont_use_me_")=infSetup_.guarantees   )).staticmethod("_guarantees")
            .def("_limitations", &stringFromArg,(boost::python::arg("_dont_use_me_")=infSetup_.limitations  )).staticmethod("_limitations")
            .def("_notes",       &stringFromArg,(boost::python::arg("_dont_use_me_")=infSetup_.notes        )).staticmethod("_notes")
            .def("_algType",     &stringFromArg,(boost::python::arg("_dont_use_me_")=infSetup_.algType      )).staticmethod("_algType")
            .def("_dependencies",&stringFromArg,(boost::python::arg("_dont_use_me_")=infSetup_.dependencies )).staticmethod("_dependencies")
            .def("_examples",    &stringFromArg,(boost::python::arg("_dont_use_me_")=infSetup_.examples     )).staticmethod("_examples")
            // BOOL
            .def("_isParameterFree",            &boolFromArg,(boost::python::arg("_dont_use_me_")=infSetup_.isParameterFree             )).staticmethod("_isParameterFree")
            .def("_isDefault",                  &boolFromArg,(boost::python::arg("_dont_use_me_")=infSetup_.isDefault                   )).staticmethod("_isDefault")
            .def("_hasInterchangeableParameter",&boolFromArg,(boost::python::arg("_dont_use_me_")=infSetup_.hasInterchangeableParameter )).staticmethod("_hasInterchangeableParameter")

            // stringvector
            .def("_hyperParameterKeywords", &stringVectorFromArg,(boost::python::arg("_dont_use_me_")=infSetup_.hyperParameterKeyWords )).staticmethod("_hyperParameterKeywords")
            .def("_hyperParameters",        &stringVectorFromArg,(boost::python::arg("_dont_use_me_")=infSetup_.hyperParameters        )).staticmethod("_hyperParameters")
            .def("_hyperParametersDoc",     &stringVectorFromArg,(boost::python::arg("_dont_use_me_")=infSetup_.hyperParametersDoc     )).staticmethod("_hyperParametersDoc")

            // PARAMETER
            .def("_parameter",&getParameter).staticmethod("_parameter")
        ;

   
    }

    static const typename INF::GraphicalModelType & graphicalModel(const INF & inf){
        return inf.graphicalModel();
    }

    static ValueType bound(const INF & inf){
        return inf.bound();
    }

    static ValueType value(const INF & inf){
        return inf.value();
    } 


    static std::string stringFromArg(const std::string & arg){
        return arg;
    }


    static StringVector stringVectorFromArg(const StringVector & arg){
        return arg;
    }

    static bool boolFromArg(const bool arg){
        return arg;
    }

    static ParameterType getParameter(){
        return ParameterType();
    }

    static opengm::InferenceTermination infer(INF & inf,const bool releaseGil){
        opengm::InferenceTermination result;
        if(releaseGil){
            releaseGIL rgil;
            result= inf.infer();
        }
        else{
            result= inf.infer();
        }
        return result;
    }

    
    static opengm::InferenceTermination infArg(const INF & inf,std::vector<LabelType> & arg,const size_t argnr){
        if(arg.size()<inf.graphicalModel().numberOfVariables()){
            arg.resize(inf.graphicalModel().numberOfVariables());
        }
        return inf.arg(arg,argnr);
    }

    static void setStartingPoint(INF & inf,const std::vector<LabelType> & start){
        inf.setStartingPoint(start.begin());
    }
};




#endif // INF_DEF_VISITOR 