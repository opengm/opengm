#ifdef WITH_CPLEX
#include <stdexcept>
#include <stddef.h>
#include <string>
#include <boost/python.hpp>
#include "nifty_iterator.hxx"
#include "inferencehelpers.hxx"
#include "export_typedes.hxx"

#include <opengm/inference/lpcplex.hxx>

// to print parameter as string
template< class PARAM>
std::string cplexParamAsString(const PARAM & param) {
   std::string p=" ";
   return p;
}



namespace pycplex{

   template<class PARAM>
   inline void set
   (
      PARAM & p,
      const bool integerConstraint,
      const int numberOfThreads,
      const double cutUp,
      const double epGap,
      const double timeLimit
   ){
      p.integerConstraint_=integerConstraint;
      p.numberOfThreads_=numberOfThreads;
      p.cutUp_=cutUp;
      p.epGap_=epGap;
      p.timeLimit_=timeLimit;
   }

   // add constraints from python numpy
   template<class INF,class VALUE_TYPE,class INDEX_TYPE>
   void addConstraintPythonNumpy
   (
      INF & inf,
      NumpyView<VALUE_TYPE,1> lpVariableIndices,
      NumpyView<VALUE_TYPE,1>  & coefficients,
      const VALUE_TYPE lowerBound,
      const VALUE_TYPE upperBound
   ) {
      // add constraint
      inf.addConstraint(lpVariableIndices.begin(),lpVariableIndices.end(),coefficients.begin(),lowerBound,upperBound);
   }

   // add constraints from python list
   template<class INF,class VALUE_TYPE,class INDEX_TYPE>
   void addConstraintPythonList
   (
      INF & inf,
      const boost::python::list & lpVariableIndices,
      const boost::python::list & coefficients,
      const VALUE_TYPE lowerBound,
      const VALUE_TYPE upperBound
   ) {
      typedef PythonFundamentalListAccessor<VALUE_TYPE,true> FloatAccessor;
      typedef opengm::AccessorIterator<FloatAccessor,true> FloatIterator;
      typedef PythonFundamentalListAccessor<INDEX_TYPE,true> IntAccessor;
      typedef opengm::AccessorIterator<IntAccessor,true> IntIterator;
      // lp-variables iterator begin and end
      IntAccessor lpVarAccessor(coefficients);
      IntIterator lpVarBegin(lpVarAccessor,0);
      IntIterator lpVarEnd(lpVarAccessor,lpVarAccessor.size());
      // coefficient iterator end
      FloatAccessor coeffAccessor(coefficients);
      FloatIterator coeffBegin(coeffAccessor,0);
      // add constraint
      inf.addConstraint(lpVarBegin,lpVarEnd,coeffBegin,lowerBound,upperBound);
   }

   template<class INF,class VALUE_TYPE,class INDEX_TYPE>
   void addConstraintsPythonListList
   (
      INF & inf,
      const boost::python::list & lpVariableIndices,
      const boost::python::list & coefficients,
      NumpyView<VALUE_TYPE,1> lowerBounds,
      NumpyView<VALUE_TYPE,1>  upperBounds
   ) {

      const size_t nVi     = static_cast<size_t>(boost::python::len(lpVariableIndices));
      const size_t nCoeff  = static_cast<size_t>(boost::python::len(coefficients));
      const size_t nLBound = static_cast<size_t>(lowerBounds.size());
      const size_t nUBound = static_cast<size_t>(upperBounds.size());
      if(nVi==nCoeff || nCoeff==1){
         // all the same length
         if( (nLBound==nUBound && nLBound == nVi)  || nLBound==1 || nUBound==1){
            // accessors for lower and upper bounds
            // loop over all constraints 
            // which should be added
            for(size_t c=0;c<nVi;++c){
               boost::python::list lpVar,coeff;
               // extract lpvar list
               {
                  boost::python::extract<boost::python::list> extractor(lpVariableIndices[c]);
                  if(extractor.check())
                     lpVar=extractor();
                  else
                     throw opengm::RuntimeError("lpVariableIndices must be a list of lists");
               }
               {
               // extract coefficients list
                  boost::python::extract<boost::python::list> extractor(coefficients[nCoeff==1 ? 0:c ]);
                  if(extractor.check())
                     lpVar=extractor();
                  else
                     throw opengm::RuntimeError("coefficients must be a list of lists");
               }
               // call funtion which adds a single constraint from a lists
               pycplex::addConstraintPythonList<INF,VALUE_TYPE,INDEX_TYPE>(inf,lpVar,coeff,lowerBounds( nLBound==1 ? 0:c ),upperBounds(nUBound==1 ? 0:c));
            }
         }
         else
            throw opengm::RuntimeError("len(lowerBounds) and len(upperBounds) must be equal to 1 or len(coefficients)");
      }
      else
         throw opengm::RuntimeError("len(coefficients) must be 1 or equal to len(lpVariableIndices)");
   }
   

   template<class INF,class VALUE_TYPE,class INDEX_TYPE>
   void addConstraintsPythonNumpy
   (
      INF & inf,
      NumpyView<INDEX_TYPE,2> & lpVariableIndices,
      NumpyView<VALUE_TYPE,2> & coefficients,
      NumpyView<VALUE_TYPE,1> lowerBounds,
      NumpyView<VALUE_TYPE,1>  upperBounds
   ) {

      const size_t nVi     = static_cast<size_t>(lpVariableIndices.shape(0));
      const size_t nCoeff  = static_cast<size_t>(coefficients.shape(0));
      const size_t nLBound = static_cast<size_t>(lowerBounds.size());
      const size_t nUBound = static_cast<size_t>(upperBounds.size());

      const size_t nVipPerConstaint    = static_cast<size_t>(lpVariableIndices.shape(1));
      const size_t nCoeffPerConstaint   = static_cast<size_t>(coefficients.shape(1));
      if(nVipPerConstaint!=nCoeffPerConstaint)
         throw opengm::RuntimeError("coefficients per constraints must be as long as variables per constraint");
      opengm::FastSequence<INDEX_TYPE> lpVar(nVipPerConstaint);
      opengm::FastSequence<INDEX_TYPE> coeff(nCoeffPerConstaint);
      if(nVi==nCoeff || nCoeff==1){
         // all the same length
         if( (nLBound==nUBound && nLBound == nVi)  || nLBound==1 || nUBound==1){
            // accessors for lower and upper bounds
            // loop over all constraints 
            // which should be added
            for(size_t c=0;c<nVi;++c){
               for(size_t i=0;i<nVipPerConstaint;++i){
                  lpVar[i]=lpVariableIndices(c,i);
                  coeff[i]=coefficients(nCoeff==1 ? 0:c ,i);
               }
               // call funtion which adds a single constraint from a lists
               inf.addConstraint(lpVar.begin(),lpVar.end(),coeff.begin(),lowerBounds( nLBound==1 ? 0:c ),upperBounds(nUBound==1 ? 0:c));
            }
         }
         else
            throw opengm::RuntimeError("len(lowerBounds) and len(upperBounds) must be equal to 1 or len(coefficients)");
      }
      else
         throw opengm::RuntimeError("len(coefficients) must be 1 or equal to len(lpVariableIndices)");
   }


   template<class INF,class VALUE_TYPE,class INDEX_TYPE>
   void addConstraintsPythonListNumpy
   (
      INF & inf,
      const boost::python::list & lpVariableIndices,
      const boost::python::list & coefficients,
      NumpyView<VALUE_TYPE,1> lowerBounds,
      NumpyView<VALUE_TYPE,1>  upperBounds
   ) {

      const size_t nVi     = static_cast<size_t>(boost::python::len(lpVariableIndices));
      const size_t nCoeff  = static_cast<size_t>(boost::python::len(coefficients));
      const size_t nLBound = static_cast<size_t>(lowerBounds.size());
      const size_t nUBound = static_cast<size_t>(upperBounds.size());
      if(nVi==nCoeff || nCoeff==1){
         // all the same length
         if( (nLBound==nUBound && nLBound == nVi)  || nLBound==1 || nUBound==1){
            // accessors for lower and upper bounds
            // loop over all constraints 
            // which should be added
            for(size_t c=0;c<nVi;++c){
               NumpyView<VALUE_TYPE,1> lpVar,coeff;
               // extract lpvar list
               {
                  boost::python::extract<NumpyView<VALUE_TYPE,1> > extractor(lpVariableIndices[c]);
                  if(extractor.check())
                     lpVar=extractor();
                  else
                     throw opengm::RuntimeError("lpVariableIndices must be a list of lists");
               }
               {
               // extract coefficients list
                  boost::python::extract<NumpyView<VALUE_TYPE,1> > extractor(coefficients[nCoeff==1 ? 0:c ]);
                  if(extractor.check())
                     lpVar=extractor();
                  else
                     throw opengm::RuntimeError("coefficients must be a list of lists");
               }
               // call funtion which adds a single constraint from a lists
               pycplex::addConstraintPythonNumpy<INF,VALUE_TYPE,INDEX_TYPE>(inf,lpVar,coeff,lowerBounds( nLBound==1 ? 0:c ),upperBounds(nUBound==1 ? 0:c));
            }
         }
         else
            throw opengm::RuntimeError("len(lowerBounds) and len(upperBounds) must be equal to 1 or len(coefficients)");
      }
      else
         throw opengm::RuntimeError("len(coefficients) must be 1 or equal to len(lpVariableIndices)");
   }



   template<class INF,class VALUE_TYPE,class INDEX_TYPE>
   void addConstraintsPythonListListOrListNumpy
   (
      INF & inf,
      const boost::python::list & lpVariableIndices,
      const boost::python::list & coefficients,
      NumpyView<VALUE_TYPE,1> lowerBounds,
      NumpyView<VALUE_TYPE,1>  upperBounds
   ) {
      bool listOfList=true;
      {
         boost::python::extract<boost::python::list> extractor(lpVariableIndices[0]);
         if(extractor.check()==false)
            listOfList=false;
      }
      {
         if(listOfList){
            boost::python::extract<boost::python::list> extractor(coefficients[0]);
            if(extractor.check()==false)
               throw opengm::RuntimeError("coefficients must be a list of lists if lpVariableIndices is a list of lists");
         }
         else{
            boost::python::extract<NumpyView<VALUE_TYPE,1> > extractor(coefficients[0]);
            if(extractor.check()==false)
               throw opengm::RuntimeError("coefficients must be a list of numpy arrays if lpVariableIndices is a list of numpy arrays");
         }
      }
      if(listOfList)
         pycplex::addConstraintsPythonListList<INF,VALUE_TYPE,INDEX_TYPE>(inf,lpVariableIndices,coefficients,lowerBounds,upperBounds);
      else
         pycplex::addConstraintsPythonListNumpy<INF,VALUE_TYPE,INDEX_TYPE>(inf,lpVariableIndices,coefficients,lowerBounds,upperBounds);
   }
}




// export function
template<class GM, class ACC>
void export_cplex() {
   using namespace boost::python;
   // import numpy c-api
   import_array();
   // Inference typedefs
   typedef typename GM::ValueType ValueType;
   typedef typename GM::IndexType IndexType;
   typedef opengm::LPCplex<GM, ACC> PyLPCplex;
   typedef typename PyLPCplex::Parameter PyLPCplexParameter;
   typedef typename PyLPCplex::VerboseVisitorType PyLPCplexVerboseVisitor;

   // export inference parameter
   class_<PyLPCplexParameter > ("LPCplexParameter", init<  >() )
      .def_readwrite("integerConstraint", &PyLPCplexParameter::integerConstraint_)
      .def_readwrite("numberOfThreads", &PyLPCplexParameter::numberOfThreads_)
      .def_readwrite("cutUp", &PyLPCplexParameter::cutUp_)
      .def_readwrite("epGap", &PyLPCplexParameter::epGap_)
      .def_readwrite("timeLimit",&PyLPCplexParameter::timeLimit_)
      .def("__str__", &cplexParamAsString<PyLPCplexParameter>)
      .def ("set", &pycplex::set<PyLPCplexParameter>, 
      (
         arg("integerConstraint")=false,
         arg("numberOfThreads")=0,
         arg("cutUp")=1.0e+75,
         arg("epGap")=0,
         arg("timeLimit")=1e+75
      ),
         "Set the parameters values.\n\n"
         "All values of the parameter have a default value.\n\n"
         "Args:\n\n"
         "TODO..\n\n"
      )
      ;
   // export inference verbose visitor via macro
   OPENGM_PYTHON_VERBOSE_VISITOR_EXPORTER(PyLPCplexVerboseVisitor, "LPCplexVerboseVisitor");
   // export inference via macro
   OPENGM_PYTHON_INFERENCE_NO_RESET_EXPORTER(PyLPCplex, "LPCplex",   
   "TODO:\n\n"
   )
   .def("addConstraint", &pycplex::addConstraintPythonNumpy<PyLPCplex,ValueType,IndexType>  )
   .def("addConstraint", &pycplex::addConstraintPythonList<PyLPCplex,ValueType,IndexType>  )
   .def("addConstraints", &pycplex::addConstraintsPythonListListOrListNumpy<PyLPCplex,ValueType,IndexType>  )
   .def("addConstraints", &pycplex::addConstraintsPythonNumpy<PyLPCplex,ValueType,IndexType>  )
   ;
}
// explicit template instantiation for the supported semi-rings
template void export_cplex<GmAdder, opengm::Minimizer>();
#endif