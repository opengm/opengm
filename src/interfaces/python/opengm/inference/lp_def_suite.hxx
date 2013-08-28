#ifndef LP_DEF_VISITOR
#define LP_DEF_VISITOR

#include <boost/python.hpp>
#include <sstream>
#include <string>
#include <boost/python/def_visitor.hpp>

#include <opengm/python/opengmpython.hxx>
#include <opengm/python/converter.hxx>
#include <opengm/python/numpyview.hxx>
#include <opengm/python/pythonfunction.hxx>

#include "gil.hxx"

#include <opengm/inference/inference.hxx>
#include <opengm/opengm.hxx>

#include <opengm/python/opengmpython.hxx>
#include <opengm/python/converter.hxx>
#include <opengm/python/numpyview.hxx>
#include <opengm/python/pythonfunction.hxx>




namespace pylp{



}




template<class INF>
class LpInferenceSuite: public boost::python::def_visitor<LpInferenceSuite<INF> >{
public:
   friend class boost::python::def_visitor_access;
   typedef typename INF::IndexType IndexType;
   typedef typename INF::LabelType LabelType;
   typedef typename INF::ValueType ValueType;
   
   LpInferenceSuite(){

   }

   template <class classT>
   void visit(classT& c) const{ 
      c
         .def("_addConstraint", &addConstraintPythonNumpy)
         .def("_addConstraints", &addConstraintsPythonNumpy)

         .def("_lpNodeVariableIndex",&INF::lpNodeVi)
         .def("_lpFactorVariableIndex_Scalar",&lpFactorViScalar)
         .def("_lpFactorVariableIndex_Numpy",&lpFactorIter)
      ;
   }


   // add constraints from python numpy
   static void addConstraintPythonNumpy
   (
      INF & inf,
      opengm::python::NumpyView<IndexType,1> lpVariableIndices,
      opengm::python::NumpyView<ValueType,1> coefficients,
      const ValueType lowerBound,
      const ValueType upperBound
   ) {
      // add constraint
      inf.addConstraint(lpVariableIndices.begin(),lpVariableIndices.end(),coefficients.begin(),lowerBound,upperBound);
   }

  

   static void addConstraintsPythonNumpy
   (
      INF & inf,
      opengm::python::NumpyView<IndexType,2> lpVariableIndices,
      opengm::python::NumpyView<ValueType,2> coefficients,
      opengm::python::NumpyView<ValueType,1> lowerBounds,
      opengm::python::NumpyView<ValueType,1>  upperBounds
   ) {

      const size_t nVi     = static_cast<size_t>(lpVariableIndices.shape(0));
      const size_t nCoeff  = static_cast<size_t>(coefficients.shape(0));
      const size_t nLBound = static_cast<size_t>(lowerBounds.size());
      const size_t nUBound = static_cast<size_t>(upperBounds.size());

      const size_t nVipPerConstaint    = static_cast<size_t>(lpVariableIndices.shape(1));
      const size_t nCoeffPerConstaint   = static_cast<size_t>(coefficients.shape(1));
      if(nVipPerConstaint!=nCoeffPerConstaint)
         throw opengm::RuntimeError("coefficients per constraints must be as long as variables per constraint");
      opengm::FastSequence<IndexType> lpVar(nVipPerConstaint);
      opengm::FastSequence<IndexType> coeff(nCoeffPerConstaint);
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



   static IndexType lpFactorIter(
      const INF & inf,
      typename INF::IndexType factorIndex,
      opengm::python::NumpyView<typename INF::LabelType> labeling 
   ){
      return inf.lpFactorVi(factorIndex,labeling.begin(),labeling.end());
   }


   static IndexType lpFactorViScalar(
      const INF & inf,
      IndexType factorIndex,
      IndexType labelingIndex
   ){
      return inf.lpFactorVi(factorIndex,labelingIndex);
   }



};





#endif // LP_DEF_VISITOR