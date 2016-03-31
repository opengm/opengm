#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>
#include <boost/python/wrapper.hpp> 

#include <stdexcept>
#include <stddef.h>
#include <vector>
#include <map>

#include <opengm/python/opengmpython.hxx>
#include <opengm/python/converter.hxx>
#include <opengm/python/numpyview.hxx>
#include <opengm/python/pythonfunction.hxx>


#include "nifty_iterator.hxx"
//#include "copyhelper.hxx"

#include "opengm/utilities/functors.hxx"
#include "opengm/functions/explicit_function.hxx"
#include "opengm/functions/absolute_difference.hxx"
#include "opengm/functions/potts.hxx"
#include "opengm/functions/pottsn.hxx"
#include "opengm/functions/pottsg.hxx"
#include "opengm/functions/squared_difference.hxx"
#include "opengm/functions/truncated_absolute_difference.hxx"
#include "opengm/functions/truncated_squared_difference.hxx"
#include "opengm/functions/sparsemarray.hxx"
#include "pyPythonFunction.hxx"


#include "functionGenBase.hxx"

using namespace boost::python;



template<class GM_ADDER,class GM_MULT>
class FunctionGeneratorBaseWrap : public FunctionGeneratorBase<GM_ADDER,GM_MULT>, public wrapper<FunctionGeneratorBase<GM_ADDER,GM_MULT> >
{
public:

   // typename GM_ADDER::FunctionIdentifier  addFunctionGmAdder(const size_t index, GM_ADDER & gm)const
   // {
   //     return this->get_override("addFunctionGmAdder")(index,gm);
   // }

   std::vector< typename GM_ADDER::FunctionIdentifier > * addFunctions( GM_ADDER & gm)const
   {
      return this->get_override("addFunctionGmMultiplier")(gm);
   }
   std::vector< typename GM_MULT::FunctionIdentifier > * addFunctions(GM_MULT & gm)const
   {
      return this->get_override("addFunctionGmMultiplier")(gm);
   }


};






template<class GM_ADDER,class GM_MULT,class FUNCTION_TYPE>
class PottsFunctionGen :
public FunctionGeneratorBase<GM_ADDER,GM_MULT>
{
public:
    typedef FUNCTION_TYPE FunctionType;
    typedef typename FUNCTION_TYPE::ValueType ValueType;
    typedef typename FUNCTION_TYPE::IndexType IndexType;
    typedef typename FUNCTION_TYPE::LabelType LabelType;
    PottsFunctionGen(
        opengm::python::NumpyView<LabelType,1> numLabels1Array,
        opengm::python::NumpyView<LabelType,1> numLabels2Array,
        opengm::python::NumpyView<ValueType,1> valEqualArray,
        opengm::python::NumpyView<ValueType,1> valNotEqualArray
    ):FunctionGeneratorBase<GM_ADDER,GM_MULT>(),
    numLabels1Array_(numLabels1Array),
    numLabels2Array_(numLabels2Array),
    valEqualArray_(valEqualArray),
    valNotEqualArray_(valNotEqualArray)
    {
        numFunctions_=std::max( 
            std::max(numLabels1Array_.shape(0),numLabels2Array_.shape(0)) , 
            std::max(valEqualArray_.shape(0),valNotEqualArray_.shape(0))
        );
    }  

   template<class GM>
   std::vector< typename GM::FunctionIdentifier > * addFunctionsGeneric(GM & gm)const{
      std::vector< typename GM::FunctionIdentifier > * fidVector = new std::vector< typename GM::FunctionIdentifier > (numFunctions_);
      for(size_t  i=0;i<numFunctions_;++i){
         const LabelType numL1=i<numLabels1Array_.size() ? numLabels1Array_(i) : numLabels1Array_(numLabels1Array_.size()-1);
         const LabelType numL2=i<numLabels2Array_.size() ? numLabels2Array_(i) : numLabels2Array_(numLabels2Array_.size()-1);
         const ValueType veq=i<valEqualArray_.size() ? valEqualArray_(i) : valEqualArray_(valEqualArray_.size()-1);
         const ValueType vneq=i<valNotEqualArray_.size() ? valNotEqualArray_(i) : valNotEqualArray_(valNotEqualArray_.size()-1);
         (*fidVector)[i]=gm.addFunction(FunctionType(numL1,numL2,veq,vneq));
      }
      return fidVector;
   }
    
   virtual std::vector< typename GM_ADDER::FunctionIdentifier > * addFunctions(GM_ADDER & gm)const{
      return this-> template addFunctionsGeneric<GM_ADDER>(gm);
   }
   virtual std::vector< typename GM_MULT::FunctionIdentifier >  * addFunctions(GM_MULT & gm)const{
      return this-> template addFunctionsGeneric<GM_MULT>(gm);
   }
private:
   opengm::python::NumpyView<LabelType,1>  numLabels1Array_;
   opengm::python::NumpyView<LabelType,1>  numLabels2Array_;
   opengm::python::NumpyView<ValueType,1>  valEqualArray_;
   opengm::python::NumpyView<ValueType,1>  valNotEqualArray_;
   size_t numFunctions_;
};


template<class GM_ADDER,class GM_MULT,class FUNCTION>
inline FunctionGeneratorBase<GM_ADDER,GM_MULT> * pottsFunctionGen(
    opengm::python::NumpyView<typename GM_ADDER::LabelType,1> numLabels1Array,
    opengm::python::NumpyView<typename GM_ADDER::LabelType,1> numLabels2Array,
    opengm::python::NumpyView<typename GM_ADDER::ValueType,1> valEqualArray,
    opengm::python::NumpyView<typename GM_ADDER::ValueType,1> valNotEqualArray
){
    FunctionGeneratorBase<GM_ADDER,GM_MULT> * ptr= new PottsFunctionGen<GM_ADDER,GM_MULT,FUNCTION>(numLabels1Array,numLabels2Array,valEqualArray,valNotEqualArray);
    return ptr;
}

template<class GM_ADDER,class GM_MULT>  
void export_function_generator(){
   boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
   import_array();
   typedef typename GM_ADDER::ValueType ValueType;
   typedef typename GM_ADDER::IndexType IndexType;
   typedef typename GM_ADDER::LabelType LabelType;

   typedef FunctionGeneratorBase<GM_ADDER,GM_MULT> PyFunctionGeneratorBase;
   typedef FunctionGeneratorBaseWrap<GM_ADDER,GM_MULT> PyFunctionGeneratorBaseWrap;

   // different function types
   typedef opengm::ExplicitFunction                      <ValueType,IndexType,LabelType> PyExplicitFunction;
   typedef opengm::PottsFunction                         <ValueType,IndexType,LabelType> PyPottsFunction;
   typedef opengm::PottsNFunction                        <ValueType,IndexType,LabelType> PyPottsNFunction;
   typedef opengm::PottsGFunction                        <ValueType,IndexType,LabelType> PyPottsGFunction;
   typedef opengm::AbsoluteDifferenceFunction            <ValueType,IndexType,LabelType> PyAbsoluteDifferenceFunction;
   typedef opengm::TruncatedAbsoluteDifferenceFunction   <ValueType,IndexType,LabelType> PyTruncatedAbsoluteDifferenceFunction;
   typedef opengm::SquaredDifferenceFunction             <ValueType,IndexType,LabelType> PySquaredDifferenceFunction;
   typedef opengm::TruncatedSquaredDifferenceFunction    <ValueType,IndexType,LabelType> PyTruncatedSquaredDifferenceFunction;
   typedef opengm::SparseFunction                        <ValueType,IndexType,LabelType> PySparseFunction; 
   typedef opengm::python::PythonFunction                <ValueType,IndexType,LabelType> PyPythonFunction; 


   typedef PottsFunctionGen<GM_ADDER,GM_MULT,PyPottsFunction> PyPottsFunctionGen;



   class_<PyFunctionGeneratorBaseWrap,boost::noncopyable>("_FunctionGeneratorBaseWrap",init<>())
   ;





   def("pottsFunctionsGen",&pottsFunctionGen<GM_ADDER,GM_MULT,PyPottsFunction>,return_value_policy<manage_new_object>(),
      (arg("numberOfLabels1"),arg("numberOfLabels2"),arg("valuesEqual"),arg("valuesNotEqual")),
      "factory function to generate a potts function generator object which can be passed to ``gm.addFunctions(functionGenerator)``");
   
   //class_<PyPottsFunctionGen, bases<PyFunctionGeneratorBase> >("_PottsFunctionGen",init<        opengm::python::NumpyView<LabelType,1> , opengm::python::NumpyView<LabelType,1> ,NumpyView<ValueType,1> ,NumpyView<ValueType,1> > () )
   //;
   
   

}

template void export_function_generator<opengm::python::GmAdder,opengm::python::GmMultiplier>();
