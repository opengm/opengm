#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/stl_iterator.hpp>
#include <stdexcept>
#include <stddef.h>

#include <opengm/python/opengmpython.hxx>
#include <opengm/python/converter.hxx>
#include <opengm/python/numpyview.hxx>
#include <opengm/python/pythonfunction.hxx>



#include <opengm/utilities/functors.hxx>
using namespace boost::python;


namespace pyvector{



   template<class VECTOR>
           boost::python::list asList(const VECTOR & vector) {
      return opengm::python::iteratorToList(vector.begin(), vector.size());
   }

   template<class VECTOR>
           boost::python::tuple asTuple(const VECTOR & vector) {
      return opengm::python::iteratorToTuple(vector.begin(), vector.size());
   }

   template<class VECTOR>
           boost::python::numeric::array asNumpy(const VECTOR & vector) {
      return  opengm::python::iteratorToNumpy(vector.begin(), vector.size());
   }

   template<class VECTOR>
           std::string asString(const VECTOR & vector) {
      std::stringstream ss;
      ss << "(";
      for (size_t i = 0; i < vector.size(); ++i) {
         //if(i!=vector.size()-1)
         ss << vector[i] << ", ";
         //else
         // ss << vector[i];
      }
      ss << ")";
      return ss.str();
   }

   template<class VECTOR,class VAL>
   inline VECTOR * construct1(
      const VAL v0
   ){
      return new VECTOR(v0);
   } 


 

   template<class VECTOR>
   inline VECTOR * constructAny(
      boost::python::object ob
   ){
      stl_input_iterator<typename VECTOR::value_type> begin(ob), end;
      return new VECTOR(begin,end);
   }

   template<class VECTOR,class N_TYPE>
   inline VECTOR * constructNumpy(
      opengm::python::NumpyView<N_TYPE> numpyView
   ){
      return new VECTOR(numpyView.begin(),numpyView.end());
   }



   ///
   template<class VECTOR>
   void range(
      VECTOR & vector,
      const typename VECTOR::value_type start,
      const typename VECTOR::value_type stop,
      const typename VECTOR::value_type step
   ){
      if(start<stop){
         size_t numElements=(stop-start)/step;
         if(numElements*step!=(stop-start)){
            ++numElements;
         }
         vector.resize(numElements);
         typename VECTOR::value_type val=start;
         for(size_t i=0;i<numElements;++i){
            vector[i]=val;
            val+=step;
         }
      }
      if(stop<start){
         size_t numElements=(start-stop)/step;
         if(numElements*step!=(start-stop)){
            ++numElements;
         }
         vector.resize(numElements);
         typename VECTOR::value_type val=start;
         for(size_t i=0;i<numElements;++i){
            vector[i]=val;
            val-=step;
         }
      }
   }

   template<class VECTOR>
   void resize(
      VECTOR & vector,
      const size_t size
   ){
      vector.resize(size);
   }


   template<class VECTOR>
   inline typename  VECTOR::value_type min_val(const VECTOR & vector){ 
      typename VECTOR::value_type min=vector.front();
      for(size_t i=0;i<vector.size();++i){
         const typename VECTOR::value_type val=vector[i];
         min = val<min ? val : min;
      }
      return min;  
   }

   template<class VECTOR>
   inline typename  VECTOR::value_type max_val(const VECTOR & vector){ 
      typename VECTOR::value_type max=vector.front();
      for(size_t i=0;i<vector.size();++i){
         const typename VECTOR::value_type val=vector[i];
         max = val>max ? val : max;
      }
      return max;  
   } 

   template<class VECTOR>
   inline typename  VECTOR::value_type sum_val(const VECTOR & vector){ 
      typename VECTOR::value_type sum=0;
      for(size_t i=0;i<vector.size();++i){
         sum+=vector[i];
      }
      return sum;  
   }


   template<class VECTOR,class INDEX_VECTOR>
   VECTOR *  getItemFromStdVector(
      const VECTOR & vector,
      const INDEX_VECTOR & indexVector
   ){
      VECTOR * returnVector = new VECTOR();
      returnVector->reserve(indexVector.size());
      for(size_t i=0;i<indexVector.size();++i){
         returnVector->push_back( vector[indexVector[i]]);
      }
      return returnVector;
   }

   template<class VECTOR,class NUMPY_VIEW>
   VECTOR *  getItemFromNumpy(
      const VECTOR & vector,
      NUMPY_VIEW indexVector
   ){
      VECTOR * returnVector = new VECTOR();
      returnVector->reserve(indexVector.size());
      for(size_t i=0;i<indexVector.size();++i){
         returnVector->push_back( vector[ indexVector(i)]);
      }
      return returnVector;
   }


   template<class VECTOR,class PYTHON >
   VECTOR *  getItemFromPython(
      const VECTOR & vector,
      PYTHON indexVector
   ){
      VECTOR * returnVector = new VECTOR();
      const size_t len=boost::python::len(indexVector);
      returnVector->reserve(len);
      stl_input_iterator<typename VECTOR::value_type> begin(indexVector), end;
      for(size_t i=0;i<len;++i){
         returnVector->push_back( vector[ *begin ]);
         ++begin;
      }
      return returnVector;
   }


   template<class VECTOR_VECTOR>
   void sortSubvectors(
      VECTOR_VECTOR & vecvec
   ){
      for(size_t i=0;i<vecvec.size();++i){
         std::sort(vecvec[i].begin(),vecvec[i].end());
      }
   }

}







template<class INDEX>
void export_vectors() {
   boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
   import_array();
   typedef std::vector<INDEX> IndexTypeStdVector;
   typedef std::vector< IndexTypeStdVector> IndexTypeStdVectorVector;

   typedef INDEX FunctionIndexType;
   typedef opengm::UInt8Type FunctionTypeIndexType;
   typedef opengm::FunctionIdentification<FunctionIndexType,FunctionTypeIndexType> PyFid;

   typedef std::vector<PyFid> FidTypeStdVector;
   typedef std::vector<std::string> StdStringStdVector;
   //------------------------------------------------------------------------------------
   // std vectors 
   //------------------------------------------------------------------------------------
   boost::python::class_<IndexTypeStdVector > ("IndexVector",init<>())
      .def("__init__", make_constructor(&pyvector::constructAny<IndexTypeStdVector> ,default_call_policies() ),
      "Construct a IndexVector from any iterable python object which returns a IndexType.\n\n"
      )
      .def(init<size_t >())
      .def(init<size_t ,INDEX>())
      .def("__init__", make_constructor(&pyvector::constructNumpy<IndexTypeStdVector,typename IndexTypeStdVector::value_type> ,default_call_policies() ),
      "Construct a IndexVector from a numpy array.\n\n"
      )

      .def("resize",&pyvector::resize<IndexTypeStdVector>)
      .def("range",&pyvector::range<IndexTypeStdVector>,
         (
            arg("start"),
            arg("stop"),
            arg("step")=1
         ),
         "doc"
      )    
     .def(boost::python::vector_indexing_suite<IndexTypeStdVector > ())
     .def("__getitem__", &pyvector::getItemFromNumpy<IndexTypeStdVector,opengm::python::NumpyView<INDEX> >, return_value_policy<manage_new_object>( ) )      
     .def("__getitem__", &pyvector::getItemFromStdVector<IndexTypeStdVector,IndexTypeStdVector>, return_value_policy<manage_new_object>( ) )      
     .def("__getitem__", &pyvector::getItemFromPython<IndexTypeStdVector, boost::python::list >, return_value_policy<manage_new_object>( ) )      
     .def("__getitem__", &pyvector::getItemFromPython<IndexTypeStdVector,boost::python::tuple >, return_value_policy<manage_new_object>( ) )      
     .add_property("size", &IndexTypeStdVector::size)
     //.def(boost::python::vector_indexing_suite<IndexTypeStdVector > ())
     .def("__str__", &pyvector::asString<IndexTypeStdVector>)
     .def("__array__", &pyvector::asNumpy<IndexTypeStdVector>)
     .def("__tuple__", &pyvector::asTuple<IndexTypeStdVector>)
     .def("__list__", &pyvector::asList<IndexTypeStdVector>)
     //.def("view", &pyvector::vectorAsopengm::python::NumpyView<IndexTypeStdVector>, with_custodian_and_ward_postcall<0, 1>(),"get dnarray view to index vector")
   ;


   boost::python::class_<FidTypeStdVector > ("FidVector")
     .def(boost::python::vector_indexing_suite<FidTypeStdVector > ())
     .def("__init__", make_constructor(&pyvector::constructAny<FidTypeStdVector> ,default_call_policies() ),
      "Construct a IndexVector from any iterable python object which returns a FunctionIdentifier.\n\n"
      )
   ;

   boost::python::class_<StdStringStdVector > ("StringVector")
     .def(boost::python::vector_indexing_suite<StdStringStdVector,true > ())
   ;

   boost::python::class_<IndexTypeStdVectorVector > ("IndexVectorVector")
     .def(init<size_t >())
     .def(init<size_t,  const typename IndexTypeStdVectorVector::value_type & >())
     .def(boost::python::vector_indexing_suite<IndexTypeStdVectorVector > ())
     .def("sort",pyvector::sortSubvectors<IndexTypeStdVectorVector>,"sort all the sub-vectors of this vector of vectors")
   ;
   


}

template void export_vectors<opengm::python::GmIndexType>();