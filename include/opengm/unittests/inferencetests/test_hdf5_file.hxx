#ifndef TEST_HDF5_FILE_HXX_
#define TEST_HDF5_FILE_HXX_

#include <opengm/inference/inference.hxx>
#include <opengm/unittests/test.hxx>
#include <opengm/unittests/inferencetests/test_base.hxx>
#ifdef WITH_HDF5
#include <opengm/datastructures/marray/marray_hdf5.hxx>
#include <opengm/graphicalmodel/graphicalmodel_hdf5.hxx>
#endif

#include <vector>
/// \cond HIDDEN_SYMBOLS
namespace opengm {
   namespace test {
      /// \brief TestHDF5File<INF> 
      /// Load a model from a hdf5-file and check if the algorithm INF show  
      /// the correct behaviour.
      /// This test is an easy way to include found bugs into the tests. 
      template <class INF>
      class TestHDF5File : public TestBase<INF> {
      protected:
         typedef typename INF::GraphicalModelType GraphicalModelType;
         typedef typename INF::AccumulatorType AccumulatorType;
         GraphicalModelType gm_;
         const std::string hdf5FileLocation_;
         const std::string datasetName_;
         TestBehaviour behaviour_;
#ifdef WITH_HDF5
         bool loadModel();
         bool loadArgMax(std::vector<typename GraphicalModelType::LabelType>&);
         bool loadArgMin(std::vector<typename GraphicalModelType::LabelType>&);
         bool loadMax(typename GraphicalModelType::ValueType&);
         bool loadMin(typename GraphicalModelType::ValueType&);
         template<class T> bool loadValue(const hid_t& groupHandle,const std::string& datasetName,T& value);
#endif
      public:
         TestHDF5File(const std::string&, const std::string&, TestBehaviour);
         virtual void test(typename INF::Parameter param);
      };

      /// \brief TestHDF5File Constructor
      /// \param hdf5FileLocation path and name of the hdf5-file containing the model
      /// \param datasetName name of the dataset of the model
      /// \param behaviour expected behaviour of the algorithm
      template <class INF>
      TestHDF5File<INF>::TestHDF5File(const std::string& hdf5FileLocation, const std::string& datasetName, TestBehaviour behaviour)
         : hdf5FileLocation_(hdf5FileLocation), datasetName_(datasetName), behaviour_(behaviour) {

      }

      /// \brief test<INF> start test with algorithm INF
      /// \param para parameters of algorithm
      template <class INF>
      void TestHDF5File<INF>::test(typename INF::Parameter param) {
         std::cout << "test started..." << std::endl;
#ifdef WITH_HDF5
         bool fail = false;
         try {
            OPENGM_TEST(loadModel());

            INF inf(gm_, param);

            OPENGM_TEST(inf.infer() == opengm::NORMAL);
            std::vector<typename INF::LabelType> state;
            OPENGM_TEST(inf.arg(state) == opengm::NORMAL);
            OPENGM_TEST(state.size() == gm_.numberOfVariables());
            
            typename GraphicalModelType::ValueType v;
            if(behaviour_ == opengm::test::OPTIMAL) {
               if(typeid(AccumulatorType) == typeid(opengm::Minimizer)) {
                  if(loadMin(v)) {
                     OPENGM_ASSERT(inf.value()==v);
                     OPENGM_ASSERT(inf.bound()<=v);
                  }
                  else{
                     std::cout << " !! file doesnt include optimal objective value !! ";
                  }
               }
               if(typeid(AccumulatorType) == typeid(opengm::Maximizer)) {
                  if(loadMax(v)) {
                     OPENGM_ASSERT(inf.value()==v); 
                     OPENGM_ASSERT(inf.bound()>=v)
                  }
                  else{
                     std::cout << " !! file doesnt include optimal objective value !! ";
                  }
               }

               throw RuntimeError("Optimal test not yet implemented");
            }
            else{
               if(typeid(AccumulatorType) == typeid(opengm::Minimizer)) {
                  if(loadMin(v)) {
                     OPENGM_ASSERT(inf.value()>=v); 
                     OPENGM_ASSERT(inf.bound()<=v);
                  }
                  else{
                     std::cout << " !! file doesnt include optimal objective value !! ";
                  }
               }
               if(typeid(AccumulatorType) == typeid(opengm::Maximizer)) {
                  if(loadMax(v)) {
                     OPENGM_ASSERT(inf.value()<=v); 
                     OPENGM_ASSERT(inf.bound()>=v)
                  }
                  else{
                     std::cout << " !! file doesnt include optimal objective value !! ";
                  }
               }


            }
         } catch (std::exception& error) {
            std::cout << error.what() << std::endl;
            fail = true;
         }

         // Check if exception has been thrown
         if(behaviour_ == opengm::test::FAIL) {
            OPENGM_TEST(fail);
         }else{
            OPENGM_TEST(!fail);
         }
         std::cout << "test done!" << std::endl;
#else
         std::cout << "hdf5 deactivated - cannot run this test!" << std::endl;
#endif
      }
#ifdef WITH_HDF5
      template <class INF>
      bool TestHDF5File<INF>::loadModel() {
         try {
            opengm::hdf5::load(gm_, hdf5FileLocation_, datasetName_);
            return true;
         } catch (std::exception& error) {
            std::cout << error.what() << std::endl;
            return false;
         }
      }  

      template <class INF>
      bool TestHDF5File<INF>::loadArgMax(std::vector<typename GraphicalModelType::LabelType>& state) {
         try {
            marray::hdf5::load(state, hdf5FileLocation_, "argmax");
            return true;
         } catch (std::exception& error) {
            std::cout << error.what() << std::endl;
            return false;
         }
      }

      template <class INF>
      bool TestHDF5File<INF>::loadArgMin(std::vector<typename GraphicalModelType::LabelType>& state) {
         try {
            marray::hdf5::load(state, hdf5FileLocation_, "argmin");
            return true;
         } catch (std::exception& error) {
            std::cout << error.what() << std::endl;
            return false;
         }
      }

      template <class INF>
      bool TestHDF5File<INF>::loadMax(typename GraphicalModelType::ValueType& value) {
         try {
            hid_t file = marray::hdf5::openFile( hdf5FileLocation_ );
            return loadValue(file, "max", value);
         } catch (std::exception& error) {
            std::cout << error.what() << std::endl;
            return false;
         }
      } 

      template <class INF>
      bool TestHDF5File<INF>::loadMin(typename GraphicalModelType::ValueType& value) {
         try { 
            hid_t file = marray::hdf5::openFile( hdf5FileLocation_ );
            return loadValue(file, "min", value);       
         } catch (std::exception& error) {
            std::cout << error.what() << std::endl;
            return false;
         }
      }

      template <class INF>
      template<class T>
      bool TestHDF5File<INF>::loadValue(
         const hid_t& groupHandle,
         const std::string& datasetName,
         T& value
         )
      {
         if(!H5Aexists(groupHandle, datasetName.c_str() ))
            return false;
         hid_t dataset = H5Dopen(groupHandle, datasetName.c_str(), H5P_DEFAULT);
         if(dataset < 0) {
            return false;
            throw std::runtime_error("Marray cannot open dataset.");
         } 
         hsize_t shape[] = {1};
         hid_t filespace = H5Dget_space(dataset);
         hid_t type = H5Dget_type(dataset);
         hid_t nativeType = H5Tget_native_type(type, H5T_DIR_DESCEND);
         hid_t memspace = H5Screate_simple(1, shape, NULL);

         if(!H5Tequal(nativeType, marray::hdf5::hdf5Type<T>())) {
            H5Dclose(dataset);
            H5Tclose(nativeType);
            H5Tclose(type);
            H5Sclose(filespace);
            return false;
            throw std::runtime_error("Data types not equal error.");
         }

         // read
         herr_t status = H5Dread(dataset, nativeType, memspace, filespace, H5P_DEFAULT, &value);
         H5Dclose(dataset);
         H5Tclose(nativeType);
         H5Tclose(type);
         H5Sclose(memspace);
         H5Sclose(filespace);
         if(status < 0) {
            return false;
            throw std::runtime_error("Marray cannot read from dataset.");
         }
         return true;
      }
#endif

   } // namespace test

} // namespace opengm
/// \endcond
#endif /* TEST_HDF5_FILE_HXX_ */
