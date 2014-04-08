#ifndef IO_BASE_HXX_
#define IO_BASE_HXX_

#include <ostream>
#include <sstream>
#include <fstream>
#include <typeinfo>

#include <opengm/datastructures/marray/marray.hxx>
#ifdef WITH_HDF5
#include <opengm/datastructures/marray/marray_hdf5.hxx>
#endif

#include "../argument/vector_argument.hxx"

namespace opengm {

namespace interface {

/*********************
 * class definitions *
 *********************/
class IOBase {
protected:
   std::ostream& standardStream_;
   std::ostream& errorStream_;
   std::ostream& logStream_;

   template <class ARGUMENT>
   bool sanityCheck(const ARGUMENT& command);

   template <class CONTAINER, class VECTYPE>
   bool sanityCheck(CONTAINER& storage, VECTYPE object);

   std::string getFileExtension(const std::string& filename);
   template <class VEC_TYPE>
   void loadVectorText(const std::string& filename, const std::string& dataset, VEC_TYPE& vec);
   #ifdef WITH_HDF5
   template <class VEC_TYPE>
   void loadVectorHDF5(const std::string& filename, const std::string& dataset, VEC_TYPE& vec);
   #endif
   template <class VEC_TYPE>
   void storeVectorText(const std::string& filename, const std::string& dataset, VEC_TYPE& vec);
   #ifdef WITH_HDF5
   template <class VEC_TYPE>
   void storeVectorHDF5(const std::string& filename, const std::string& dataset, VEC_TYPE& vec);
   #endif
   template <class VEC_TYPE>
   void printVector(VEC_TYPE& vec);
   #ifdef WITH_HDF5
   static herr_t file_info(hid_t loc_id, const char *name, void *opdata);
   #endif
   template <class MARRAY>
   void loadMArrayText(const std::string& filename, const std::string& dataset, MARRAY& array);
   #ifdef WITH_HDF5
   template <class MARRAY>
   void loadMArrayHDF5(const std::string& filename, const std::string& dataset, MARRAY& array);
   #endif
   template <class MARRAY>
   void storeMArrayText(const std::string& filename, const std::string& dataset, MARRAY& array);
   #ifdef WITH_HDF5
   template <class MARRAY>
   void storeMArrayHDF5(const std::string& filename, const std::string& dataset, MARRAY& array);
   #endif
public:
   IOBase(std::ostream& standardStreamIn, std::ostream& errorStreamIn, std::ostream& logStreamIn);

   template <class ARGUMENT>
   bool read(const ARGUMENT& command);
   template <class VECTOR, class CONTAINER>
   bool read(const VectorArgument<VECTOR, CONTAINER>& command);
   template <class VECTOR>
   bool read(const VectorArgument<VECTOR>& command);

   template <class ARGUMENT>
   bool write(const ARGUMENT& command);

   template <class ARGUMENT>
   bool info(const ARGUMENT& command);

   void separateFilename(const std::string& completeFilename, std::string& filename, std::string& dataset);
   template <class VEC_TYPE>
   void loadVector(const std::string& completeFilename, VEC_TYPE& vec);
   template <class VEC_TYPE>
   void storeVector(const std::string& completeFilename, VEC_TYPE& vec);
   template <class MARRAY>
   void loadMArray(const std::string& completeFilename, MARRAY& array);
   template <class MARRAY>
   void storeMArray(const std::string& completeFilename, MARRAY& array);
   template <class GM>
   void modelInfo(GM& graphicalModel, std::ostream& stream);

   bool fileExists(const std::string& filename) const;
   std::ostream& standardStream();
   std::ostream& errorStream();
   std::ostream& logStream();
};

/***********************
 * class documentation *
 ***********************/
//TODO add documentation

/******************
 * implementation *
 ******************/

/**
 * @brief Constructor for the base interface.
 * @param[in] argc Number of arguments provided by the user
 * @param[in] argv List of the aruments provided by the user.
 */
IOBase::IOBase(std::ostream& standardStreamIn, std::ostream& errorStreamIn, std::ostream& logStreamIn)
   : standardStream_(standardStreamIn), errorStream_(errorStreamIn), logStream_(logStreamIn) { }

/**
 * @brief Reads the specified command from the user input.
 * @note This function is specialized for each type of ArgumentBase<>. The
 *       default implementation only throws an error of unknown type.
 * @tparam ARGUMENT Type of the command argument.
 * @param[in] command The selected command.
 * @return True if read was successful, otherwise false.
 */
template <class ARGUMENT>
bool IOBase::read(const ARGUMENT& command) {
   const std::string error((std::string)"Trying to read command of type " + (std::string)typeid(ARGUMENT).name() + (std::string)" which is not supported.");
   errorStream_ << error << std::endl;
   throw RuntimeError(error);
   return false;
}

template <class VECTOR, class CONTAINER>
bool IOBase::read(const VectorArgument<VECTOR, CONTAINER>& command) {
   const std::string error((std::string)"Trying to read command of type " + (std::string)typeid(VectorArgument<VECTOR, CONTAINER>).name() + (std::string)" which is not supported.");
   errorStream_ << error << std::endl;
   throw RuntimeError(error);
   return false;
}

template <class VECTOR>
bool IOBase::read(const VectorArgument<VECTOR>& command) {
   const std::string error((std::string)"Trying to read command of type " + (std::string)typeid(VectorArgument<VECTOR>).name() + (std::string)" which is not supported.");
   errorStream_ << error << std::endl;
   throw RuntimeError(error);
   return false;
}

/**
 * @brief Writes the specified command.
 * @note This function is specialized for each type of ArgumentBase<>. The
 *       default implementation only throws an error of unknown type.
 * @tparam ARGUMENT Type of the command argument.
 * @param[in] command The selected command.
 * @return True if write was successful, otherwise false.
 */
template <class ARGUMENT>
bool IOBase::write(const ARGUMENT& command) {
   const std::string error((std::string)"Trying to write command of type " + (std::string)typeid(ARGUMENT).name() + (std::string)" which is not supported.");
   errorStream_ << error << std::endl;
   throw RuntimeError(error);
   return false;
}

/**
 * @brief Gives informations about the specified command.
 * @note This function is specialized for each type of ArgumentBase<>. The
 *       default implementation only throws an error of unknown type.
 * @tparam ARGUMENT Type of the command argument.
 * @param[in] command The selected command.
 * @return True if info was successful, otherwise false.
 */
template <class ARGUMENT>
bool IOBase::info(const ARGUMENT& command) {
   const std::string error((std::string)"Trying to get info from command of type " + (std::string)typeid(ARGUMENT).name() + (std::string)" which is not supported.");
   errorStream_ << error << std::endl;
   throw RuntimeError(error);
   return false;
}

template <class ARGUMENT>
bool IOBase::sanityCheck(const ARGUMENT& command) {
   if(command.isRequired() && !command.isSet()){
      std::string error = (std::string)"Error: Argument \"" + command.getLongName() + "\" not set, but required.";
      if(!command.GetPermittedValues().empty()) {
         std::stringstream stream;
         command.printValidValues(stream);
         error += (std::string)"\n Valid values are: \n";
         error += stream.str();
      }
      errorStream_ << error << std::endl;
      throw RuntimeError(error);
   } else if(command.hasDefaultValue()){
      standardStream_ << "Warning: Argument \"" << command.getLongName() << "\" not set. Using default value: ";
      command.printDefaultValue(standardStream_);
      standardStream_ << std::endl;
      command(command.getDefaultValue(), false);
      return true;
   } else {
      return false;
   }
}

template <class VEC_TYPE>
void IOBase::loadVector(const std::string& completeFilename, VEC_TYPE& vec) {
  std::string separatedFilename;
  std::string dataset;
  std::string fileExtension;

  separateFilename(completeFilename, separatedFilename, dataset);
  fileExtension = getFileExtension(separatedFilename);
  if(fileExtension == "txt") {
    loadVectorText(separatedFilename, dataset, vec);
  } else if(fileExtension == "h5") {
#ifdef WITH_HDF5
    loadVectorHDF5(separatedFilename, dataset, vec);
#else
    std::cerr << "HDF5 support currently disabled. Rebuild with HDF5 support enabled and try again" << std::endl;
    std::abort();
#endif
  } else {
    std::cout << "warning: unknown file extension. Assuming, the selected file is a text file" << std::endl;
    loadVectorText(separatedFilename, dataset, vec);
  }
}

template <class VEC_TYPE>
void IOBase::storeVector(const std::string& completeFilename, VEC_TYPE& vec) {
  if(completeFilename == "PRINT ON SCREEN") {
    printVector(vec);
  } else {
    std::string separatedFilename;
    std::string dataset;
    std::string fileExtension;

    separateFilename(completeFilename, separatedFilename, dataset);
    fileExtension = getFileExtension(separatedFilename);
    if(fileExtension == "txt") {
      storeVectorText(separatedFilename, dataset, vec);
    } else if(fileExtension == "h5") {
#ifdef WITH_HDF5
      storeVectorHDF5(separatedFilename, dataset, vec);
#else
    std::cerr << "HDF5 support currently disabled. Rebuild with HDF5 support enabled and try again" << std::endl;
    std::abort();
#endif
    } else {
      std::cout << "warning: unknown file extension. Assuming, the selected file is a text file" << std::endl;
      storeVectorText(separatedFilename, dataset, vec);
    }
  }
}

template <class MARRAY>
void IOBase::loadMArray(const std::string& completeFilename, MARRAY& array) {
  std::string separatedFilename;
  std::string dataset;
  std::string fileExtension;

  separateFilename(completeFilename, separatedFilename, dataset);
  fileExtension = getFileExtension(separatedFilename);
  if(fileExtension == "txt") {
    loadMArrayText(separatedFilename, dataset, array);
  } else if(fileExtension == "h5") {
#ifdef WITH_HDF5
    loadMArrayHDF5(separatedFilename, dataset, array);
#else
    std::cerr << "HDF5 support currently disabled. Rebuild with HDF5 support enabled and try again" << std::endl;
    std::abort();
#endif
  } else {
    std::cout << "warning: unknown file extension. Assuming, the selected file is a text file" << std::endl;
    loadMArrayText(separatedFilename, dataset, array);
  }
}

template <class MARRAY>
void IOBase::storeMArray(const std::string& completeFilename, MARRAY& array) {
  if(completeFilename == "PRINT ON SCREEN") {
    printVector(array);
  } else {
    std::string separatedFilename;
    std::string dataset;
    std::string fileExtension;

    separateFilename(completeFilename, separatedFilename, dataset);
    fileExtension = getFileExtension(separatedFilename);
    if(fileExtension == "txt") {
      storeMArrayText(separatedFilename, dataset, array);
    } else if(fileExtension == "h5") {
#ifdef WITH_HDF5
      storeMArrayHDF5(separatedFilename, dataset, array);
#else
    std::cerr << "HDF5 support currently disabled. Rebuild with HDF5 support enabled and try again" << std::endl;
    std::abort();
#endif
    } else {
      std::cout << "warning: unknown file extension. Assuming, the selected file is a text file" << std::endl;
      storeMArrayText(separatedFilename, dataset, array);
    }
  }
}

//TODO adjust modelinfo() to new structure of graphical model
template <class GM>
void IOBase::modelInfo(GM& graphicalModel, std::ostream& stream) {
   stream << "Model Info" <<std::endl;
   stream << "----------" <<std::endl;
   stream << std::endl;
   stream << "Number of Variables   : " << graphicalModel.numberOfVariables() << std::endl;
   typename GM::IndexType min = 1000000000;
   typename GM::IndexType max = 0;
   for (typename GM::IndexType i=0; i<graphicalModel.numberOfVariables();++i){
      if(min>graphicalModel.numberOfLabels(i)) min = graphicalModel.numberOfLabels(i);
      if(max<graphicalModel.numberOfLabels(i)) max = graphicalModel.numberOfLabels(i);
   }
   if(min==max) std::cout<<"Number of States      : "<< min << std::endl;
   else         std::cout<<"Number of States      : ["<<min<<","<<max<<"]" << std::endl;
   stream << "Model-Order           : " << graphicalModel.factorOrder() << std::endl;
   stream << "Number of Factors     : " << graphicalModel.numberOfFactors() << std::endl;
   stream << "Number of Function Types   : "<< GM::NrOfFunctionTypes << std::endl;
/*      stream << "Number of Functions   : "<< graphicalModel.numberOfFunctions(0) << std::endl;
   stream << "Number of Functions   : "<< graphicalModel.numberOfFunctions(1) << std::endl;
   stream << "Number of Functions   : "<< graphicalModel.numberOfFunctions(2) << std::endl;*/
/*      stream << "-  Number of Explicite Functions : " << graphicalModel.numberOfExplicitFunctions() << std::endl;
   stream << "-  Number of Sparse Functions    : " << graphicalModel.numberOfSparseFunctions() << std::endl;
   std::cout<<"-  Number of Implicit Functions  : " << graphicalModel.numberOfImplicitFunctions() << std::endl;
   std::cout<<"Number of Constraints : " << graphicalModel.numberOfConstraints() << std::endl;
*/
   stream << "Acyclic Model         : " << graphicalModel.isAcyclic() << std::endl;
   //stream << "Submodular Model      : ???" << std::endl;
   marray::Vector<size_t> chain;
   stream << "Chain Model           : " << graphicalModel.isChain(chain) << std::endl;
   marray::Matrix<size_t> grid;
   stream << "Grid Model            : " << graphicalModel.isGrid(grid) << std::endl;
   //stream << "Planar Model          : ???" << std::endl;
   //stream << "Treewidth             : ???" << std::endl;
   //stream << "Diameter              : ???" << std::endl;
   //stream << "Node-Degree           : ???" << std::endl;
   stream << std::endl;
}

std::ostream& IOBase::standardStream() {
   return standardStream_;
}

std::ostream& IOBase::errorStream() {
   return errorStream_;
}
std::ostream& IOBase::logStream() {
   return logStream_;
}

void IOBase::separateFilename(const std::string& completeFilename, std::string& filename, std::string& dataset) {
  size_t colonPosition = completeFilename.find(':');
  if(colonPosition == std::string::npos) {
    std::cout << "warning: no dataset specified" << std::endl;
    filename = completeFilename;
    dataset = "";
  } else {
    filename = completeFilename.substr(0, colonPosition);
    dataset = completeFilename.substr(colonPosition + 1);
  }
}

std::string IOBase::getFileExtension(const std::string& filename) {
  size_t dotPosition = filename.rfind('.');
  if(dotPosition == std::string::npos) {
    std::cout << "warning: no file extension found. Assuming, the selected file is a text file" << std::endl;
    return "txt";
  } else {
    return filename.substr(dotPosition + 1);
  }
}

template <class VEC_TYPE>
void IOBase::loadVectorText(const std::string& filename, const std::string& dataset, VEC_TYPE& vec) {
  //std::cout << "loading vector from textfile..." << std::endl;
  std::ifstream vector_file(filename.c_str(), std::ifstream::in);
  if(!vector_file) {
    std::cerr << "error: could not open file: " << filename << std::endl;
    std::abort();
  }
  typename VEC_TYPE::value_type current_value;
  std::string current_line;
  std::string datasetIntern = dataset;
  if(datasetIntern.empty())
  {
    std::cout << "warning: dataset not specified, taking first vector found in file." << std::endl;
    while(!vector_file.eof()) {
      getline(vector_file, current_line);

      if(current_line.size() == 0) {
        continue;
      }

      if(current_line.at(0) == '%') {
        datasetIntern = current_line.substr(1);
        break;
      }
    }
    if(datasetIntern == "") {
      std::cerr << "no vector found" << std::endl;
      std::abort();
    }
  }
  //reset position
  vector_file.seekg (0, std::ios::beg);
  vector_file.clear();

  bool datasetFound = false;

  while(!vector_file.eof() && !datasetFound) {
    getline(vector_file, current_line);

    if(current_line.size() == 0) {
      continue;
    }
    //search desired dataset
    if(current_line.at(0) == '%') {
      if(datasetIntern == current_line.substr(1)) {
        getline(vector_file, current_line);
        if(vector_file.eof()) {
          std::cerr << "error: vector with specified dataset was not found" << std::endl;
          std::abort();
        } else {
          datasetFound = true;
        }
      } else {
        continue;
      }
    }
    if(datasetFound) {
      size_t current_position = 0;
      while(true) {
        current_value = std::atof(&current_line[current_position]);
        vec.push_back(current_value);
        current_position = current_line.find(';', current_position);
        if(current_position == std::string::npos) {
          vec.pop_back();
          break;
        }
        current_position++;
      }
    }

  }

  if(!datasetFound) {
    std::cerr << "error: vector with specified dataset was not found" << std::endl;
    std::abort();
  }
  vector_file.close();
}

#ifdef WITH_HDF5
template <class VEC_TYPE>
void IOBase::loadVectorHDF5(const std::string& filename, const std::string& dataset, VEC_TYPE& vec) {
  //std::cout << "loading vector from HDF5 file..." << std::endl;
  hid_t handle = marray::hdf5::openFile(filename);
  if(dataset.empty()) {
    std::cout << "warning: dataset not specified, choose one of the following datasets and try again." << std::endl;
    std::cout << "All possible Datasets are: " << std::endl;
    H5Giterate(handle, "/", NULL, file_info, NULL);
    abort();
  }
  //FIXME andres::hdf5::load() doesn't support std::vector
  marray::Vector<typename VEC_TYPE::value_type> w;
  marray::hdf5::load(handle, dataset, w);
  for(size_t i = 0; i < w.size(); i++) {
    vec.push_back(w(i));
  }
  marray::hdf5::closeFile(handle);
}
#endif

template <class VEC_TYPE>
void IOBase::storeVectorText(const std::string& filename, const std::string& dataset, VEC_TYPE& vec) {
  //std::cout << "storing vector in text file..." << std::endl;

  //check if dataset is empty
  if(dataset.empty()) {
    std::cout << "warning: no dataset specified to store vector, using default dataset \"vector\"" << std::endl;
    std::string datasetNew = "vector";
    std::cout << "new dataset name: \"" << datasetNew << "\"" << std::endl;
    storeVectorText(filename, datasetNew, vec);
    return;
  }

  //check if dataset already exists
  std::ifstream vector_fileIn(filename.c_str(), std::ifstream::in);
  if(vector_fileIn.is_open()) {
    std::string current_line;
    while(!vector_fileIn.eof()) {
      getline(vector_fileIn, current_line);

      if(current_line.empty()) {
        continue;
      }

      if(current_line.at(0) == '%') {
        if(dataset == current_line.substr(1)) {
          vector_fileIn.close();
          std::cout << "warning: dataset already exists, appending \"_new\" to desired dataset name" << std::endl;
          std::string datasetNew = dataset + "_new";
          std::cout << "new dataset name: \"" << datasetNew << "\"" << std::endl;
          storeVectorText(filename, datasetNew, vec);
          return;
        }
      }
    }
    vector_fileIn.close();
  }

  std::ofstream vector_file(filename.c_str(), std::ofstream::out|std::ios::app);
  if(!vector_file) {
    std::cerr << "error: could not open file: " << filename << std::endl;
    std::abort();
  }
  vector_file << "%" << dataset << std::endl;

  for(typename VEC_TYPE::const_iterator iter = vec.begin(); iter != vec.end(); iter++) {
    vector_file << *iter << "; ";
  }
  vector_file << std::endl;
  vector_file.close();
}

#ifdef WITH_HDF5
template <class VEC_TYPE>
void IOBase::storeVectorHDF5(const std::string& filename, const std::string& dataset, VEC_TYPE& vec) {
  //std::cout << "storing vector in HDF5 file..." << std::endl;
  std::string datasetIntern = dataset;
  if(datasetIntern.empty()) {
    std::cout << "warning: no dataset specified to store vector, using default dataset \"vector\"" << std::endl;
    datasetIntern = "vector";
  } 
  if(vec.size()==0) {
    std::cout << "warning: vector has size 0 and will be skipped." << std::endl;
    return;
  }
  hid_t handle;
  //check if file already exists and create file if it doesn't exist
  if(fileExists(filename)) {
     handle = marray::hdf5::openFile(filename, marray::hdf5::READ_WRITE);
  } else {
     handle = marray::hdf5::createFile(filename);
  }

  //check if dataset with same name already exists and if so, rename dataset
  try {
    marray::hdf5::save(handle, datasetIntern, vec);
  } catch(...) {
    marray::hdf5::closeFile(handle); 
    std::cout << "warning: was not able to store dataset" << datasetIntern <<"! Maybe it exist or data are not valid." << std::endl;
/*
    std::cout << "warning: dataset already exists, appending \"_new\" to desired dataset name" << std::endl;
    std::string datasetNew = datasetIntern + "_new";
    std::cout << "new dataset name: \"" << datasetNew << "\"" << std::endl;
    storeVectorHDF5(filename, datasetNew, vec);
*/
    return;
  }

  marray::hdf5::closeFile(handle);
}
#endif

template <class MARRAY>
void IOBase::loadMArrayText(const std::string& filename, const std::string& dataset, MARRAY& array) {
   throw(RuntimeError("Load MArray only supports HDF5 file format"));
}

#ifdef WITH_HDF5
template <class MARRAY>
void IOBase::loadMArrayHDF5(const std::string& filename, const std::string& dataset, MARRAY& array) {
  //std::cout << "loading vector from HDF5 file..." << std::endl;
  hid_t handle = marray::hdf5::openFile(filename);
  if(dataset.empty()) {
    std::cout << "warning: dataset not specified, choose one of the following datasets and try again." << std::endl;
    std::cout << "All possible Datasets are: " << std::endl;
    H5Giterate(handle, "/", NULL, file_info, NULL);
    abort();
  }
  marray::hdf5::load(handle, dataset, array);
  marray::hdf5::closeFile(handle);
}
#endif

template <class MARRAY>
void IOBase::storeMArrayText(const std::string& filename, const std::string& dataset, MARRAY& array) {
   throw(RuntimeError("Store MArray only supports HDF5 file format"));
}

#ifdef WITH_HDF5
template <class MARRAY>
void IOBase::storeMArrayHDF5(const std::string& filename, const std::string& dataset, MARRAY& array) {
  //std::cout << "storing vector in HDF5 file..." << std::endl;
  std::string datasetIntern = dataset;
  if(datasetIntern.empty()) {
    std::cout << "warning: no dataset specified to store vector, using default dataset \"marray\"" << std::endl;
    datasetIntern = "marray";
  }
  hid_t handle;
  //check if file already exists and create file if it doesn't exist
  if(fileExists(filename)) {
     handle = marray::hdf5::openFile(filename, marray::hdf5::READ_WRITE);
  } else {
     handle = marray::hdf5::createFile(filename);
  }

  //check if dataset with same name already exists and if so, rename dataset
  try {
    marray::hdf5::save(handle, datasetIntern, array);
  } catch(...) {
    marray::hdf5::closeFile(handle);
    std::cout << "warning: dataset already exists, appending \"_new\" to desired dataset name" << std::endl;
    std::string datasetNew = datasetIntern + "_new";
    std::cout << "new dataset name: \"" << datasetNew << "\"" << std::endl;
    storeMArrayHDF5(filename, datasetNew, array);
    return;
  }

  marray::hdf5::closeFile(handle);
}
#endif

template <class VEC_TYPE>
void IOBase::printVector(VEC_TYPE& vec) {
  typename VEC_TYPE::const_iterator iter;
  const size_t max_values_output = 10;
  size_t num_values_output = 0;
  std::cout << "printig vector... (only the first " << max_values_output << " elements are shown)" << std::endl;
  for(iter = vec.begin(); iter != vec.end(); iter++) {
    std::cout << *iter << "; ";
    num_values_output++;
    if(num_values_output == max_values_output) {
      break;
    }
  }
  std::cout << std::endl;
}

#ifdef WITH_HDF5
/*
 * Operator function for H5Giterate used in loadVectorHDF5().
 */
herr_t IOBase::file_info(hid_t loc_id, const char *name, void *opdata)
{
    H5G_stat_t statbuf;

    /*
     * Get type of the object and display its name and type.
     * The name of the object is passed to this function by
     * the Library. Some magic :-)
     */
    H5Gget_objinfo(loc_id, name, false, &statbuf);
    switch (statbuf.type) {
    case H5G_GROUP:
        std::cout << "Group: " << name << std::endl;
        break;
    case H5G_DATASET:
        std::cout << "Dataset: " << name << std::endl;
        break;
    case H5G_TYPE:
        std::cout << "Named Datatype: " << name << std::endl;
        break;
    default:
         printf(" Unable to identify an object ");
    }
    return 0;
 }

#endif

bool IOBase::fileExists(const std::string& filename) const {
   std::ifstream file(filename.c_str(), std::ifstream::in);
   if(file.is_open()) {
      file.close();
      return true;
   }
   return false;
}

} // namespace interface

} // namespace opengm

#endif /* IO_BASE_HXX_ */
