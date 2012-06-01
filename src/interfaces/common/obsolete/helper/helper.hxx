#ifndef HELPER_HXX_
#define HELPER_HXX_

#include <iostream>
#include <fstream>
#include <opengm/datastructures/marray/marray.hxx>

#ifdef WITH_HDF5
//#include <hdf5.h>
//#include <H5Exception.h>
#include <opengm/datastructures/marray/marray_hdf5.hxx>
#endif

namespace opengm {

namespace interface {
/************************
 * forward declarations *
 ************************/
void seperateFilename(const std::string& completeFilename, std::string& filename, std::string& dataset);
std::string getFileExtension(const std::string& filename);
template <class VEC_TYPE>
void loadVector(const std::string& filename, VEC_TYPE& vec);
template <class VEC_TYPE>
void loadVectorText(const std::string& filename, const std::string& dataset, VEC_TYPE& vec);
#ifdef WITH_HDF5
template <class VEC_TYPE>
void loadVectorHDF5(const std::string& filename, const std::string& dataset, VEC_TYPE& vec);
#endif
template <class VEC_TYPE>
void storeVector(const std::string& filename, std::vector<size_t>& vec);
template <class VEC_TYPE>
void storeVectorText(const std::string& filename, const std::string& dataset, VEC_TYPE& vec);
#ifdef WITH_HDF5
template <class VEC_TYPE>
void storeVectorHDF5(const std::string& filename, const std::string& dataset, VEC_TYPE& vec);
#endif
template <class VEC_TYPE>
void printVector(VEC_TYPE& vec);
#ifdef WITH_HDF5
herr_t file_info(hid_t loc_id, const char *name, void *opdata);
#endif
template <class GM>
void modelInfo(GM& graphicalModel, std::ostream& stream);

/******************
 * implementation *
 ******************/

void separateFilename(const std::string& completeFilename, std::string& filename, std::string& dataset) {
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

std::string getFileExtension(const std::string& filename) {
  size_t dotPosition = filename.rfind('.');
  if(dotPosition == std::string::npos) {
    std::cout << "warning: no file extension found. Assuming, the selected file is a text file" << std::endl;
    return "txt";
  } else {
    return filename.substr(dotPosition + 1);
  }
}

template <class VEC_TYPE>
void loadVector(const std::string& filename, VEC_TYPE& vec) {
  std::string separatedFilename;
  std::string dataset;
  std::string fileExtension;

  separateFilename(filename, separatedFilename, dataset);
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
void loadVectorText(const std::string& filename, const std::string& dataset, VEC_TYPE& vec) {
  std::cout << "loading vector from textfile..." << std::endl;
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
void loadVectorHDF5(const std::string& filename, const std::string& dataset, VEC_TYPE& vec) {
  std::cout << "loading vector from HDF5 file..." << std::endl;
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
void storeVector(const std::string& filename, VEC_TYPE& vec) {
  if(filename == "PRINT ON SCREEN") {
    printVector(vec);
  } else {
    std::string separatedFilename;
    std::string dataset;
    std::string fileExtension;

    separateFilename(filename, separatedFilename, dataset);
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

template <class VEC_TYPE>
void storeVectorText(const std::string& filename, const std::string& dataset, VEC_TYPE& vec) {
  std::cout << "storing vector in text file..." << std::endl;

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
void storeVectorHDF5(const std::string& filename, const std::string& dataset, VEC_TYPE& vec) {
  std::cout << "storing vector in HDF5 file..." << std::endl;
  std::string datasetIntern = dataset;
  if(datasetIntern.empty()) {
    std::cout << "warning: no dataset specified to store vector, using default dataset \"vector\"" << std::endl;
    datasetIntern = "vector";
  }
  hid_t handle;
  //check if file already exists and create file if it doesn't exist
  try {
    //H5::Exception::dontPrint();
    handle = marray::hdf5::openFile(filename, marray::hdf5::READ_WRITE);
  } catch(...) {
    handle = marray::hdf5::createFile(filename);
  }
  //check if dataset with same name already exists and if so, rename dataset
  try {
    marray::hdf5::save(handle, datasetIntern, vec);
  } catch(...) {
    marray::hdf5::closeFile(handle);
    std::cout << "warning: dataset already exists, appending \"_new\" to desired dataset name" << std::endl;
    std::string datasetNew = datasetIntern + "_new";
    std::cout << "new dataset name: \"" << datasetNew << "\"" << std::endl;
    storeVectorHDF5(filename, datasetNew, vec);
    return;
  }

  marray::hdf5::closeFile(handle);
}
#endif

template <class VEC_TYPE>
void printVector(VEC_TYPE& vec) {
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
herr_t file_info(hid_t loc_id, const char *name, void *opdata)
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

//TODO adjust modelinfo() to new structure of graphical model
template <class GM>
void modelInfo(GM& graphicalModel, std::ostream& stream) {
   stream << "Model Info" <<std::endl;
   stream << "----------" <<std::endl;
   stream << std::endl;
   stream << "Number of Variables   : " << graphicalModel.numberOfVariables() << std::endl;
   size_t min = 1000000000;
   size_t max = 0;
   for (size_t i=0; i<graphicalModel.numberOfVariables();++i){
      if(min>graphicalModel.numberOfStates(i)) min = graphicalModel.numberOfStates(i);
      if(max<graphicalModel.numberOfStates(i)) max = graphicalModel.numberOfStates(i);
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
   stream << "Submodular Model      : ???" << std::endl;
   marray::Vector<size_t> chain;
   stream << "Chain Model           : " << graphicalModel.isChain(chain) << std::endl;
   marray::Matrix<size_t> grid;
   stream << "Grid Model            : " << graphicalModel.isGrid(grid) << std::endl;
   stream << "Planar Model          : ???" << std::endl;
   stream << "Treewidth             : ???" << std::endl;
   stream << "Diameter              : ???" << std::endl;
   stream << "Node-Degree           : ???" << std::endl;
   stream << std::endl;
}

} //namespace interface

} // namespace opengm
#endif /* HELPER_HXX_ */
