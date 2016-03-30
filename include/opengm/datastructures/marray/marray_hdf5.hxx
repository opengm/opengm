#pragma once
#ifndef MARRAY_HDF5_HXX
#define MARRAY_HDF5_HXX

// compat fix for buggy hdf5 1.8 versions
#include <H5version.h>
#if (H5_VERS_MAJOR == 1 && H5_VERS_MINOR >= 8 && defined(H5_USE_16_API_DEFAULT))
#define H5Gcreate_vers 2
#define H5Gopen_vers 2
#define H5Dopen_vers 2
#define H5Dcreate_vers 2
#define H5Acreate_vers 2
#endif

#include <cstring>

#include "marray.hxx"
#include "hdf5.h"

namespace marray {
/// HDF5 import/export support.
namespace hdf5 {

// assertion testing

// \cond suppress doxygen
template<bool B> class HandleCheck;
template<> class HandleCheck<false> {
public:
    HandleCheck()
        { counter_ = H5Fget_obj_count(H5F_OBJ_ALL, H5F_OBJ_ALL); }
    void check()
        { marray_detail::Assert( counter_ == H5Fget_obj_count(H5F_OBJ_ALL, H5F_OBJ_ALL)); }
private:
    ssize_t counter_;
};
template<> class HandleCheck<true> {
public:
    void check() {}
};
// \endcond

// namespace variables

const char reverseShapeAttributeName[14] = "reverse-shape";

// prototypes

enum FileAccessMode {READ_ONLY, READ_WRITE};
enum HDF5Version {DEFAULT_HDF5_VERSION, LATEST_HDF5_VERSION};

inline hid_t createFile(const std::string&, HDF5Version = DEFAULT_HDF5_VERSION);
inline hid_t openFile(const std::string&, FileAccessMode = READ_ONLY, HDF5Version = DEFAULT_HDF5_VERSION);
inline void closeFile(const hid_t&);

inline hid_t createGroup(const hid_t&, const std::string&);
inline hid_t openGroup(const hid_t&, const std::string&);
inline void closeGroup(const hid_t&);

template<class T>
    void save(const hid_t&, const std::string&, const Marray<T>&);
template<class T, bool isConst>
    void save(const hid_t&, const std::string&, const View<T, isConst>&);
template<class T>
    void save(const hid_t&, const std::string&, const std::vector<T>&);
template<class T, class BaseIterator, class ShapeIterator>
    void saveHyperslab(const hid_t&, const std::string&,
        BaseIterator, BaseIterator, ShapeIterator, const Marray<T>&);
template<class T, class ShapeIterator>
    void create(const hid_t&, const std::string&, ShapeIterator,
        ShapeIterator, CoordinateOrder);

template<class T>
    void load(const hid_t&, const std::string&, Marray<T>&);
template<class T>
    void loadShape(const hid_t&, const std::string&, Vector<T>&);
template<class T, class BaseIterator, class ShapeIterator>
    void loadHyperslab(const hid_t&, const std::string&,
        BaseIterator, BaseIterator, ShapeIterator, Marray<T>&);

// type conversion from C++ to HDF5

// \cond suppress doxygen
template<class T>
inline hid_t uintTypeHelper() {
   switch(sizeof(T)) {
       case 1:
           return H5T_STD_U8LE;
       case 2:
           return H5T_STD_U16LE;
       case 4:
           return H5T_STD_U32LE;
       case 8:
           return H5T_STD_U64LE;
       default:
           throw std::runtime_error("No matching HDF5 type.");
   }
}

template<class T>
inline hid_t intTypeHelper() {
   switch(sizeof(T)) {
       case 1:
           return H5T_STD_I8LE;
       case 2:
           return H5T_STD_I16LE;
       case 4:
           return H5T_STD_I32LE;
       case 8:
           return H5T_STD_I64LE;
       default:
           throw std::runtime_error("No matching HDF5 type.");
   }
}

template<class T>
inline hid_t floatingTypeHelper() {
   switch(sizeof(T)) {
       case 4:
           return H5T_IEEE_F32LE;
       case 8:
           return H5T_IEEE_F64LE;
       default:
           throw std::runtime_error("No matching HDF5 type.");
   }
}

template<class T>
inline hid_t hdf5Type();

template<> inline hid_t hdf5Type<unsigned char>()
    { return uintTypeHelper<unsigned char>(); }
template<> inline hid_t hdf5Type<unsigned short>()
    { return uintTypeHelper<unsigned short>(); }
template<> inline hid_t hdf5Type<unsigned int>()
    { return uintTypeHelper<unsigned int>(); }
template<> inline hid_t hdf5Type<unsigned long>()
    { return uintTypeHelper<unsigned long>(); }
template<> inline hid_t hdf5Type<unsigned long long>()
    { return uintTypeHelper<unsigned long long>(); }

template<> inline hid_t hdf5Type<char>()
    { return uintTypeHelper<char>(); }
template<> inline hid_t hdf5Type<signed char>()
    { return intTypeHelper<signed char>(); }
template<> inline hid_t hdf5Type<short>()
    { return intTypeHelper<short>(); }
template<> inline hid_t hdf5Type<int>()
    { return intTypeHelper<int>(); }
template<> inline hid_t hdf5Type<long>()
    { return intTypeHelper<long>(); }
template<> inline hid_t hdf5Type<long long>()
    { return intTypeHelper<long long>(); }

template<> inline hid_t hdf5Type<float>()
    { return floatingTypeHelper<float>(); }
template<> inline hid_t hdf5Type<double>()
    { return floatingTypeHelper<double>(); }
// \endcond

// implementation

/// Create and close an HDF5 dataset to store Marray data.
///
/// \param groupHandle Handle of the parent HDF5 file or group.
/// \param datasetName Name of the HDF5 dataset.
/// \param begin Iterator to the beginning of a sequence that determines the shape of the dataset.
/// \param end Iterator to the end of a sequence that determines the shape of the dataset.
/// \param coordinateOrder Coordinate order of the Marray.
///
/// \sa save(), saveHyperslab()
///
template<class T, class ShapeIterator>
void create(
    const hid_t& groupHandle,
    const std::string& datasetName,
    ShapeIterator begin,
    ShapeIterator end,
    CoordinateOrder coordinateOrder
) {
    marray_detail::Assert(MARRAY_NO_ARG_TEST || groupHandle >= 0);
    HandleCheck<MARRAY_NO_DEBUG> handleCheck;

    // build dataspace
    hid_t datatype = H5Tcopy(hdf5Type<T>());
    size_t dimension = std::distance(begin, end);
    Vector<hsize_t> shape((size_t)(dimension));
    if(coordinateOrder == FirstMajorOrder) {
        // copy shape as is
        for(size_t j=0; j<dimension; ++j) {
            shape[j] = hsize_t(*begin);
            ++begin;
        }
    }
    else {
        // reverse shape
        for(size_t j=0; j<dimension; ++j) {
            shape[dimension-j-1] = hsize_t(*begin);
            ++begin;
        }
    }
    hid_t dataspace = H5Screate_simple(dimension, &shape[0], NULL);
    if(dataspace < 0) {
        H5Tclose(datatype);
        throw std::runtime_error("Marray cannot create dataspace.");
    }


    // create new dataset
    hid_t dataset = H5Dcreate(groupHandle, datasetName.c_str(), datatype,
        dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if(dataset < 0) {
        H5Sclose(dataspace);
        H5Tclose(datatype);
        throw std::runtime_error("Marray cannot create dataset.");
    }

    // write attribute to indicate whether shape is reversed
    if(coordinateOrder == LastMajorOrder) {
        hsize_t attributeShape[1] = {1};
        hid_t attributeDataspace = H5Screate_simple(1, attributeShape, NULL);
        if(attributeDataspace < 0) {
            H5Dclose(dataset);
            H5Sclose(dataspace);
            H5Tclose(datatype);
            throw std::runtime_error("Marray cannot create dataspace.");
        }
        hid_t attribute = H5Acreate(dataset, reverseShapeAttributeName,
            H5T_STD_U8LE, attributeDataspace, H5P_DEFAULT, H5P_DEFAULT);
        if(attribute < 0) {
            H5Sclose(attributeDataspace);
            H5Dclose(dataset);
            H5Sclose(dataspace);
            H5Tclose(datatype);
            throw std::runtime_error("Marray cannot create attribute.");
        }
        unsigned int data = 1;
        herr_t err = H5Awrite(attribute, H5T_STD_U8LE, &data);
        H5Aclose(attribute);
        H5Sclose(attributeDataspace);
        if(err < 0) {
            H5Dclose(dataset);
            H5Sclose(dataspace);
            H5Tclose(datatype);
            throw std::runtime_error("Marray cannot create write to attribute.");
        }
    }

    // clean up
    H5Dclose(dataset);
    H5Sclose(dataspace);
    H5Tclose(datatype);
    handleCheck.check();
}

/// Save an Marray as an HDF5 dataset.
///
/// \param groupHandle Handle of the parent HDF5 file or group.
/// \param datasetName Name of the HDF5 dataset.
/// \param in Marray.
///
/// \sa saveHyperslab()
///
template<class T>
void save(
    const hid_t& groupHandle,
    const std::string& datasetName,
    const Marray<T>& in
) {
    marray_detail::Assert(MARRAY_NO_ARG_TEST || groupHandle >= 0);
    HandleCheck<MARRAY_NO_DEBUG> handleCheck;

    // build dataspace
    hid_t datatype = H5Tcopy(hdf5Type<T>());
    Vector<hsize_t> shape(in.dimension());
    if(in.coordinateOrder() == FirstMajorOrder) {
        // copy shape as is
        for(size_t j=0; j<in.dimension(); ++j) {
            shape[j] = hsize_t(in.shape(j));
        }
    }
    else {
        // reverse shape
        for(size_t j=0; j<in.dimension(); ++j) {
            shape[size_t(in.dimension()-j-1)] = hsize_t(in.shape(j));
        }
    }
    hid_t dataspace = H5Screate_simple(in.dimension(), &shape[0], NULL);
    if(dataspace < 0) {
        H5Tclose(datatype);
        throw std::runtime_error("Marray cannot create dataspace.");
    }

    // create new dataset
    hid_t dataset = H5Dcreate(groupHandle, datasetName.c_str(), datatype,
        dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if(dataset < 0) {
        H5Sclose(dataspace);
        H5Tclose(datatype);
        throw std::runtime_error("Marray cannot create dataset.");
    }

    // write attribute to indicate whether shape is reversed
    if(in.coordinateOrder() == LastMajorOrder) {
        hsize_t attributeShape[1] = {1};
        hid_t attributeDataspace = H5Screate_simple(1, attributeShape, NULL);
        if(attributeDataspace < 0) {
            H5Dclose(dataset);
            H5Sclose(dataspace);
            H5Tclose(datatype);
            throw std::runtime_error("Marray cannot create dataspace.");
        }
        hid_t attribute = H5Acreate(dataset, reverseShapeAttributeName,
            H5T_STD_U8LE, attributeDataspace, H5P_DEFAULT, H5P_DEFAULT);
        if(attribute < 0) {
            H5Sclose(attributeDataspace);
            H5Dclose(dataset);
            H5Sclose(dataspace);
            H5Tclose(datatype);
            throw std::runtime_error("Marray cannot create attribute.");
        }
        unsigned int data = 1;
        herr_t err = H5Awrite(attribute, H5T_STD_U8LE, &data);
        H5Aclose(attribute);
        H5Sclose(attributeDataspace);
        if(err < 0) {
            H5Dclose(dataset);
            H5Sclose(dataspace);
            H5Tclose(datatype);
            throw std::runtime_error("Marray cannot create write to attribute.");
        }
    }

    // write
    herr_t status = H5Dwrite(dataset, datatype, H5S_ALL, H5S_ALL,
        H5P_DEFAULT, &in(0));
    H5Dclose(dataset);
    H5Sclose(dataspace);
    H5Tclose(datatype);
    if(status < 0) {
        throw std::runtime_error("Marray cannot write to dataset.");
    }

    handleCheck.check();
}

/// Save a View as an HDF5 dataset.
/// 
/// \param groupHandle Handle of the parent HDF5 file or group.
/// \param datasetName Name of the HDF5 dataset.
/// \param in View.
///
/// \sa saveHyperslab()
///
template<class T, bool isConst>
inline void save(
    const hid_t& groupHandle,
    const std::string& datasetName,
    const View<T, isConst>& in
) {
    Marray<T> m = in;
    save(groupHandle, datasetName, m);
}

/// Save an std::vector as an HDF5 dataset.
/// 
/// \param groupHandle Handle of the parent HDF5 file or group.
/// \param datasetName Name of the HDF5 dataset.
/// \param in std::vector.
///
/// \sa saveHyperslab()
///
template<class T>
void save(
    const hid_t& groupHandle,
    const std::string& datasetName,
    const std::vector<T>& in
)
{
    marray::Vector<T> v(in.size());
    for(size_t j=0; j<in.size(); ++j) {
        v[j] = in[j];
    }
    save(groupHandle, datasetName, v);
}

/// Load an Marray from an HDF5 dataset.
///
/// \param groupHandle Handle of the parent HDF5 file or group.
/// \param datasetName Name of the HDF5 dataset.
/// \param out Marray.
///
/// \sa loadHyperslab()
///
template<class T>
void load(
    const hid_t& groupHandle,
    const std::string& datasetName,
    Marray<T>& out
) {
    marray_detail::Assert(MARRAY_NO_ARG_TEST || groupHandle >= 0);
    HandleCheck<MARRAY_NO_DEBUG> handleCheck;

    hid_t dataset = H5Dopen(groupHandle, datasetName.c_str(), H5P_DEFAULT);
    if(dataset < 0) {
        throw std::runtime_error("Marray cannot open dataset.");
    }
    hid_t filespace = H5Dget_space(dataset);
    hid_t type = H5Dget_type(dataset);
    hid_t nativeType = H5Tget_native_type(type, H5T_DIR_DESCEND);
    if(!H5Tequal(nativeType, hdf5Type<T>())) {
        H5Dclose(dataset);
        H5Tclose(nativeType);
        H5Tclose(type);
        H5Sclose(filespace);
        throw std::runtime_error("Data types not equal error.");
    }
    int dimension = H5Sget_simple_extent_ndims(filespace);
    Vector<hsize_t> shape(dimension);
    herr_t status = H5Sget_simple_extent_dims(filespace, &shape[0], NULL);
    if(status < 0) {
        H5Dclose(dataset);
        H5Tclose(nativeType);
        H5Tclose(type);
        H5Sclose(filespace);
        throw std::runtime_error("H5Sget_simple_extent_dims error.");
    }
    hid_t memspace = H5Screate_simple(dimension, &shape[0], NULL);

    // resize marray
    marray::Vector<size_t> newShape((size_t)(dimension));
    for(size_t j=0; j<newShape.size(); ++j) {
        newShape(j) = (size_t)(shape[j]);
    }
    if(H5Aexists(dataset, reverseShapeAttributeName) > 0) {
        // reverse shape
        out = Marray<T>(SkipInitialization, newShape.rbegin(), 
            newShape.rend(), LastMajorOrder);
    }
    else {
        // don't reverse shape
        out = Marray<T>(SkipInitialization, newShape.begin(),
            newShape.end(), FirstMajorOrder);
    }

    // read
    status = H5Dread(dataset, nativeType, memspace, filespace,
        H5P_DEFAULT, &out(0));
    H5Dclose(dataset);
    H5Tclose(nativeType);
    H5Tclose(type);
    H5Sclose(memspace);
    H5Sclose(filespace);
    if(status < 0) {
        throw std::runtime_error("Marray cannot read from dataset.");
    }

    handleCheck.check();
}

/// Load the shape of an HDF5 dataset.
///
/// \param groupHandle Handle of the parent HDF5 file or group.
/// \param datasetName Name of the HDF5 dataset.
/// \param out Shape.
///
/// \sa load()
///
template<class T>
void loadShape(
   const hid_t& groupHandle,
   const std::string& datasetName,
   Vector<T>& out
) {
    marray_detail::Assert(MARRAY_NO_ARG_TEST || groupHandle >= 0);
    HandleCheck<MARRAY_NO_DEBUG> handleCheck;

    // load shape from HDF5 file
    hid_t dataset = H5Dopen(groupHandle, datasetName.c_str(), H5P_DEFAULT);
    if(dataset < 0) {
        throw std::runtime_error("Marray cannot open dataset.");
    }
    hid_t filespace = H5Dget_space(dataset);
    hsize_t dimension = H5Sget_simple_extent_ndims(filespace);
    hsize_t* shape = new hsize_t[(size_t)(dimension)];
    herr_t status = H5Sget_simple_extent_dims(filespace, shape, NULL);
    if(status < 0) {
        H5Dclose(dataset);
        H5Sclose(filespace);
        delete[] shape;
        throw std::runtime_error("Marray cannot get extension of dataset.");
    }

    // write shape to out
    out = Vector<T>((size_t)(dimension));
    if(H5Aexists(dataset, reverseShapeAttributeName) > 0) {
        for(size_t j=0; j<out.size(); ++j) {
           out[out.size()-j-1] = T(shape[j]);
        }
    }
    else {
        for(size_t j=0; j<out.size(); ++j) {
            out[j] = T(shape[j]);
        }
    }

    // clean up
    delete[] shape;
    H5Dclose(dataset);
    H5Sclose(filespace);
    handleCheck.check();
}

/// Load a hyperslab from an HDF5 dataset into an Marray.
/// 
/// \param groupHandle Handle of the parent HDF5 file or group.
/// \param datasetName Name of the HDF5 dataset.
/// \param baseBegin Iterator to the beginning of the sequence that determines the first coordinate of the hyperslab.
/// \param baseEnd Iterator to the end of the sequence that determines the first coordinate of the hyperslab.
/// \param shapeBegin Iterator to the beginning of the sequence that determines the shape of the hyperslab.
/// \param out Marray.
///
/// \sa saveHyperslab(), create()
///
template<class T, class BaseIterator, class ShapeIterator>
void loadHyperslab(
    const hid_t& groupHandle,
    const std::string& datasetName,
    BaseIterator baseBegin,
    BaseIterator baseEnd,
    ShapeIterator shapeBegin,
    Marray<T>& out
) {
    marray_detail::Assert(MARRAY_NO_ARG_TEST || groupHandle >= 0);
    HandleCheck<MARRAY_NO_DEBUG> handleCheck;

    // open dataset
    hid_t dataset = H5Dopen(groupHandle, datasetName.c_str(), H5P_DEFAULT);
    if(dataset < 0) {
        throw std::runtime_error("Marray cannot open dataset.");
    }

    // determine shape of hyperslab and array
    size_t size = std::distance(baseBegin, baseEnd);
    Vector<hsize_t> offset(size);
    Vector<hsize_t> slabShape(size);
    Vector<hsize_t> marrayShape(size);
    CoordinateOrder coordinateOrder;
    if(H5Aexists(dataset, reverseShapeAttributeName) > 0) {
        // reverse base and shape
        coordinateOrder = LastMajorOrder;
        size_t j = size-1;
        size_t k = 0;
        for(;;) {
            offset[j] = hsize_t(*baseBegin);
            slabShape[j] = hsize_t(*shapeBegin);
            marrayShape[k] = slabShape[j];
            if(j == 0) {
                break;
            }
            else {
                ++baseBegin;
                ++shapeBegin;
                ++k;
                --j;
            }
        }
    } 
    else {
        // don't reverse base and shape
        coordinateOrder = FirstMajorOrder;
        for(size_t j=0; j<size; ++j) {
            offset[j] = hsize_t(*baseBegin);
            slabShape[j] = hsize_t(*shapeBegin);
            marrayShape[j] = slabShape[j];
            ++baseBegin;
            ++shapeBegin;
        }
    }
    
    // select dataspace hyperslab
    hid_t datatype = H5Dget_type(dataset);
    
    if(!H5Tequal(datatype, hdf5Type<T>())) {
        throw std::runtime_error("data type of stored hdf5 dataset and passed array do not match in loadHyperslab");
    }
    
    hid_t dataspace = H5Dget_space(dataset);
    herr_t status = H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, 
        &offset[0], NULL, &slabShape[0], NULL);
    if(status < 0) {
        H5Tclose(datatype);
        H5Sclose(dataspace);
        H5Dclose(dataset);
        throw std::runtime_error("Marray cannot select hyperslab. Check offset and shape!");
    }

    // select memspace hyperslab
    hid_t memspace = H5Screate_simple(int(size), &marrayShape[0], NULL);
    Vector<hsize_t> offsetOut(size, 0); // no offset
    status = H5Sselect_hyperslab(memspace, H5S_SELECT_SET, &offsetOut[0],
        NULL, &marrayShape[0], NULL);
    if(status < 0) {
        H5Sclose(memspace); 
        H5Tclose(datatype);
        H5Sclose(dataspace);
        H5Dclose(dataset);
        throw std::runtime_error("Marray cannot select hyperslab. Check offset and shape!");
    }

    // read from dataspace into memspace
    out = Marray<T>(SkipInitialization, &marrayShape[0], 
        (&marrayShape[0])+size, coordinateOrder);
    status = H5Dread(dataset, datatype, memspace, dataspace,
        H5P_DEFAULT, &(out(0)));

    // clean up
    H5Sclose(memspace); 
    H5Tclose(datatype);
    H5Sclose(dataspace);
    H5Dclose(dataset);
    if(status < 0) {
        throw std::runtime_error("Marray cannot read from dataset.");
    }
    handleCheck.check();
}

/// Save an Marray as a hyperslab into an HDF5 dataset.
/// 
/// \param groupHandle Handle of the parent HDF5 file or group.
/// \param datasetName Name of the HDF5 dataset.
/// \param baseBegin Iterator to the beginning of the sequence that determines the first coordinate of the hyperslab.
/// \param baseEnd Iterator to the end of the sequence that determines the first coordinate of the hyperslab.
/// \param shapeBegin Iterator to the beginning of the sequence that determines the shape of the hyperslab.
/// \param in Marray.
///
/// \sa loadHyperslab(), create()
///
template<class T, class BaseIterator, class ShapeIterator>
void 
saveHyperslab(
    const hid_t& groupHandle,
    const std::string& datasetName,
    BaseIterator baseBegin,
    BaseIterator baseEnd,
    ShapeIterator shapeBegin,
    const Marray<T>& in
) {
    marray_detail::Assert(MARRAY_NO_ARG_TEST || groupHandle >= 0);
    HandleCheck<MARRAY_NO_DEBUG> handleCheck;

    // open dataset
    hid_t dataset = H5Dopen(groupHandle, datasetName.c_str(), H5P_DEFAULT);
    if(dataset < 0) {
        throw std::runtime_error("Marray cannot open dataset.");
    }

    // determine hyperslab shape
    Vector<hsize_t> memoryShape(in.dimension());
    for(size_t j=0; j<in.dimension(); ++j) {
        memoryShape[j] = in.shape(j);
    }
    size_t size = std::distance(baseBegin, baseEnd);
    Vector<hsize_t> offset(size);
    Vector<hsize_t> slabShape(size);
    bool reverseShapeAttribute = 
        (H5Aexists(dataset, reverseShapeAttributeName) > 0);
    if(reverseShapeAttribute && in.coordinateOrder() == LastMajorOrder) {
        // reverse base and shape
        size_t j = size-1;
        for(;;) {
            offset[j] = hsize_t(*baseBegin);
            slabShape[j] = hsize_t(*shapeBegin);
            if(j == 0) {
                break;
            }
            else {
                ++baseBegin;
                ++shapeBegin;
                --j;
            }
        }
    }
    else if(!reverseShapeAttribute && in.coordinateOrder() == FirstMajorOrder) {
        for(size_t j=0; j<size; ++j) {
            offset[j] = hsize_t(*baseBegin);
            slabShape[j] = hsize_t(*shapeBegin);
            ++baseBegin;
            ++shapeBegin;
        }
    }
    else {
        H5Dclose(dataset);
        throw std::runtime_error("Marray cannot write to HDF5 file. A different order was used when the file was created.");
    }

    // select dataspace hyperslab
    hid_t datatype = H5Dget_type(dataset);
    hid_t dataspace = H5Dget_space(dataset);
    herr_t status = H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, 
        &offset[0], NULL, &slabShape[0], NULL);
    if(status < 0) {
        H5Tclose(datatype);
        H5Sclose(dataspace);
        H5Dclose(dataset);
        throw std::runtime_error("Marray cannot select hyperslab. Check offset and shape!");
    }

    // select memspace hyperslab
    hid_t memspace = H5Screate_simple(int(in.dimension()), &memoryShape[0], NULL);
    Vector<hsize_t> memoryOffset(int(in.dimension()), 0); // no offset
    status = H5Sselect_hyperslab(memspace, H5S_SELECT_SET, &memoryOffset[0], NULL,
        &memoryShape[0], NULL);
    if(status < 0) {
        H5Sclose(memspace); 
        H5Tclose(datatype);
        H5Sclose(dataspace);
        H5Dclose(dataset);
        throw std::runtime_error("Marray cannot select hyperslab. Check offset and shape!");
    }

    // write from memspace to dataspace
    status = H5Dwrite(dataset, datatype, memspace, dataspace, H5P_DEFAULT, &(in(0)));

    // clean up
    H5Sclose(memspace); 
    H5Tclose(datatype);
    H5Sclose(dataspace);
    H5Dclose(dataset);
    if(status < 0) {
        throw std::runtime_error("Marray cannot write to dataset.");
    }
    handleCheck.check();
}

/// Create an HDF5 file.
///
/// \param filename Name of the file.
/// \param hdf5version HDF5 version tag.
///
/// \returns HDF5 handle
///
/// \sa openFile(), closeFile()
///
inline hid_t
createFile
(
    const std::string& filename,
    HDF5Version hdf5version
)
{
    hid_t version = H5P_DEFAULT;
    if(hdf5version == LATEST_HDF5_VERSION) {
        version = H5Pcreate(H5P_FILE_ACCESS);
        H5Pset_libver_bounds(version, H5F_LIBVER_LATEST, H5F_LIBVER_LATEST);
    }

    hid_t fileHandle = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, version);
    if(fileHandle < 0) {
        throw std::runtime_error("Could not create HDF5 file: " + filename);
    }

    return fileHandle;
}

/// Open an HDF5 file.
///
/// \param filename Name of the file.
/// \param fileAccessMode File access mode.
/// \param hdf5version HDF5 version tag.
///
/// \returns HDF5 handle
///
/// \sa closeFile(), createFile()
///
inline hid_t
openFile
(
    const std::string& filename,
    FileAccessMode fileAccessMode,
    HDF5Version hdf5version
)
{
    hid_t access = H5F_ACC_RDONLY;
    if(fileAccessMode == READ_WRITE) {
        access = H5F_ACC_RDWR;
    }

    hid_t version = H5P_DEFAULT;
    if(hdf5version == LATEST_HDF5_VERSION) {
        version = H5Pcreate(H5P_FILE_ACCESS);
        H5Pset_libver_bounds(version, H5F_LIBVER_LATEST, H5F_LIBVER_LATEST);
    }

    hid_t fileHandle = H5Fopen(filename.c_str(), access, version);
    if(fileHandle < 0) {
        throw std::runtime_error("Could not open HDF5 file: " + filename);
    }

    return fileHandle;
}

/// Close an HDF5 file
/// 
/// \param handle Handle to the HDF5 file.
///
/// \sa openFile(), createFile()
///
inline void closeFile
(
    const hid_t& handle
)
{
    H5Fclose(handle);
}

/// Create an HDF5 group.
///
/// \param parentHandle HDF5 handle on the parent group or file.
/// \param groupName Name of the group.
/// \returns HDF5 handle on the created group
///
/// \sa openGroup(), closeGroup()
///
inline hid_t 
createGroup
(
    const hid_t& parentHandle,
    const std::string& groupName
)
{ 
    hid_t groupHandle = H5Gcreate(parentHandle, groupName.c_str(), 
        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if(groupHandle < 0) {
        throw std::runtime_error("Could not create HDF5 group.");
    }
    return groupHandle;
}

/// Open an HDF5 group.
///
/// \param parentHandle HDF5 handle on the parent group or file.
/// \param groupName Name of the group.
/// \returns HDF5 handle on the opened group.
///
/// \sa createGroup(), closeGroup()
///
inline hid_t 
openGroup
(
    const hid_t& parentHandle,
    const std::string& groupName
)
{ 
    hid_t groupHandle = H5Gopen(parentHandle, groupName.c_str(), H5P_DEFAULT);
    if(groupHandle < 0) {
        throw std::runtime_error("Could not open HDF5 group.");
    }
    return groupHandle;
}

/// Close an HDF5 group.
///
/// \param handle HDF5 handle on group to close.
///
/// \sa openGroup(), createGroup()
///
inline void 
closeGroup
(
    const hid_t& handle
)
{
    H5Gclose(handle);
}

} // namespace hdf5
} // namespace marray

#endif // #ifndef MARRAY_HDF5_HXX

