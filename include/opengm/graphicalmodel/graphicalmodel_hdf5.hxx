#pragma once
#ifndef OPENGM_GRAPHICALMODEL_HDF5_HXX
#define OPENGM_GRAPHICALMODEL_HDF5_HXX

#include <string>
#include <iostream>
#include <sstream>
#include <typeinfo>

#include "opengm/opengm.hxx"
#include "opengm/utilities/metaprogramming.hxx"
#include "opengm/datastructures/marray/marray.hxx"
#include "opengm/datastructures/marray/marray_hdf5.hxx"

#include "opengm/graphicalmodel/graphicalmodel_factor.hxx"
#include "opengm/graphicalmodel/graphicalmodel.hxx"
#include "opengm/functions/sparsemarray.hxx"
#include "opengm/functions/potts.hxx"
#include "opengm/functions/pottsn.hxx"
#include "opengm/functions/absolute_difference.hxx"
#include "opengm/functions/squared_difference.hxx"
#include "opengm/functions/truncated_absolute_difference.hxx"
#include "opengm/functions/truncated_squared_difference.hxx"

namespace opengm {

/// Fiel I/O of graphical models using the HDF5 binary data format   
namespace hdf5 {

/// \cond HIDDEN_SYMBOLS
template<class T>
struct IsValidTypeForHdf5Save {
   typedef  opengm::meta::Bool<
      opengm::meta::Not<
         opengm::meta::IsInvalidType<T>::value
      >::value
   > tmpBoolAType;
   typedef opengm::meta::Bool<
      std::numeric_limits<T>::is_specialized
   > tmpBoolBType;
   enum Values {
      value=meta::And<tmpBoolAType::value,tmpBoolBType::value>::value
   };
};

struct StoredValueTypeInfo{
   enum Values{
      AsFloat=0,
      AsDouble=1,
      AsUInt=2,
      AsInt=3
   };
};

template<class GM, size_t IX, size_t DX, bool END>
struct GetFunctionRegistration;

template<class GM, size_t IX, size_t DX>
struct GetFunctionRegistration<GM, IX, DX, false> {
   static size_t get(const size_t functionIndex) {
      if(IX == functionIndex) {
         typedef typename meta::TypeAtTypeList<typename GM::FunctionTypeList, IX>::type TypeAtIX ;
         return FunctionRegistration< TypeAtIX >::Id;
      }
      else {
         return GetFunctionRegistration
         <
            GM,
            meta::Increment<IX>::value,
            DX,
            meta::EqualNumber<meta::Increment<IX>::value, DX>::value
         >::get(functionIndex);
      }
   }
};

template<class GM, size_t IX, size_t DX>
struct GetFunctionRegistration<GM, IX, DX, true>{
   static size_t get(const size_t functionIndex) {
      throw RuntimeError("Incorrect function type id.");
   }
};

template<class GM, size_t IX, size_t DX, bool END>
struct SaveAndLoadFunctions;

template<class GM, size_t IX, size_t DX>
struct SaveAndLoadFunctions<GM, IX, DX, false>
{
   template<class HDF5_HANDLE>
   static void save
   (
      HDF5_HANDLE handle,
      const GM& gm,
      const opengm::UInt64Type storeValueTypeAs
   ) {
      if(meta::FieldAccess::template byIndex<IX>(gm.functionDataField_).functionData_.functions_.size() != 0)
      {
         // create group
         std::stringstream ss;
         typedef typename meta::TypeAtTypeList<typename GM::FunctionTypeList, IX>::type TypeAtIX;
         ss << "function-id-" << (FunctionRegistration<TypeAtIX>::Id);
         hid_t group = marray::hdf5::createGroup(handle, ss.str());

         // loop over all functions of this type
         size_t indexCounter = 0;
         size_t valueCounter = 0;
         for(size_t i=0; i<meta::FieldAccess::template byIndex<IX>(gm.functionDataField_).functionData_.functions_.size(); ++i) {
            indexCounter += FunctionSerialization<TypeAtIX>::indexSequenceSize(
               meta::FieldAccess::template byIndex<IX> (gm.functionDataField_).functionData_.functions_[i]);
            valueCounter += FunctionSerialization<TypeAtIX>::valueSequenceSize(
               meta::FieldAccess::template byIndex<IX> (gm.functionDataField_).functionData_.functions_[i]);
         }
         marray::Vector<typename GM::ValueType> valueVector(valueCounter);
         marray::Vector<opengm::UInt64Type> indexVector(indexCounter);
         typename marray::Vector<typename GM::ValueType>::iterator valueIter = valueVector.begin();
         typename marray::Vector<opengm::UInt64Type>::iterator indexIter = indexVector.begin();
         for(size_t i=0; i<meta::FieldAccess::template byIndex<IX>(gm.functionDataField_).functionData_.functions_.size(); ++i) {
            FunctionSerialization<TypeAtIX>::serialize(
               meta::FieldAccess::template byIndex<IX> (gm.functionDataField_).functionData_.functions_[i], indexIter, valueIter);
            indexIter+=FunctionSerialization<TypeAtIX>::indexSequenceSize(
               meta::FieldAccess::template byIndex<IX> (gm.functionDataField_).functionData_.functions_[i]);
            valueIter+=FunctionSerialization<TypeAtIX>::valueSequenceSize(
               meta::FieldAccess::template byIndex<IX> (gm.functionDataField_).functionData_.functions_[i]);
         }
         marray::hdf5::save(group, std::string("indices"), indexVector);
         OPENGM_ASSERT(storeValueTypeAs<4);
         typedef typename GM::ValueType GmValueType;
         if(storeValueTypeAs==static_cast<opengm::UInt64Type>(StoredValueTypeInfo::AsFloat)) {
            typedef opengm::detail_types::Float StorageType;
            if(opengm::meta::Compare<GmValueType,StorageType>::value==true) {
               marray::hdf5::save(group, std::string("values"), valueVector);
            }
            else{
               marray::Vector<StorageType> tmpValueVector=valueVector;
               marray::hdf5::save(group, std::string("values"), tmpValueVector);
            }
         }
         else if(storeValueTypeAs==static_cast<opengm::UInt64Type>(StoredValueTypeInfo::AsDouble)) {
            typedef opengm::detail_types::Double StorageType;
            if(opengm::meta::Compare<GmValueType,StorageType>::value==true) {
               marray::hdf5::save(group, std::string("values"), valueVector);
            }
            else{
               marray::Vector<StorageType> tmpValueVector=valueVector;
               marray::hdf5::save(group, std::string("values"), tmpValueVector);
            }
         }
         else if(storeValueTypeAs==static_cast<opengm::UInt64Type>(StoredValueTypeInfo::AsUInt)) {
            typedef opengm::detail_types::UInt64Type StorageType;
            if(opengm::meta::Compare<GmValueType,StorageType>::value==true) {
               marray::hdf5::save(group, std::string("values"), valueVector);
            }
            else{
               marray::Vector<StorageType> tmpValueVector=valueVector;
               marray::hdf5::save(group, std::string("values"), tmpValueVector);
            }
         }
         else if (storeValueTypeAs==static_cast<opengm::UInt64Type>(StoredValueTypeInfo::AsInt)) {
            typedef opengm::detail_types::Int64Type StorageType;
            if(opengm::meta::Compare<GmValueType,StorageType>::value==true) {
               marray::hdf5::save(group, std::string("values"), valueVector);
            }
            else{
               marray::Vector<StorageType> tmpValueVector=valueVector;
               marray::hdf5::save(group, std::string("values"), tmpValueVector);
            }
         }
         marray::hdf5::closeGroup(group);
      }

      // save functions of the next type in the typelist
      typedef typename opengm::meta::Increment<IX>::type NewIX;
      SaveAndLoadFunctions<GM, NewIX::value, DX, opengm::meta::EqualNumber<NewIX::value, DX>::value >::save(handle, gm,storeValueTypeAs);
   }

   template<class HDF5_HANDLE>
   static void load
   (
      HDF5_HANDLE handle,
      GM& gm,
      const  std::vector<opengm::UInt64Type>& numberOfFunctions,
      const  std::vector<opengm::UInt64Type>& functionIndexLookup,
      const  std::vector<bool> & useFunction,
      const opengm::UInt64Type loadValueTypeAs,
      bool oldFormat=false
   ) {
      if(useFunction[IX]==true) {
         size_t mappedIndex;
         bool foundIndex=false;
         for(size_t i=0;i<functionIndexLookup.size();++i) {
            if(functionIndexLookup[i]==IX ) {
               mappedIndex=i;
               foundIndex=true;
               break;
            }
         }
         if(!foundIndex) {
            throw RuntimeError("Could not load function.");
         }

         if(numberOfFunctions[mappedIndex] != 0) {
            // create subgroup
            std::stringstream ss;
            typedef typename meta::TypeAtTypeList<typename GM::FunctionTypeList, IX>::type TypeAtIX ;
            ss << "function-id-" << (FunctionRegistration<TypeAtIX>::Id);
            hid_t group = marray::hdf5::openGroup(handle, ss.str());
            marray::Vector<typename GM::ValueType> serializationValues;
            marray::Vector<opengm::UInt64Type> serializationIndicies;
            {
               std::string subDatasetName("indices");
               marray::hdf5::load(group, subDatasetName, serializationIndicies);
            }
            {
               std::string subDatasetName("values");
               OPENGM_ASSERT(loadValueTypeAs<4);
               typedef typename GM::ValueType GmValueType;
               if(oldFormat==false) {
                  if(loadValueTypeAs==static_cast<opengm::UInt64Type>(StoredValueTypeInfo::AsFloat)) {
                     typedef opengm::detail_types::Float StorageType;
                     if(opengm::meta::Compare<GmValueType,StorageType>::value==true) {
                        marray::hdf5::load(group, subDatasetName, serializationValues);
                     }
                     else{
                        marray::Vector<StorageType> tmpSerializationValues;
                        marray::hdf5::load(group, subDatasetName, tmpSerializationValues);
                        serializationValues=tmpSerializationValues;
                     }
                  }
                  else if(loadValueTypeAs==static_cast<opengm::UInt64Type>(StoredValueTypeInfo::AsDouble)) {
                     typedef opengm::detail_types::Double StorageType;
                     if(opengm::meta::Compare<GmValueType,StorageType>::value==true) {
                        marray::hdf5::load(group, subDatasetName, serializationValues);
                     }
                     else{
                        marray::Vector<StorageType> tmpSerializationValues;
                        marray::hdf5::load(group, subDatasetName, tmpSerializationValues);
                        serializationValues=tmpSerializationValues;
                     }
                  }
                  else if(loadValueTypeAs==static_cast<opengm::UInt64Type>(StoredValueTypeInfo::AsUInt)) {
                     typedef opengm::detail_types::UInt64Type StorageType;
                     if(opengm::meta::Compare<GmValueType,StorageType>::value==true) {
                        marray::hdf5::load(group, subDatasetName, serializationValues);
                     }
                     else{
                        marray::Vector<StorageType> tmpSerializationValues;
                        marray::hdf5::load(group, subDatasetName, tmpSerializationValues);
                        serializationValues=tmpSerializationValues;
                     }
                  }
                  else if (loadValueTypeAs==static_cast<opengm::UInt64Type>(StoredValueTypeInfo::AsInt)) {
                     typedef opengm::detail_types::Int64Type StorageType;
                     if(opengm::meta::Compare<GmValueType,StorageType>::value==true) {
                        marray::hdf5::load(group, subDatasetName, serializationValues);
                     }
                     else{
                        marray::Vector<StorageType> tmpSerializationValues;
                        marray::hdf5::load(group, subDatasetName, tmpSerializationValues);
                        serializationValues=tmpSerializationValues;
                     }
                  }
               }
               else{
                  marray::hdf5::load(group, subDatasetName, serializationValues);
               }

            }
            // resize function
            gm.template functions<IX>().resize(numberOfFunctions[mappedIndex]);
            typename marray::Vector<opengm::UInt64Type>::const_iterator indexIter=serializationIndicies.begin();
            typename marray::Vector<typename GM::ValueType>::const_iterator valueIter=serializationValues.begin();
            // fill function with data
            for(size_t i=0; i<meta::FieldAccess::template byIndex<IX> (gm.functionDataField_).functionData_.functions_.size(); ++i) {
               FunctionSerialization<TypeAtIX>::deserialize(
                  indexIter, valueIter, meta::FieldAccess::template byIndex<IX> (gm.functionDataField_).functionData_.functions_[i]);
               indexIter += FunctionSerialization<TypeAtIX>::indexSequenceSize(
                  meta::FieldAccess::template byIndex<IX> (gm.functionDataField_).functionData_.functions_[i]);
               valueIter+=FunctionSerialization<TypeAtIX>::valueSequenceSize(
                  meta::FieldAccess::template byIndex<IX> (gm.functionDataField_).functionData_.functions_[i]);
            }
            marray::hdf5::closeGroup(group);
         }
      }

      // load functions of the next type in the typelist
      typedef typename opengm::meta::Increment<IX>::type NewIX;
      SaveAndLoadFunctions<GM, NewIX::value, DX, opengm::meta::EqualNumber<NewIX::value, DX>::value >::load
      (handle, gm, numberOfFunctions,functionIndexLookup,useFunction,loadValueTypeAs,oldFormat);
   }
};

template<class GM, size_t IX, size_t DX>
struct SaveAndLoadFunctions<GM, IX, DX, true> {
   template<class HDF5_HANDLE>
   static void save
   (
      HDF5_HANDLE handle,
      const GM& gm,
      const opengm::UInt64Type storeValueTypeAs
   )
   {

   }

   template<class HDF5_HANDLE>
   static void load
   (
      HDF5_HANDLE handle,
      GM& gm,
      const  std::vector<opengm::UInt64Type>& numberOfFunctions,
      const  std::vector<opengm::UInt64Type>& functionIndexLookup,
      const  std::vector<bool> & useFunction ,
      const opengm::UInt64Type loadValueTypeAs,
      bool oldFormat=false
   )
   {

   }
};
/// \endcond

/// \brief save a graphical model to an HDF5 file
/// \param gm graphical model to save
/// \param filepath to save as
/// \param name of dataset within the HDF5 file
template<class GM>
void save
(
   const GM& gm,
   const std::string& filepath,
   const std::string& datasetName
)
{
   typedef typename GM::ValueType ValueType;
   typedef typename GM::FactorType FactorType;

   if(IsValidTypeForHdf5Save<typename GM::ValueType>::value==false) {
      throw opengm::RuntimeError( std::string("ValueType  has no support for hdf5 export") );
   }
   hid_t file = marray::hdf5::createFile(filepath, marray::hdf5::DEFAULT_HDF5_VERSION);
   hid_t group = marray::hdf5::createGroup(file, datasetName);
   std::vector<UInt64Type> serializationIndicies;
   opengm::UInt64Type storeValueTypeAs;
   // float
   if(opengm::meta::Compare<opengm::detail_types::Float,ValueType>::value==true) {
      storeValueTypeAs=static_cast<opengm::UInt64Type>(StoredValueTypeInfo::AsFloat);
   }
   //double
   else if(opengm::meta::Compare<opengm::detail_types::Double,ValueType>::value==true) {
      storeValueTypeAs=static_cast<opengm::UInt64Type>(StoredValueTypeInfo::AsDouble);
   }
   // long double
   else if(opengm::meta::Compare<opengm::detail_types::LongDouble,ValueType>::value==true) {
      throw RuntimeError(std::string("ValueType \" long double\" has no support for hdf5 export"));
   }
   // bool
   else if(opengm::meta::Compare<opengm::detail_types::Bool,ValueType>::value==true) {
      storeValueTypeAs = static_cast<opengm::UInt64Type> (StoredValueTypeInfo::AsUInt);
   }
   // unsigned integers
   else if(std::numeric_limits<ValueType>::is_integer==true && std::numeric_limits<ValueType>::is_signed==false) {
      storeValueTypeAs = static_cast<opengm::UInt64Type> (StoredValueTypeInfo::AsUInt);
   }
   // signed integers
   else if(std::numeric_limits<ValueType>::is_integer==true && std::numeric_limits<ValueType>::is_signed==true) {
      storeValueTypeAs = static_cast<opengm::UInt64Type> (StoredValueTypeInfo::AsInt);
   }
   else{
       throw RuntimeError(std::string("ValueType has no support for hdf5 export"));
   }
   //opengm::UInt64Type
   // save meta data
   {
      std::string subDatasetName("header");
      serializationIndicies.push_back(VERSION_MAJOR);
      serializationIndicies.push_back(VERSION_MINOR);
      serializationIndicies.push_back(gm.numberOfVariables());
      serializationIndicies.push_back(gm.factors_.size());
      serializationIndicies.push_back(GM::NrOfFunctionTypes);
      for(size_t i=0; i<GM::NrOfFunctionTypes; ++i) {
         const size_t fRegId=GetFunctionRegistration
         <
            GM,
            0,
            GM::NrOfFunctionTypes,
            meta::EqualNumber<GM::NrOfFunctionTypes, 0>::value
         >::get(i);
         serializationIndicies.push_back(fRegId);
         serializationIndicies.push_back(gm.numberOfFunctions(i));
      }
      serializationIndicies.push_back(storeValueTypeAs);
      marray::hdf5::save(group, subDatasetName, serializationIndicies);
   }

   // save numbers of states
   {
      std::string subDatasetName("numbers-of-states");
      serializationIndicies.resize(gm.numberOfVariables());
      for(size_t i=0;i<gm.numberOfVariables();++i) {
         serializationIndicies[i]=
            static_cast<opengm::UInt64Type>(gm.numberOfLabels(i));
      }
      marray::hdf5::save(group, subDatasetName, serializationIndicies);
   }
   serializationIndicies.clear();

   // save all functions
   SaveAndLoadFunctions<GM, 0, GM::NrOfFunctionTypes, opengm::meta::EqualNumber<GM::NrOfFunctionTypes, 0>::value >::save(group, gm,storeValueTypeAs);

   // save all factors
   {
      std::string subDatasetName("factors");
      for(size_t i = 0; i < gm.factors_.size(); ++i) {
         serializationIndicies.push_back(static_cast<opengm::UInt64Type>(gm.factors_[i].functionIndex_));
         serializationIndicies.push_back(static_cast<opengm::UInt64Type>(gm.factors_[i].functionTypeId_));
         serializationIndicies.push_back(static_cast<opengm::UInt64Type>(gm.factors_[i].numberOfVariables()));
         for(size_t j = 0; j < gm.factors_[i].numberOfVariables(); ++j) {
            //serializationIndicies.push_back(static_cast<opengm::UInt64Type> (gm.factors_[i].variableIndices_[j]));
            serializationIndicies.push_back(static_cast<opengm::UInt64Type> (gm.factors_[i].variableIndex(j)));
         }
      }
      if(serializationIndicies.size() != 0)marray::hdf5::save(group, subDatasetName, serializationIndicies);
   }
   marray::hdf5::closeGroup(group);
   marray::hdf5::closeFile(file);
}

template<class GM>
void load
(
   GM& gm,
   const std::string& filepath,
   const std::string& datasetName
)
{
   typedef typename GM::ValueType ValueType;
   typedef typename GM::FactorType FactorType;
   hid_t file = marray::hdf5::openFile(filepath, marray::hdf5::READ_ONLY, marray::hdf5::DEFAULT_HDF5_VERSION);
   hid_t group =marray::hdf5::openGroup(file, datasetName);
   marray::Vector<opengm::UInt64Type> serializationIndicies;
   std::vector<opengm::UInt64Type> numberOfFunctions;
   std::vector<opengm::UInt64Type> functionIndexLookup;
   std::vector<bool> useFunction(GM::NrOfFunctionTypes,false);
   marray::Vector<opengm::UInt64Type> typeRegisterId;
   //size_t numberOfVariables;
   bool oldFormat=false;
   opengm::UInt64Type loadValueTypeAs=0;
   {
      std::string subDatasetName("header");
      marray::hdf5::load(group, subDatasetName, serializationIndicies);
      OPENGM_CHECK_OP(serializationIndicies.size() ,>, 5," ")
      //OPENGM_CHECK_OP(serializationIndicies.size() ,<=, 5 + 2 * GM::NrOfFunctionTypes+1," ")
      //OPENGM_ASSERT( serializationIndicies.size() > 5 && serializationIndicies.size() <= 5 + 2 * GM::NrOfFunctionTypes+1);
      if(!(serializationIndicies.size() > 5 && serializationIndicies.size() <= 5 + 2 * GM::NrOfFunctionTypes)) {
      }
      if(serializationIndicies[0] != 2 || serializationIndicies[1] != 0) {
         throw RuntimeError("This version of the HDF5 file format is not supported by this version of OpenGM.");
      }
      //numberOfVariables=serializationIndicies[2];
      //gm.numbersOfStates_.resize(serializationIndicies[2]);
      gm.factors_.resize(serializationIndicies[3], FactorType(&gm));
      numberOfFunctions.resize(serializationIndicies[4]);
      functionIndexLookup.resize(serializationIndicies[4]);
      typeRegisterId.resize(serializationIndicies[4]);
      for(size_t i=0; i<numberOfFunctions.size(); ++i) {
         //const size_t fRegId=GetFunctionRegistration<GM, 0, GM::NrOfFunctionTypes, meta::EqualNumber<GM::NrOfFunctionTypes, 0>::value>::get(i);
         typeRegisterId[i]=serializationIndicies[5 + 2 * i];
         numberOfFunctions[i]=serializationIndicies[5 + 2*i + 1];
      }

      if(serializationIndicies.size()!=5+2*numberOfFunctions.size()+1) {
         if(serializationIndicies.size()==5+2*numberOfFunctions.size()) {
            oldFormat=true;
         }
         else{
            throw RuntimeError(std::string("error in hdf5 file"));
         }
      }
      else{
         loadValueTypeAs=serializationIndicies[serializationIndicies.size()-1];
         OPENGM_ASSERT(loadValueTypeAs<4);
      }
      // check if saved function (type list) is a subset of the typelist of the
      // gm in which we want to load
      for(size_t i=0; i<numberOfFunctions.size(); ++i) {
         opengm::UInt64Type regIdToFind=typeRegisterId[i];
         bool foundId=false;
         for(size_t j=0; j<GM::NrOfFunctionTypes; ++j) {
            opengm::UInt64Type regIdInList=GetFunctionRegistration<GM, 0, GM::NrOfFunctionTypes, meta::EqualNumber<GM::NrOfFunctionTypes, 0>::value>::get(j);
            if(regIdToFind==regIdInList ) {
               foundId=true;
               functionIndexLookup[i]=j;
               useFunction[j]=true;
               break;
            }
         }
         if(foundId==false && numberOfFunctions[i]!=0) {
             std::stringstream ss;
             ss << "The HDF5 file contains the function type "
                << regIdToFind
                << " which is not contained in the type list in the C++ code.";
            throw RuntimeError(ss.str());
         }
      }
   }
   //if(numberOfVariables != 0) {
   std::string subDatasetName("numbers-of-states");
   marray::hdf5::load(group, subDatasetName, serializationIndicies);
   gm.space_.assignDense(serializationIndicies.begin(), serializationIndicies.end());
   OPENGM_ASSERT(serializationIndicies.size() == gm.numberOfVariables());
   //}
   SaveAndLoadFunctions<GM, 0, GM::NrOfFunctionTypes, opengm::meta::EqualNumber<GM::NrOfFunctionTypes, 0>::value >::load
   (group, gm, numberOfFunctions,functionIndexLookup,useFunction,loadValueTypeAs,oldFormat);

   gm.factorsVis_.clear();

   if(gm.factors_.size() != 0) {

      std::string subDatasetName("factors");
      marray::hdf5::load(group, subDatasetName, serializationIndicies);
      size_t sIndex = 0;
      for(size_t i = 0; i < gm.factors_.size(); ++i) {
         gm.factors_[i].functionIndex_ = static_cast<opengm::UInt64Type> (serializationIndicies[sIndex]);
         sIndex++;
         gm.factors_[i].functionTypeId_ =
            functionIndexLookup[static_cast<opengm::UInt64Type> (serializationIndicies[sIndex])];
         sIndex++;


         //factorsVis_
         const opengm::UInt64Type order = static_cast<opengm::UInt64Type> (serializationIndicies[sIndex]);
         const opengm::UInt64Type indexInVisVector  = static_cast<opengm::UInt64Type> (gm.factorsVis_.size());

         gm.factors_[i].vis_.assign(gm.factorsVis_,indexInVisVector,order);
         gm.order_ = std::max( gm.order_,order);

         //gm.factors_[i].order_=static_cast<opengm::UInt64Type> (serializationIndicies[sIndex]);
         //gm.factors_[i].indexInVisVector_=static_cast<opengm::UInt64Type> (gm.factorsVis_.size());

         sIndex++;
         for(size_t j = 0; j < gm.factors_[i].numberOfVariables(); ++j) {
            gm.factorsVis_.push_back( static_cast<opengm::UInt64Type> (serializationIndicies[sIndex]));
            sIndex++;
         }



      }
   }

   marray::hdf5::closeGroup(group);
   marray::hdf5::closeFile(file);
   gm.variableFactorAdjaceny_.resize(gm.numberOfVariables());
   // adjacenies

   for(size_t i=0; i<gm.numberOfFactors(); ++i) {
      for(size_t vi=0;vi<gm[i].numberOfVariables(); ++vi) {
         gm.variableFactorAdjaceny_[ gm[i].variableIndex(vi) ].insert(i);
      }
      //typedef functionwrapper::AddFactorUpdateAdjacencyWrapper<GM::NrOfFunctionTypes> WrapperType;
      //WrapperType::op(gm.functionDataField_, gm[i].functionIndex(), i, gm[i].functionType());
   }
   //gm.initializeFactorFunctionAdjacency();
}
      
} // namespace hdf5
} // namespace opengm

#endif // #ifndef OPENGM_GRAPHICALMODEL_HDF5_HXX
