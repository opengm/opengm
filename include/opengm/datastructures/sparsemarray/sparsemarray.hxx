#pragma once
#ifndef OPENGM_SPARSEMARRAY
#define OPENGM_SPARSEMARRAY

#include <algorithm>
#include <iostream>
#include <map>
#include "opengm/functions/function_properties_base.hxx"


namespace opengm {

template<class T,class I,class L,class CONTAINER=std::map<I,T> >
class SparseFunction   : public FunctionBase<SparseFunction<T, I, L,CONTAINER>, T, I, L> {
public:
    typedef CONTAINER ContainerType;
    typedef typename ContainerType::key_type KeyType;
    typedef typename ContainerType::mapped_type MappedType;
    typedef std::pair<KeyType,MappedType> KeyValPairType;
    typedef T ValueType;
    typedef I IndexType;
    typedef L LabelType;


    typedef typename ContainerType::const_iterator ConstContainerIteratorType;
    typedef typename ContainerType::iterator ContainerIteratorType;


    SparseFunction():
      dimension_(0),
      defaultValue_(0),
      container_(),
      shape_(),
      strides_(){
    }


    //constructors
    template<class SHAPE_ITERATOR>
    SparseFunction(SHAPE_ITERATOR shapeBegin,SHAPE_ITERATOR shapeEnd ,const ValueType defaultValue):
    dimension_(std::distance(shapeBegin,shapeEnd)),
    defaultValue_(defaultValue),
    container_(){
        shape_.resize(dimension_);
        strides_.resize(dimension_);
        // compute strides
        LabelType strideVal=1;
        for(unsigned short  dim=0;dim<dimension_;++dim){
            shape_[dim]=*shapeBegin;
            strides_[dim]=strideVal;
            strideVal*=shape_[dim];
            ++shapeBegin;
        }
    }

    size_t size()const{
      size_t size =1;
      for(unsigned short  dim=0;dim<dimension_;++dim){
            size*=static_cast<size_t>(shape_[dim]);
      }
      return size;
    }

    const ContainerType & container()const{
        return container_;
    }
    ContainerType & container(){
        return container_;
    }
    const size_t dimension()const{
        return dimension_;
    }
    const LabelType shape(const IndexType i)const{
        return shape_[i];
    }



    template<class COORDINATE_ITERATOR>
    void keyToCoordinate(const KeyType key, COORDINATE_ITERATOR coordinate )const{
        typedef typename std::iterator_traits<COORDINATE_ITERATOR>::value_type CoordType;
        KeyType keyRest=key;
        if(dimension_!=1){
            for(unsigned short  d=0;d<dimension_;++d){
                const unsigned short dim=(dimension_-1)-d;
                const KeyType c=keyRest/static_cast<KeyType>( strides_[dim] );
                keyRest=keyRest-c*static_cast<KeyType>( strides_[dim] );
                coordinate[dim]=static_cast<CoordType>(c);
            }
        }
        else{
            *coordinate=static_cast<CoordType>(key);
        }
    }


    template<class COORDINATE_ITERATOR>
    KeyType coordinateToKey(COORDINATE_ITERATOR coordinate)const{
        KeyType key=static_cast<KeyType>(0);
        for(unsigned short  dim=0;dim<dimension_;++dim){
            key+=strides_[dim]*static_cast<KeyType>(*coordinate);
            ++coordinate;
        }
        return key;
    }

    template<class COORDINATE_ITERATOR,size_t DIM>
    KeyType coordinateToKeyWithDim(COORDINATE_ITERATOR coordinate )const{
        KeyType key=static_cast<KeyType>(0);
        for(unsigned short  dim=0;dim<DIM;++dim){
            key+=strides_[dim]*static_cast<KeyType>(coordinate[dim]);
        }
        return key;
    }

    template<class COORDINATE_ITERATOR>
    ValueType operator()(COORDINATE_ITERATOR coordinate)const{
        typedef COORDINATE_ITERATOR CoordType;
        KeyType key;//=coordinateToKey(coordinate);

        switch (dimension_)
        {
            case 1:
                return valueFromKey(coordinateToKeyWithDim<CoordType,1>(coordinate));
            case 2:
                return valueFromKey(coordinateToKeyWithDim<CoordType,2>(coordinate));
            case 3:
                return valueFromKey(coordinateToKeyWithDim<CoordType,3>(coordinate));
            case 4:
                return valueFromKey(coordinateToKeyWithDim<CoordType,4>(coordinate));
            case 5:
                return valueFromKey(coordinateToKeyWithDim<CoordType,5>(coordinate));
            case 6:
                return valueFromKey(coordinateToKeyWithDim<CoordType,6>(coordinate));
            case 7:
                return valueFromKey(coordinateToKeyWithDim<CoordType,7>(coordinate));
            case 8:
                return valueFromKey(coordinateToKeyWithDim<CoordType,8>(coordinate));
            case 9:
                return valueFromKey(coordinateToKeyWithDim<CoordType,9>(coordinate));
            case 10:
                return valueFromKey(coordinateToKeyWithDim<CoordType,10>(coordinate));
            case 11:
                return valueFromKey(coordinateToKeyWithDim<CoordType,11>(coordinate));
            case 12:
                return valueFromKey(coordinateToKeyWithDim<CoordType,12>(coordinate));
            case 13:
                return valueFromKey(coordinateToKeyWithDim<CoordType,13>(coordinate));
            case 14:
                return valueFromKey(coordinateToKeyWithDim<CoordType,14>(coordinate));
            case 15:
                return valueFromKey(coordinateToKeyWithDim<CoordType,15>(coordinate));
            case 16:
                return valueFromKey(coordinateToKeyWithDim<CoordType,16>(coordinate));
            default:
                return valueFromKey(coordinateToKey(coordinate));
        }
    }

    ValueType defaultValue()const{
      return defaultValue_;
    }

    ValueType valueFromKey(const KeyType key)const{
        ConstContainerIteratorType iter=container_.find(key);
        if(iter!=container_.end()){
            return iter->second;
        }
        else{
            return defaultValue_;
        }
    }

    template<class COORDINATE_ITERATOR>
    void insert(COORDINATE_ITERATOR coordinate,const ValueType value){
        container_.insert(KeyValPairType(coordinateToKey(coordinate),value));
    }

private:
    unsigned short dimension_;
    ValueType defaultValue_;        
    ContainerType container_;
    std::vector<LabelType> shape_;
    std::vector<size_t> strides_;

};
}

#endif