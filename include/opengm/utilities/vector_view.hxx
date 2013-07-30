#ifndef OPENGM_VECTOR_VIEW
#define OPENGM_VECTOR_VIEW

namespace opengm{

template<class VECTOR,class INDEX_TYPE>
class VectorView{

public:



	typedef VECTOR VectorType;
	typedef INDEX_TYPE IndexType;
	typedef typename VectorType::value_type ValueType;

	typedef typename VectorType::const_iterator const_iterator;
	typedef typename VectorType::iterator iterator;

	VectorView(){}


	VectorView(
		const VectorType & vector
	) :	vectorPtr_(&vector),
		start_(0),
		size_(0){

	}


	VectorView(
		const VectorType & vector,
		const IndexType start,
		const IndexType size
	) :	vectorPtr_(&vector),
		start_(start),
		size_(size){

	}

	void assign(
		const VectorType & vector,
		const IndexType start,
		const IndexType size
	){
		vectorPtr_=&vector;
		start_=start;
		size_=size;
	}


	void assignPtr(
		const VectorType & vector
	){
		vectorPtr_=&vector;
	}


	iterator begin(){
		return vectorPtr_->begin()+start_;
	}
	iterator end(){
		return vectorPtr_->begin()+start_+size_;
	}

	const_iterator begin()const{
		return vectorPtr_->begin()+start_;
	}
	const_iterator end()const{
		return vectorPtr_->begin()+start_+size_;
	}


	const IndexType size()const{
		return size_;
	}
	const ValueType & operator[](const IndexType i)const{
		return (*vectorPtr_)[start_+i];
	}
	ValueType & operator[](const IndexType i){
		return (*vectorPtr_)[start_+i];
	}


private:
	const VectorType * vectorPtr_;
	IndexType start_;
	IndexType size_;

};

}


#endif