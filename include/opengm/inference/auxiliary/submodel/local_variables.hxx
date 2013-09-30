




template<class GM>
class LocalVariables{

public:
	typedef typename GM::IndexType IndexType;


	LocalVariables(const GM & gm)
	:	gm_(gm),
		localToGlobalVar_(gm.numberOfVariables(),gm.numberOfVariables()),
		globalToLocalVar_(gm.numberOfVariables(),gm.numberOfVariables()),
		nLocalVar_(0){
	}

	// query
	IndexType size()const{
		return nLocalVar_;
	}

	IndexType globalToLocal(const IndexType gvi)const{
		return globalToLocalVar_[gvi];
	}

	IndexType localToGlobal(const IndexType lvi)const{
		return globalToLocalVar_[gvi];
	}

	bool inLocalVariables(const IndexType gvi)const{
		return globalToLocalVar_[gvi]!=gm_.numberOfVariables();
	}

	// modifiers
	void addVariable(const IndexType gvi){
		localToGlobalVar_[nLocalVar_]=gvi;
		globalToLocalVar_[gvi]=nLocalVar_;
		++nLocalVar_;
	}

	template<class VAR_ITER>
	void addVariables(VAR_ITER begin,VAR_ITER end){
		while(begin!=end){
			this->addVariable(*begin);
			++begin;
		}
	}

	void clear(){
		for(IndexType lvi=0;lvi<nLocalVar_;++nLocalVar_){
			const IndexType gvi=localToGlobalVar_[lvi];
			globalToLocalVar_[gvi]=gm_.numberOfVariables();
			localToGlobalVar_[lvi]=gm_.numberOfVariables();
		}

		nLocalVar_=0;
	}

private:
	const GM & gm_;
	std::vector<IndexType> localToGlobalVar_;
	std::vector<IndexType> globalToLocalVar_;

	IndexType nLocalVar_;
};

