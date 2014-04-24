#ifndef OUTPUT_DEBUG_UTILS_HXX_
#define OUTPUT_DEBUG_UTILS_HXX_

#ifdef WITH_HDF5
#include <opengm/opengm.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/graphicalmodel/space/simplediscretespace.hxx>
#include <opengm/graphicalmodel/graphicalmodel_hdf5.hxx>
#endif

#ifdef TRWS_DEBUG_OUTPUT

#include <iostream>
#include <fstream>
#include <vector>
#include <valarray>
#include <string>
#include <list>

namespace OUT{

template<class M>
class nullstreamT: public std::ostream
{
public:
	static nullstreamT* Instance()
	{
		if (!_pInstance) _pInstance = new nullstreamT();
		return _pInstance;
	};

	template <class T>
	nullstreamT& operator << (T& t){return *this;}
private:
	nullstreamT(): std::ios(0), std::ostream(0){};  // Private so that it can  not be called
	nullstreamT(nullstreamT const&){};             // copy constructor is private
	nullstreamT& operator=(nullstreamT const&){return *this;};  // assignment operator is private
	static nullstreamT* _pInstance;
};

template<class M>
nullstreamT<M>* nullstreamT<M>::_pInstance=0;

typedef nullstreamT<int> nullstream;

	template<typename ArrayType>
	std::ostream& operator << (std::ostream& logger,const std::vector<ArrayType>& arr)
	{
		for (size_t i=0;i<arr.size();++i)
			logger << arr[i]<<"; ";
		logger <<std::endl;
		return logger;
	};

	template<typename ArrayType>
	std::ostream& operator << (std::ostream& logger,const std::list<ArrayType>& lst)
	{
		typename std::list<ArrayType>::const_iterator beg=lst.begin(), end=lst.end();
		for (;beg!=end;++beg)
			logger << *beg<<"; ";
		logger <<std::endl;
		return logger;
	};

	template<typename ArrayType>
	std::ostream& operator << (std::ostream& logger,const std::vector<std::vector<ArrayType> >& arr)
	{
		for (size_t i=0;i<arr.size();++i)
			logger << arr[i];
		return logger;
	};

	template<typename T>
	std::ostream& operator << (std::ostream& logger,const std::valarray<T> & arr)
	{
		for (size_t i=0;i<arr.size();++i)
			logger << arr[i]<<", ";
		return logger;
	};

	template<typename Type1,typename Type2>
	std::ostream& operator << (std::ostream& logger, const std::pair<Type1,Type2>& p)
	{
		logger <<"("<<p.first<<","<<p.second<<")";
		return logger;
	};

	template<class Iterator>
	void saveContainer(std::ostream& fout, Iterator begin, Iterator end)
	{
		for ( ; begin!=end; ++begin)
			fout <<std::scientific<<*begin<<"; ";
		fout << std::endl;
	}

	template<class Iterator>
	void saveContainer(const std::string& filename, Iterator begin, Iterator end)
	{
		std::ofstream fout(filename.c_str());
		saveContainer(fout, begin, end);
		fout.close();
	};
};
#endif //TRWS_DEBUG_OUTPUT

#ifdef	WITH_HDF5

namespace opengm{
template<class GM> void store_into_explicit(const GM& gm, const std::string& file,const std::string& modelname="gm" )
{
	typedef typename GM::FunctionIdentifier             FunctionIdentifierType;
	typedef typename GM::IndexType IndexType;
	typedef typename GM::LabelType LabelType;
	typedef typename GM::ValueType ValueType;

	typedef GraphicalModel<ValueType, Adder, ExplicitFunction<ValueType,IndexType,LabelType>, typename GM::SpaceType> ExplicitModel;

	std::vector<LabelType> numLabels(gm.numberOfVariables());
	for(IndexType varId=0; varId<gm.numberOfVariables(); ++varId){
		numLabels[varId] = gm.numberOfLabels(varId);        
	}
	ExplicitModel explicitGm = ExplicitModel( typename GM::SpaceType(numLabels.begin(), numLabels.end() ));

	std::vector< std::vector<IndexType> > factorIndices(gm.numberOfFactors());
	for(IndexType factorId=0; factorId<gm.numberOfFactors(); ++factorId)
		for(IndexType varId=0; varId < gm.numberOfVariables(factorId); ++varId)
			factorIndices[factorId].push_back( gm.variableOfFactor(factorId,varId) );
	
	for(IndexType factorId=0; factorId<gm.numberOfFactors(); ++factorId) {

		// implemented storing only for unary and pairwise factors
		if( factorIndices[factorId].size() == 1 ) {
			const LabelType shape[] = { numLabels[factorIndices[factorId][0]] };
			ExplicitFunction<double> function(shape, shape + 1);
			for(size_t s = 0; s<shape[0]; ++s){
				std::vector<LabelType> it(1);
				it[0] = s;
				function(s) = gm[factorId](it.begin());
			}
			FunctionIdentifierType funcId = explicitGm.addFunction( function );
			explicitGm.addFactor( funcId, factorIndices[factorId].begin(), factorIndices[factorId].end());
		} else if(  factorIndices[factorId].size() == 2 ) {
			const LabelType shape[] = { numLabels[factorIndices[factorId][0]], numLabels[factorIndices[factorId][1]] };
			ExplicitFunction<double> function(shape, shape + 2);
			for(LabelType s1 = 0; s1<shape[0]; ++s1){
				for(LabelType s2 = 0; s2<shape[1]; ++s2){
					std::vector<LabelType> it(2);
					it[0] = s1;
					it[1] = s2;
					function(s1,s2) = gm[factorId](it.begin());
				}
			}
			FunctionIdentifierType funcId = explicitGm.addFunction( function );
			explicitGm.addFactor( funcId, factorIndices[factorId].begin(), factorIndices[factorId].end());
		} else throw std::runtime_error("store_into_explicit: Factors of order > 2 are not supported yet.");

	}

	hdf5::save(explicitGm, file, modelname);
}
}//namespace opengm

#endif

#endif
