#ifndef OUTPUT_DEBUG_UTILS_HXX_
#define OUTPUT_DEBUG_UTILS_HXX_

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
#endif
