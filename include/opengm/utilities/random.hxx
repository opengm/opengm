#pragma once
#ifndef OPENGM_RANDOM_HXX
#define OPENGM_RANDOM_HXX

/// \cond HIDDEN_SYMBOLS

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "opengm/opengm.hxx"

namespace opengm {

template<class T>
   class RandomUniformBase;
template<class T>
   class RandomUniformFloatingPoint;
template<class T>
   class RandomUniformInteger;
template<class T>
   class RandomUniformUnknown;
template<class T>
   class RandomUniform;

template<class T>
class RandomUniformBase
{
public:
   RandomUniformBase& operator=(const RandomUniformBase& src)
	{
		if(this!=&src) {
			low_=src.low_;
			high_=src.high_;
		}
		return *this;
	}

   RandomUniformBase(const T low = 0, const T high = 1)
	:  low_(low), 
      high_(high) 
   {}

private:
	T low_;
	T high_;

template<class> friend class RandomUniformInteger;
template<class> friend class RandomUniformFloatingPoint;
template<class> friend class RandomUniformUnknown;
template<class> friend class RandomUniform;
};

template<class T>
class RandomUniformInteger
: RandomUniformBase<T>
{
public:
	RandomUniformInteger(const T low = 0, const T high = 1)
	:  RandomUniformBase<T>(low, high)
   {}

	RandomUniformInteger(const T low, const T high, const size_t seed)
	:  RandomUniformBase<T>(low, high)
	{
		srand(time_t(seed));
	}

	T operator()() const
	{
      // intended semantics: random number is drawn from [low_, high)
      // however, not all std random number generators behave like this
      // thus this trivial work-around:
      T randomNumber = this->high_;
      while(randomNumber == this->high_) {
         randomNumber = static_cast<T>((rand() % (this->high_ - this->low_)) + this->low_);
      }
		return randomNumber;
	}

template<class> friend class RandomUniform;
};

template<class T>
class RandomUniformFloatingPoint
:  RandomUniformBase<T>
{
public:
	RandomUniformFloatingPoint(const T low = 0, const T high = 1)
	:  RandomUniformBase<T>(low, high)
	{}
	
   RandomUniformFloatingPoint(const T low, const T high, const size_t seed)
	:  RandomUniformBase<T>(low, high)
	{
		srand(time_t(seed));
	}
	
   T operator()() const
	{
      // intended semantics: random number is drawn from [low_, high)
      // however, not all std random number generators behave like this
      // thus this trivial work-around:
      T randomNumber = this->high_;
      while(randomNumber == this->high_) {
         randomNumber = (static_cast<T>(rand()) / static_cast<T>(RAND_MAX)) * (this->high_ - this->low_) + this->low_;
      }
		return randomNumber;
	}

template<class> friend class RandomUniform;
};

template<class T>
class RandomUniformUnknown : RandomUniformBase<T> {
template<class> friend class RandomUniform;
};

template<class T>
class RandomUniform
: public
		opengm::meta::If
		<
			opengm::meta::IsFundamental<T>::value, 
			typename opengm::meta::If
			<
				opengm::meta::IsFloatingPoint<T>::value, 
				opengm::RandomUniformFloatingPoint<T>, 
				opengm::RandomUniformInteger<T>
			>::type, 
			RandomUniformUnknown<T>
		>::type
{
private:
	typedef typename opengm::meta::If
	<
		opengm::meta::IsFundamental<T>::value, 
		typename opengm::meta::If
		<
			opengm::meta::IsFloatingPoint<T>::value, 
			opengm::RandomUniformFloatingPoint<T>, 
			opengm::RandomUniformInteger<T>
		>::type, 
		RandomUniformUnknown<T>
	>::type BaseType;

public:
	RandomUniform(const T low = 0, const T high = 1)
	:  BaseType(low, high)
	{}

	RandomUniform(const T low, const T high, const size_t seed)
   :  BaseType(low, high, seed)
	{}
};

template<class T, class U>
class RandomDiscreteWeighted
{
public:
   RandomDiscreteWeighted() {}

	template<class Iterator>
	RandomDiscreteWeighted(Iterator begin, Iterator end)
	:  probSum_(std::distance(begin, end)), 
      randomFloatingPoint_(0, 1)
	{
		U sum = 0;
      // normalization
		for(size_t i=0;i<probSum_.size();++i) {
			probSum_[i]=static_cast<U>(*begin);
			++begin;
			sum+=probSum_[i];
		}
		probSum_[0]/=sum;
		for(size_t i=1;i<probSum_.size();++i) {
			probSum_[i]/=sum;
			probSum_[i]+=probSum_[i-1];
		}
	}

	template<class Iterator>
	RandomDiscreteWeighted(Iterator begin, Iterator end, size_t seed)
	:  probSum_(std::distance(begin, end)), 
      randomFloatingPoint_(0, 1, seed)
	{
		U sum=0;
		// normalization
		for(size_t i=0;i<probSum_.size();++i) {
			probSum_[i]=static_cast<U>(*begin);
			++begin;
			sum+=probSum_[i];
		}
		probSum_[0]/=sum;
		// accumulatives
		for(size_t i=1;i<probSum_.size();++i) {
			probSum_[i]/=sum;
			probSum_[i]+=probSum_[i-1];
		}
	}

   T operator()() const
	{
		U rnd = randomFloatingPoint_();
		if(rnd < probSum_[0]) {
         return 0;
      }
		for(size_t i=0;i<probSum_.size()-1;++i) {
			if(probSum_[i]<=rnd && rnd<probSum_[i+1]) {
				return static_cast<T>(i+1);
			}
		}
		return static_cast<T>(probSum_.size() - 1);
	}

private:
	std::vector<U> probSum_;
	opengm::RandomUniform<U> randomFloatingPoint_;
};

} // namespace opengm

/// \endcond

#endif // #ifndef OPENGM_RANDOM_HXX
