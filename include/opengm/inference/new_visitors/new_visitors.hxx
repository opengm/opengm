#include <iostream>

template<class INFERENCE>
class EmptyVisitor{
public:
	EmptyVisitor(){
	}
	void begin(INFERENCE & inf){
	}
	bool operator()(INFERENCE & inf){
		return true;
	}
	void end(INFERENCE & inf){
	}
};



template<class INFERENCE>
class VerboseVisitor{
public:
	VerboseVisitor(){
	}
	void begin(INFERENCE & inf){
		std::cout<<"value "<<inf.value()<<" bound "<<inf.bound()<<"\n";
	}
	bool operator()(INFERENCE & inf){
		std::cout<<"value "<<inf.value()<<" bound "<<inf.bound()<<"\n";
		return true;
	}
	void end(INFERENCE & inf){
		std::cout<<"value "<<inf.value()<<" bound "<<inf.bound()<<"\n";
	}
};



template<class INFERENCE>
class TimingVisitor{
public:
	typedef typename  INFERENCE::ValueType ValueType;
	
	TimingVisitor() 
	:
		iteration_(0)
		times_(),
		values_(),
		bounds_(),
		iterations_(),
		timer_()
	{
		timer_.tic();
	}

	void begin(INFERENCE & inf){
		// stop timer
		timer_.toc();
		// store values
		const ValueType val=inf.value();
		const ValueType bound=inf.bound();
        times_.push_back(timer_.elapsedTime());
        values_.push_back(values_);
        bounds_.push_back(bounds_);

        std::cout<<"value "<<val<<" bound "<<bound<<"\n";

        ++iteration_;
		// restart timer
		timer_.tic():
	}

	bool operator()(INFERENCE & inf){
		// stop timer
		timer_.toc();
		// store values
		const ValueType val=inf.value();
		const ValueType bound=inf.bound();
        times_.push_back(timer_.elapsedTime());
        values_.push_back(values_);
        bounds_.push_back(bounds_);

        std::cout<<"value "<<val<<" bound "<<bound<<"\n";

        ++iteration_;
		// restart timer
		timer_.tic():
		return true;
	}

	void end(INFERENCE & inf){
		// stop timer
		timer_.toc();
		// store values
		const ValueType val=inf.value();
		const ValueType bound=inf.bound();
        times_.push_back(timer_.elapsedTime());
        values_.push_back(values_);
        bounds_.push_back(bounds_);

        std::cout<<"value "<<val<<" bound "<<bound<<"\n";

	}
private:
	iteration_;
	std::vector<float  > times_;
	std::vector<float  > values_;
	std::vector<float  > bounds_;
	std::vector<size_t > iterations_;
	opengm::Timer timer_;
};
