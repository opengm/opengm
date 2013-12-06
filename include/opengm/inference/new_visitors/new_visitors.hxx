#ifndef OPENGM_NEW_VISITOR_HXX
#define OPENGM_NEW_VISITOR_HXX

#include <iostream>
#include <opengm/opengm.hxx>
#include <opengm/utilities/timer.hxx>

namespace opengm{
namespace visitors{

struct VisitorReturnFlag{
	enum VisitorReturnFlagValues{
		continueInf			=0,
		stopInfBoundReached	=1,
		stopInfTimeout    	=2
	};
};


template<class INFERENCE>
class EmptyVisitor{
public:
	EmptyVisitor(){
	}
	void begin(INFERENCE & inf){
	}
	size_t operator()(INFERENCE & inf){
		return static_cast<size_t>(VisitorReturnFlag::continueInf);
	}
	void end(INFERENCE & inf){
	}
};



template<class INFERENCE>
class VerboseVisitor{
public:
	VerboseVisitor(const size_t visithNth=1,const bool multiline=false)
	: 	iteration_(0),
		visithNth_(visithNth),
		multiline_(multiline){
	}
	void begin(INFERENCE & inf){
		std::cout<<"begin: value "<<inf.value()<<" bound "<<inf.bound()<<"\n";
		++iteration_;
	}
	size_t operator()(INFERENCE & inf){
		if((iteration_)%visithNth_==0){
			std::cout<<"step: "<<iteration_<<" value "<<inf.value()<<" bound "<<inf.bound()<<"\n";
		}
		++iteration_;
		return static_cast<size_t>(VisitorReturnFlag::continueInf);
	}
	void end(INFERENCE & inf){
		std::cout<<"value "<<inf.value()<<" bound "<<inf.bound()<<"\n";
	}
private:
	size_t iteration_;
	size_t visithNth_;
	bool   multiline_;
};



template<class INFERENCE>
class TimingVisitor{
public:
	typedef typename  INFERENCE::ValueType ValueType;
	
	TimingVisitor(
		const size_t visithNth=1,
		const size_t reserve=0,
		const bool 	 verbose=true,
		const bool   multiline=true,
		const double timeLimit=std::numeric_limits<double>::infinity()
	) 
	:
		protocolMap_(),
		times_(),
		values_(),
		bounds_(),
		iterations_(),
		timer_(),
		iteration_(0),
		visithNth_(visithNth),
		multiline_(multiline),
		timeLimit_(timeLimit),
 		totalTime_(0.0)
	{
		// allocate all protocolated items
		ctime_		= & protocolMap_["ctime"]    ;
		times_      = & protocolMap_["times"]    ;
		values_     = & protocolMap_["values"]   ;
		bounds_     = & protocolMap_["bounds"]   ;
		iterations_ = & protocolMap_["iteration"];

		// reservations
		if(reserve>0){
			times_->reserve(reserve);
			values_->reserve(reserve);
			bounds_->reserve(reserve);
			iterations_->reserve(reserve);
		}

		// start timer to measure time from
		// constructor call to "begin" call
		timer_.tic();
	}

	void begin(INFERENCE & inf){

		// stop timer which measured time from
		// constructor call to this "begin" call
		timer_.toc();
		// store values bound time and iteration number  
		const ValueType val=inf.value();
		const ValueType bound=inf.bound();
		ctime_->push_back(timer_.elapsedTime());
        times_->push_back(0);
        values_->push_back(val);
        bounds_->push_back(bound);
        iterations_->push_back(double(iteration_));

        // print step
        if(verbose_)
        	std::cout<<"value "<<val<<" bound "<<bound<<"\n";
        // increment iteration
        ++iteration_;
		// restart timer
		timer_.reset();
		timer_.tic();
	}

	size_t operator()(INFERENCE & inf){

		if(iteration_%visithNth_==0){
			// stop timer
			timer_.toc();

			// store values bound time and iteration number  
			const ValueType val 	=inf.value();
			const ValueType bound 	=inf.bound();
			const double 	t    	= timer_.elapsedTime();
	        times_->push_back(t);
	        values_->push_back(val);
	        bounds_->push_back(bound);
	        iterations_->push_back(double(iteration_));
	        // increment total time
	        totalTime_+=t;
	        if(verbose_){
	        	std::cout<<"step: "<<iteration_<<" value "<<val<<" bound "<<bound<<" [ "<<totalTime_ << "]" <<"\n";
	        }
	        // restart timer
	        timer_.reset();
			timer_.tic();
    	}
        ++iteration_;

        // check is time limit reached
		if(totalTime_<timeLimit_){
			return static_cast<size_t>(VisitorReturnFlag::continueInf);
		}
		else{
			if(verbose_)
				std::cout<<"timeout reached\n";
			return static_cast<size_t>(VisitorReturnFlag::stopInfTimeout);
		}
	}


	void end(INFERENCE & inf){
		// stop timer
		timer_.toc();
		// store values bound time and iteration number  
		const ValueType val=inf.value();
		const ValueType bound=inf.bound();
 		times_->push_back(timer_.elapsedTime());
        values_->push_back(val);
        bounds_->push_back(bound);
        iterations_->push_back(double(iteration_));
        if(verbose_){
        	std::cout<<"value "<<val<<" bound "<<bound<<"\n";
        }
	}


	// timing visitor specific interface

	const std::map< std::string, std::vector<double  > > & protocolMap()const{
		return protocolMap_;
	}

	const std::vector<double> & getConstructionTime()const{
		return *ctime_;
	}
	const std::vector<double> & getTimes			()const{
		return *times_;
	}
	const std::vector<double> & getValues			()const{
		return *values_;
	}
	const std::vector<double> & getBounds			()const{
		return *bounds_;
	}
	const std::vector<double> & getIterations		()const{
		return *iterations_;
	}	


private:

	std::map< std::string, std::vector<double  > >  protocolMap_;

	std::vector<double  > * ctime_;
	std::vector<double  > * times_;
	std::vector<double  > * values_;
	std::vector<double  > * bounds_;
	std::vector<double  > * iterations_;
	opengm::Timer timer_;
	opengm::Timer totalTimer_;
	size_t iteration_;
	size_t visithNth_;
	bool verbose_;
	bool   multiline_;

	double timeLimit_;
	double totalTime_;
};
}
}

#endif //OPENGM_NEW_VISITOR_HXX