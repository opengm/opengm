#ifndef OPENGM_VISITOR_HXX
#define OPENGM_VISITOR_HXX

#include <iostream>
#include <map>
#include <cmath>
#include <opengm/opengm.hxx>
#include <opengm/utilities/timer.hxx>

namespace opengm{
namespace visitors{

struct VisitorReturnFlag{
  const static size_t ContinueInf          = 0;
   const static size_t StopInfBoundReached  = 1;
   const static size_t StopInfTimeout       = 2;
};


template<class INFERENCE>
class EmptyVisitor{
public:
  EmptyVisitor(){
  }
  void begin(INFERENCE & inf){}
  size_t operator()(INFERENCE & inf){
    return VisitorReturnFlag::ContinueInf;
  }
  void end(INFERENCE & inf){
  }

  void addLog(const std::string & logName){}
  void log(const std::string & logName,const double logValue){}
};

template<class INFERENCE>
class ExplicitEmptyVisitor{
public:
   ExplicitEmptyVisitor(){
   }
   void begin(INFERENCE & inf, const typename INFERENCE::ValueType value, const typename INFERENCE::ValueType bound){}
   size_t operator()(INFERENCE & inf, const typename INFERENCE::ValueType value, const typename INFERENCE::ValueType bound){
      return VisitorReturnFlag::ContinueInf;
   }
   void end(INFERENCE & inf, const typename INFERENCE::ValueType value, const typename INFERENCE::ValueType bound){
   }
};

template<class INFERENCE>
class VerboseVisitor{
public:
  VerboseVisitor(const size_t visithNth=1,const bool multiline=false)
  :   iteration_(0),
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
    return VisitorReturnFlag::ContinueInf;
  }
  void end(INFERENCE & inf){
    std::cout<<"value "<<inf.value()<<" bound "<<inf.bound()<<"\n";
  }

  void addLog(const std::string & logName){}
  void log(const std::string & logName,const double logValue){
    if((iteration_)%visithNth_==0){
      std::cout<<logName<<" "<<logValue<<"\n";
    }
  }


private:
  size_t iteration_;
  size_t visithNth_;
  bool   multiline_;
};

template<class INFERENCE>
class ExplicitVerboseVisitor{
public:
   ExplicitVerboseVisitor(const size_t visithNth=1,const bool multiline=false)
   :  iteration_(0),
      visithNth_(visithNth),
      multiline_(multiline){
   }
   void begin(INFERENCE & inf, const typename INFERENCE::ValueType value, const typename INFERENCE::ValueType bound){
      std::cout<<"begin: value "<< value <<" bound "<< bound <<"\n";
      ++iteration_;
   }
   size_t operator()(INFERENCE & inf, const typename INFERENCE::ValueType value, const typename INFERENCE::ValueType bound){
      if((iteration_)%visithNth_==0){
         std::cout<<"step: "<<iteration_<<" value "<< value <<" bound "<< bound <<"\n";
      }
      ++iteration_;
      return VisitorReturnFlag::ContinueInf;
   }
   void end(INFERENCE & inf, const typename INFERENCE::ValueType value, const typename INFERENCE::ValueType bound){
      std::cout<<"value "<< value <<" bound "<< bound <<"\n";
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
    const bool   verbose=true,
    const bool   multiline=true,
    const double timeLimit=std::numeric_limits<double>::infinity(),
    const double gapLimit=0.0
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
    verbose_(verbose),
    multiline_(multiline),
    timeLimit_(timeLimit),
    gapLimit_(gapLimit),
    totalTime_(0.0)
  {
    // allocate all protocolated items
    ctime_    = & protocolMap_["ctime"]    ;
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
      const ValueType val   =inf.value();
      const ValueType bound   =inf.bound();
      const double  t     = timer_.elapsedTime();
         times_->push_back(t);
         values_->push_back(val);
         bounds_->push_back(bound);
         iterations_->push_back(double(iteration_));

         for(size_t el=0;el<extraLogs_.size();++el){
          protocolMap_[extraLogs_[el]].push_back(  std::numeric_limits<double>::quiet_NaN() );
         }

         // increment total time
         totalTime_+=t;
         if(verbose_){
         std::cout<<"step: "<<iteration_<<" value "<<val<<" bound "<<bound<<" [ "<<totalTime_ << "]" <<"\n";
         }

         // check if gap limit reached
      if(std::fabs(bound - val) <= gapLimit_){
           if(verbose_)
              std::cout<<"gap limit reached\n";
           // restart timer
           timer_.reset();
           timer_.tic();
           return VisitorReturnFlag::StopInfBoundReached;
         }
      // check if time limit reached
         if(totalTime_ > timeLimit_) {
           if(verbose_)
              std::cout<<"timeout reached\n";
           // restart timer
           timer_.reset();
           timer_.tic();
           return VisitorReturnFlag::StopInfTimeout;
         }
         // restart timer
         timer_.reset();
         timer_.tic();
      }
      ++iteration_;
      return VisitorReturnFlag::ContinueInf;
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


  void addLog(const std::string & logName){
    protocolMap_[logName]=std::vector<double>();
    extraLogs_.push_back(logName);
  }
  void log(const std::string & logName,const double logValue){
    if((iteration_)%visithNth_==0){
      timer_.toc();
      if(verbose_){
        std::cout<<logName<<" "<<logValue<<"\n";
      }
      protocolMap_[logName].back()=logValue;        
      // start timer
      timer_.tic();
    }
  }

  // timing visitor specific interface

  const std::map< std::string, std::vector<double  > > & protocolMap()const{
    return protocolMap_;
  }

  const std::vector<double> & getConstructionTime()const{
    return *ctime_;
  }
  const std::vector<double> & getTimes      ()const{
    return *times_;
  }
  const std::vector<double> & getValues     ()const{
    return *values_;
  }
  const std::vector<double> & getBounds     ()const{
    return *bounds_;
  }
  const std::vector<double> & getIterations   ()const{
    return *iterations_;
  } 


private:

  std::map< std::string, std::vector<double  > >  protocolMap_;
  std::vector<std::string> extraLogs_;
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
  double gapLimit_;
  double totalTime_;
};

template<class INFERENCE>
class ExplicitTimingVisitor{
public:
   typedef typename  INFERENCE::ValueType ValueType;

   ExplicitTimingVisitor(
      const size_t visithNth=1,
      const size_t reserve=0,
      const bool   verbose=true,
      const bool   multiline=true,
      const double timeLimit=std::numeric_limits<double>::infinity(),
      const double gapLimit=0.0
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
      verbose_(verbose),
      multiline_(multiline),
      timeLimit_(timeLimit),
      gapLimit_(gapLimit),
      totalTime_(0.0)
   {
      // allocate all protocolated items
      ctime_      = & protocolMap_["ctime"]    ;
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

   void begin(INFERENCE & inf, const typename INFERENCE::ValueType value, const typename INFERENCE::ValueType bound){
      // stop timer which measured time from
      // constructor call to this "begin" call
      timer_.toc();
      // store values bound time and iteration number
      ctime_->push_back(timer_.elapsedTime());
      times_->push_back(0);
      values_->push_back(value);
      bounds_->push_back(bound);
      iterations_->push_back(double(iteration_));

      // print step
      if(verbose_)
         std::cout<<"value "<<value<<" bound "<<bound<<"\n";
      // increment iteration
      ++iteration_;
      // restart timer
      timer_.reset();
      timer_.tic();
   }

   size_t operator()(INFERENCE & inf, const typename INFERENCE::ValueType value, const typename INFERENCE::ValueType bound){

      if(iteration_%visithNth_==0){
         // stop timer
         timer_.toc();

         // store values bound time and iteration number
         const double   t     = timer_.elapsedTime();
         times_->push_back(t);
         values_->push_back(value);
         bounds_->push_back(bound);
         iterations_->push_back(double(iteration_));
         // increment total time
         totalTime_+=t;
         if(verbose_){
         std::cout<<"step: "<<iteration_<<" value "<<value<<" bound "<<bound<<" [ "<<totalTime_ << "]" <<"\n";
         }

         // check if gap limit reached
         if(std::fabs(bound - value) <= gapLimit_){
           if(verbose_)
              std::cout<<"gap limit reached\n";
           // restart timer
           timer_.reset();
           timer_.tic();
           return VisitorReturnFlag::StopInfBoundReached;
         }
         // check if time limit reached
         if(totalTime_ > timeLimit_) {
           if(verbose_)
              std::cout<<"timeout reached\n";
           // restart timer
           timer_.reset();
           timer_.tic();
           return VisitorReturnFlag::StopInfTimeout;
         }
         // restart timer
         timer_.reset();
         timer_.tic();
      }
      ++iteration_;
      return VisitorReturnFlag::ContinueInf;
   }


   void end(INFERENCE & inf, const typename INFERENCE::ValueType value, const typename INFERENCE::ValueType bound){
      // stop timer
      timer_.toc();
      // store values bound time and iteration number
      times_->push_back(timer_.elapsedTime());
        values_->push_back(value);
        bounds_->push_back(bound);
        iterations_->push_back(double(iteration_));
        if(verbose_){
         std::cout<<"value "<<value<<" bound "<<bound<<"\n";
        }
   }


   // timing visitor specific interface

   const std::map< std::string, std::vector<double  > > & protocolMap()const{
      return protocolMap_;
   }

   const std::vector<double> & getConstructionTime()const{
      return *ctime_;
   }
   const std::vector<double> & getTimes         ()const{
      return *times_;
   }
   const std::vector<double> & getValues        ()const{
      return *values_;
   }
   const std::vector<double> & getBounds        ()const{
      return *bounds_;
   }
   const std::vector<double> & getIterations    ()const{
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
   double gapLimit_;
   double totalTime_;
};

}
}

#endif //OPENGM_VISITOR_HXX
