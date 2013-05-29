#ifndef ASTAR_CALLER_HXX_
#define ASTAR_CALLER_HXX_

#include <opengm/opengm.hxx>
#include <opengm/inference/astar.hxx>

#include "inference_caller_base.hxx"
#include "../argument/argument.hxx"

namespace opengm {

namespace interface {

template <class IO, class GM, class ACC>
class AStarCaller : public InferenceCallerBase<IO, GM, ACC, AStarCaller<IO, GM, ACC> > {
public:
   typedef InferenceCallerBase<IO, GM, ACC, AStarCaller<IO, GM, ACC> > BaseClass;
   typedef AStar<GM, ACC> A_Star;
   typedef typename A_Star::VerboseVisitorType VerboseVisitorType;
   typedef typename A_Star::EmptyVisitorType EmptyVisitorType;
   typedef typename A_Star::TimingVisitorType TimingVisitorType;

   const static std::string name_;
   AStarCaller(IO& ioIn);
   virtual ~AStarCaller();
protected:
   using BaseClass::addArgument;
   using BaseClass::io_;
   using BaseClass::infer;
   typedef typename BaseClass::OutputBase OutputBase;
   virtual void runImpl(GM& model, OutputBase& output, const bool verbose);

   typename A_Star::Parameter astarParameter_;
   std::string selectedHeuristic_;
};

template <class IO, class GM, class ACC>
inline AStarCaller<IO, GM, ACC>::AStarCaller(IO& ioIn)
   : BaseClass(name_, "detailed description of AStar caller...", ioIn) {
   addArgument(Size_TArgument<>(astarParameter_.maxHeapSize_, "", "maxheapsize", "maximum size of the heap", astarParameter_.maxHeapSize_));
   addArgument(Size_TArgument<>(astarParameter_.numberOfOpt_, "", "numopt", "number of optimizations", astarParameter_.numberOfOpt_));
   addArgument(ArgumentBase<typename A_Star::ValueType>(astarParameter_.objectiveBound_, "", "objectivebound", "boundary for the objective", astarParameter_.objectiveBound_));
   std::vector<std::string> possibleHeuristics;
   possibleHeuristics.push_back("DEFAULT");
   possibleHeuristics.push_back("FAST");
   possibleHeuristics.push_back("STANDARD");
   addArgument(StringArgument<>(selectedHeuristic_, "", "heuristic", "selects the desired cost heuristic", possibleHeuristics.at(0), possibleHeuristics));
   // TODO is astarParameter_.treeFactorIds_ really always of type std::vector<size_t> not IndexType? Bug in astar.hxx???
   addArgument(VectorArgument<std::vector<size_t> >(astarParameter_.treeFactorIds_, "", "factorids", "location of the file containing a vector which specifies the desired tree factor ids", false));
   addArgument(VectorArgument<std::vector<typename A_Star::IndexType> >(astarParameter_.nodeOrder_, "", "nodeorder", "location of the file containing a vector which specifies the desired node order", false));
}

template <class IO, class GM, class ACC>
inline AStarCaller<IO, GM, ACC>::~AStarCaller() {

}

template <class IO, class GM, class ACC>
inline void AStarCaller<IO, GM, ACC>::runImpl(GM& model, OutputBase& output, const bool verbose) {
   std::cout << "running AStar caller" << std::endl;

   if(selectedHeuristic_ == "DEFAULT") {
     astarParameter_.heuristic_= A_Star::Parameter::DEFAULTHEURISTIC;
   } else if(selectedHeuristic_ == "FAST") {
     astarParameter_.heuristic_= A_Star::Parameter::FASTHEURISTIC;
   } else if(selectedHeuristic_ == "STANDARD") {
     astarParameter_.heuristic_= A_Star::Parameter::STANDARDHEURISTIC;
   } else {
     throw RuntimeError("Unknown heuristic for AStar");
   }

   this-> template infer<A_Star, TimingVisitorType, typename A_Star::Parameter>(model, output, verbose, astarParameter_);
/*   A_Star astar(model, astarParameter_);

   std::vector<size_t> states;
   std::cout << "Inferring!" << std::endl;
   if(!(astar.infer() == NORMAL)) {
      std::string error("AStar did not solve the problem.");
      io_.errorStream() << error << std::endl;
      throw RuntimeError(error);
   }
   std::cout << "writing states in vector!" << std::endl;
   if(!(astar.arg(states) == NORMAL)) {
      std::string error("AStar could not return optimal argument.");
      io_.errorStream() << error << std::endl;
      throw RuntimeError(error);
   }

   io_.read(outputfile);
   io_.storeVector(outputfile.getValue(), states);*/
}

template <class IO, class GM, class ACC>
const std::string AStarCaller<IO, GM, ACC>::name_ = "ASTAR";

} // namespace interface

} // namespace opengm

#endif /* ASTAR_CALLER_HXX_ */
