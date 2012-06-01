#ifndef LIBDAI_JUNCTIONTREE_CALLER
#define LIBDAI_JUNCTIONTREE_CALLER

#include <opengm/opengm.hxx>
#include <opengm/inference/external/libdai/junction_tree.hxx>

#include "inference_caller_base.hxx"
#include "../argument/argument.hxx"

namespace opengm {

namespace interface {

template <class IO, class GM, class ACC>
class LibDaiJunctionTreeCaller : public InferenceCallerBase<IO, GM, ACC> {
protected:
   using InferenceCallerBase<IO, GM, ACC>::addArgument;
   using InferenceCallerBase<IO, GM, ACC>::io_;
   using InferenceCallerBase<IO, GM, ACC>::infer;
   virtual void runImpl(GM& model, StringArgument<>& outputfile, const bool verbose);
   typedef external::libdai::JunctionTree<GM, ACC> LibDai_JunctionTree;
   typedef typename LibDai_JunctionTree::VerboseVisitorType VerboseVisitorType;
   typedef typename LibDai_JunctionTree::EmptyVisitorType EmptyVisitorType;
   typedef typename LibDai_JunctionTree::TimingVisitorType TimingVisitorType;
   typename LibDai_JunctionTree::Parameter jtParameter_;
   std::string selectedUpdateRule_;
   std::string selectedHeuristic_;
public:
   const static std::string name_;
   LibDaiJunctionTreeCaller(IO& ioIn);
};

template <class IO, class GM, class ACC>
inline LibDaiJunctionTreeCaller<IO, GM, ACC>::LibDaiJunctionTreeCaller(IO& ioIn)
   : InferenceCallerBase<IO, GM, ACC>(name_, "detailed description of LibDaiJunctionTreeCaller caller...", ioIn) {
   addArgument(Size_TArgument<>(jtParameter_.verbose_, "", "verboseLevel", "Libdai verbose level", size_t(jtParameter_.verbose_)));
   std::vector<std::string> possibleUpdateRule;
   possibleUpdateRule.push_back(std::string("HUGIN"));
   possibleUpdateRule.push_back(std::string("SHSH"));
   addArgument(StringArgument<>(selectedUpdateRule_, "", "updateRule", "selects the update rule", possibleUpdateRule.at(0), possibleUpdateRule));
   std::vector<std::string> possibleHeuristic;
   possibleHeuristic.push_back(std::string("MINFILL"));
   possibleHeuristic.push_back(std::string("WEIGHTEDMINFILL"));
   possibleHeuristic.push_back(std::string("MINWEIGHT"));
   possibleHeuristic.push_back(std::string("MINNEIGHBORS"));
   addArgument(StringArgument<>(selectedHeuristic_, "", "heuristic", "selects heuristic", possibleHeuristic.at(0), possibleHeuristic));
}

template <class IO, class GM, class ACC>
inline void LibDaiJunctionTreeCaller<IO, GM, ACC>::runImpl(GM& model, StringArgument<>& outputfile, const bool verbose) {
   std::cout << "running LibDaiJunctionTree caller" << std::endl;

   if(selectedUpdateRule_ == std::string("HUGIN")) {
     jtParameter_.updateRule_= LibDai_JunctionTree::HUGIN;
   }
   else if(selectedUpdateRule_ == std::string("SHSH")) {
     jtParameter_.updateRule_= LibDai_JunctionTree::SHSH;
   } 
   else {
     throw RuntimeError("Unknown update rule for libdai-junction-tree");
   }
   
   if(selectedHeuristic_ == std::string("MINFILL")) {
     jtParameter_.heuristic_= LibDai_JunctionTree::MINFILL;
   }
   else if(selectedHeuristic_ == std::string("WEIGHTEDMINFILL")) {
     jtParameter_.heuristic_= LibDai_JunctionTree::WEIGHTEDMINFILL;
   }
   else if(selectedHeuristic_ == std::string("MINWEIGHT")) {
     jtParameter_.heuristic_= LibDai_JunctionTree::MINWEIGHT;
   } 
   else if(selectedHeuristic_ == std::string("MINNEIGHBORS")) {
     jtParameter_.heuristic_= LibDai_JunctionTree::MINNEIGHBORS;
   } 
   else {
     throw RuntimeError("Unknown update rule for libdai-junction-tree");
   }

   this-> template infer<LibDai_JunctionTree, TimingVisitorType, typename LibDai_JunctionTree::Parameter>(model, outputfile, verbose, jtParameter_);

}

template <class IO, class GM, class ACC>
const std::string LibDaiJunctionTreeCaller<IO, GM, ACC>::name_ = "LIBDAI-JUNCTIONTREE";

} // namespace interface

} // namespace opengm

#endif /* LIBDAI_JUNCTIONTREE_CALLER */
