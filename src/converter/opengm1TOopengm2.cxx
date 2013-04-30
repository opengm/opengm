#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <map>

#include <opengm/utilities/metaprogramming.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/graphicalmodel/graphicalmodel_hdf5.hxx>
#include <opengm/operations/multiplier.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/functions/explicit_function.hxx>
#include <opengm/functions/potts.hxx>
#include <opengm/functions/pottsn.hxx>
#include <opengm/functions/pottsg.hxx>
#include <opengm/functions/truncated_absolute_difference.hxx>
#include <opengm/functions/truncated_squared_difference.hxx>

#include <boost/shared_ptr.hpp>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

struct ConverterDispatcher {
   virtual void exec(
      std::string sourceFileName, 
      std::string destinationFileName,
      std::string inputModelName,
      std::string outputModelName,
      size_t maxCacheEntries,
      bool verbose
      ) = 0;
};

template<class ValueType, class OperationType>
struct ConcreteConverterDispatcher : ConverterDispatcher {
   virtual void exec(
      std::string sourceFileName, 
      std::string destinationFileName,
      std::string inputModelName,
      std::string outputModelName,
      size_t maxCacheEntries,
      bool verbose
   ) {
      typedef size_t LabelType;
      typedef size_t IndexType;
      typedef opengm::GraphicalModel<
         ValueType,
         OperationType,
         typename opengm::meta::TypeListGenerator< opengm::ExplicitFunction<ValueType> >::type,
         opengm::DiscreteSpace<IndexType, LabelType>,
         false
      > GmType;
      typedef opengm::ExplicitFunction<ValueType> ExplicitFunctionType;
      typedef typename GmType::FunctionIdentifier FunctionIdentifier;

      hid_t file = marray::hdf5::openFile(sourceFileName, 
         marray::hdf5::READ_ONLY,
         marray::hdf5::DEFAULT_HDF5_VERSION);

      marray::Marray<ValueType> in;
      marray::hdf5::load(file, inputModelName, in);

      size_t numberOfVariables = static_cast<IndexType>(in(0));
      size_t numberOfFactors = static_cast<IndexType>(in(1)); 
      GmType gm(opengm::DiscreteSpace<IndexType, LabelType >(&in(2), &in(2+numberOfVariables) ));
      //std::vector<size_t> stateSpaceDimension(&in(2), &in(2+numberOfVariables));
      //GmType gm(stateSpaceDimension.begin(), stateSpaceDimension.end());

      std::vector<std::pair<ExplicitFunctionType, FunctionIdentifier> > cache;

      size_t p = 2+numberOfVariables-1;
      for(size_t j=0; j<numberOfFactors; ++j) {
         ++p;
         size_t numberOfVariablesOfFactor = static_cast<IndexType>(in(p));
         std::vector<IndexType> variableIndicesOfFactor(numberOfVariablesOfFactor);
         std::vector<LabelType> variableDimensionsOfFactor(numberOfVariablesOfFactor);
         size_t tableSize = 1;
         for(size_t k=0; k<numberOfVariablesOfFactor; ++k) {
            ++p;
            variableIndicesOfFactor[k] = static_cast<IndexType>(in(p));
            variableDimensionsOfFactor[k] = gm.numberOfLabels(variableIndicesOfFactor[k]);
            tableSize *= gm.numberOfLabels(variableIndicesOfFactor[k]);
            //variableDimensionsOfFactor[k] = stateSpaceDimension[variableIndicesOfFactor[k]];
            //tableSize *= stateSpaceDimension[variableIndicesOfFactor[k]];
         }
         ExplicitFunctionType ef(variableDimensionsOfFactor.begin(),variableDimensionsOfFactor.end());
         for(size_t k=0; k<tableSize; ++k) {
            ++p;
            ef(k) = in(p);
         }

         // Optionally try to be a bit smarter here and 
         // match pairwise or higher factors that were already encountered
         // (usually corresponding to a regularizer)
         if (numberOfVariablesOfFactor >= 2) {
            // linear scan (perhaps replace by some tree structured thing some time...) of the cache
            bool found=false;
            for (size_t ci=0; ci<cache.size(); ++ci) {
               if (cache[ci].first.size() == ef.size()) {
                  if (std::equal(cache[ci].first.begin(), cache[ci].first.end(), ef.begin())) {
                     found = true;
                     gm.addFactor(cache[ci].second, variableIndicesOfFactor.begin(), variableIndicesOfFactor.end());
                     break;
                  }
               }
            }
            if (found) {
               if (verbose) 
                  std::cout << "H";
               continue;
            } else {
               if (verbose) 
                  std::cout << ".";
            }
         }
         FunctionIdentifier fi = gm.addFunction(ef);
         if(numberOfVariablesOfFactor >= 2 && cache.size() < maxCacheEntries) {
            // keep cache reasonably small
            cache.push_back(std::make_pair(ef, fi));
         }
         gm.addFactor(fi, variableIndicesOfFactor.begin(), variableIndicesOfFactor.end());
      }
      OPENGM_ASSERT(p == in.size()-1);
      opengm::hdf5::save(gm, destinationFileName, outputModelName);
      marray::hdf5::closeFile(file);
      if (verbose)
         std::cout << std::endl 
         << "Done, converted " << numberOfFactors << " factors in " 
         << numberOfVariables << " variables"
         << " using " << cache.size() << " cache entries for aggregation."
         << std::endl;    
   }
};


int main(int argc, char** argv) {
   po::options_description desc("OpenGM legacy HDF5 model file converter");
   desc.add_options()
      ("help,h", "output help message")
      ("verbose,v", "verbose operation")
      ("input", po::value<std::string>(), "source model file (in old HDF5 format) to process")
      ("output", po::value<std::string>(), "target model file to write result to (in new HDF5 format)")
      ("additive,a", "assume additive (log-domain/energy-like) potentials (default)")
      ("multiplicative,m", "assume multiplicative (probability) potentials")
      ("compress,c", po::value<size_t>()->default_value(10), "cache N pairwise/higher-order potentials to match and aggregate (e.g. regularizer terms)")
      ("inputmodelname", po::value<std::string>()->default_value("model"), "name of input model within file")
      ("outputmodelname", po::value<std::string>()->default_value("model"), "name of output model within file")
      ("type", po::value<std::string>()->default_value("double"), "type code (value type of input)")
      ;

   po::variables_map vm;
   po::positional_options_description posopt;
   posopt.add("input", 1);
   posopt.add("output", 1);

   try {
      po::store(po::command_line_parser(argc, argv).
         options(desc).positional(posopt).run(), vm);
      po::notify(vm);
   } catch (po::unknown_option &e) {
      std::cerr << e.what() << std::endl;
      std::cerr << desc << std::endl;
      return 1;
   }
   bool minargs = (vm.count("input") > 0) && (vm.count("output") > 0);

   if (!minargs || vm.count("help")) {
      std::cerr << desc << std::endl;
      return 1;
   }
   const bool verbose = vm.count("verbose") > 0;

   const std::string sourceFileName(vm["input"].as<std::string>());
   const std::string destinationFileName(vm["output"].as<std::string>());

   const std::string inputModelName(vm["inputmodelname"].as<std::string>());
   const std::string outputModelName(vm["outputmodelname"].as<std::string>());
   const size_t maxCacheEntries(vm["compress"].as<size_t>());

   const bool multiplicative = vm.count("multiplicative");
   const bool additive = vm.count("additive");
   if (multiplicative && additive) {
      std::cerr << "Error: conflicting options." << std::endl;
      std::cerr << desc << std::endl;
      return 1;
   }

   typedef boost::shared_ptr<ConverterDispatcher> ptr;
   std::map<std::pair<std::string, bool>, ptr > typemap;
   typemap[std::make_pair("double",false)] = ptr(new ConcreteConverterDispatcher<double, opengm::Adder>());
   typemap[std::make_pair("float",false)] = ptr(new ConcreteConverterDispatcher<float, opengm::Adder>());
   typemap[std::make_pair("int",false)] = ptr(new ConcreteConverterDispatcher<int, opengm::Adder>());
   typemap[std::make_pair("double",true)] = ptr(new ConcreteConverterDispatcher<double, opengm::Multiplier>());
   typemap[std::make_pair("float",true)] = ptr(new ConcreteConverterDispatcher<float, opengm::Multiplier>());
   typemap[std::make_pair("int",true)] = ptr(new ConcreteConverterDispatcher<int, opengm::Multiplier>());

   ptr selected = typemap[std::make_pair(vm["type"].as<std::string>(), multiplicative)];
   if (selected) 
      selected->exec(sourceFileName, destinationFileName, inputModelName, outputModelName, maxCacheEntries, verbose);
   else {
      std::cerr << "Error: type " << vm["type"].as<std::string>() << " unsupported." << std::endl;
      return 1;
   }

   return 0;
}
