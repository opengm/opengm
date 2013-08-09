#ifndef ARGUMENT_EXECUTER_HXX_
#define ARGUMENT_EXECUTER_HXX_

namespace opengm {

namespace interface {

/*********************
 * class definitions *
 *********************/

template <class IO>
class ArgumentExecuter {
protected:
   class AdapterBase {
   public:
      virtual ~AdapterBase();
      virtual void printHelp(std::ostream& stream, bool verbose) const = 0;
      virtual void read() const = 0;
   };

   template<typename ARGUMENT>
   class Adapter : public AdapterBase {
   protected:
      IO& io_;
      ARGUMENT* argument_;
   public:
      explicit Adapter(IO& ioIn, ARGUMENT* argumentIn);
      virtual void printHelp(std::ostream& stream, bool verbose) const;
      virtual void read() const;
      virtual ~Adapter();
   };

   IO& io_;
   std::vector<AdapterBase*> arguments_;
   size_t maxArguments_;
public:
   ArgumentExecuter(IO& ioIn, size_t maxArgumentsIn = 100);
   template<typename ARGUMENT>
   ARGUMENT& addArgument(const ARGUMENT& argumentIn);
   size_t size() const;
   void printHelp(std::ostream& stream, bool verbose);
   void read();
   ~ArgumentExecuter();
};

/***********************
 * class documentation *
 ***********************/
//TODO add documentation

/******************
 * implementation *
 ******************/

template <class IO>
inline ArgumentExecuter<IO>::AdapterBase::~AdapterBase() {

}

template <class IO>
template<typename ARGUMENT>
inline ArgumentExecuter<IO>::Adapter<ARGUMENT>::Adapter(IO& ioIn, ARGUMENT* argumentIn)
   : io_(ioIn), argument_(argumentIn) {

}

template <class IO>
template<typename ARGUMENT>
inline void ArgumentExecuter<IO>::Adapter<ARGUMENT>::printHelp(std::ostream& stream, bool verbose) const {
   argument_->printHelp(stream, verbose);
}

template <class IO>
template<typename ARGUMENT>
inline void ArgumentExecuter<IO>::Adapter<ARGUMENT>::read() const {
   io_.read(*argument_);
}

template <class IO>
template<typename ARGUMENT>
inline ArgumentExecuter<IO>::Adapter<ARGUMENT>::~Adapter() {
   delete argument_;
}

template <class IO>
inline ArgumentExecuter<IO>::ArgumentExecuter(IO& ioIn, size_t maxArgumentsIn)
   : io_(ioIn), maxArguments_(maxArgumentsIn) {
   arguments_.reserve(maxArguments_);
}

template <class IO>
template<typename ARGUMENT>
inline ARGUMENT& ArgumentExecuter<IO>::addArgument(const ARGUMENT& argumentIn) {
   if(arguments_.size() >= maxArguments_) {
      throw RuntimeError("To many arguments added to ArgumentExecuter.");
   }
   ARGUMENT* argument = new ARGUMENT(argumentIn);
   arguments_.push_back(new Adapter<ARGUMENT>(io_, argument));
   return *argument;
}

template <class IO>
inline size_t ArgumentExecuter<IO>::size() const {
   return arguments_.size();
}

template <class IO>
inline void ArgumentExecuter<IO>::printHelp(std::ostream& stream, bool verbose) {
   for (typename std::vector<AdapterBase*>::iterator iter = arguments_.begin(); iter != arguments_.end(); ++iter) {
      (*iter)->printHelp(stream, verbose);
   }
}

template <class IO>
inline void ArgumentExecuter<IO>::read() {
   for (typename std::vector<AdapterBase*>::iterator iter = arguments_.begin(); iter != arguments_.end(); ++iter) {
      (*iter)->read();
   }
}

template <class IO>
inline ArgumentExecuter<IO>::~ArgumentExecuter(){
   for (typename std::vector<AdapterBase*>::iterator i = arguments_.begin(); i != arguments_.end(); ++i) {
      delete *i;
   }
}

} // namespace interface

} // namespace opengm

#endif /* ARGUMENT_EXECUTER_HXX_ */
