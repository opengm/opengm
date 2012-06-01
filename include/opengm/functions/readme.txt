1. Minimum Interface for a Function
2. Registering a function for the OpenGM file format
3. Built-in functions
4. Serialization and de-serialization


/////////////////////////////////////////////////////////////
1. Minimum Interface for a Function
/////////////////////////////////////////////////////////////

template<class T>
class CustomFunction : public opengm::FunctionBase<Function<T>, T> {
   typedef T ValueType;

   template<class Iterator>
       ValueType operator()(Iterator) const; // function evaluation
   size_t shape(const size_t) const; // number of labels of the indicated input variable
   size_t dimension() const; // number of input variables
   size_t size(); // number of parameters
}


/////////////////////////////////////////////////////////////
2. Registering a function for the OpenGM file format
/////////////////////////////////////////////////////////////

To add a custom function to the OpenGM file format, one needs to specialize
the class template 'FunctionRegistration':

template<class T>
struct FunctionRegistration<CustomFunction<T> > {
   enum ID { Id=1 };
};

The IDs of built-in function types start at
opengm::FUNCTION_TYPE_ID_OFFSET  = 16000.
Custom functions should have smaller IDs.


/////////////////////////////////////////////////////////////
3. Built-in functions
/////////////////////////////////////////////////////////////

This list contains only those functions that can be stored to a file.
Note that there are functions, such as 'ViewFunction', that cannot be stored.

Function Name                                FunctionRegistration Id

opengm::Marray                               16000
opengm::SparseMarray                         16001
opengm::AbsoluteDifferenceFunction           16002
opengm::TruncatedAbsoluteDifferenceFunction  16003
opengm::SquaredDifferenceFunction            16004
opengm::TruncatedSquaredDifferenceFunction   16005
opengm::Potts                                16006
opengm::PottsN                               16007
opengm::ConstantFunction                     16008
opengm::StaticSingleSideFunction             16009
opengm::DynamicSingleSideFunction            16010
opengm::PottsG                               16011


/////////////////////////////////////////////////////////////
4. Serialization and de-serialization
/////////////////////////////////////////////////////////////

To add a custom function to the OpenGM file format, one also needs to specialize
the 'FunctionSerialization' class template:

template<class T>
class FunctionSerialization<CustomFunction<T> > {
public:
   typedef typename Function<T>::ValueType ValueType;
   
   static size_t indexSequenceSize(const Function<T> &); // number of indices to be stored
   static size_t valueSequenceSize(const Function<T> &); // number of values to be stored

   template<class INDEX_OUTPUT_ITERATOR, class VALUE_OUTPUT_ITERATOR>
      static void serialize(const Function<T&, INDEX_OUTPUT_ITERATOR, VALUE_OUTPUT_ITERATOR);
  template<class INDEX_INPUT_ITERATOR ,class VALUE_INPUT_ITERATOR>
     static void deserialize(INDEX_INPUT_ITERATOR, VALUE_INPUT_ITERATOR, Function<T>&);
};

