#pragma once
#ifndef OPENGM_ACCUMULATION_HXX
#define OPENGM_ACCUMULATION_HXX

#include "opengm/datastructures/fast_sequence.hxx"

/// \cond HIDDEN_SYMBOLS

namespace opengm {
   
template<class Value, class State, class Accumulator>
class Accumulation {
public:
   /// value type of the accumulation
   typedef Value ValueType;
   /// state type of the accumulation
   typedef State LabelType;
   Accumulation();
   ValueType value() const;
   size_t size() const;
   Value neutral() const;
   void state(FastSequence<size_t>&) const;
   template<class OUT_CONTAINER>
   void state(OUT_CONTAINER &) const;
   LabelType state(size_t) const;
   void clear();
   void operator()(const ValueType&, const opengm::FastSequence<size_t>&);
   template<class CONTAINER>
   void operator()(const ValueType&, const CONTAINER&);
   void operator()(const ValueType&);
private:
   ValueType value_;
   opengm::FastSequence<size_t> state_;
};
   
template<class Value, class State, class Accumulator>
inline
Accumulation<Value, State, Accumulator>::Accumulation()
:  value_(Accumulator::template neutral<ValueType>()),
   state_(opengm::FastSequence<size_t>())
{}
   
template<class Value, class State, class Accumulator>
inline typename Accumulation<Value, State, Accumulator>::ValueType
Accumulation<Value, State, Accumulator>::value() const
{
   return(value_);
}
   
template<class Value, class State, class Accumulator>
inline size_t
Accumulation<Value, State, Accumulator>::size() const
{
   return state_.size();
}
   
template<class Value, class State, class Accumulator>
inline typename Accumulation<Value, State, Accumulator>::ValueType
Accumulation<Value, State, Accumulator>::neutral() const
{
   return state_.size();
}
   
template<class Value, class State, class Accumulator>
inline void
Accumulation<Value, State, Accumulator>::state
(
   opengm::FastSequence<size_t>& out
) const
{
   out = state_;
}
   
template<class Value, class State, class Accumulator>
template<class OUT_CONTAINER>
inline void
Accumulation<Value, State, Accumulator>::state
(
   OUT_CONTAINER& out
) const
{
   out.resize(state_.size());
   for(size_t i=0;i<state_.size();++i) {
      out[i]=state_[i];
   }
}
   
template<class Value, class State, class Accumulator>
inline typename Accumulation<Value, State, Accumulator>::LabelType
Accumulation<Value, State, Accumulator>::state
(
   size_t index
) const
{
   return state_(index);
}
   
template<class Value, class State, class Accumulator>
inline void
Accumulation<Value, State, Accumulator>::clear()
{
   Accumulator::neutral(value_);
   state_.resize(0);
}
   
template<class Value, class State, class Accumulator>
inline void
Accumulation<Value, State, Accumulator>::operator()
(
   const ValueType& value,
   const opengm::FastSequence<size_t>& state
) {
   if(Accumulator::bop(value, value_)) {
      state_ = state;
      OPENGM_ASSERT(state_.size()==state.size());
   }
   Accumulator::op(value, value_, value_);
   //OPENGM_ASSERT(state_.size()==state.size());
}
   
template<class Value, class State, class Accumulator>
template<class CONTAINER>
inline void
Accumulation<Value, State, Accumulator>::operator()
(
   const ValueType& value,
   const CONTAINER & state
)
{
   if(Accumulator::bop(value, value_)) {
      state_.resize(state.size());
      for(size_t i=0;i<state.size();++i) {
         state_[i] = state[i];
      }
   }
   Accumulator::op(value, value_, value_);
   OPENGM_ASSERT(state_.size()==state.size());
}
   
template<class Value, class State, class Accumulator>
inline void
Accumulation<Value, State, Accumulator>::operator()
(
   const ValueType& value
)
{
   Accumulator::op(value, value_, value_);
}

} // namespace opengm

/// \endcond

#endif // #ifndef OPENGM_ACCUMULATION_HXX
