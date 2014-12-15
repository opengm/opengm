#ifndef OPENGM_LEARNING_WEIGHTS
#define OPENGM_LEARNING_WEIGHTS


namespace opengm{
namespace learning{

   template<class T>
   class Weights{
   public:
      typedef T ValueType;

      Weights(const size_t numberOfParameters=0)
      : weights_(numberOfParameters){

      }

      ValueType getWeight(const size_t pi)const{
         OPENGM_ASSERT_OP(pi,<,weights_.size());
         return weights_[pi];
      }

      void setWeight(const size_t pi,const ValueType value){
         OPENGM_ASSERT_OP(pi,<,weights_.size());
         weights_[pi]=value;
      }

      ValueType operator[](const size_t pi)const{
         return getWeight(pi);
      }

      size_t numberOfWeights()const{
         return weights_.size();
      }

   private:

      std::vector<ValueType> weights_;
   };
} // namespace learning
} // namespace opengm


#endif /* OPENGM_LEARNING_WEIGHTS */
