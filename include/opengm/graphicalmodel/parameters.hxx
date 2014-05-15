#ifndef OPENGM_PARAMETERS
#define OPENGM_PARAMETERS


namespace opengm{

   template<class T,class I>
   class Parameters{
   public:
      typedef T ValueType;
      typedef I IndexType;


      Parameters(const IndexType numberOfParameters=0)
      : params_(numberOfParameters){

      }

      ValueType getParameter(const size_t pi)const{
         OPENGM_ASSERT_OP(pi,<,params_.size());
         return params_[pi];
      }

      void setParameter(const size_t pi,const ValueType value){
         OPENGM_ASSERT_OP(pi,<,params_.size());
         params_[pi]=value;
      }

      ValueType operator[](const size_t pi)const{
         return getParameter(pi);
      }

      size_t numberOfParameters()const{
         return params_.size();
      }

   private:

      std::vector<ValueType> params_;
   };
}


#endif /* OPENGM_PARAMETERS */