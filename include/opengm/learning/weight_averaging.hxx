#ifndef OPENGM_LEARNING_WEIGHT_AVERAGING_HXX
#define OPENGM_LEARNING_WEIGHT_AVERAGING_HXX



namespace opengm{
namespace learning{


    template<class T>
    class WeightAveraging{
    public:
        WeightAveraging(Weights<T> & weights, int order=2)
        :   weights_(&weights),
            order_(order),
            iteration_(1){
        }
        WeightAveraging()
        :   weights_(NULL),
            order_(2),
            iteration_(1){
        }
        void setWeights(Weights<T> & weights){
            weights_ = &weights;
        }

        template<class U>
        void operator()(const Weights<U> & weights){
            const T t = static_cast<T>(iteration_);
            if(order_ == -1){
                *weights_ = weights;
            }
            else if(order_ == 0){
                throw opengm::RuntimeError("running average is not yet implemented");
            }
            else if(order_==1){
                const T rho = 2.0 / (t + 2.0);
                for(size_t i=0; i<weights_->size(); ++i){
                    (*weights_)[i] =  (*weights_)[i]*(1.0 - rho) + weights[i]*rho;
                }
            }
            else if(order_ == 2){
                const T rho = 6.0 * (t+1.0) / ( (t+2.0)*(2.0*t + 3.0) );
                for(size_t i=0; i<weights_->size(); ++i){
                    (*weights_)[i] =  (*weights_)[i]*(1.0 - rho) + weights[i]*rho;
                }
            }
            else{
                throw opengm::RuntimeError("order must be -1,0,1 or 2");
            }
            ++iteration_;
        }
        const Weights<T> & weights()const{
            return weights_;
        }
    private:
        Weights<T>  * weights_;
        int order_;
        size_t iteration_;
    };



}   // end namespace opengm
}   // end namespace opengm


#endif /*OPENGM_LEARNING_WEIGHT_AVERAGING_HXX*/
