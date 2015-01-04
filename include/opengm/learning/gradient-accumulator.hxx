#ifndef OPENGM_LEARNING_GRADIENT_ACCUMULATOR_H__
#define OPENGM_LEARNING_GRADIENT_ACCUMULATOR_H__

namespace opengm {
namespace learning {

/**
 * Model function visitor to accumulate the gradient for each model weight, 
 * given a configuration.
 */
template <typename ModelWeights, typename ConfigurationType>
class GradientAccumulator {
    typedef typename ConfigurationType::const_iterator ConfIter;

public:

    /**
     * How to accumulate the gradient on the provided ModelWeights.
     */
    enum Mode {

        Add,

        Subtract
    };

    /**
     * @param gradient
     *              ModelWeights reference to store the gradients. Gradient 
     *              values will only be added (or subtracted, if mode == 
     *              Subtract), so you have to make sure gradient is properly 
     *              initialized to zero.
     *
     * @param configuration
     *              Configuration of the variables in the model, to evaluate the 
     *              gradient for.
     *
     * @param mode
     *              Add or Subtract the weight gradients from gradient.
     */
    GradientAccumulator(ModelWeights& gradient, const ConfigurationType& configuration, Mode mode = Add) :
        _gradient(gradient),
        _configuration(configuration),
        _mode(mode) {}

    template <typename Iterator, typename FunctionType>
    void operator()(Iterator begin, Iterator end, const FunctionType& function) {

        typedef opengm::SubsetAccessor<Iterator, ConfIter> Accessor;
        typedef opengm::AccessorIterator<Accessor, true> Iter;
        const Accessor accessor(begin, end, _configuration.begin());

        for (int i = 0; i < function.numberOfWeights(); i++) {

            int index = function.weightIndex(i);

            double g = function.weightGradient(i, Iter(accessor, 0));

            if (_mode == Add)
                _gradient[index] += g;
            else
                _gradient[index] -= g;
        }
    }

private:

    ModelWeights& _gradient;
    const ConfigurationType& _configuration;
    Mode _mode;
};


template<class GM, class LABEL_ITER>
struct FeatureAccumulator{

    typedef typename GM::LabelType LabelType;
    typedef typename GM::IndexType IndexType;
    typedef typename GM::ValueType ValueType;
    


    FeatureAccumulator(const size_t nW)
    :   accWeights_(nW),
        gtLabel_(),
        mapLabel_(){
            
        for(size_t i=0; i<accWeights_.size(); ++i){
            accWeights_[i] = 0.0;
        }
    }

    void setLabels(const LABEL_ITER gtLabel, const LABEL_ITER mapLabel){
        gtLabel_ = gtLabel;
        mapLabel_  = mapLabel;
    }

    void resetWeights(){
        //accFeaturesGt_ = 0.0;
        //accWeights_ = 0.0;
        for(size_t i=0; i<accWeights_.size(); ++i){
            accWeights_[i] = 0.0;
        }
    }
    double getWeight(const size_t wi)const{
        return accWeights_[wi];
    }
    template<class Iter, class F>
    void operator()(Iter begin, Iter end, const F & f){

        typedef opengm::SubsetAccessor<Iter, LABEL_ITER> Accessor;
        typedef opengm::AccessorIterator<Accessor, true> Iter;


        // get the number of weights_
        const size_t nWeights = f.numberOfWeights();
        if(nWeights>0){
            // loop over all weights
            for(size_t wi=0; wi<nWeights; ++wi){
                // accumulate features for both labeling
                const size_t gwi = f.weightIndex(wi);


                const Accessor accessorGt(begin, end, gtLabel_);
                const Accessor accessorMap(begin, end, mapLabel_);
                
                // for test label
                accWeights_[gwi] += f.weightGradient(wi, Iter(accessorMap, 0));

                // for gt label
                accWeights_[gwi] -= f.weightGradient(wi, Iter(accessorGt, 0));
            }
        }
    }

    void accumulateFromOther(const FeatureAccumulator & otherAcc){
        for(size_t i=0; i<accWeights_.size(); ++i){
            accWeights_[i] += otherAcc.accWeights_[i];
        }
    }

    void accumulateModelFeatures(
        const GM & gm, 
        const LABEL_ITER & gtLabel,
        const LABEL_ITER & mapLabel
    ){
        gtLabel_ = gtLabel;
        mapLabel_  = mapLabel;

        // iterate over all factors
        // and accumulate features
        for(size_t fi=0; fi<gm.numberOfFactors(); ++fi){
            gm[fi].callViFunctor(*this);
        }
    }
    opengm::learning::Weights<double>  accWeights_;
    LABEL_ITER gtLabel_;
    LABEL_ITER mapLabel_;
};




}} // namespace opengm::learning

#endif // OPENGM_LEARNING_GRADIENT_ACCUMULATOR_H__

