#ifndef OPENGM_LEARNING_WEIGHTS
#define OPENGM_LEARNING_WEIGHTS

#include <opengm/opengm.hxx>

namespace opengm{
namespace learning{

    /*
    template<class T>
    class Weights {
    public:

        typedef T ValueType;

        Weights(const size_t numberOfWeights=0)
        :   weights_(numberOfWeights)
        {

        }

        ValueType getWeight(const size_t pi)const{
            OPENGM_ASSERT_OP(pi,<,weights_.size());
            return weights_[pi];
        }

        void setWeight(const size_t pi,const ValueType value){
            OPENGM_ASSERT_OP(pi,<,weights_.size());
            weights_[pi] = value;
        }

        const ValueType& operator[](const size_t pi)const{
            return weights_[pi];
        }

        ValueType& operator[](const size_t pi) {
            return weights_[pi];
        }

        size_t numberOfWeights()const{
            return weights_.size();
        }

        size_t size()const{
            return weights_.size();
        }

    private:

        std::vector<ValueType> weights_;
    };
    */
    template<class T>
    class Weights : public marray::Vector<T>
    {
    public:

        typedef T ValueType;

        Weights(const size_t numberOfWeights=0)
        :   marray::Vector<T>(numberOfWeights)
        {

        }

        ValueType getWeight(const size_t pi)const{
            OPENGM_ASSERT_OP(pi,<,this->size());
            return (*this)[pi];
        }

        void setWeight(const size_t pi,const ValueType value){
            OPENGM_ASSERT_OP(pi,<,this->size());
            (*this)[pi] = value;
        }


        size_t numberOfWeights()const{
            return this->size();
        }

    private:

        //std::vector<ValueType> weights_;
    };


    template<class T>
    class WeightRegularizer{
    public:
        enum RegularizationType{
            NoRegularizer=-1,
            L1Regularizer=1,
            L2Regularizer=2
        };

        WeightRegularizer(const int regularizationNorm, const double lambda=1.0)
        :   regularizationType_(),
            lambda_(lambda){
            if(regularizationNorm==-1){
                regularizationType_ = NoRegularizer;
            }
            else if(regularizationNorm==1){
                regularizationType_ = L1Regularizer;
            }
            else if(regularizationNorm==2){
                regularizationType_ = L2Regularizer;
            }
            else{
                throw opengm::RuntimeError("regularizationNorm must be -1 (NONE), 1 (L1) or 2 (L2)");
            }
        }
        WeightRegularizer(const RegularizationType regularizationType=L2Regularizer, const double lambda=1.0)
        :   regularizationType_(regularizationType),
            lambda_(lambda){

        }

        double lambda()const{
            return lambda_;
        }

        RegularizationType regularizationType()const{
            return regularizationType_;
        }

        int regularizerNorm()const{
            return static_cast<int>(regularizationType_);
        }

        double evaluate(const Weights<T> & weights){
            if(regularizationType_== NoRegularizer){
                return 0.0;
            }
            else if(regularizationType_ == L1Regularizer){
                double val = 0.0;
                for(size_t wi=0; wi<weights.size(); ++wi){
                    val += std::abs(weights[wi]);
                }
                return val*lambda_;
            }
            else { //if(regularizationType_ == L2Regularizer){
                double val = 0.0;
                for(size_t wi=0; wi<weights.size(); ++wi){
                    val += std::pow(weights[wi], 2);
                }
                return val*lambda_;
            }
        }

    private:
        RegularizationType regularizationType_;
        double lambda_;
    };


    template<class T>
    class WeightConstraints{
    public:

        WeightConstraints(const size_t nWeights = 0)
        :   wLowerBounds_(nWeights,-1.0*std::numeric_limits<T>::infinity()),
            wUpperBounds_(nWeights, 1.0*std::numeric_limits<T>::infinity()),
            cLowerBounds_(),
            cUpperBounds_(),
            cOffset_(0),
            cStart_(),
            cSize_(),
            cIndices_(),
            cCoeff_(){

        }
        template<class ITER_LB, class ITER_UB>
        WeightConstraints(ITER_LB lbBegin, ITER_LB lbEnd, ITER_UB ubBegin)
        :   wLowerBounds_(lbBegin,lbEnd),
            wUpperBounds_(ubBegin, ubBegin + std::distance(lbBegin, lbEnd)),
            cLowerBounds_(),
            cUpperBounds_(),
            cOffset_(0),
            cStart_(),
            cSize_(),
            cIndices_(),
            cCoeff_()
        {

        }   
        // query
        size_t numberOfConstraints()const{
            return cStart_.size();
        }

        T weightLowerBound(const size_t wi)const{
            return wLowerBounds_[wi];
        }
        T weightUpperBound(const size_t wi)const{
            return wUpperBounds_[wi];
        }

        const std::vector<T> & weightLowerBounds()const{
            return wLowerBounds_;
        }
        const std::vector<T> & weightUpperBounds()const{
            return wUpperBounds_;
        }


        size_t constraintSize(const size_t ci)const{
            return cSize_[ci];
        }
        T constraintLowerBound(const size_t ci)const{
            return cLowerBounds_[ci];
        }
        T constraintUpperBound(const size_t ci)const{
            return cUpperBounds_[ci];
        }

        const std::vector<size_t> & constraintSizes()const{
            return cLowerBounds_;
        }
        const std::vector<T> & constraintLowerBounds()const{
            return cLowerBounds_;
        }
        const std::vector<T> & constraintUpperBounds()const{
            return cUpperBounds_;
        }

        //  modification
        template<class ITER_LB>
        void setLowerBounds(ITER_LB lbBegin, ITER_LB lbEnd){
            wLowerBounds_.assign(lbBegin, lbEnd);
        }

        template<class ITER_UB>
        void setUpperBounds(ITER_UB ubBegin, ITER_UB ubEnd){
            wUpperBounds_.assign(ubBegin, ubEnd);
        }

        template<class ITER_INDICES, class ITER_COEFF>
        void addConstraint(ITER_INDICES indicesBegin, ITER_INDICES indicesEnd, ITER_COEFF coeffBegin, const T lowerBound, const T upperBound){
            // length of this constraint
            const size_t cSize = std::distance(indicesBegin, indicesEnd);
            // store length of constraint
            cSize_.push_back(cSize);

            // store offset / index in 'cIndices_' and 'cCoeff_'
            cStart_.push_back(cOffset_);

            // increment the cOffset_ for the next constraint which
            // could be added by the user
            cOffset_ +=cSize;

            // copy indices and coefficients
            for( ;indicesBegin!=indicesEnd; ++indicesBegin,++coeffBegin){
                cIndices_.push_back(*indicesBegin);
                cCoeff_.push_back(*coeffBegin);
            }
        }

    private:
        // w upper-lower bound
        std::vector<T> wLowerBounds_;
        std::vector<T> wUpperBounds_;
        // constraints 
        std::vector<T> cLowerBounds_;
        std::vector<T> cUpperBounds_;

        size_t cOffset_;
        std::vector<size_t> cStart_;
        std::vector<size_t> cSize_;
        std::vector<size_t> cIndices_;
        std::vector<T>      cCoeff_;
    };


} // namespace learning
} // namespace opengm






#endif /* OPENGM_LEARNING_WEIGHTS */
