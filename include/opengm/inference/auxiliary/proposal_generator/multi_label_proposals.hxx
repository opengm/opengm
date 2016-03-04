#ifndef OPENGM_MULTI_LABEL_PROPOSAL_HXX
#define OPENGM_MULTI_LABEL_PROPOSAL_HXX



namespace opengm{
    namespace proposals{


        template<class GM, class ACC>
        class LabelHistory{
        public:
            typedef ACC AccumulationType;
            typedef GM GraphicalModelType;
            OPENGM_GM_TYPE_TYPEDEFS;
            typedef std::vector<LabelType > LabelVec;
            typedef std::vector<ValueType > ValueVec;
            typedef std::vector< LabelVec > LabelVecVec;

            LabelHistory(const GM & gm, size_t size)
            :   gm_(gm),
                labels_(size,LabelVec(gm.numberOfVariables)),
                valueVec_(size),
                topIndex_(0),
                entries_(0){
            }

            LabelVec & topArg(){
                return labels_[topIndex_];
            }

            LabelVec & topValue(){
                return valueVec_[topIndex_];
            }

            void setTopValue(const ValueType & value){
                valueVec_[topIndex_]=value;
            }

            LabelVec & topArg(int histIndex){
                int i = topIndex_ - histIndex;
                if(i>=0)
                    return labels_[i];
                else
                    return labels_[labels_.size()-1];
            }

            LabelVec & topArg(int histIndex){
                return valueVec_[topIndex_];
            }

            void nextTop(){
                entries_ = std::min(++entries_, int(labels_.size()));
                topIndex_ = topIndex_+1 < labels_.size() ? topIndex_+1 : 0;
            }

            size_t histSize()const{
                return entries_;
            }

            const GM & gm_;
            LabelVecVec labels_;
            ValueVec valueVec_;
            int topIndex_;
            int entries_;
        };


        template<class GM, class ACC>
        class AlphaExpansion{
            
        public:

            typedef ACC AccumulationType;
            typedef GM GraphicalModelType;
            OPENGM_GM_TYPE_TYPEDEFS;

            typedef LabelHistory<GM, ACC> History;
            typedef std::vector<LabelType > LabelVec;
            typedef std::vector<ValueType > ValueVec;
            typedef std::vector< LabelVec > LabelVecVec;

            class Parameter{
            public:
                Parameter(
                    size_t          r=0,
                    size_t          alphaIncrement=1,
                    bool            randomAlphaToLabel=false,
                    bool            autoSeed=true,
                    int             seed=-1
                )
                :
                    r_(r),
                    alphaIncrement_(alphaIncrement),
                    randomAlphaToLabel_(randomAlphaToLabel),
                    autoSeed_(autoSeed),
                    seed_(seed){
                }
                size_t          r_;
                size_t          alphaIncrement_;
                bool            randomAlphaToLabel_;
                bool            autoSeed_;
                unsigned int    seed_;
            };   

            // interface        
            AlphaExpansion(const GM & gm, Parameter & param & Parameter())
            :   gm_(gm),
                param_(param),
                maxLabel_(gm.maxNumberOfLabels()){

                for(size_t i=0; i<maxLabel_; i)
                    alphaToLabel_[i] = i;
                if(param.randomAlphaToLabel_){
                    if(!param_.seed_>=0 || param_.autoSeed_){
                        if(autoSeed_){
                            std::srand ( unsigned ( std::time(0) ) );
                        }
                        else{
                            std::srand ( unsigned ( param_.seed_ ) );
                        }
                    }
                    std::random_shuffle ( myvector.begin(), myvector.end() );
                }
            }

            size_t numProposals(){
                return param_.r_*2 + 1;
            }

            Int64Type iterPerRound()const{
                return static_cast<Int64Type>( (maxLabel_ / alphaIncrement_) + 1);
            }

            size_t requiredHistorySize()const{
                return 0;
            }

            void reset(){
                currentAlpha_ = 0;
            } 

            void getProposals(
                const LabelVec & current,
                const History  & bestHistory,
                const bool improvementViaLastProposal,
                LabelVecVec & proposals
            ){
                const Int64Type sl = -1 * param_.r_ + currentAlpha_;
                const Int64Type el =  1 + param_.r_ + currentAlpha_;
                for(size_t vi=0; vi<gm_.numberOfVariables(); ++vi){
                    for(Int64Type a=mr, pi=0 ; a<el; ++a,++pi){
                        if(a>0 && alphaToLabel_[a]<gm_.maxNumberOfLabels(vi))
                            proposals[pi][vi] = alphaToLabel_[a];
                        else
                            proposals[pi][vi] = current[vi];
                    }
                }
                currentAlpha_+=alphaIncrement_;
            }



        private:
            const GM & gm_;
            Parameter param_;
            std::vector<LabelType> alphaToLabel_;
            size_t maxLabel_;
            LabelType currentAlpha_;
        };

    }
}



#endif /*OPENGM_MULTI_LABEL_PROPOSAL_HXX*/
