#ifndef DENOISE_HXX
#define	DENOISE_HXX

#include <vector>
#include <map>


#include <opengm/operations/adder.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/opengm.hxx>
#include <opengm/functions/truncated_squared_difference.hxx>


template<class VALUE_TYPE,class PARAM_TYPE>
class Denoise{
   public:
      typedef VALUE_TYPE ValueType;
      typedef PARAM_TYPE ParamType;
      typedef vigra::BImage ImageType;
      typedef vigra::BImage ResultImageType;
      typedef typename opengm::meta::TypeListGenerator<
         opengm::TruncatedSquaredDifferenceFunction<ValueType> ,
         opengm::ExplicitFunction<ValueType>
      >::type FunctionTypeList;
      typedef opengm::GraphicalModel<
         ValueType,
         opengm::Adder,
         FunctionTypeList
      > GraphicalModelType;
      typedef typename GraphicalModelType::LabelType LabelType;
      Denoise(const ImageType & ,const ImageType &,ParamType ,ParamType,const bool);
      void buildModel(GraphicalModelType &)const;
      template<class ITERATOR>
      void statesToImage(ITERATOR ,ITERATOR,ResultImageType & )const;
      const std::vector<LabelType> & startingPoint()const;
   private:
      vigra::BImage image_;
      vigra::BImage mask_;
      ParamType lambda_;
      ParamType truncateAt_;
      std::vector<LabelType> startingPoint_;
      bool verbose_;
};



template<class VALUE_TYPE,class PARAM_TYPE>
Denoise<VALUE_TYPE,PARAM_TYPE>::Denoise
(
   const typename Denoise<VALUE_TYPE,PARAM_TYPE>::ImageType& image,
   const typename Denoise<VALUE_TYPE,PARAM_TYPE>::ImageType& mask,
   PARAM_TYPE lambda,
   PARAM_TYPE truncateAt,
   const bool verbose
): image_(image),
   mask_(mask),
   lambda_(lambda),
   truncateAt_(truncateAt),
   verbose_(verbose){
   startingPoint_.resize(image_.width()*image_.height());
   for(size_t x=0;x<image_.width();++x)
   for(size_t y=0;y<image_.height();++y){
      startingPoint_[x+y*image_.width()]=image_(x,y);
   }

}

template<class VALUE_TYPE,class PARAM_TYPE>
inline const std::vector<typename Denoise<VALUE_TYPE,PARAM_TYPE>::LabelType> &
Denoise<VALUE_TYPE,PARAM_TYPE>::startingPoint()const{
   return startingPoint_;
}

template<class VALUE_TYPE,class PARAM_TYPE>
void Denoise<VALUE_TYPE,PARAM_TYPE>::buildModel
(
   typename Denoise<VALUE_TYPE,PARAM_TYPE>::GraphicalModelType& gm
)const{
   {
      std::vector<size_t> nos(image_.width()*image_.height(),256);
      gm.assign(opengm::DiscreteSpace<size_t,size_t> (nos.begin(),nos.end()));
   }
   typedef typename GraphicalModelType::FunctionIdentifier IdType;
   //add single side functions
   parser::ProgressPrinter pp1(image_.height()*image_.width(),1000,"adding 1. order factors ");
   
   const size_t shape[]={256};
   for(size_t y=0;y<image_.height();++y)
   for(size_t x=0;x<image_.width();++x){
      const size_t variableIndex[]={x+image_.width()*y};
      if(verbose_)pp1(variableIndex[0]);
      if(mask_(x,y)!=0){
         const size_t imgValue=static_cast<size_t>(image_(x,y));
         opengm::ExplicitFunction<ValueType> f(shape,shape+1);
         for(size_t l=0;l<256;++l){   
            size_t e=l>imgValue ? l-imgValue : imgValue-l;
            e*=e;
            f(&l)=static_cast<ValueType>(e);
         }
         IdType id=gm.addSharedFunction(f);
         gm.addFactor(id,variableIndex,variableIndex+1);
      }
    
      else{

      }
   }
   size_t counter=0;
   for(size_t y=0;y<image_.height();++y)
   for(size_t x=0;x<image_.width();++x){
      if(x<image_.width()-1)  ++counter;
      if(y<image_.height()-1) ++counter;
   }
   //high order function
   parser::ProgressPrinter pp2(counter,1000,"adding 2. order factors ");
   opengm::TruncatedSquaredDifferenceFunction<ValueType>  f(256,256,truncateAt_,lambda_);
   IdType id=gm.addFunction(f);
   counter=0;
   for(size_t y=0;y<image_.height();++y)
   for(size_t x=0;x<image_.width();++x){
      size_t variableIndex[2];
      variableIndex[0]=x+image_.width()*y;
      //right neighbour pixel
      if(x<image_.width()-1){
         if(verbose_)pp2(counter++);
         variableIndex[1]=x+1+image_.width()*y;
         gm.addFactor(id,variableIndex,variableIndex+2);
      }
      //down neighbour pixel
      if(y<image_.height()-1){
         if(verbose_)pp2(counter++);
         variableIndex[1]=x+image_.width()*(y+1);
         gm.addFactor(id,variableIndex,variableIndex+2);
      }
   }
}

template<class VALUE_TYPE,class PARAM_TYPE>
template<class ITERATOR>
void Denoise<VALUE_TYPE,PARAM_TYPE>::statesToImage(ITERATOR begin,ITERATOR end,ResultImageType & image )const{
   image.resize(image_.width(),image_.height());
   for(size_t y=0;y<image_.height();++y)
   for(size_t x=0;x<image_.width();++x){
      image(x,y)=begin[x+y*image_.width()];
   }
}

#endif	/* DENOISE_HXX */

