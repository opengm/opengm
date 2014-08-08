#ifndef PHOTOMONTAGE_MERGING_HXX
#define	PHOTOMONTAGE_MERGING_HXX

#define A_INFINITY 100000

#include <vector>
#include <cmath>
#include <opengm/operations/adder.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/opengm.hxx>
#include <opengm/functions/squared_difference.hxx>
#include <opengm/functions/truncated_squared_difference.hxx>
//#include <opengm/functions/singlesidefunction.hxx>



template<class VALUE_TYPE>
class PhotomontageMerging{
   public:
      typedef VALUE_TYPE ValueType;
      typedef vigra::BRGBImage ImageType;
      typedef vigra::BRGBImage ResultImageType;     
      typedef opengm::GraphicalModel<
         ValueType,
         opengm::Adder,
         opengm::ExplicitFunction<ValueType> 
      > GraphicalModelType;
      typedef typename GraphicalModelType::LabelType LabelType;
      PhotomontageMerging(const std::vector<ImageType> &,const vigra::BRGBImage & ,const bool );
      void buildModel(GraphicalModelType &)const;
      template<class ITERATOR>
      void statesToImage(ITERATOR ,ITERATOR,ResultImageType & )const;
      const std::vector<LabelType> & startingPoint(void)const;
   private:
      bool isBlack(const size_t,const size_t ,const size_t)const;
      float pairEnergySimple(const size_t,const size_t ,const size_t,const size_t,const size_t,const size_t)const;
      size_t pairEnergyWithGradient(const size_t,const size_t ,const size_t,const size_t)const;
      std::vector<vigra::BRGBImage> images_;
      vigra::BRGBImage costImage_;
      size_t dimX_;
      size_t dimY_;
      size_t dimL_;
      std::vector<LabelType> startingPoint_;
      
      std::vector<vigra::FImage> xGradMags_;
      std::vector<vigra::FImage> yGradMags_;
      bool verbose_;
};

template<class VALUE_TYPE>
PhotomontageMerging<VALUE_TYPE>::PhotomontageMerging
(
   const std::vector< vigra::BRGBImage >  & images,
   const vigra::BRGBImage & costImage,
   const bool verbose
): images_(images),
   costImage_(costImage),
   dimX_(images_[0].width()),
   dimY_(images_[1].height()),
   dimL_(images_.size()),
   verbose_(verbose){
   startingPoint_.resize(dimX_*dimY_);
   for(size_t x=0;x<dimX_;++x){
      for(size_t y=0;y<dimY_;++y){
         for(size_t l=0;l<dimL_;++l){
            if(isBlack(x,y,l) ){
               startingPoint_[x+y*dimX_]=l;
               break;
            }
         }
      }
   }
   xGradMags_.resize(dimL_);
   yGradMags_.resize(dimL_);
   for(size_t l=0;l<dimL_;++l){
      xGradMags_[l].resize(dimX_,dimY_);
      yGradMags_[l].resize(dimX_,dimY_);
   }
   for(size_t x=0;x<dimX_;++x){
      for(size_t y=0;y<dimY_;++y){
         for(size_t l=0;l<dimL_;++l){
            //compute horizontal grad. mag.
            {
               if (x == 0 || x == dimX_ - 1 || y == dimY_ - 1)
                  xGradMags_[l](x,y)= -1.f;
               else if (isBlack(x - 1, y,l) || isBlack(x, y,l) || isBlack(x + 1, y,l) ||
                        isBlack(x - 1, y + 1,l) || isBlack(x, y + 1,l) || isBlack(x + 1, y + 1,l))
                  xGradMags_[l](x,y)= -1.f;
               else{
                  float sum = 0, d;
                  for (int c = 0; c < 3; ++c) { // iterate over colors
                     d =   float(images_[l](x-1,y)[c]) + 2.f*float(images_[l](x,y)[c]) + float(images_[l](x+1,y)[c]) -
                        float(images_[l](x-1,y+1)[c]) - 2.f * images_[l](x,y+1)[c] - float(images_[l](x+1,y+1)[c]);
                     d /= 3.f;
                     sum += d*d;
                  }
                  d = std::sqrt(sum);
                  //assert(finite(d) && !isnan(d));
                  xGradMags_[l](x,y)=d;
               }
            }
            // compute vertical grad. mag.
            {
               if (y == 0 || x == dimX_- 1 || y == dimY_ - 1)
                  yGradMags_[l](x,y)= -1.f;
               else if (isBlack(x , y -1,l) || isBlack(x, y,l) || isBlack(x, y+1,l) ||
                        isBlack(x + 1, y -1,l) || isBlack(x+1, y ,l) || isBlack(x + 1, y + 1,l))
                  yGradMags_[l](x,y)=  -1.f;
               else{
                  float sum = 0, d;
                  for (int c = 0; c < 3; ++c) { // iterate over colors
                     d = float(images_[l](x , y-1)[c]) + 2.f*float(images_[l](x, y)[c])+ float(images_[l](x , y+1)[c]) - 
                        float(images_[l](x + 1, y-1)[c]) - 2.f *float(images_[l](x+1, y )[c]) - float(images_[l](x + 1, y + 1)[c]);
                     d /= 3.f;
                     sum += d*d;
                  }
                  d = std::sqrt(sum);
                  //assert(finite(d) && !isnan(d));
                  //if (d!=0) printf("horiz %f\n",d);
                  yGradMags_[l](x,y)=d;
               }
            }
         }
      }
   }
}

template<class VALUE_TYPE>
inline const std::vector<typename PhotomontageMerging<VALUE_TYPE>::LabelType> &
PhotomontageMerging<VALUE_TYPE>::startingPoint()const{
   return startingPoint_;
}

template<class VALUE_TYPE>
inline bool
PhotomontageMerging<VALUE_TYPE>::isBlack
(
   const size_t x,
   const size_t y,
   const size_t l
)const{
   OPENGM_ASSERT(l<dimL_);
   OPENGM_ASSERT(x<dimX_);
   OPENGM_ASSERT(y<dimY_);
   return (images_[l](x,y)[0]==0 && images_[l](x,y)[1]==0  && images_[l](x,y)[2]==0);
}

template<class VALUE_TYPE>
inline float
PhotomontageMerging<VALUE_TYPE>::pairEnergySimple
(
   const size_t x1,
   const size_t y1,
   const size_t x2,
   const size_t y2,
   const size_t l1,
   const size_t l2
)const{
   float distP=0,distQ=0;
   for (int c=0; c<3; ++c) {
      const float dP = float(images_[l1](x1,y1)[c]) - float(images_[l2](x1,y1)[c]);
      const float dQ = float(images_[l1](x2,y2)[c]) - float(images_[l2](x2,y2)[c]);
      distP += dP*dP;
      distQ += dQ*dQ;
   }
   distP=std::sqrt(static_cast<float>(distP));
   distQ=std::sqrt(static_cast<float>(distQ));
   return (distP+distQ);
}

template<class VALUE_TYPE>
void PhotomontageMerging<VALUE_TYPE>::buildModel
(
   typename PhotomontageMerging<VALUE_TYPE>::GraphicalModelType& gm
)const{
   {
      std::vector<size_t> nos(dimX_*dimY_,dimL_);
      gm.assign(opengm::DiscreteSpace< > (nos.begin(),nos.end()));
   }
   typedef typename GraphicalModelType::FunctionIdentifier IdType;
   //add single side functions
   parser::ProgressPrinter pp1(dimY_*dimX_,1000,"adding 1. order factors ");
   
   for(size_t y=0,counter=0;y<dimY_;++y){
      for(size_t x=0;x<dimX_;++x,++counter){
         if(verbose_)pp1(counter);
         const size_t variableIndex[]={x+dimX_*y};
         opengm::ExplicitFunction<ValueType> f(&dimL_,&dimL_+1);
         for(size_t l=0;l<dimL_;++l){       
            if(costImage_(x,y)[0]== 255 ){
               f(l)=static_cast<ValueType>(0);
            }
            else{
               if( size_t(costImage_(x,y)[0]/size_t(20)) == l){
                  f(l)=static_cast<ValueType>(0);
                    
               }
               else{
                  f(l)=static_cast<ValueType>(A_INFINITY);
               }
            }
         }

         IdType id=gm.addSharedFunction(f);
         gm.addFactor(id,variableIndex,variableIndex+1);
      }
   }
   
   //high order functions and factors
   size_t numHighFactors=0;
   
   for(size_t y=0;y<images_[0].height();++y){
      for(size_t x=0;x<images_[0].width();++x ){
         size_t xx[]={1,0};
         size_t yy[]={0,1};
         for(size_t n=0;n<2;++n){
            if(x+xx[n]<images_[0].width() && y+yy[n]<images_[0].height()){
               ++numHighFactors;
            }
         }
      }
   }
   parser::ProgressPrinter pp2(numHighFactors,1000,"adding 2. order factors ");
   numHighFactors=0;
   for(size_t y=0;y<images_[0].height();++y){
      for(size_t x=0;x<images_[0].width();++x ){
         const size_t shape[]={images_.size(),images_.size()};
         opengm::ExplicitFunction<ValueType> f(shape,shape+2);
         size_t variableIndex[2];
         size_t xx[]={1,0};
         size_t yy[]={0,1};
         for(size_t n=0;n<2;++n){
            if(x+xx[n]<images_[0].width() && y+yy[n]<images_[0].height()){
               if(verbose_)pp2(numHighFactors);
               ++numHighFactors;
               
               bool allbad1=true;
               bool allbad2=true;
               for(size_t l11=0;l11<dimL_;++l11){
                  if( isBlack(x,y,l11)==false)
                     allbad1=false;
               }
               for(size_t l22=0;l22<dimL_;++l22){
                  if( isBlack(x+xx[n],y+yy[n],l22)==false)
                     allbad2=false;
               }
               for(size_t l1=0;l1<dimL_;++l1){
                  for(size_t l2=0;l2<dimL_;++l2){
                     if(l1==l2){
                        f(l1,l2)=0;
                        continue;
                     }
                     else{ 
                        f(l1,l2)=(int(pairEnergySimple(x,y,x+xx[n],y+yy[n],l1,l2)));
                        f(l1,l2)+=int(1);
                     }
                     float grad=0;
                     if (x != x+xx[n]) {
                        // vertical cut
                        float a =yGradMags_[l1](std::min(x, x+xx[n]),y);
                        float b = yGradMags_[l2](std::min(x, x+xx[n]),y);
                           
                        if (a <0 || b  <0)
                           grad = 1.f;
                        else 
                           grad = .5f*(a + b);
                     } 
                     else { 
                        // horizontal cut
                        float a = xGradMags_[l1](x,std::min(y, y+yy[n]));
                        float b = xGradMags_[l2](x,std::min(y, y+yy[n]));
                        if (a <0 || b  <0)
                           grad = 1.f;
                        else 
                           grad = .5f * (a + b);
                     }
                     if (grad == 0){
                        f(l1,l2) = A_INFINITY;
                     }
                     else {
                        if(grad<0){
                           throw opengm::RuntimeError("error \n");
                        }
                        f(l1,l2) = (int) (float(  int(f(l1,l2)) * 100) / grad);
                     }
                        
                     if(f(l1,l2) > A_INFINITY) 
                        f(l1,l2) = A_INFINITY;
                     std::cout.precision(8);
                  }
               }
               IdType id=false==false ? gm.addFunction(f):gm.addSharedFunction(f);
               variableIndex[0]=x+dimX_*y;
               variableIndex[1]=x+xx[n]+dimX_*(y+yy[n]);
               gm.addFactor(id,variableIndex,variableIndex+2);
            }
         }
      }
   }
}

template<class VALUE_TYPE>
template<class ITERATOR>
void PhotomontageMerging<VALUE_TYPE>::statesToImage(ITERATOR begin,ITERATOR end,ResultImageType & image )const{
   image.resize(dimX_,dimY_);
   for(size_t y=0;y<dimY_;++y)
   for(size_t x=0;x<dimX_;++x){
      image(x,y)=images_[begin[x+y*dimX_]](x,y);
   }
}



#endif	/* PHOTOMONTAGE_MERGING_HXX */

