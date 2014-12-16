/***********************************************************************
 * Author:       Joerg Hendrik Kappes
 * Date:         07.07.2014
 *
 * Description:
 * ------------
 * This class prociveds a simple IO for PGM-Files (8-bit gray-valued images)
 * It is header-only and require no additional libraries 
 *
 ************************************************************************/

#ifndef OPENGM_PGM_IMAGE_HXX
#define OPENGM_PGM_IMAGE_HXX

//#include <math.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>

namespace opengm{
   template <typename T>
   class PGMImage {
   public:
      PGMImage();
      PGMImage(int height, int width);
      PGMImage(const PGMImage<T>& i);

      // Access with boundary condition (boundary of image is contiued outside)
      const T& operator()(int row, int col) const;
      T& operator()(int row, int col); 
  
      // Direct access
//      const T& operator[](int row, int col) const {return data_[row*width_ + col];}
//      T& operator[](int row, int col)             {return data_[row*width_ + col];}
      T&  operator()(int i)                       {return data_[i]; }
      const T&  operator()(int i) const           {return data_[i]; } 
      int height() const                          {return height_; }
      int width() const                           {return width_; }

      // File-I/O
      void readPGM(const std::string filename);
      void writePGM(const std::string filename) const;

   private:
      std::vector<T> data_;
      int height_;
      int width_;
   };

   typedef PGMImage<unsigned char> PGMImage_uc;

   template <typename T>
   inline PGMImage<T>::PGMImage(): height_(0), width_(0){ }

   template <typename T>
   inline PGMImage<T>::PGMImage(int height, int width)
      : height_(height), width_(width)
   {
      data_.resize(height*width,0);
   }

   template <typename T>
   inline PGMImage<T>::PGMImage(const PGMImage<T>& i)  : data_(i.data_), height_(i.height_), width_(i.width_){ }


   template <typename T>
   inline const T& PGMImage<T>::operator()(int row, int col) const {
      if (row >= height_) row = height_-1;
      else if (row < 0)        row = 0;
      if (col >= width_)  col = width_-1;
      else if (col < 0)        col = 0;
      return data_[row*width_ + col];
   }

   template <typename T>
   inline T& PGMImage<T>::operator()(int row, int col) {
      if (row >= height_) row = height_-1;
      else if (row < 0)        row = 0;
      if (col >= width_)  col = width_-1;
      else if (col < 0)        col = 0;
      return data_[row*width_ + col];
   }

   template <typename T>
   inline void PGMImage<T>::readPGM(const std::string filename)
   {
      std::ifstream in(filename.c_str(),std::ios::in | std::ios::binary);
      std::string control;
      in >> control;
      if (control.compare(0,2,"P5")!=0) {
         std::cerr << "Magic number "<<control<<" does not match PGM" << std::endl;
         return;
      }
      int  max;
      in >> width_;
      in >> height_;
      data_.resize(width_*height_);
      in.ignore(1);
      in >> max;
      in.ignore(1);

      for (int i = 0; i < height_; ++i) {
         in.read((char *)(&data_[i*width_]), width_);
      }
      return;
   }

   template <typename T>
   inline void PGMImage<T>::writePGM(const std::string filename) const 
   {
      std::ofstream out(filename.c_str(), std::ios::out | std::ios::binary);
      out << "P5\n";
      out << width_ << " ";
      out << height_ << '\n';
      out << 255 << '\n';
      for (int i = 0; i < height_; ++i) {
         out.write((char *)(&data_[i*width_]), width_);
      }
   }


}
#endif
