#ifndef MYMATH_HXX
#define	MYMATH_HXX
#include <math.h>
#include <stdlib.h>

template<class T>
inline void randGauss2d
(	
	const T x0,
	const T y0,
	const T sigmaX,
	const T sigmaY,
	T & x, 
	T & y
)
{
	const T r1=T(rand())/RAND_MAX;
	const T r2=T(rand())/RAND_MAX;
	x=x0+sigmaX*sqrt(T(-2.0f)*log(r1))*cos(T(2.0f*3.14159265)*r2);
	y=y0+sigmaY*sqrt(T(-2.0f)*log(r1))*sin(T(2.0f*3.14159265)*r2);
}

template<class T>
inline void limitedRandGauss2d
(
	const T lx,
	const T hx,
	const T ly ,
	const T hy,
	const T x0,
	const T y0,
	const T sigmaX,
	const T sigmaY,
	T & x, 
	T & y
)
{
	do{
		randGauss2d(x0,y0,sigmaX,sigmaY,x,y);
	}
	while( (lx<x && x<hx && ly<y && y< hy)==false);
}

#endif	/* MYMATH_HXX */

