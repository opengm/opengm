#ifndef OPENGM_UNSIGNED_INTEGER_POW_HXX_
#define OPENGM_UNSIGNED_INTEGER_POW_HXX_

namespace opengm {

size_t unsignedIntegerPow(size_t base, size_t exponent) {
   if(base == 1) {
      return 1;
   } else if((base == 0) && (exponent > 0)) {
      return 0;
   } else {
      // exponentiation by squaring
      size_t result = 1;
      while(exponent) {
         // check if exponent is odd
         if(exponent % 2 == 1) {
            result *= base;
         }

         // halve exponent
         exponent /= 2;

         // square base
         base *= base;
      }
      return result;
   }
}

} // namespace opengm

/*! \file unsigned_integer_pow.hxx
 *  \brief Provides implementation for the power function of unsigned integer
 *         values.
 */

/*! \fn size_t opengm::unsignedIntegerPow(size_t base, size_t exponent)
 *  \brief Unsigned integer power function.
 *
 *  \param[in] base The base for the power function computation.
 *  \param[in] exponent The exponent for the power function computation.
 *
 *  \return Returns the value of the power function base^exponent.
 *
 *  \warning Might overflow for even small values of base and exponent due to
 *           the limited capacities of size_t.
 */


#endif /* OPENGM_UNSIGNED_INTEGER_POW_HXX_ */
