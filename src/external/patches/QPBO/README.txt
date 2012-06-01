Changes applied to QPBO-LIB version 1.3 to get it usable for openGM:

1. Added some typecasts from "const char*" to "char*" to fix compiler error.
2. Added some explicit typecasts to fix compiler error.
2. Renamed defines in header file "block.h" by adding a leading "QPBO_" to avoid include conflicts with openGM and other libraries.
3. The whole library was set into namespace "kolmogorov::qpbo" to avoid naming conflicts with openGM and other libraries.