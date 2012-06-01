Changes applied to MaxFlow-LIB version 3.02 to get it usable for openGM:

1. The whole library was set into namespace "maxflowLib" to avoid naming conflicts with openGM and other libraries.
2. Renamed defines in all header files by adding a leading "MAXFLOW_" to avoid include conflicts with openGM and other libraries.
3. Added "template class Graph<size_t,size_t,size_t>;" to file "instances.inc" to enable support for data type size_t.
4. Added file "maxflowlib.h" which handels local includes to fix problems with other libraries witch have identical names for header files.