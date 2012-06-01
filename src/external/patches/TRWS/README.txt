Changes applied to TRW-S-LIB version 1.3 to get it usable for openGM:

1. Added missing includes in file "example.cpp" and changed signature of "void main();" to "int main();" to fix compiler error.
2. removed some printfs.
3. Added some typecasts from "const char*" to "char*" to fix compiler error.
4. Modified file "instances.inc" //TODO WHY???