Changes applied to ConicBundle-LIB version 0.3.11 to get it usable for openGM:

1. Changed Makefile to build release version of ConicBundle-LIB.
2. Removed echoflags -e in Makefile as option "-e" was written into "include/CBconfig.hxx" instead of being regarded as a parameter for echo.