# This module finds cplex.
#
# User can give CPLEX_ROOT_DIR as a hint stored in the cmake cache.
#
# It sets the following variables:
#  CPLEX_FOUND              - Set to false, or undefined, if cplex isn't found.
#  CPLEX_INCLUDE_DIRS       - include directory
#  CPLEX_LIBRARIES          - library files

## config
set(CPLEX_ROOT_DIR "" CACHE PATH "CPLEX root directory")

if(WIN32)
  if(NOT CPLEX_WIN_VERSION)
    set(CPLEX_WIN_VERSION 125 CACHE STRING "Cplex version integer code. Necessary on Windows to guess root dir and to determine the library name, i.e. cplex125.lib")
  endif(NOT CPLEX_WIN_VERSION)

  if(NOT CPLEX_WIN_VS_VERSION)
    set(CPLEX_WIN_VS_VERSION 2010 CACHE STRING "Cplex Visual Studio version, for instance 2008 or 2010.")
  endif(NOT CPLEX_WIN_VS_VERSION)

  if(NOT CPLEX_WIN_LINKAGE)
    set(CPLEX_WIN_LINKAGE mda CACHE STRING "Cplex linkage variant on Windows. One of these: mda (dll, release), mdd (dll, debug), mta (static, release), mtd (static, debug)")
  endif(NOT CPLEX_WIN_LINKAGE)

  if(NOT CPLEX_WIN_BITNESS)
    set(CPLEX_WIN_BITNESS x64 CACHE STRING "On Windows: x86 or x64 (32bit resp. 64bit)")
  endif(NOT CPLEX_WIN_BITNESS)

  # now, generate platform string
  set(CPLEX_WIN_PLATFORM "${CPLEX_WIN_BITNESS}_windows_vs${CPLEX_WIN_VS_VERSION}/stat_${CPLEX_WIN_LINKAGE}")

else(WIN32)
  set(CPLEX_WIN_PLATFORM "")
endif(WIN32)

## cplex root dir guessing
# windows: trying to guess the root dir from a 
# env variable set by the cplex installer
set(WIN_ROOT_GUESS $ENV{CPLEX_STUDIO_DIR${CPLEX_WIN_VERSION}})

FIND_PATH(CPLEX_INCLUDE_DIR
  ilcplex/cplex.h
  HINTS ${CPLEX_ROOT_DIR}/cplex/include
        ${CPLEX_ROOT_DIR}/include
	${WIN_ROOT_GUESS}/cplex/include
  PATHS "C:/ILOG/CPLEX121"
        "/server/opt/cplex121"
	"~/usr/cplex121"
        ENV C_INCLUDE_PATH
        ENV C_PLUS_INCLUDE_PATH
        ENV INCLUDE_PATH
  )

FIND_PATH(CONCERT_INCLUDE_DIR
  ilconcert/iloenv.h 
  HINTS ${CPLEX_ROOT_DIR}/concert/include
        ${CPLEX_ROOT_DIR}/include
        ${WIN_ROOT_GUESS}/concert/include
  PATHS "/server/opt/concert29/include"
        "~/usr/concert29/include"
        ENV C_INCLUDE_PATH
        ENV C_PLUS_INCLUDE_PATH
        ENV INCLUDE_PATH
  )

FIND_LIBRARY(CPLEX_LIBRARY
  NAMES cplex cplex${CPLEX_WIN_VERSION}
  HINTS ${CPLEX_ROOT_DIR}/cplex/lib/${CPLEX_WIN_PLATFORM} #windows
        ${WIN_ROOT_GUESS}/cplex/lib/${CPLEX_WIN_PLATFORM} #windows
  PATHS "C:/ILOG/CPLEX121/lib/msvc7/stat_mda" #windows
        "/server/opt/cplex121/bin" #unix
        "/server/opt/cplex121/lib/x86-64_debian4.0_4.1/static_pic" #unix
	"~/usr/cplex121/lib/x86-64_debian4.0_4.1/static_pic" #unix
        ENV LIBRARY_PATH #unix
        ENV LD_LIBRARY_PATH #unix
  )

FIND_LIBRARY(ILOCPLEX_LIBRARY
  ilocplex
  HINTS ${CPLEX_ROOT_DIR}/cplex/lib/${CPLEX_WIN_PLATFORM} #windows
        ${WIN_ROOT_GUESS}/cplex/lib/${CPLEX_WIN_PLATFORM} #windows  
  PATHS "C:/ILOG/CPLEX121/lib/msvc7/stat_mda"
        "/server/opt/cplex121/bin"
        "/server/opt/cplex121/lib/x86-64_debian4.0_4.1/static_pic"
	"~/usr/cplex121/lib/x86-64_debian4.0_4.1/static_pic"
        ENV LIBRARY_PATH
        ENV LD_LIBRARY_PATH
  )

FIND_LIBRARY(CONCERT_LIBRARY
  concert
  HINTS ${CPLEX_ROOT_DIR}/concert/lib/${CPLEX_WIN_PLATFORM} #windows
        ${WIN_ROOT_GUESS}/concert/lib/${CPLEX_WIN_PLATFORM} #windows  
  PATHS "/server/opt/concert29/lib/x86-64_debian4.0_4.1/static_pic"
        "~/usr/concert29/lib/x86-64_debian4.0_4.1/static_pic"
        ENV LIBRARY_PATH
        ENV LD_LIBRARY_PATH
  )

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(CPLEX DEFAULT_MSG 
 CPLEX_LIBRARY CPLEX_INCLUDE_DIR ILOCPLEX_LIBRARY CONCERT_LIBRARY CONCERT_INCLUDE_DIR)
	
if(WIN32)
	FIND_PATH(CPLEX_BIN_DIR
	  cplex121.dll
	  PATHS "C:/ILOG/CPLEX91/bin/x86_win32"
	  )
else()
	FIND_PATH(CPLEX_BIN_DIR
	  libcplex121.so
	  PATHS "~/usr/cplex121/bin/x86-64_debian4.0_4.1"
	  )
endif()

IF(CPLEX_FOUND)
  SET(CPLEX_INCLUDE_DIRS ${CPLEX_INCLUDE_DIR} ${CONCERT_INCLUDE_DIR})
  SET(CPLEX_LIBRARIES ${CONCERT_LIBRARY} ${ILOCPLEX_LIBRARY} ${CPLEX_LIBRARY} )
  IF(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    SET(CPLEX_LIBRARIES "${CPLEX_LIBRARIES};m;pthread")
  ENDIF(CMAKE_SYSTEM_NAME STREQUAL "Linux")
ENDIF(CPLEX_FOUND)

MARK_AS_ADVANCED(CPLEX_LIBRARY CPLEX_INCLUDE_DIR ILOCPLEX_LIBRARY CONCERT_INCLUDE_DIR CONCERT_LIBRARY CPLEX_BIN_DIR)

IF(CPLEX_FOUND)
  SET(LEMON_HAVE_LP TRUE)
  SET(LEMON_HAVE_MIP TRUE)
  SET(LEMON_HAVE_CPLEX TRUE)
ENDIF(CPLEX_FOUND)

