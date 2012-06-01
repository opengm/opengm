SET(CPLEX_ROOT_DIR "" CACHE PATH "CPLEX root directory")
	
FIND_PATH(CPLEX_INCLUDE_DIR
  ilcplex/cplex.h
  PATHS "C:/ILOG/CPLEX121/include"
        "/server/opt/cplex121/include"
	"~/usr/cplex121/include"
        ENV C_INCLUDE_PATH
        ENV C_PLUS_INCLUDE_PATH
        ENV INCLUDE_PATH
  HINTS ${CPLEX_ROOT_DIR}/include
  )

FIND_PATH(CONCERT_INCLUDE_DIR
  ilconcert/iloenv.h 
  PATHS "/server/opt/concert29/include"
        "~/usr/concert29/include"
        ENV C_INCLUDE_PATH
        ENV C_PLUS_INCLUDE_PATH
        ENV INCLUDE_PATH
  HINTS ${CPLEX_ROOT_DIR}/include
  )

FIND_LIBRARY(CPLEX_LIBRARY
  cplex ilocplex
  PATHS "C:/ILOG/CPLEX121/lib/msvc7/stat_mda"
        "/server/opt/cplex121/bin"
        "/server/opt/cplex121/lib/x86-64_debian4.0_4.1/static_pic"
	"~/usr/cplex121/lib/x86-64_debian4.0_4.1/static_pic"
        ENV LIBRARY_PATH
        ENV LD_LIBRARY_PATH
  HINTS ${CPLEX_ROOT_DIR}/bin
  )
FIND_LIBRARY(ILOCPLEX_LIBRARY
  ilocplex
  PATHS "C:/ILOG/CPLEX121/lib/msvc7/stat_mda"
        "/server/opt/cplex121/bin"
        "/server/opt/cplex121/lib/x86-64_debian4.0_4.1/static_pic"
	"~/usr/cplex121/lib/x86-64_debian4.0_4.1/static_pic"
        ENV LIBRARY_PATH
        ENV LD_LIBRARY_PATH
  HINTS ${CPLEX_ROOT_DIR}/bin
  )


FIND_LIBRARY(CONCERT_LIBRARY
  concert
  PATHS "/server/opt/concert29/lib/x86-64_debian4.0_4.1/static_pic"
        "~/usr/concert29/lib/x86-64_debian4.0_4.1/static_pic"
        ENV LIBRARY_PATH
        ENV LD_LIBRARY_PATH
  HINTS ${CPLEX_ROOT_DIR}/bin
  )
	
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(CPLEX DEFAULT_MSG CPLEX_LIBRARY CPLEX_INCLUDE_DIR)
	
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
  SET(CPLEX_INCLUDE_DIRS ${CPLEX_INCLUDE_DIR})
  SET(CPLEX_LIBRARIES ${CPLEX_LIBRARY})
  IF(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    SET(CPLEX_LIBRARIES "${CPLEX_LIBRARIES};m;pthread")
  ENDIF(CMAKE_SYSTEM_NAME STREQUAL "Linux")
ENDIF(CPLEX_FOUND)

MARK_AS_ADVANCED(CPLEX_LIBRARY CPLEX_INCLUDE_DIR CPLEX_BIN_DIR)

IF(CPLEX_FOUND)
  SET(LEMON_HAVE_LP TRUE)
  SET(LEMON_HAVE_MIP TRUE)
  SET(LEMON_HAVE_CPLEX TRUE)
ENDIF(CPLEX_FOUND)

