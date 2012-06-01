#pragma once
#ifndef OPENGM_MACROS
#define OPENGM_MACROS

#include <string>

#include "opengm/opengm.hxx"

#define STRING_TO_ENUM_1( CLASS_OF_ENUM,NAME_STRING,PARAM_NAME,ENTRY_1,ENUM_OUT) \
if       (NAME_STRING==std::string( #ENTRY_1)) ENUM_OUT=CLASS_OF_ENUM::ENTRY_1 \
else     (throw RuntimeError(std::string("Wrong value for parameter")+std::string(#PARAM_NAME))) 

#define STRING_TO_ENUM_2( CLASS_OF_ENUM,NAME_STRING,PARAM_NAME,ENTRY_1,ENTYR_2,ENUM_OUT) \
if       (NAME_STRING==std::string( #ENTRY_1)) ENUM_OUT=CLASS_OF_ENUM::ENTRY_1 \
else if  (NAME_STRING==std::string( #ENTRY_2)) ENUM_OUT=CLASS_OF_ENUM::ENTRY_2 \
else     (throw RuntimeError(std::string("Wrong value for parameter")+std::string(#PARAM_NAME))) 

#define STRING_TO_ENUM_3( CLASS_OF_ENUM,NAME_STRING,PARAM_NAME,ENTRY_1,ENTYR_2,ENTYR_3,ENUM_OUT) \
if       (NAME_STRING==std::string( #ENTRY_1)) {ENUM_OUT=CLASS_OF_ENUM::ENTRY_1;} \
else if  (NAME_STRING==std::string( #ENTRY_2)) {ENUM_OUT=CLASS_OF_ENUM::ENTRY_2;} \
else if  (NAME_STRING==std::string( #ENTRY_3)) {ENUM_OUT=CLASS_OF_ENUM::ENTRY_3;} \
else     {throw RuntimeError(std::string("unsupported value for parameter ")+std::string(#PARAM_NAME));}

#define STRING_TO_ENUM_4( CLASS_OF_ENUM,NAME_STRING,PARAM_NAME,ENTRY_1,ENTYR_2,ENTYR_3,ENTYR_4,ENUM_OUT) \
if       (NAME_STRING==std::string( #ENTRY_1)) {ENUM_OUT=CLASS_OF_ENUM::ENTRY_1;} \
else if  (NAME_STRING==std::string( #ENTRY_2)) {ENUM_OUT=CLASS_OF_ENUM::ENTRY_2;} \
else if  (NAME_STRING==std::string( #ENTRY_3)) {ENUM_OUT=CLASS_OF_ENUM::ENTRY_3;} \
else if  (NAME_STRING==std::string( #ENTRY_4)) {ENUM_OUT=CLASS_OF_ENUM::ENTRY_4;} \
else     {throw RuntimeError(std::string("unsupported value for parameter ")+std::string(#PARAM_NAME));}

#define STRING_TO_ENUM_5( CLASS_OF_ENUM,NAME_STRING,PARAM_NAME,ENTRY_1,ENTYR_2,ENTYR_3,ENTYR_4,ENTYR_5,ENUM_OUT) \
if       (NAME_STRING==std::string( #ENTRY_1)) {ENUM_OUT=CLASS_OF_ENUM::ENTRY_1;} \
else if  (NAME_STRING==std::string( #ENTRY_2)) {ENUM_OUT=CLASS_OF_ENUM::ENTRY_2;} \
else if  (NAME_STRING==std::string( #ENTRY_3)) {ENUM_OUT=CLASS_OF_ENUM::ENTRY_3;} \
else if  (NAME_STRING==std::string( #ENTRY_4)) {ENUM_OUT=CLASS_OF_ENUM::ENTRY_4;} \
else if  (NAME_STRING==std::string( #ENTRY_5)) {ENUM_OUT=CLASS_OF_ENUM::ENTRY_5;} \
else     {throw RuntimeError(std::string("unsupported value for parameter ")+std::string(#PARAM_NAME));}

#define STRING_TO_ENUM_6( CLASS_OF_ENUM,NAME_STRING,PARAM_NAME,ENTRY_1,ENTYR_2,ENTYR_3,ENTYR_4,ENTYR_5,ENTYR_6,ENUM_OUT) \
if       (NAME_STRING==std::string( #ENTRY_1)) {ENUM_OUT=CLASS_OF_ENUM::ENTRY_1;} \
else if  (NAME_STRING==std::string( #ENTRY_2)) {ENUM_OUT=CLASS_OF_ENUM::ENTRY_2;} \
else if  (NAME_STRING==std::string( #ENTRY_3)) {ENUM_OUT=CLASS_OF_ENUM::ENTRY_3;} \
else if  (NAME_STRING==std::string( #ENTRY_4)) {ENUM_OUT=CLASS_OF_ENUM::ENTRY_4;} \
else if  (NAME_STRING==std::string( #ENTRY_5)) {ENUM_OUT=CLASS_OF_ENUM::ENTRY_5;} \
else if  (NAME_STRING==std::string( #ENTRY_6)) {ENUM_OUT=CLASS_OF_ENUM::ENTRY_6;} \
else     {throw RuntimeError(std::string("unsupported value for parameter ")+std::string(#PARAM_NAME));}

#define STRING_TO_ENUM_7( CLASS_OF_ENUM,NAME_STRING,PARAM_NAME,ENTRY_1,ENTYR_2,ENTYR_3,ENTYR_4,ENTYR_5,ENTYR_6,ENTYR_7,ENUM_OUT) \
if       (NAME_STRING==std::string( #ENTRY_1)) {ENUM_OUT=CLASS_OF_ENUM::ENTRY_1;} \
else if  (NAME_STRING==std::string( #ENTRY_2)) {ENUM_OUT=CLASS_OF_ENUM::ENTRY_2;} \
else if  (NAME_STRING==std::string( #ENTRY_3)) {ENUM_OUT=CLASS_OF_ENUM::ENTRY_3;} \
else if  (NAME_STRING==std::string( #ENTRY_4)) {ENUM_OUT=CLASS_OF_ENUM::ENTRY_4;} \
else if  (NAME_STRING==std::string( #ENTRY_5)) {ENUM_OUT=CLASS_OF_ENUM::ENTRY_5;} \
else if  (NAME_STRING==std::string( #ENTRY_6)) {ENUM_OUT=CLASS_OF_ENUM::ENTRY_6;} \
else if  (NAME_STRING==std::string( #ENTRY_7)) {ENUM_OUT=CLASS_OF_ENUM::ENTRY_7;} \
else     {throw RuntimeError(std::string("unsupported value for parameter ")+std::string(#PARAM_NAME));}

#define STRING_TO_ENUM_8( CLASS_OF_ENUM,NAME_STRING,PARAM_NAME,ENTRY_1,ENTYR_2,ENTYR_3,ENTYR_4,ENTYR_5,ENTYR_6,ENTYR_7,ENTYR_8,ENUM_OUT) \
if       (NAME_STRING==std::string( #ENTRY_1)) {ENUM_OUT=CLASS_OF_ENUM::ENTRY_1;} \
else if  (NAME_STRING==std::string( #ENTRY_2)) {ENUM_OUT=CLASS_OF_ENUM::ENTRY_2;} \
else if  (NAME_STRING==std::string( #ENTRY_3)) {ENUM_OUT=CLASS_OF_ENUM::ENTRY_3;} \
else if  (NAME_STRING==std::string( #ENTRY_4)) {ENUM_OUT=CLASS_OF_ENUM::ENTRY_4;} \
else if  (NAME_STRING==std::string( #ENTRY_5)) {ENUM_OUT=CLASS_OF_ENUM::ENTRY_5;} \
else if  (NAME_STRING==std::string( #ENTRY_6)) {ENUM_OUT=CLASS_OF_ENUM::ENTRY_6;} \
else if  (NAME_STRING==std::string( #ENTRY_7)) {ENUM_OUT=CLASS_OF_ENUM::ENTRY_7;} \
else if  (NAME_STRING==std::string( #ENTRY_8)) {ENUM_OUT=CLASS_OF_ENUM::ENTRY_8;} \
else     {throw RuntimeError(std::string("unsupported value for parameter ")+std::string(#PARAM_NAME));}

#define STRING_TO_ENUM_9( CLASS_OF_ENUM,NAME_STRING,PARAM_NAME,ENTRY_1,ENTYR_2,ENTYR_3,ENTYR_4,ENTYR_5,ENTYR_6,ENTYR_7,ENTYR_8,ENTYR_9,ENUM_OUT) \
if       (NAME_STRING==std::string( #ENTRY_1)) {ENUM_OUT=CLASS_OF_ENUM::ENTRY_1;} \
else if  (NAME_STRING==std::string( #ENTRY_2)) {ENUM_OUT=CLASS_OF_ENUM::ENTRY_2;} \
else if  (NAME_STRING==std::string( #ENTRY_3)) {ENUM_OUT=CLASS_OF_ENUM::ENTRY_3;} \
else if  (NAME_STRING==std::string( #ENTRY_4)) {ENUM_OUT=CLASS_OF_ENUM::ENTRY_4;} \
else if  (NAME_STRING==std::string( #ENTRY_5)) {ENUM_OUT=CLASS_OF_ENUM::ENTRY_5;} \
else if  (NAME_STRING==std::string( #ENTRY_6)) {ENUM_OUT=CLASS_OF_ENUM::ENTRY_6;} \
else if  (NAME_STRING==std::string( #ENTRY_7)) {ENUM_OUT=CLASS_OF_ENUM::ENTRY_7;} \
else if  (NAME_STRING==std::string( #ENTRY_8)) {ENUM_OUT=CLASS_OF_ENUM::ENTRY_8;} \
else if  (NAME_STRING==std::string( #ENTRY_9)) {ENUM_OUT=CLASS_OF_ENUM::ENTRY_9;} \
else     {throw RuntimeError(std::string("unsupported value for parameter ")+std::string(#PARAM_NAME));}

#define STRING_TO_ENUM_10( CLASS_OF_ENUM,NAME_STRING,PARAM_NAME,ENTRY_1,ENTYR_2,ENTYR_3,ENTYR_4,ENTYR_5,ENTYR_6,ENTYR_7,ENTYR_8,ENTYR_9,ENTYR_10,ENUM_OUT) \
if       (NAME_STRING==std::string( #ENTRY_1)) {ENUM_OUT=CLASS_OF_ENUM::ENTRY_1;} \
else if  (NAME_STRING==std::string( #ENTRY_2)) {ENUM_OUT=CLASS_OF_ENUM::ENTRY_2;} \
else if  (NAME_STRING==std::string( #ENTRY_3)) {ENUM_OUT=CLASS_OF_ENUM::ENTRY_3;} \
else if  (NAME_STRING==std::string( #ENTRY_4)) {ENUM_OUT=CLASS_OF_ENUM::ENTRY_4;} \
else if  (NAME_STRING==std::string( #ENTRY_5)) {ENUM_OUT=CLASS_OF_ENUM::ENTRY_5;} \
else if  (NAME_STRING==std::string( #ENTRY_6)) {ENUM_OUT=CLASS_OF_ENUM::ENTRY_6;} \
else if  (NAME_STRING==std::string( #ENTRY_7)) {ENUM_OUT=CLASS_OF_ENUM::ENTRY_7;} \
else if  (NAME_STRING==std::string( #ENTRY_8)) {ENUM_OUT=CLASS_OF_ENUM::ENTRY_8;} \
else if  (NAME_STRING==std::string( #ENTRY_9)) {ENUM_OUT=CLASS_OF_ENUM::ENTRY_9;} \
else if  (NAME_STRING==std::string( #ENTRY_10)) {ENUM_OUT=CLASS_OF_ENUM::ENTRY_10;} \
else     {throw RuntimeError(std::string("unsupported value for parameter ")+std::string(#PARAM_NAME));}

#endif // #ifndef OPENGM_MACROS
