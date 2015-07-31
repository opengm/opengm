//******************************************************
//** Author: Joerg Hendrik Kappes
//** 
//** The MomoryInfo class provides a interface to access memory informations
//** for the current process or system-wide across diffrent platforms.
//** It provides interfaces to get the:
//**  - current physically used memory by the process
//**  - current virtual    used memory by the process
//**  - maximal physically used memory by the process so far
//**  - maximal virtual    used memory by the process so far
//**
//**  - used physical memory of the system
//**  - free physical memory of the system
//**  - total physical memory of the system 
//**
//******************************************************

#pragma once
#ifndef SYS_MEMORYINFO_HXX
#define SYS_MEMORYINFO_HXX

#include <iostream>
#include <stdexcept>
#include "stdlib.h"
#include "stdio.h"
#include "string.h"

// uncomment this line if U have problems with memorylogging -> this will disable it.
#define SYS_MEMORYINFO_ON

#if ( defined(__APPLE__) &&  defined(SYS_MEMORYINFO_ON) )
#   define SYS_MEMORYINFO_MAC
#   include <mach/vm_statistics.h>
#   include <mach/mach_types.h> 
#   include <mach/mach_init.h>
#   include <mach/mach_host.h>
#   include <mach/mach.h>
#   include <limits>
#elif (defined(SYS_MEMORYINFO_ON) && (defined(_WIN32) || defined(WIN32) || defined(__CYGWIN__) || defined(__MINGW32__) || defined(_WIN64)) )
#   define SYS_MEMORYINFO_WINDOWS 
#   include <windows.h>
#   include <psapi.h>
#   undef min
#   undef max
#elif (defined(SYS_MEMORYINFO_ON))
#   define SYS_MEMORYINFO_LINUX
#   include <unistd.h>
#   include "sys/types.h"
#   include "sys/sysinfo.h"
#else
#endif
namespace sys {

   class MemoryInfo{
   public:
      inline static double usedVirtualMem();
      inline static double usedPhysicalMem();
      inline static double usedVirtualMemMax();
      inline static double usedPhysicalMemMax();

      inline static double usedSystemMem();
      inline static double freeSystemMem();
      inline static double systemMem();
   private:
      
      inline static int parseLine(char* line){
         int i = strlen(line);
         while (*line < '0' || *line > '9') line++;
         line[i-3] = '\0';
         i = atoi(line);
         return i;
      }
   };

//**************************************
//**
//**   Implementation for Linux
//**
//*************************************
#ifdef  SYS_MEMORYINFO_LINUX  
   double MemoryInfo::usedVirtualMem(){
      FILE* file = fopen("/proc/self/status", "r");
      int result = -1;
      char line[128];
      while (fgets(line, 128, file) != NULL){
         if (strncmp(line, "VmSize:", 7) == 0){
            result = parseLine(line);
            break;
         }
      }
      fclose(file);
      return static_cast<double>(result);
   }
   double MemoryInfo::usedPhysicalMem(){
      FILE* file = fopen("/proc/self/status", "r");
      int result = -1;
      char line[128];
      while (fgets(line, 128, file) != NULL){
         if (strncmp(line, "VmRSS:", 6) == 0){
            result = parseLine(line);
            break;
         }
      }
      fclose(file);
      return static_cast<double>(result);
   } 
   double MemoryInfo::usedVirtualMemMax(){
      FILE* file = fopen("/proc/self/status", "r");
      int result = -1;
      char line[128];
      while (fgets(line, 128, file) != NULL){
         if (strncmp(line, "VmPeak:", 7) == 0){
            result = parseLine(line);
            break;
         }
      }
      fclose(file);
      return static_cast<double>(result);
   } 
   double MemoryInfo::usedPhysicalMemMax(){
      FILE* file = fopen("/proc/self/status", "r");
      int result = -1;
      char line[128];
      while (fgets(line, 128, file) != NULL){
         if (strncmp(line, "VmHWM:", 6) == 0){
            result = parseLine(line);
            break;
         }
      }
      fclose(file);
      return static_cast<double>(result);
   }  
   double MemoryInfo::usedSystemMem(){
      struct sysinfo memInfo;
      sysinfo(&memInfo);
      return static_cast<double>(memInfo.totalram - memInfo.freeram)*memInfo.mem_unit/1024.0;
   }
   double MemoryInfo::freeSystemMem(){
      struct sysinfo memInfo;
      sysinfo(&memInfo);
      return static_cast<double>(memInfo.freeram)*memInfo.mem_unit/1024.0;
   }
   double MemoryInfo::systemMem(){
      struct sysinfo memInfo;
      sysinfo(&memInfo);
      return static_cast<double>(memInfo.totalram)*memInfo.mem_unit/1024.0;
      //long pages = sysconf(_SC_PHYS_PAGES);
      //long page_size = sysconf(_SC_PAGE_SIZE);
      //return static_cast<double>(pages * page_size)/1024.0;   
   }

//**************************************************
//**
//**  Implementation for Windows
//**
//**************************************************
# elif defined(  SYS_MEMORYINFO_WINDOWS )
   double MemoryInfo::usedPhysicalMem(){
      PROCESS_MEMORY_COUNTERS_EX pmc;
      GetProcessMemoryInfo(GetCurrentProcess(),(PROCESS_MEMORY_COUNTERS*) &pmc, sizeof(pmc));
      return  static_cast<double>(pmc.WorkingSetSize)/1024.0;
   }
   double MemoryInfo::usedVirtualMem(){
      PROCESS_MEMORY_COUNTERS_EX pmc;
      GetProcessMemoryInfo(GetCurrentProcess(),(PROCESS_MEMORY_COUNTERS*) &pmc, sizeof(pmc));
      return  static_cast<double>(pmc.PrivateUsage)/1024.0;
   }
   double MemoryInfo::usedPhysicalMemMax(){
      PROCESS_MEMORY_COUNTERS_EX pmc;
      GetProcessMemoryInfo(GetCurrentProcess(),(PROCESS_MEMORY_COUNTERS*) &pmc, sizeof(pmc));
      return  static_cast<double>(pmc.PeakWorkingSetSize)/1024.0;
   }
   double MemoryInfo::usedVirtualMemMax(){
      PROCESS_MEMORY_COUNTERS_EX pmc;
      GetProcessMemoryInfo(GetCurrentProcess(),(PROCESS_MEMORY_COUNTERS*) &pmc, sizeof(pmc));
      return  static_cast<double>(pmc.PeakPagefileUsage)/1024.0;
   } 
   double MemoryInfo::usedSystemMem(){
      MEMORYSTATUSEX memInfo;
      memInfo.dwLength = sizeof(MEMORYSTATUSEX);
      GlobalMemoryStatusEx(&memInfo);
      return  static_cast<double>(memInfo.ullTotalPhys-memInfo.ullAvailPhys)/1024.0;
   } 
   double MemoryInfo::freeSystemMem(){
      MEMORYSTATUSEX memInfo;
      memInfo.dwLength = sizeof(MEMORYSTATUSEX);
      GlobalMemoryStatusEx(&memInfo);
      return  static_cast<double>(memInfo.ullAvailPhys)/1024.0;
   }
   double MemoryInfo::systemMem(){
      MEMORYSTATUSEX memInfo;
      memInfo.dwLength = sizeof(MEMORYSTATUSEX);
      GlobalMemoryStatusEx(&memInfo);
      return  static_cast<double>(memInfo.ullTotalPhys)/1024.0;
   }

//**************************************************
//**
//**  Implementation for MAC
//**
//************************************************** 
# elif defined(  SYS_MEMORYINFO_MAC )
  double MemoryInfo::usedPhysicalMem(){
    struct mach_task_basic_info info;
    mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
    if ( task_info( mach_task_self( ), MACH_TASK_BASIC_INFO,(task_info_t)&info, &infoCount ) != KERN_SUCCESS )
      return std::numeric_limits<double>::quiet_NaN();
    else
      return static_cast<double>(info.resident_size)/1024.0;
   }
   double MemoryInfo::usedVirtualMem(){
     struct mach_task_basic_info info;
     mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
     if ( task_info( mach_task_self( ), MACH_TASK_BASIC_INFO,(task_info_t)&info, &infoCount ) != KERN_SUCCESS )
       return std::numeric_limits<double>::quiet_NaN();
     else
       return static_cast<double>(info.virtual_size)/1024.0;
   }
   double MemoryInfo::usedPhysicalMemMax(){
     struct rusage rusage;
     getrusage( RUSAGE_SELF, &rusage );
     return static_cast<double>(rusage.ru_maxrss)/1024;
     //return std::numeric_limits<double>::quiet_NaN(); 
   }
   double MemoryInfo::usedVirtualMemMax(){
      return std::numeric_limits<double>::quiet_NaN();
   } 
   double MemoryInfo::usedSystemMem(){
      return std::numeric_limits<double>::quiet_NaN();
   } 
   double MemoryInfo::freeSystemMem(){
      return std::numeric_limits<double>::quiet_NaN();
   }
   double MemoryInfo::systemMem(){
      return std::numeric_limits<double>::quiet_NaN();
   }
#else
  double MemoryInfo::usedPhysicalMem(){
      return std::numeric_limits<double>::quiet_NaN();
   }
   double MemoryInfo::usedVirtualMem(){
      return std::numeric_limits<double>::quiet_NaN();
   }
   double MemoryInfo::usedPhysicalMemMax(){
      return std::numeric_limits<double>::quiet_NaN();
   }
   double MemoryInfo::usedVirtualMemMax(){
      return std::numeric_limits<double>::quiet_NaN();
   } 
   double MemoryInfo::usedSystemMem(){
      return std::numeric_limits<double>::quiet_NaN();
   } 
   double MemoryInfo::freeSystemMem(){
      return std::numeric_limits<double>::quiet_NaN();
   }
   double MemoryInfo::systemMem(){
      return std::numeric_limits<double>::quiet_NaN();
   }
#endif
   
} // namespace sys


#endif // SYS_MEMORYINFO_HXX
