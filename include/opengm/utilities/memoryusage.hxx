#pragma once
#ifndef OPENGM_MEMORYUSAGE_HXX
#define OPENGM_MEMORYUSAGE_HXX

#include <stdexcept>
#include "stdlib.h"
#include "stdio.h"
#include "string.h"

# if  ( defined(__APPLE__))
#   define OPENGM_MEMORY_MAC
# elif (defined(_WIN32) || defined(WIN32) || defined(__CYGWIN__) || defined(__MINGW32__) || defined(_WIN64))
#   define OPENGM_MEMORY_WINDOWS 
#   include <windows.h>
#   include <psapi.h>
#   undef min
#   undef max
# else
#   define OPENGM_MEMORY_LINUX
#   include <unistd.h>
#   include "sys/types.h"
#   include "sys/sysinfo.h"
# endif
namespace opengm {

   class MemoryUsage{
   public:
      double usedVirtualMem() const; 
      double usedPhysicalMem() const; 
      double usedVirtualMemMax() const; 
      double usedPhysicalMemMax() const;

      double usedSystemMem() const;
      double freeSystemMem() const;
      double systemMem() const;
   private:
      
      int parseLine(char* line) const {
         int i = strlen(line);
         while (*line < '0' || *line > '9') line++;
         line[i-3] = '\0';
         i = atoi(line);
         return i;
      }
   };
#ifdef  OPENGM_MEMORY_LINUX  
   double MemoryUsage::usedVirtualMem() const{
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
   double MemoryUsage::usedPhysicalMem() const{
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
   double MemoryUsage::usedVirtualMemMax() const{
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
   double MemoryUsage::usedPhysicalMemMax() const{
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
   double MemoryUsage::usedSystemMem() const{
      struct sysinfo memInfo;
      sysinfo(&memInfo);
      return static_cast<double>(memInfo.totalram - memInfo.freeram)*memInfo.mem_unit/1024.0;
   }
   double MemoryUsage::freeSystemMem() const{
      struct sysinfo memInfo;
      sysinfo(&memInfo);
      return static_cast<double>(memInfo.freeram)*memInfo.mem_unit/1024.0;
   }
   double MemoryUsage::systemMem() const{
      struct sysinfo memInfo;
      sysinfo(&memInfo);
      return static_cast<double>(memInfo.totalram)*memInfo.mem_unit/1024.0;
      //long pages = sysconf(_SC_PHYS_PAGES);
      //long page_size = sysconf(_SC_PAGE_SIZE);
      //return static_cast<double>(pages * page_size)/1024.0;   
   }
# elif defined(  OPENGM_MEMORY_WINDOWS )
   double MemoryUsage::usedPhysicalMem() const {
      PROCESS_MEMORY_COUNTERS_EX pmc;
      GetProcessMemoryInfo(GetCurrentProcess(),(PROCESS_MEMORY_COUNTERS*) &pmc, sizeof(pmc));
      return  static_cast<double>(pmc.WorkingSetSize)/1024.0;
   }
   double MemoryUsage::usedVirtualMem() const{
      PROCESS_MEMORY_COUNTERS_EX pmc;
      GetProcessMemoryInfo(GetCurrentProcess(),(PROCESS_MEMORY_COUNTERS*) &pmc, sizeof(pmc));
      return  static_cast<double>(pmc.PrivateUsage)/1024.0;
   }
   double MemoryUsage::usedPhysicalMemMax() const {
      PROCESS_MEMORY_COUNTERS_EX pmc;
      GetProcessMemoryInfo(GetCurrentProcess(),(PROCESS_MEMORY_COUNTERS*) &pmc, sizeof(pmc));
      return  static_cast<double>(pmc.PeakWorkingSetSize)/1024.0;
   }
   double MemoryUsage::usedVirtualMemMax() const{
      PROCESS_MEMORY_COUNTERS_EX pmc;
      GetProcessMemoryInfo(GetCurrentProcess(),(PROCESS_MEMORY_COUNTERS*) &pmc, sizeof(pmc));
      return  static_cast<double>(pmc.PeakPagefileUsage)/1024.0;
   } 
   double MemoryUsage::usedSystemMem() const{
      MEMORYSTATUSEX memInfo;
      memInfo.dwLength = sizeof(MEMORYSTATUSEX);
      GlobalMemoryStatusEx(&memInfo);
      return  static_cast<double>(memInfo.ullTotalPhys-memInfo.ullAvailPhys)/1024.0;
   } 
   double MemoryUsage::freeSystemMem() const{
      MEMORYSTATUSEX memInfo;
      memInfo.dwLength = sizeof(MEMORYSTATUSEX);
      GlobalMemoryStatusEx(&memInfo);
      return  static_cast<double>(memInfo.ullAvailPhys)/1024.0;
   }
   double MemoryUsage::systemMem() const{
      MEMORYSTATUSEX memInfo;
      memInfo.dwLength = sizeof(MEMORYSTATUSEX);
      GlobalMemoryStatusEx(&memInfo);
      return  static_cast<double>(memInfo.ullTotalPhys)/1024.0;
   }
#else
  double MemoryUsage::usedPhysicalMem() const {
      return std::numeric_limits<double>::quiet_NaN();
   }
   double MemoryUsage::usedVirtualMem() const{
      return std::numeric_limits<double>::quiet_NaN();
   }
   double MemoryUsage::usedPhysicalMemMax() const {
      return std::numeric_limits<double>::quiet_NaN();
   }
   double MemoryUsage::usedVirtualMemMax() const{
      return std::numeric_limits<double>::quiet_NaN();
   } 
   double MemoryUsage::usedSystemMem() const{
      return std::numeric_limits<double>::quiet_NaN();
   } 
   double MemoryUsage::freeSystemMem() const{
      return std::numeric_limits<double>::quiet_NaN();
   }
   double MemoryUsage::systemMem() const{
      return std::numeric_limits<double>::quiet_NaN();
   }
#endif
   
} // namespace opengm


#endif // OPENGM_MEMORYUSAGE_HXX
