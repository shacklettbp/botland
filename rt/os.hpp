#pragma once

#include "types.hpp"
#include "rt.hpp"

namespace bot {

struct OSMemStats {
  u64 freeMem;
  u64 totalMem;
};

struct VirtualMemProperties {
  u64 pageSize;
};

struct CPUProperties {
  i32 numThreads;
};

CPUProperties osCPUProperties(); 
void osPinThread(i32 os_thread_idx);

OSMemStats osMemStats();
VirtualMemProperties virtualMemProperties();

void * virtualMemReserveRegion(u64 num_bytes);
void virtualMemReleaseRegion(void *ptr, u64 num_bytes);

void virtualMemCommitRegion(void *ptr, u64 offset, u64 num_bytes);
void virtualMemDecommitRegion(void *ptr, u64 offset, u64 num_bytes);

char * readFile(Runtime &rt, MemArena &arena, const char *path,
                u64 *num_bytes_out, u64 alignment = 1);

char * readFileAsString(Runtime &rt, MemArena &arena, const char *path,
                        u64 *num_bytes_out, u64 alignment = 1);

bool writeFile(const char *path, char *data, u64 num_bytes);

const char * getCodePath(Runtime &rt, void (*fn)(), u64 *path_len_out);

}
