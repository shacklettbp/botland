#include "os.hpp"

#include "macros.hpp"
#include "err.hpp"

#include <fstream>

#if defined(BOT_OS_LINUX)
#include <pthread.h>
#include <sys/sysinfo.h>
#elif defined(BOT_OS_MACOS)
#include <sys/sysctl.h>
#elif defined(BOT_OS_WINDOWS)
#endif

#if defined(BOT_OS_LINUX) or defined(BOT_OS_MACOS)
#include <sys/mman.h>
#include <unistd.h>
#include <dlfcn.h>
#elif defined(_WIN32)
#include <windows.h>
#endif

namespace bot {

CPUProperties osCPUProperties()
{
  auto getNumThreadsErr = []()
  {
    FATAL("Failed to get number of CPU threads from OS");
  };

  i32 num_threads;
#if defined(BOT_OS_MACOS)
  {
    int num_processors = sysconf(_SC_NPROCESSORS_ONLN);

    if (num_processors <= 0) {
      getNumThreadsErr();
    }

    num_threads = (i32)num_processors;
  }
#elif defined(BOT_OS_LINUX)
    cpu_set_t cpuset;
    pthread_getaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
    int num_active_threads = CPU_COUNT(&cpuset);
    if (num_active_threads <= 0) {
      getNumThreadsErr();
    }

    num_threads = (i32)num_active_threads;
#elif defined(BOT_OS_WINDOWS)
    SYSTEM_INFO sys_info;
    GetSystemInfo(&sys_info);
    num_threads = (i32)sys_info.dwNumberOfProcessors;
#else
    BOT_UNIMPLEMENTED();
#endif

  return CPUProperties {
    .numThreads = num_threads,
  };
}

void osPinThread([[maybe_unused]] i32 worker_id)
{
#ifdef BOT_OS_LINUX
  cpu_set_t cpu_set;
  pthread_getaffinity_np(pthread_self(), sizeof(cpu_set), &cpu_set);

  const int max_threads = CPU_COUNT(&cpu_set);

  if (worker_id > max_threads) [[unlikely]] {
    FATAL("Tried setting thread affinity to %d when %d is max",
      worker_id, max_threads);
  }

  cpu_set_t worker_set;
  CPU_ZERO(&worker_set);
  
  // This is needed in case there was already cpu masking via
  // a different call to setaffinity or via cgroup (SLURM)
  for (i32 thread_idx = 0, available_threads = 0;
       thread_idx < (i32)CPU_SETSIZE; thread_idx++) {
    if (CPU_ISSET(thread_idx, &cpu_set)) {
      if ((available_threads++) == worker_id) {
        CPU_SET(thread_idx, &worker_set);
        break;
      }
    }
  }

  int res = pthread_setaffinity_np(
    pthread_self(), sizeof(worker_set), &worker_set);

  if (res != 0) {
    FATAL("Failed to set thread affinity to %d", worker_id);
  }
#else
  (void)worker_id;
#endif
}


OSMemStats osMemStats()
{
#if defined(BOT_OS_LINUX)
  struct sysinfo info;
  if (sysinfo(&info) == -1) {
    FATAL("Failed to get system memory info");
  }

  return {
    .freeMem = (u64)info.totalram,
    .totalMem = (u64)info.totalram,
  };
#elif defined(BOT_OS_MACOS)
  int mib[] = { CTL_HW, HW_MEMSIZE };
  int64_t value = 0;
  size_t length = sizeof(value);
  
  if(sysctl(mib, 2, &value, &length, NULL, 0) == -1) {
    FATAL("Failed to get system memory info");
  }

  return {
    .freeMem = (u64)value,
    .totalMem = (u64)value,
  };
#elif defined(BOT_OS_WINDOWS)
  FATAL("Unimplemented")
#else
  FATAL("Unimplemented")
#endif
}

VirtualMemProperties virtualMemProperties()
{
#if defined(BOT_OS_LINUX) or defined(BOT_OS_MACOS)
  u32 size = sysconf(_SC_PAGESIZE);
  return {
    .pageSize = (u64)size,
  };
#elif defined(_WIN32)
  SYSTEM_INFO sys_info;
  GetSystemInfo(&sys_info);

  return {
    .pageSize = (u64)sys_info.dwPageSize,
  };
#else
  BOT_UNIMPLEMENTED();
#endif
}

void * virtualMemReserveRegion(u64 num_bytes)
{
#if defined(BOT_OS_LINUX) or defined(BOT_OS_MACOS)
#ifdef BOT_OS_LINUX
  constexpr int mmap_init_flags = MAP_PRIVATE | MAP_ANON | MAP_NORESERVE;
#elif defined(BOT_OS_MACOS)
  constexpr int mmap_init_flags = MAP_PRIVATE | MAP_ANON;
#endif

  void *base =
    mmap(nullptr, num_bytes, PROT_NONE, mmap_init_flags, -1, 0);

  if (base == MAP_FAILED) [[unlikely]] {
    FATAL("Failed to allocate %lu bytes of virtual address space",
          num_bytes);
  }
#elif _WIN32
  void *base = VirtualAlloc(nullptr, num_bytes, MEM_RESERVE,
                            PAGE_NOACCESS);

  if (base == nullptr) [[unlikely]] {
    FATAL("Failed to allocate %lu bytes of virtual address space",
          num_bytes);
  }
#else
  BOT_UNIMPLEMENTED();
#endif

  return base;
}

void virtualMemReleaseRegion(void *ptr, u64 num_bytes)
{
#if defined(BOT_OS_LINUX) or defined(BOT_OS_MACOS)
  munmap(ptr, num_bytes);
#elif defined(_WIN32)
  (void)num_bytes;
  VirtualFree(ptr, 0, MEM_RELEASE);
#else
  BOT_UNIMPLEMENTED();
#endif
}

void virtualMemCommitRegion(void *ptr, u64 offset, u64 num_bytes)
{
  void *start = (char *)ptr + offset;

#if defined(BOT_OS_LINUX) or defined(BOT_OS_MACOS)
  void *res = mmap(start, num_bytes, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANON | MAP_FIXED, -1, 0);
  bool fail = res == MAP_FAILED;
#elif defined(BOT_OS_WINDOWS)
  void *res = VirtualAlloc(start, num_bytes, MEM_COMMIT, PAGE_READWRITE);
  bool fail = res == nullptr;
#else
  BOT_UNIMPLEMENTED();
#endif

  if (fail) [[unlikely]] {
    FATAL("Failed to commit virtual memory @ %p, offset %zu, num_bytes %zu",
          ptr, offset, num_bytes);
  }
}

void virtualMemDecommitRegion(void *ptr, u64 offset, u64 num_bytes)
{
  void *start = (char *)ptr + offset;

#if defined(BOT_OS_LINUX) or defined(BOT_OS_MACOS)
  // FIXME MADV_FREE instead
  int res = madvise(start, num_bytes,
#ifdef BOT_OS_LINUX
                    MADV_DONTNEED);
#elif defined(BOT_OS_MACOS)
                    MADV_FREE);
#endif
  bool fail = res != 0;

#elif defined(BOT_OS_WINDOWS)
  int res = VirtualFree(start, num_bytes, MEM_DECOMMIT);
  bool fail = res == 0;
#else
  BOT_UNIMPLEMENTED();
#endif

  if (fail) {
    FATAL("Failed to decommit virtual memory @ %p, offset %zu, num_bytes %zu",
          start, offset, num_bytes);
  }
}

char * readFile(Runtime &rt, MemArena &arena, const char *path,
                u64 *num_bytes_out, u64 alignment)
{
  std::ifstream src_file(path, std::ios::binary);
  if (!src_file.is_open()) {
    return nullptr;
  }

  src_file.seekg(0, std::ios::end);
  u64 num_bytes = src_file.tellg();
  src_file.seekg(0, std::ios::beg);

  char *data = (char *)rt.arenaAlloc(arena, num_bytes, alignment);
  src_file.read(data, num_bytes);

  *num_bytes_out = num_bytes;
  return data;
}

char * readFileAsString(Runtime &rt, MemArena &arena, const char *path,
                        u64 *num_bytes_out, u64 alignment)
{
  std::ifstream src_file(path, std::ios::binary);
  if (!src_file.is_open()) {
    return nullptr;
  }

  src_file.seekg(0, std::ios::end);
  u64 num_bytes = src_file.tellg();
  src_file.seekg(0, std::ios::beg);

  char *data = (char *)rt.arenaAlloc(arena, num_bytes + 1, alignment);
  src_file.read(data, num_bytes);
  data[num_bytes] = 0;

  *num_bytes_out = num_bytes + 1;
  return data;
}

bool writeFile(const char *path, char *data, u64 num_bytes)
{
  std::ofstream out_file(path, std::ios::binary);
  if (!out_file.is_open()) {
    return false;
  }

  out_file.write(data, num_bytes);

  return true;
}

const char * getCodePath(Runtime &rt, void (*fn)(),
                         u64 *path_len_out)
{
#if defined(BOT_OS_WINDOWS)
  HMODULE h_module = nullptr;
  if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                          GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                          static_cast<LPCSTR>(fn), &h_mopdule)) {
    return nullptr;
  }

  char *path = nullptr;
  u64 buf_size = MAX_PATH;
  while (true) {
    path = rt.resultAllocN<char>(buf_size);

    DWORD path_size = GetModuleFileNameA(h_module, path, buf_size);

    if (path_size == 0) {
      return nullptr;
    }

    if (path_size != buf_size) {
      *path_len_out = (u64)path_size;
      break;
    }

    buf_size *= 2;
  }

  return path;
#elif defined(BOT_OS_LINUX) or defined(BOT_OS_MACOS)
  Dl_info dl_info;
  if (dladdr((void *)fn, &dl_info) == 0 || !dl_info.dli_fname) {
    return nullptr;
  }

  int path_len = strlen(dl_info.dli_fname);
  if (path_len == 0) {
    return nullptr;
  }

  char *path = rt.resultAllocN<char>(path_len + 1);
  memcpy(path, dl_info.dli_fname, path_len);
  path[path_len] = '\0';

  *path_len_out = path_len;
  return path;
#endif
}

}
