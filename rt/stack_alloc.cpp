#include <rt/stack_alloc.hpp>
#include <rt/utils.hpp>
#include <rt/err.hpp>

namespace bot {

static void * allocAligned(size_t num_bytes, size_t alignment)
{
#if defined(_LIBCPP_VERSION)
    return std::aligned_alloc(alignment, num_bytes);
#elif defined(BOT_CXX_MSVC)
    return _aligned_malloc(num_bytes, alignment);
#else
    static_assert(false);
#endif
}

static void deallocAligned(void *ptr)
{
#if defined(_LIBCPP_VERSION)
    free(ptr);
#elif defined(BOT_CXX_MSVC)
    _aligned_free(ptr);
#else
    static_assert(false);
#endif
}

StackAlloc::StackAlloc(u64 chunk_size)
    : first_chunk_(nullptr),
      cur_chunk_(first_chunk_),
      chunk_offset_((uint32_t)chunk_size),
      chunk_size_((uint32_t)chunk_size)
{
    chk(isPower2((uint64_t)chunk_size));
}

void StackAlloc::pop(AllocFrame frame)
{
    if ((uintptr_t)frame.ptr == chunk_size_) {
        release();
        return;
    }

    uintptr_t mask = chunk_size_ - 1;
    uintptr_t cur_offset = (uintptr_t)frame.ptr & mask;

    void *chunk_start = (char *)frame.ptr - cur_offset;
    auto *metadata = (ChunkMetadata *)chunk_start;

    ChunkMetadata *free_chunk = metadata->next;
    while (free_chunk != nullptr) {
        ChunkMetadata *next = free_chunk->next;
        deallocAligned(free_chunk);
        free_chunk = next;
    }

    metadata->next = nullptr;

    cur_chunk_ = (char *)metadata;
    chunk_offset_ = (uint32_t)cur_offset;
}

void StackAlloc::release()
{
    auto *metadata = (ChunkMetadata *)first_chunk_;
    while (metadata != nullptr) {
        auto *next = metadata->next;
        deallocAligned(metadata);
        metadata = next;
    }

    first_chunk_ = nullptr;
    cur_chunk_ = nullptr;
    chunk_offset_ = chunk_size_;
}

char * StackAlloc::newChunk(u64 num_bytes, u64 alignment)
{
    // FIXME: replace malloc here
    void *new_chunk = allocAligned(num_bytes, alignment);

    auto *metadata = (ChunkMetadata *)new_chunk;
    metadata->next = nullptr;

    return (char *)new_chunk;
}

}
