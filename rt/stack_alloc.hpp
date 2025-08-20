#pragma once

#include <rt/types.hpp>
#include <rt/utils.hpp>

namespace bot {

struct AllocFrame {
    void *ptr;
};

class StackAlloc {
public:
    StackAlloc(u64 chunk_size = 32768);
    StackAlloc(const StackAlloc &) = delete;
    inline StackAlloc(StackAlloc &&o);
    inline ~StackAlloc();

    StackAlloc & operator=(const StackAlloc &) = delete;
    inline StackAlloc & operator=(StackAlloc &&o);

    inline AllocFrame push();
    void pop(AllocFrame frame);

    void release();

    inline void * alloc(u64 num_bytes, u64 alignment);

    template <typename T>
    T * alloc();

    template <typename T>
    T * allocN(u64 num_elems);

private:
    struct ChunkMetadata {
        ChunkMetadata *next;
    };

    static char * newChunk(u64 num_bytes, u64 alignment);

    char *first_chunk_;
    char *cur_chunk_;
    u64 chunk_offset_;
    u64 chunk_size_;
};

}

#include "stack_alloc.inl"
