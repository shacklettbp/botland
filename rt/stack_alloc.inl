namespace bot {

StackAlloc::StackAlloc(StackAlloc &&o)
  : first_chunk_(o.first_chunk_),
    cur_chunk_(o.cur_chunk_),
    chunk_offset_(o.chunk_offset_),
    chunk_size_(o.chunk_size_)
{
    o.first_chunk_ = nullptr;
    o.cur_chunk_ = nullptr;
    o.chunk_offset_ = chunk_size_;
}

StackAlloc::~StackAlloc()
{
    release();
}

StackAlloc & StackAlloc::operator=(StackAlloc &&o)
{
    first_chunk_ = o.first_chunk_;
    cur_chunk_ = o.cur_chunk_;
    chunk_offset_ = o.chunk_offset_;
    chunk_size_ = o.chunk_size_;

    o.first_chunk_ = nullptr;
    o.cur_chunk_ = nullptr;
    o.chunk_offset_ = chunk_size_;

    return *this;
}

AllocFrame StackAlloc::push()
{
    return AllocFrame {
        cur_chunk_ + chunk_offset_,
    };
}

void * StackAlloc::alloc(u64 num_bytes, u64 alignment)
{
    u64 alloc_offset =
        (i64)roundToAlignment(chunk_offset_, (u64)alignment);
    u64 new_offset = alloc_offset + num_bytes;

    if (new_offset <= chunk_size_) [[likely]] {
        void *start = cur_chunk_ + alloc_offset;
        chunk_offset_ = new_offset;

        return start;
    }

    alloc_offset =
        (u64)roundToAlignment(sizeof(ChunkMetadata), (u64)alignment);
    new_offset = alloc_offset + num_bytes;

    // FIXME: it would be good to handle oversized chunks differently,
    // round up to chunk_size_ multiple and still use leftover bytes?
    u64 alloc_size;
    if (new_offset > chunk_size_) [[unlikely]] {
        alloc_size = roundToAlignment(new_offset, chunk_size_);
        new_offset = chunk_size_;
    } else {
        alloc_size = chunk_size_;
    }

    char *new_chunk = StackAlloc::newChunk(
        alloc_size, std::max(alignment, chunk_size_));

    if (first_chunk_ == nullptr) {
      first_chunk_ = new_chunk;
    } else {
      auto *cur_metadata = (ChunkMetadata *)cur_chunk_;
      cur_metadata->next = (ChunkMetadata *)new_chunk;
    }

    cur_chunk_ = new_chunk;
    chunk_offset_ = new_offset;

    return new_chunk + alloc_offset;
}

template <typename T>
T * StackAlloc::alloc()
{
    return (T *)alloc(sizeof(T), alignof(T));
}

template <typename T>
T * StackAlloc::allocN(u64 num_elems)
{
    return (T *)alloc(sizeof(T) * num_elems, alignof(T));
}

}
