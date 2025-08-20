#pragma once

#include <rt/types.hpp>
#include <rt/span.hpp>

namespace bot {

char * readBinaryFile(const char *path,
                      size_t buffer_alignment,
                      size_t *out_num_bytes);

}
