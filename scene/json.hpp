#pragma once

#include <simbotson.h>

#include <rt/err.hpp>

namespace bot::json {

void checkSIMBOTSONResult(simbotson::error_code err,
                         const char *file, int line,
                         const char *func_name);

#define REQ_JSON(err) ::bot::json::checkSIMBOTSONResult(err, __FILE__, \
    __LINE__, MADRONA_COMPILER_FUNCTION_NAME);

}
