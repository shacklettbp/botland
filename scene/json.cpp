#if 0
#include "json.hpp"
#include <brt/err.hpp>

namespace madrona::json {

void checkSIMBOTSONResult(
    simbotson::error_code err,
    const char *file,
    int line,
    const char *func_name)
{
  if (err) {
    FATAL(file, line, func_name, "Failed to parse JSON: %s",
        simbotson::error_message(err));
  }
}

}
#endif
