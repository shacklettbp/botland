namespace bot {

void checkAssert(bool cond, const char *str, const char *file, int line,
                 const char *funcname)
{
#ifdef BOT_DISABLE_ASSERTS
  (void)cond;
  (void)str;
  (void)file;
  (void)line;
#else
  [[noreturn]] void failAssert(const char *str, const char *file, int line,
                               const char *funcname);

  if (!cond) [[unlikely]] {
    failAssert(str, file, line, funcname);
  }
#endif
}

}
