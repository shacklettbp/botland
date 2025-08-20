/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#pragma once

#include "macros.hpp"
#include "types.hpp"

#include <array>
#include <initializer_list>

namespace bot {

template <typename T>
class Span {
public:
  Span() : ptr(nullptr), n(0) {}

  Span(T *ptr, i64 num_elems)
    : ptr(ptr), n(num_elems)
  {}

  template <i64 N>
  Span(T (&arr)[N])
    : ptr(arr), n(N)
  {}

  template <typename U, i64 N>
  Span(const std::array<U, N> &arr)
      requires(std::is_same_v<std::remove_cv_t<T>, std::remove_cv_t<U>>)
    : ptr(arr.data()), n(N)
  {}

  // GCC correctly warns that the below constructor is dangerous, but it's
  // convenient as long as the Span doesn't outlive the current expression
#ifdef BOT_CXX_GCC
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Winit-list-lifetime"
#endif
  Span(std::initializer_list<std::remove_cv_t<T>> const && init BOT_LFBOUND)
      requires(std::is_const_v<T>)
    : ptr(init.begin()), n(init.size())
  {}
#ifdef BOT_CXX_GCC
#pragma GCC diagnostic pop
#endif

  template <typename U>
  Span(const U &u)
    : ptr(u.data()), n(u.size())
  {}

  constexpr T * data() const { return ptr; }

  constexpr i64 size() const { return n; }

  T & operator[](i64 idx) const { return ptr[idx]; }

  T * begin() const { return ptr; }
  T * end() const { return ptr + n; }

  T *ptr;
  i64 n;
};

}
